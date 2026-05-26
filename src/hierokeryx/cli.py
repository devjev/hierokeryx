"""Typer CLI for hierokeryx: schema, extract, resolve, pipeline, review, inspect."""

from __future__ import annotations

import glob as glob_mod
import json
import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from hierokeryx import pipeline
from hierokeryx.confidence import route_for_review
from hierokeryx.eval import evaluate, load_gold, sweep_thresholds
from hierokeryx.eval.report import EvalReport
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.anthropic_client import AnthropicClient
from hierokeryx.llm.protocol import LLMClient
from hierokeryx.llm.standard_gateway_client import StandardGatewayClient
from hierokeryx.models import Document
from hierokeryx.resolve.crossdoc import resolve_crossdoc
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.review.jsonl import write_review_dir
from hierokeryx.review.lint import lint_review_dir, lint_review_file
from hierokeryx.schema import DEFAULT_SCHEMA, load_schema, save_schema

app = typer.Typer(
    name="hkx",
    help="hierokeryx — entity extraction and resolution with GLiNER + LLM coref and file-based HITL.",
    no_args_is_help=True,
    add_completion=False,
)
schema_app = typer.Typer(no_args_is_help=True, help="Manage entity schemas.")
review_app = typer.Typer(no_args_is_help=True, help="Export, lint, and re-import HITL review files.")
app.add_typer(schema_app, name="schema")
app.add_typer(review_app, name="review")

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _make_llm_client(provider: str) -> LLMClient:
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "gateway":
        return StandardGatewayClient()
    raise typer.BadParameter(f"Unknown --provider {provider!r}; use 'anthropic' or 'gateway'.")


def _read_documents(input_path: str) -> list[Document]:
    """Read documents from a file, directory, or glob pattern."""
    p = Path(input_path)
    paths: list[Path] = []
    if p.is_dir():
        paths = sorted([q for q in p.rglob("*") if q.is_file() and q.suffix in {".txt", ".md"}])
    elif p.is_file():
        paths = [p]
    else:
        matched = sorted(glob_mod.glob(input_path, recursive=True))
        paths = [Path(m) for m in matched if Path(m).is_file()]
    if not paths:
        raise typer.BadParameter(f"No documents matched {input_path!r}")
    docs = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        docs.append(Document(id=path.stem, text=text, source=str(path)))
    return docs


# ---- schema commands -------------------------------------------------------


@schema_app.command("init")
def schema_init(
    out: Path = typer.Option(Path("schema.yaml"), "--out", "-o", help="Where to write the schema."),
    force: bool = typer.Option(False, "--force", help="Overwrite if the file exists."),
) -> None:
    """Write a starter EntitySchema (Person / Organization / Location)."""
    if out.exists() and not force:
        console.print(f"[yellow]Refusing to overwrite {out} (use --force).[/yellow]")
        raise typer.Exit(1)
    save_schema(DEFAULT_SCHEMA, out)
    console.print(f"Wrote starter schema to [green]{out}[/green]")


@schema_app.command("validate")
def schema_validate(
    path: Path = typer.Argument(..., exists=True, help="YAML or JSON schema file."),
) -> None:
    """Validate a schema file."""
    schema = load_schema(path)
    console.print(f"OK: [green]{path}[/green] — {len(schema.types)} type(s), fingerprint {schema.fingerprint()}")
    for t in schema.types:
        console.print(f"  - [cyan]{t.name}[/cyan]: {t.description}")


# ---- extract / resolve / pipeline ------------------------------------------


@app.command()
def extract(
    input_path: str = typer.Argument(..., help="File, directory, or glob of .txt/.md documents."),
    schema_path: Path = typer.Option(..., "--schema", "-s", exists=True),
    out: Path = typer.Option(..., "--out", "-o", help="Workdir to write extractions to."),
    gliner_model: str = typer.Option("urchade/gliner_large-v2.5", "--gliner-model"),
    threshold: float = typer.Option(0.4, "--threshold"),
    provider: str = typer.Option("gateway", "--provider", help="LLM provider: 'gateway' (default) or 'anthropic'."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run GLiNER + within-doc coref, writing extractions to `out/extractions/`."""
    _setup_logging(verbose)
    schema = load_schema(schema_path)
    docs = _read_documents(input_path)
    extractor = GLiNERExtractor(model_id=gliner_model, threshold=threshold)
    llm = _make_llm_client(provider)

    out.mkdir(parents=True, exist_ok=True)
    (out / "extractions").mkdir(exist_ok=True)
    save_schema(schema, out / "schema.yaml")

    n_entities = 0
    for doc in docs:
        result = pipeline.run_one(doc, schema, extractor=extractor, llm_client=llm)
        path = out / "extractions" / f"{doc.id}.json"
        pipeline.save_extraction(result, path)
        n_entities += len(result.entities)
        console.print(f"  [cyan]{doc.id}[/cyan]: {len(result.entities)} entities → {path.name}")

    console.print(f"\n[green]Done.[/green] {len(docs)} docs, {n_entities} entities.")


@app.command()
def resolve(
    workdir: Path = typer.Argument(..., exists=True, file_okay=False),
    merge_threshold: float = typer.Option(0.82, "--threshold"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM tie-break on borderline merges."),
    provider: str = typer.Option("gateway", "--provider", help="LLM provider: 'gateway' (default) or 'anthropic'."),
    against: Path | None = typer.Option(
        None,
        "--against",
        help="Resolve incrementally against an existing workdir's registry + centroids.",
        exists=True,
        file_okay=False,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Cluster entities across documents already in `workdir/extractions/`."""
    _setup_logging(verbose)
    schema = load_schema(workdir / "schema.yaml")
    results = pipeline.load_extractions_dir(workdir / "extractions")
    llm = None if no_llm else _make_llm_client(provider)
    embedder = SentenceTransformerEmbedder()

    if against is not None:
        existing_registry = pipeline.load_registry(against / "registry.json")
        existing_centroids = pipeline.load_centroids(against)
        updated, registry, centroids = pipeline.resolve_incremental(
            results,
            schema,
            existing_registry=existing_registry,
            existing_centroids=existing_centroids,
            llm_client=llm,
            embedder=embedder,
            merge_threshold=merge_threshold,
        )
    else:
        updated, registry = resolve_crossdoc(
            results,
            schema,
            llm_client=llm,
            embedder=embedder,
            merge_threshold=merge_threshold,
        )
        centroids = pipeline.compute_centroids(updated, embedder)
    for result in updated:
        pipeline.save_extraction(result, workdir / "extractions" / f"{result.document.id}.json")
    pipeline.save_registry(registry, workdir / "registry.json")
    pipeline.save_centroids(workdir, centroids)
    console.print(f"[green]Resolved[/green]: {len(registry.clusters)} cross-doc cluster(s)")


@app.command("resolve-centroids-rebuild")
def resolve_centroids_rebuild(
    workdir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Recompute `registry_embeddings.npz` from the workdir's extractions.

    Use to retrofit a workdir that pre-dates centroid persistence, so it can
    be the target of `hkx resolve --against`.
    """
    results = pipeline.load_extractions_dir(workdir / "extractions")
    embedder = SentenceTransformerEmbedder()
    centroids = pipeline.compute_centroids(results, embedder)
    pipeline.save_centroids(workdir, centroids)
    console.print(
        f"[green]Wrote[/green] centroid sidecar for {len(centroids.cluster_ids)} cluster(s)"
    )


@app.command(name="pipeline")
def pipeline_cmd(
    input_path: str = typer.Argument(..., help="File, directory, or glob of documents."),
    schema_path: Path = typer.Option(..., "--schema", "-s", exists=True),
    out: Path = typer.Option(..., "--out", "-o", help="Workdir for all artifacts."),
    review_threshold: float = typer.Option(0.7, "--review-threshold"),
    merge_threshold: float = typer.Option(0.82, "--merge-threshold"),
    gliner_model: str = typer.Option("urchade/gliner_large-v2.5", "--gliner-model"),
    no_llm_tiebreak: bool = typer.Option(False, "--no-llm-tiebreak"),
    only_flagged_review: bool = typer.Option(True, "--only-flagged/--all-for-review"),
    provider: str = typer.Option("gateway", "--provider", help="LLM provider: 'gateway' (default) or 'anthropic'."),
    against: Path | None = typer.Option(
        None,
        "--against",
        help="Resolve incrementally against an existing workdir's registry + centroids.",
        exists=True,
        file_okay=False,
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """End-to-end pipeline: extract, coref, cross-doc resolve, write HITL review files."""
    _setup_logging(verbose)
    schema = load_schema(schema_path)
    docs = _read_documents(input_path)
    extractor = GLiNERExtractor(model_id=gliner_model)
    llm = _make_llm_client(provider)

    run = pipeline.run(
        documents=docs,
        schema=schema,
        workdir=out,
        extractor=extractor,
        llm_client=llm,
        embedder=SentenceTransformerEmbedder(),
        review_threshold=review_threshold,
        merge_threshold=merge_threshold,
        only_flagged_review=only_flagged_review,
        incremental_from=against,
    )

    console.print()
    console.print(f"[green]Pipeline complete[/green] → {out}")
    console.print(f"  documents: {len(docs)}")
    console.print(f"  entities:  {sum(len(r.entities) for r in run.extraction_results)}")
    console.print(f"  clusters:  {len(run.registry.clusters)}")
    console.print(f"  flagged:   {len(run.flagged)} ({len(run.review_paths)} review file(s))")


# ---- review commands -------------------------------------------------------


@review_app.command("export")
def review_export(
    workdir: Path = typer.Argument(..., exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out", "-o"),
    only_flagged: bool = typer.Option(True, "--only-flagged/--all"),
    review_threshold: float = typer.Option(0.7, "--threshold"),
) -> None:
    """Write JSONL review files from a workdir's current extractions."""
    results = pipeline.load_extractions_dir(workdir / "extractions")
    entities = [e for r in results for e in r.entities]
    flagged = route_for_review(entities, cluster_threshold=review_threshold)
    paths = write_review_dir(results, out, flagged=flagged, only_flagged=only_flagged)
    console.print(f"Wrote [green]{len(paths)}[/green] review file(s) under {out}")


@review_app.command("lint")
def review_lint(
    path: Path = typer.Argument(..., exists=True),
    workdir: Path | None = typer.Option(None, "--workdir", help="Use original docs and schema for full validation."),
) -> None:
    """Validate review files. Catches span text mismatches, unknown types, etc."""
    if workdir is not None:
        schema = load_schema(workdir / "schema.yaml")
        extractions = {r.document.id: r for r in pipeline.load_extractions_dir(workdir / "extractions")}
        documents = {r.document.id: r.document for r in extractions.values()}
    else:
        schema = None
        documents = None
        extractions = None

    if path.is_file():
        errors = lint_review_file(
            path,
            document=documents.get(path.stem) if documents else None,
            schema=schema,
            original=extractions.get(path.stem) if extractions else None,
        )
        if errors:
            for e in errors:
                console.print(f"[red]ERROR[/red] {e}")
            raise typer.Exit(1)
        console.print(f"[green]OK[/green] {path}")
    else:
        all_errors = lint_review_dir(
            path,
            documents=documents,
            schema=schema,
            extractions=extractions,
        )
        failed = {k: v for k, v in all_errors.items() if v}
        for _doc_id, errs in failed.items():
            for err in errs:
                console.print(f"[red]ERROR[/red] {err}")
        if failed:
            console.print(f"[red]{sum(len(v) for v in failed.values())} error(s) across {len(failed)} file(s)[/red]")
            raise typer.Exit(1)
        console.print(f"[green]OK[/green] {len(all_errors)} file(s)")


@review_app.command("import")
def review_import(
    review_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    workdir: Path = typer.Option(..., "--workdir", exists=True, file_okay=False),
) -> None:
    """Replay edited review JSONL files back into the workdir's extractions."""
    updated = pipeline.import_reviewed(workdir, review_dir)
    console.print(f"Replayed reviews for [green]{len(updated)}[/green] document(s)")


# ---- eval ------------------------------------------------------------------


@app.command()
def eval(
    workdir: Path = typer.Argument(..., exists=True, file_okay=False),
    gold_path: Path = typer.Option(..., "--gold", "-g", exists=True, dir_okay=False),
    sweep: bool = typer.Option(False, "--sweep", help="Sweep merge/borderline thresholds."),
    json_out: Path | None = typer.Option(None, "--json-out", help="Write full report(s) as JSON."),
) -> None:
    """Score a workdir's cluster assignments against a JSONL gold file."""
    results = pipeline.load_extractions_dir(workdir / "extractions")
    gold = load_gold(gold_path)

    if sweep:
        schema = load_schema(workdir / "schema.yaml")
        reports = sweep_thresholds(results, gold, schema)
        if not reports:
            console.print("[yellow]No scorable entities in extraction results.[/yellow]")
            raise typer.Exit(1)
        _render_sweep(reports)
        if json_out:
            json_out.write_text(
                json.dumps([r.to_dict() for r in reports], indent=2) + "\n",
                encoding="utf-8",
            )
        return

    report = evaluate(results, gold)
    _render_report(report)
    if json_out:
        json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")


def _render_report(report: EvalReport) -> None:
    table = Table(title="Eval report")
    table.add_column("metric")
    table.add_column("precision", justify="right")
    table.add_column("recall", justify="right")
    table.add_column("f1", justify="right")
    table.add_row(
        "pairwise",
        f"{report.pairwise.precision:.3f}",
        f"{report.pairwise.recall:.3f}",
        f"{report.pairwise.f1:.3f}",
    )
    table.add_row(
        "bcubed",
        f"{report.bcubed.precision:.3f}",
        f"{report.bcubed.recall:.3f}",
        f"{report.bcubed.f1:.3f}",
    )
    console.print(table)
    console.print(
        f"scored {report.n_entities_scored} entities — "
        f"system clusters: {report.n_clusters_system}, "
        f"gold clusters: {report.n_clusters_gold}"
    )


def _render_sweep(reports: list[EvalReport]) -> None:
    table = Table(title="Threshold sweep (pairwise F1 / BCubed F1)")
    table.add_column("merge", justify="right")
    table.add_column("borderline", justify="right")
    table.add_column("pairwise_f1", justify="right")
    table.add_column("bcubed_f1", justify="right")
    for r in reports:
        table.add_row(
            f"{r.config.get('merge_threshold', 0):.2f}",
            f"{r.config.get('borderline_threshold', 0):.2f}",
            f"{r.pairwise.f1:.3f}",
            f"{r.bcubed.f1:.3f}",
        )
    console.print(table)
    best = max(reports, key=lambda r: r.pairwise.f1)
    console.print(
        f"\n[green]Best by pairwise F1:[/green] merge={best.config.get('merge_threshold')}, "
        f"borderline={best.config.get('borderline_threshold')} → "
        f"pairwise_f1={best.pairwise.f1:.3f}, bcubed_f1={best.bcubed.f1:.3f}"
    )


# ---- inspect ---------------------------------------------------------------


@app.command()
def inspect(
    workdir: Path = typer.Argument(..., exists=True, file_okay=False),
) -> None:
    """Pretty-print a summary of a workdir."""
    schema_path = workdir / "schema.yaml"
    manifest_path = workdir / "manifest.json"
    extractions_dir = workdir / "extractions"

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        console.print(f"[bold]Run at[/bold]: {manifest.get('created_at')}")
        console.print(f"  schema fingerprint: {manifest.get('schema_fingerprint')}")
        console.print(f"  documents: {manifest.get('n_documents')}")
        console.print(f"  entities:  {manifest.get('n_entities')}")
        console.print(f"  flagged:   {manifest.get('n_flagged')}")

    if not extractions_dir.exists():
        console.print("[yellow]No extractions found.[/yellow]")
        return

    results = pipeline.load_extractions_dir(extractions_dir)
    table = Table(title="Top entities by mention count")
    table.add_column("doc")
    table.add_column("entity")
    table.add_column("type")
    table.add_column("mentions", justify="right")
    table.add_column("confidence", justify="right")
    table.add_column("cluster")

    rows = []
    for r in results:
        for e in r.entities:
            rows.append((r.document.id, e.canonical, e.type, len(e.mentions), e.confidence, e.cluster_id or "-"))
    rows.sort(key=lambda x: (-x[3], -x[4]))
    for row in rows[:20]:
        table.add_row(row[0], row[1], row[2], str(row[3]), f"{row[4]:.2f}", row[5] or "-")
    console.print(table)

    if schema_path.exists():
        schema = load_schema(schema_path)
        console.print(f"\n[bold]Schema[/bold]: {', '.join(schema.type_names)} (fingerprint {schema.fingerprint()})")


def main() -> None:
    try:
        app()
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]ERROR[/red]: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()

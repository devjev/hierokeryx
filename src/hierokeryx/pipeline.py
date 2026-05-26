"""End-to-end orchestrator: extract → coref → cross-doc resolve → HITL routing.

The pipeline writes intermediate artifacts to a `workdir/` so any stage is
re-runnable independently:

```
workdir/
├── schema.yaml          # the EntitySchema used for the run
├── manifest.json        # run metadata (versions, fingerprint, timestamps)
├── extractions/
│   └── <doc_id>.json    # per-document ExtractionResult (post-coref)
├── registry.json        # cross-doc EntityRegistry
└── review/
    └── <doc_id>.jsonl   # HITL review (one file per flagged doc)
```

Re-importing reviewed JSONL files writes the merged result back into
`extractions/`.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from hierokeryx.confidence import route_for_review
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.protocol import LLMClient
from hierokeryx.models import (
    Document,
    EntityRegistry,
    EntitySchema,
    ExtractionResult,
    ReviewItem,
)
from hierokeryx.resolve.centroids import (
    compute_centroids,
    load_centroids,
    save_centroids,
)
from hierokeryx.resolve.coref import resolve_within_doc
from hierokeryx.resolve.crossdoc import resolve_crossdoc, resolve_incremental
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.review.apply import apply_review
from hierokeryx.review.jsonl import (
    _safe_filename,
    read_review_dir,
    write_review_dir,
)
from hierokeryx.schema import save_schema

__all__ = [
    "PipelineRun",
    "run",
    "run_one",
    "import_reviewed",
    "save_extraction",
    "load_extraction",
    "save_registry",
    "load_registry",
    "load_extractions_dir",
    "compute_centroids",
    "load_centroids",
    "save_centroids",
    "resolve_incremental",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRun:
    workdir: Path
    schema: EntitySchema
    extraction_results: list[ExtractionResult]
    registry: EntityRegistry
    flagged: list[ReviewItem]
    review_paths: list[Path]
    extraction_paths: list[Path] = field(default_factory=list)


def run(
    documents: list[Document],
    schema: EntitySchema,
    workdir: str | Path,
    *,
    extractor: GLiNERExtractor | None = None,
    llm_client: LLMClient,
    embedder: SentenceTransformerEmbedder | None = None,
    review_threshold: float = 0.7,
    span_threshold: float = 0.5,
    merge_threshold: float = 0.82,
    borderline_threshold: float = 0.75,
    only_flagged_review: bool = True,
    incremental_from: str | Path | None = None,
) -> PipelineRun:
    """Run the full pipeline: extract → coref → cross-doc → HITL routing.

    If `incremental_from` is set, load that workdir's registry + centroid
    sidecar and resolve new entities against it instead of clustering from
    scratch. The merged registry and updated centroids are written into
    `workdir`.
    """
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    (workdir_path / "extractions").mkdir(exist_ok=True)
    (workdir_path / "review").mkdir(exist_ok=True)

    save_schema(schema, workdir_path / "schema.yaml")

    extractor = extractor or GLiNERExtractor()
    extraction_results, extraction_paths = _extract_phase(documents, schema, extractor, llm_client, workdir_path)

    embedder = embedder or SentenceTransformerEmbedder()

    if incremental_from is not None:
        existing_registry = load_registry(Path(incremental_from) / "registry.json")
        existing_centroids = load_centroids(incremental_from)
        updated, registry, centroids = resolve_incremental(
            extraction_results,
            schema,
            existing_registry=existing_registry,
            existing_centroids=existing_centroids,
            llm_client=llm_client,
            embedder=embedder,
            merge_threshold=merge_threshold,
            borderline_threshold=borderline_threshold,
        )
    elif len(extraction_results) > 1:
        updated, registry = resolve_crossdoc(
            extraction_results,
            schema,
            llm_client=llm_client,
            embedder=embedder,
            merge_threshold=merge_threshold,
            borderline_threshold=borderline_threshold,
        )
        centroids = compute_centroids(updated, embedder)
    else:
        # Single doc → all entities are singleton clusters, just tag them.
        updated, registry = resolve_crossdoc(
            extraction_results,
            schema,
            llm_client=None,  # no need for tie-break with one doc
            embedder=embedder,
        )
        centroids = compute_centroids(updated, embedder)

    # Re-save extractions with cluster_ids attached.
    extraction_paths = []
    for result in updated:
        path = workdir_path / "extractions" / f"{_safe_filename(result.document.id)}.json"
        save_extraction(result, path)
        extraction_paths.append(path)

    save_registry(registry, workdir_path / "registry.json")
    save_centroids(workdir_path, centroids)

    flagged_entities = [
        e for result in updated for e in result.entities
    ]
    flagged = route_for_review(
        flagged_entities,
        cluster_threshold=review_threshold,
        span_threshold=span_threshold,
    )
    review_paths = write_review_dir(
        updated,
        workdir_path / "review",
        flagged=flagged,
        only_flagged=only_flagged_review,
    )

    _write_manifest(
        workdir_path,
        schema=schema,
        n_documents=len(documents),
        n_entities=sum(len(r.entities) for r in updated),
        n_flagged=len(flagged),
    )

    return PipelineRun(
        workdir=workdir_path,
        schema=schema,
        extraction_results=updated,
        registry=registry,
        flagged=flagged,
        review_paths=review_paths,
        extraction_paths=extraction_paths,
    )


def import_reviewed(
    workdir: str | Path,
    review_dir: str | Path | None = None,
) -> list[ExtractionResult]:
    """Replay edited review JSONL files into the workdir's extractions."""
    workdir_path = Path(workdir)
    review_root = Path(review_dir) if review_dir is not None else workdir_path / "review"
    extractions_dir = workdir_path / "extractions"

    parsed = read_review_dir(review_root)
    updated: list[ExtractionResult] = []
    for doc_id, (header, lines) in parsed.items():
        extraction_path = extractions_dir / f"{_safe_filename(doc_id)}.json"
        if not extraction_path.exists():
            logger.warning("No extraction file for reviewed doc_id=%r at %s", doc_id, extraction_path)
            continue
        original = load_extraction(extraction_path)
        applied = apply_review(original, header, lines)
        save_extraction(applied, extraction_path)
        updated.append(applied)
    return updated


def run_one(
    document: Document,
    schema: EntitySchema,
    *,
    extractor: GLiNERExtractor | None = None,
    llm_client: LLMClient,
) -> ExtractionResult:
    """Convenience: extract + within-doc coref for a single document.

    Skips cross-doc resolution and HITL routing — use `run()` for those.
    """
    extractor = extractor or GLiNERExtractor()
    mentions = extractor.extract(document, schema)
    entities = resolve_within_doc(document, mentions, schema, llm_client)
    return ExtractionResult(
        document=document,
        entities=entities,
        schema_version=schema.fingerprint(),
        model_versions={
            "gliner": extractor.model_id,
            "llm_coref": getattr(llm_client, "coref_model", "unknown"),
        },
    )


# Persistence helpers --------------------------------------------------------

def save_extraction(result: ExtractionResult, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_extraction(path: str | Path) -> ExtractionResult:
    return ExtractionResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_registry(registry: EntityRegistry, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(registry.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_registry(path: str | Path) -> EntityRegistry:
    return EntityRegistry.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_extractions_dir(directory: str | Path) -> list[ExtractionResult]:
    d = Path(directory)
    return [load_extraction(p) for p in sorted(d.glob("*.json"))]


# Internals ------------------------------------------------------------------

def _extract_phase(
    documents: list[Document],
    schema: EntitySchema,
    extractor: GLiNERExtractor,
    llm_client: LLMClient,
    workdir: Path,
) -> tuple[list[ExtractionResult], list[Path]]:
    results: list[ExtractionResult] = []
    paths: list[Path] = []
    for doc in documents:
        logger.info("Extracting %s (%d chars)", doc.id, len(doc.text))
        mentions = extractor.extract(doc, schema)
        entities = resolve_within_doc(doc, mentions, schema, llm_client)
        result = ExtractionResult(
            document=doc,
            entities=entities,
            schema_version=schema.fingerprint(),
            model_versions={
                "gliner": extractor.model_id,
                "llm_coref": getattr(llm_client, "coref_model", "unknown"),
            },
        )
        path = workdir / "extractions" / f"{_safe_filename(doc.id)}.json"
        save_extraction(result, path)
        results.append(result)
        paths.append(path)
    return results, paths


def _write_manifest(
    workdir: Path,
    *,
    schema: EntitySchema,
    n_documents: int,
    n_entities: int,
    n_flagged: int,
) -> None:
    manifest = {
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "schema_fingerprint": schema.fingerprint(),
        "schema_version": schema.version,
        "n_documents": n_documents,
        "n_entities": n_entities,
        "n_flagged": n_flagged,
    }
    (workdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

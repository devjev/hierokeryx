"""Score a workdir's cross-doc clustering against a JSONL gold file.

Reads `examples-workdir/batch1/` (produced by `incremental_pipeline.py`)
and scores its registry against a hand-written gold cluster assignment.
Prints both pairwise and BCubed P/R/F1, then runs a threshold sweep.

Run inside the project's nix dev shell:

    nix develop
    uv run python examples/incremental_pipeline.py   # produce a workdir
    uv run python examples/evaluate_with_gold.py
"""

from __future__ import annotations

from pathlib import Path

from hierokeryx import pipeline
from hierokeryx.eval import GoldEntry, evaluate, save_gold, sweep_thresholds
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.schema import load_schema


def main() -> None:
    workdir = Path("examples-workdir") / "batch1"
    if not workdir.exists():
        raise SystemExit(
            f"Run examples/incremental_pipeline.py first to produce {workdir}/"
        )

    results = pipeline.load_extractions_dir(workdir / "extractions")
    schema = load_schema(workdir / "schema.yaml")

    # Build a gold file matching the first batch — every Curie/Pierre mention
    # in curie-long should land in its own cluster. Adjust the entity ids to
    # match what your run actually produced.
    actual_ids = [
        (r.document.id, e.id, e.canonical)
        for r in results
        for e in r.entities
    ]
    print("Entity ids in workdir (for hand-crafting a gold file):")
    for doc_id, entity_id, canonical in actual_ids:
        print(f"  {doc_id}/{entity_id}: {canonical!r}")

    # Toy gold: assume every entity in the workdir is its own gold cluster.
    # In a real eval you would write this by hand, grouping mentions you
    # know co-refer.
    gold = [
        GoldEntry(
            doc_id=doc_id,
            entity_id=entity_id,
            gold_cluster_id=f"g_{doc_id}_{entity_id[:6]}",
        )
        for doc_id, entity_id, _ in actual_ids
    ]
    gold_path = Path("examples-workdir") / "toy-gold.jsonl"
    save_gold(gold, gold_path)
    print(f"\nWrote {len(gold)} gold records → {gold_path}")

    # Single-shot evaluation.
    report = evaluate(results, gold)
    print("\n--- Eval report ---")
    print(f"scored: {report.n_entities_scored} entities")
    print(f"system clusters: {report.n_clusters_system}")
    print(f"gold clusters:   {report.n_clusters_gold}")
    print(
        f"pairwise: p={report.pairwise.precision:.3f} "
        f"r={report.pairwise.recall:.3f} f1={report.pairwise.f1:.3f}"
    )
    print(
        f"bcubed:   p={report.bcubed.precision:.3f} "
        f"r={report.bcubed.recall:.3f} f1={report.bcubed.f1:.3f}"
    )

    # Threshold sweep.
    print("\n--- Threshold sweep ---")
    reports = sweep_thresholds(
        results,
        gold,
        schema,
        embedder=SentenceTransformerEmbedder(),
    )
    for r in reports:
        print(
            f"  merge={r.config['merge_threshold']:.2f} "
            f"borderline={r.config['borderline_threshold']:.2f} "
            f"pairwise_f1={r.pairwise.f1:.3f} bcubed_f1={r.bcubed.f1:.3f}"
        )
    best = max(reports, key=lambda r: r.pairwise.f1)
    print(
        f"\nBest by pairwise F1: merge={best.config['merge_threshold']}, "
        f"borderline={best.config['borderline_threshold']} → "
        f"pairwise_f1={best.pairwise.f1:.3f}"
    )


if __name__ == "__main__":
    main()

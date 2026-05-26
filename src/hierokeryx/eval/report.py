"""High-level evaluation: produce an `EvalReport` for a workdir against gold,
plus a small threshold-sweep helper for tuning.

`sweep_thresholds` re-runs `cluster_by_type` over a grid of
`(merge_threshold, borderline_threshold)` pairs and returns one
`EvalReport` per grid point. It deliberately runs without an LLM client —
sweeps are meant to be cheap; LLM tie-breaks would dominate cost and add
noise across runs.

Scaling caveat: each grid point pays an O(N²) pairwise similarity cost in
`_threshold_cluster`. Fine for dev sets up to a few thousand entities; for
larger sets, cache the similarity matrix and re-threshold (not done here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hierokeryx.eval.gold import GoldEntry
from hierokeryx.eval.metrics import bcubed_prf, pairwise_prf
from hierokeryx.models import EntitySchema, ExtractionResult
from hierokeryx.resolve.cluster import cluster_by_type
from hierokeryx.resolve.embed import (
    SentenceTransformerEmbedder,
    encode_extraction_results,
)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvalReport:
    n_entities_scored: int
    n_clusters_system: int
    n_clusters_gold: int
    pairwise: PRF
    bcubed: PRF
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_entities_scored": self.n_entities_scored,
            "n_clusters_system": self.n_clusters_system,
            "n_clusters_gold": self.n_clusters_gold,
            "pairwise": {
                "precision": self.pairwise.precision,
                "recall": self.pairwise.recall,
                "f1": self.pairwise.f1,
            },
            "bcubed": {
                "precision": self.bcubed.precision,
                "recall": self.bcubed.recall,
                "f1": self.bcubed.f1,
            },
            "config": self.config,
        }


def evaluate(
    extraction_results: list[ExtractionResult],
    gold: list[GoldEntry],
    *,
    config: dict[str, Any] | None = None,
) -> EvalReport:
    """Score a list of (already-resolved) extraction results against gold.

    Entities are matched to gold by `(doc_id, entity_id)`. Entities present
    in the results but missing from gold are excluded from metrics (and
    vice-versa).
    """
    system: dict[str, str] = {}
    for result in extraction_results:
        for entity in result.entities:
            if entity.cluster_id is None:
                continue
            system[f"{result.document.id}/{entity.id}"] = entity.cluster_id
    gold_map: dict[str, str] = {
        f"{g.doc_id}/{g.entity_id}": g.gold_cluster_id for g in gold
    }

    common = system.keys() & gold_map.keys()
    sys_restricted = {k: system[k] for k in common}
    gold_restricted = {k: gold_map[k] for k in common}

    p, r, f = pairwise_prf(sys_restricted, gold_restricted)
    bp, br, bf = bcubed_prf(sys_restricted, gold_restricted)

    return EvalReport(
        n_entities_scored=len(common),
        n_clusters_system=len(set(sys_restricted.values())),
        n_clusters_gold=len(set(gold_restricted.values())),
        pairwise=PRF(p, r, f),
        bcubed=PRF(bp, br, bf),
        config=config or {},
    )


DEFAULT_MERGE_GRID: tuple[float, ...] = (0.78, 0.82, 0.86)
DEFAULT_BORDERLINE_GRID: tuple[float, ...] = (0.70, 0.75, 0.80)


def sweep_thresholds(
    extraction_results: list[ExtractionResult],
    gold: list[GoldEntry],
    schema: EntitySchema,
    *,
    embedder: SentenceTransformerEmbedder | None = None,
    merge_grid: tuple[float, ...] = DEFAULT_MERGE_GRID,
    borderline_grid: tuple[float, ...] = DEFAULT_BORDERLINE_GRID,
) -> list[EvalReport]:
    """Re-cluster over a grid of thresholds and score each setting against gold.

    Runs without an LLM (LLM tie-break is not deterministic and would
    dominate cost). The same embeddings are reused across grid points; only
    the union-find threshold changes between runs.
    """
    embedder = embedder or SentenceTransformerEmbedder()
    entity_index, embeddings = encode_extraction_results(extraction_results, embedder)
    if embeddings.size == 0:
        return []

    reports: list[EvalReport] = []
    for merge_t in merge_grid:
        for border_t in borderline_grid:
            if border_t > merge_t:
                continue
            assignments = cluster_by_type(
                entity_index,
                embeddings,
                schema,
                llm_client=None,
                merge_threshold=merge_t,
                borderline_threshold=border_t,
            )
            tagged: list[ExtractionResult] = []
            for result in extraction_results:
                new_entities = []
                for entity in result.entities:
                    a = assignments.get(entity.id)
                    if a is None:
                        new_entities.append(entity)
                    else:
                        new_entities.append(
                            entity.model_copy(update={"cluster_id": a.cluster_id})
                        )
                tagged.append(result.model_copy(update={"entities": new_entities}))
            report = evaluate(
                tagged,
                gold,
                config={
                    "merge_threshold": merge_t,
                    "borderline_threshold": border_t,
                    "embedder_id": embedder.model_id,
                },
            )
            reports.append(report)
    return reports

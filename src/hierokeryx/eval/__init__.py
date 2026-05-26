"""Evaluation of cross-document entity resolution against gold cluster labels.

Compute pairwise and BCubed precision / recall / F1 against a JSONL gold file
that maps `(doc_id, entity_id)` to an opaque `gold_cluster_id`. Metrics are
computed over the intersection of entities present in both the system
extractions and the gold file, so partial gold sets are supported.

The threshold sweep helper re-runs `cluster_by_type` over a small grid of
`merge_threshold` / `borderline_threshold` values to support threshold tuning.
Sweeps never call the LLM tie-break — they're meant to be cheap.
"""

from hierokeryx.eval.gold import GoldEntry, load_gold, save_gold
from hierokeryx.eval.metrics import bcubed_prf, pairwise_prf
from hierokeryx.eval.report import PRF, EvalReport, evaluate, sweep_thresholds

__all__ = [
    "PRF",
    "EvalReport",
    "GoldEntry",
    "bcubed_prf",
    "evaluate",
    "load_gold",
    "pairwise_prf",
    "save_gold",
    "sweep_thresholds",
]

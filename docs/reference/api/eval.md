# `hierokeryx.eval`

Evaluate cross-document entity resolution against gold cluster labels.

Includes pairwise and BCubed precision / recall / F1, a JSONL gold-file
format, and a threshold-sweep helper for tuning. See the
[evaluation how-to](../../how-to/evaluating-resolution.md) for
end-to-end usage.

## `hierokeryx.eval.gold`

::: hierokeryx.eval.gold
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - GoldEntry
        - load_gold
        - save_gold

## `hierokeryx.eval.metrics`

::: hierokeryx.eval.metrics
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - pairwise_prf
        - bcubed_prf

## `hierokeryx.eval.report`

::: hierokeryx.eval.report
    options:
      show_root_heading: true
      show_root_full_path: true
      members:
        - PRF
        - EvalReport
        - evaluate
        - sweep_thresholds

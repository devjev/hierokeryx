# Confidence math

Every entity in the pipeline carries a single `confidence` field in
`[0, 1]`. This page explains how that number is computed and how it
drives the review router.

## Two stages, two formulas

Confidence is updated twice: once after within-document coreference, and
once after cross-document resolution.

### Within-doc

```text
confidence = 0.5 × mean(mention.score) + 0.5 × llm_cluster_confidence
```

- `mention.score` is GLiNER's per-span confidence.
- `llm_cluster_confidence` is the LLM's self-reported confidence for the
  cluster it just produced.

Source:
[`within_doc_confidence`][hierokeryx.confidence.within_doc_confidence].

### Cross-doc

```text
confidence = 0.4 × base_confidence
           + 0.4 × llm_decision_confidence
           + 0.2 × margin
```

Where:

- `base_confidence` is the within-doc score carried over.
- `llm_decision_confidence` is the LLM's self-reported merge confidence
  (or `1.0` if the merge was decided by threshold alone, no LLM call).
- `margin = max(0, top_similarity − second_similarity)` — the cosine
  margin between the assigned cluster's centroid and the next-best
  rival's. A small margin signals a borderline merge.

Source:
[`crossdoc_confidence`][hierokeryx.confidence.crossdoc_confidence].

## Why this shape

A few principles drove the choice:

- **Ensemble, not pick-the-winner.** GLiNER and the LLM disagree
  productively — if either is uncertain, the entity should land lower
  in the queue.
- **LLM self-report is signal, not gospel.** LLM confidences are well
  known to be miscalibrated. Weighting them at 0.4–0.5 instead of 1.0
  means a single overconfident LLM call can't drown out an obviously
  weak GLiNER span.
- **Margin matters for clustering.** The difference between "best
  match is 0.85, runner-up is 0.30" and "best match is 0.85, runner-up
  is 0.83" is everything for a cluster-assignment decision. The
  `margin` term captures this without an extra hyperparameter.

The weights are tuned by hand on a few hundred labeled examples. They
are not magic and you may want to override them — see the constants at
the top of [`hierokeryx.confidence`](../reference/api/confidence.md)
(`WITHIN_DOC_*` and `CROSSDOC_*`). They're module-level so they're easy
to find and rebind for experimentation; we may make them configurable
in v0.2.

## How routing uses confidence

[`route_for_review`][hierokeryx.confidence.route_for_review] decides
which entities get a JSONL row. Reason precedence, highest first:

1. **`ambiguous_merge`** — confidence below
   `cluster_threshold` AND the entity has a `cluster_id` assigned. The
   cross-doc merge was uncertain.
2. **`low_cluster_conf`** — confidence below `cluster_threshold` AND no
   `cluster_id`. The within-doc coref was uncertain.
3. **`low_span_conf`** — confidence is fine but at least one mention
   scored below `span_threshold`. GLiNER is unsure about the span itself.

The reason annotation is exported into the JSONL `reason` field, so a
reviewer can prioritise by failure mode.

## What numbers mean in practice

Rough calibration on a few thousand entities from mixed English text:

| Range         | Typical content                                  |
|---------------|--------------------------------------------------|
| `0.90 – 1.00` | High-precision: unambiguous names, common types. |
| `0.70 – 0.90` | The healthy middle.                              |
| `0.50 – 0.70` | Borderline — most flagged entities live here.    |
| `< 0.50`      | Almost always a bug: bad span, wrong type.       |

If your distribution looks very different (e.g., everything piles up at
`0.95`), one of these is probably true:

- Your schema is too easy — every mention is unambiguous.
- The LLM's self-reported confidence is uniformly high (a known
  miscalibration on some models). Consider weighting the LLM term
  down.

## Knobs and where they live

| Knob                       | Default | Where                                                       |
|----------------------------|---------|-------------------------------------------------------------|
| `cluster_threshold`        | `0.7`   | `route_for_review` arg, `--review-threshold` flag           |
| `span_threshold`           | `0.5`   | `route_for_review` arg                                      |
| `WITHIN_DOC_MENTION_WEIGHT`| `0.5`   | module constant in `hierokeryx.confidence`                   |
| `WITHIN_DOC_LLM_WEIGHT`    | `0.5`   | module constant                                             |
| `CROSSDOC_BASE_WEIGHT`     | `0.4`   | module constant                                             |
| `CROSSDOC_LLM_WEIGHT`      | `0.4`   | module constant                                             |
| `CROSSDOC_MARGIN_WEIGHT`   | `0.2`   | module constant                                             |

The two thresholds are runtime-configurable; the weights are not (yet).

## Calibration tips

If you have ground-truth labels for a subset of your corpus:

1. Export *all* entities (`--all-for-review`), not just flagged.
2. Sort by confidence; plot precision (or whatever metric you care
   about) against the score.
3. Find the operating point — the confidence at which precision falls
   below your bar.
4. Set `--review-threshold` to that point.

This gives you a calibrated `confidence` threshold without retuning the
ensemble weights.

# Evaluating resolution quality

The three load-bearing thresholds in cross-doc resolution —
`merge_threshold` (default 0.82), `borderline_threshold` (0.75), and
`review_threshold` (0.7) — are sensible defaults, not universal truths.
The eval harness lets you measure precision, recall, and F1 against a
small labeled set so threshold choices stop being vibes.

The same harness is the natural verification tool for
[incremental resolution](cross-doc-resolution.md#incremental-resolution-against-an-existing-workdir):
batch resolve, score; batch the first half then incrementally resolve
the second half, score; if the F1s agree within a couple of points,
incremental is working.

## The gold format

One JSONL record per labeled entity. The gold `cluster_id` is opaque —
only consistency across records matters, not the value itself:

```jsonl
{"doc_id": "doc_curie",       "entity_id": "e_marie_curie",  "gold_cluster_id": "g_marie_curie"}
{"doc_id": "doc_curie_short", "entity_id": "e_marie_curie",  "gold_cluster_id": "g_marie_curie"}
{"doc_id": "doc_curie",       "entity_id": "e_pierre_curie", "gold_cluster_id": "g_pierre_curie"}
{"doc_id": "doc_curie_short", "entity_id": "e_pierre_curie", "gold_cluster_id": "g_pierre_curie"}
```

Blank lines and `#`-prefixed lines are ignored. Duplicate
`(doc_id, entity_id)` pairs fail loudly.

Building a gold file:

1. Run the pipeline against a small representative slice of your
   corpus.
2. Open `workdir/extractions/*.json` and copy the entity ids you care
   about. Entity ids are stable across re-runs against the same
   document text (they hash sorted mention ids — see
   [`make_entity_id`][hierokeryx.models.make_entity_id]).
3. For each entity, write the JSONL record with the gold cluster id
   you'd expect a perfect system to produce.

You don't need 100% coverage. Entities present in the system output
but not in the gold file are excluded from metrics (and vice-versa).
A focused 20-50 line gold set is more useful than a sloppy one with
thousands of records.

## Metrics

Both standard ER metrics, both computed over the intersection of
entities present in system and gold:

- **Pairwise P/R/F1**. Considers every unordered pair of entities.
  TP = pair co-clustered in both; FP = co-clustered only in system;
  FN = co-clustered only in gold. Most intuitive, sensitive to
  cluster sizes.
- **BCubed P/R/F1**. Per-mention precision and recall, then averaged.
  More robust to skewed cluster-size distributions. The standard
  choice in coref / ER literature.

Pairwise is easier to explain; BCubed is what most ER papers report.
Look at both — divergence between them usually signals one or two big
clusters dominating the pairwise count.

## CLI

```bash
# Score a workdir's current clustering
hkx eval --workdir wd/ --gold path/to/gold.jsonl
```

Output:

```
                Eval report
┏━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ metric   ┃ precision ┃ recall ┃    f1 ┃
┡━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ pairwise │     1.000 │  1.000 │ 1.000 │
│ bcubed   │     1.000 │  1.000 │ 1.000 │
└──────────┴───────────┴────────┴───────┘
scored 8 entities — system clusters: 4, gold clusters: 4
```

For CI / programmatic use, dump the full report as JSON:

```bash
hkx eval --workdir wd/ --gold gold.jsonl --json-out report.json
```

## Tuning thresholds with `--sweep`

Re-cluster across a small grid of `(merge_threshold, borderline_threshold)`
values, score each, and report the best by pairwise F1:

```bash
hkx eval --workdir wd/ --gold gold.jsonl --sweep
```

Output:

```
       Threshold sweep (pairwise F1 / BCubed F1)
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃  merge ┃ borderline ┃ pairwise_f1 ┃ bcubed_f1 ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│   0.78 │       0.70 │       0.857 │     0.890 │
│   0.78 │       0.75 │       0.857 │     0.890 │
│   0.82 │       0.70 │       1.000 │     1.000 │
│   0.82 │       0.75 │       1.000 │     1.000 │
│   0.82 │       0.80 │       1.000 │     1.000 │
│   0.86 │       0.70 │       0.800 │     0.812 │
│   0.86 │       0.75 │       0.800 │     0.812 │
│   0.86 │       0.80 │       0.800 │     0.812 │
└────────┴────────────┴─────────────┴───────────┘

Best by pairwise F1: merge=0.82, borderline=0.75 → pairwise_f1=1.000, bcubed_f1=1.000
```

The default grid is `merge ∈ {0.78, 0.82, 0.86}` × `borderline ∈ {0.70,
0.75, 0.80}`, skipping pairs where `borderline > merge`. The sweep
never calls the LLM (LLM tie-break is non-deterministic and would
dominate cost) — the embeddings are computed once and re-thresholded.

**Scaling caveat.** Each grid point pays an O(N²) pairwise-similarity
cost in the underlying union-find. Fine up to a few thousand entities;
slow beyond that. For larger eval sets, sweep a coarser grid.

## Library API

```python
from hierokeryx import pipeline
from hierokeryx.eval import evaluate, load_gold, sweep_thresholds
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.schema import load_schema

results = pipeline.load_extractions_dir("wd/extractions")
gold = load_gold("path/to/gold.jsonl")
schema = load_schema("wd/schema.yaml")

# Single-shot evaluation
report = evaluate(results, gold)
print(report.pairwise.f1, report.bcubed.f1)

# Threshold sweep
reports = sweep_thresholds(
    results, gold, schema,
    embedder=SentenceTransformerEmbedder(),
)
best = max(reports, key=lambda r: r.pairwise.f1)
print(best.config, best.pairwise.f1)
```

See the [`hierokeryx.eval` API reference](../reference/api/eval.md) for the full surface.

## Comparing batch vs incremental

A common smoke test:

```bash
# Baseline: a full batch resolve over the whole corpus.
hkx pipeline docs/ -s schema.yaml -o wd-all/
hkx eval --workdir wd-all/ --gold gold.jsonl

# Incremental: first half batch, second half incremental against it.
hkx pipeline docs/half1 -s schema.yaml -o wd-1/
hkx pipeline docs/half2 -s schema.yaml -o wd-2/ --against wd-1/
hkx eval --workdir wd-2/ --gold gold.jsonl
```

Expect a small F1 gap (a few points at most) — incremental is greedy
and order-dependent, batch is global. A larger gap means either the
incremental algorithm is misbehaving on your data, or the gold set is
too small to discriminate. Grow the gold set first.

# Tune confidence thresholds

`hierokeryx` exposes four knobs that govern how aggressively the pipeline
merges entities and how often it asks a human. Defaults are tuned for
"high precision, moderate recall, sensible HITL volume" but every corpus
is different.

## The four knobs

| Threshold              | CLI flag                | Default | Range typically tuned    |
|------------------------|-------------------------|---------|--------------------------|
| GLiNER span threshold  | `--threshold` (extract) | `0.4`   | `0.3 – 0.6`              |
| Cross-doc merge        | `--merge-threshold`     | `0.82`  | `0.75 – 0.90`            |
| Cross-doc borderline   | (library only)          | `0.75`  | `merge_threshold – 0.05` |
| Review (within-doc)    | `--review-threshold`    | `0.7`   | `0.5 – 0.85`             |

Span threshold is on the GLiNER side; the other three live in
[`hierokeryx.confidence`](../reference/api/confidence.md) and
[`hierokeryx.resolve.cluster`](../reference/api/resolve.md).

## Span threshold (`--threshold`)

Below this, GLiNER drops the candidate entirely. Lower = more recall, more
noise. Higher = fewer but more confident spans.

- `0.3` for noisy domains where you want to catch everything and let the
  LLM filter.
- `0.5+` for clean text where you only want high-confidence spans (legal
  documents, scientific papers).

Effect is per-entity-type. If you see false positives concentrated on one
type, narrow that type's description and examples in the schema before
touching the threshold.

## Merge threshold (`--merge-threshold`)

Cosine similarity between two entity embeddings at or above this value
unions their clusters. Embeddings come from the sentence-transformer in
[`hierokeryx.resolve.embed`](../reference/api/resolve.md).

- `0.85+` for tight clusters (named scientific entities, exact product
  IDs).
- `0.78 – 0.82` for default-quality corpora.
- `0.75` only when you know name variation is high (e.g., translit­erated
  proper nouns) and you're prepared to review more merges.

Below the merge threshold, but above `borderline_threshold`, the LLM is
asked to adjudicate. If you've disabled the LLM (`--no-llm-tiebreak`),
borderline pairs stay split.

## Borderline threshold

Library-only at the moment — set when you call
[`resolve_crossdoc`][hierokeryx.resolve.crossdoc.resolve_crossdoc]
directly. Default is `merge_threshold - 0.07`. The band between
`borderline_threshold` and `merge_threshold` is where the LLM does its
work; widening it costs tokens but improves recall.

## Review threshold (`--review-threshold`)

Entities with within-doc + cross-doc confidence below this value get
flagged for human review.

- `0.5` for a once-over before publishing to production.
- `0.7` (default) for "trust but verify" workflows.
- `0.85` for high-stakes domains (medical, legal, financial) where you
  want to look at almost everything.

See [Confidence math](../concepts/confidence-math.md) for how the score
is composed.

## A tuning workflow

1. **Run with defaults** on a representative sample of 20–50 documents.
2. `hkx review export workdir/ --out review/ --all` — export *every*
   entity, not just flagged.
3. Hand-label a few hundred entities as correct / incorrect / borderline.
4. Sort by confidence; look at the distribution of incorrect entities
   against the score. The threshold you want is the score at which
   roughly 50% of items below are incorrect.
5. Adjust `--review-threshold` to that point. Re-run.

For corpora where you already have ground truth, plumb confidence into a
precision-recall curve and pick the operating point that matches your
review budget.

## Library API

```python
from hierokeryx import pipeline
from hierokeryx.resolve.embed import SentenceTransformerEmbedder

run = pipeline.run(
    documents=docs,
    schema=schema,
    workdir="workdir/",
    llm_client=client,
    embedder=SentenceTransformerEmbedder(),
    review_threshold=0.6,
    span_threshold=0.5,
    merge_threshold=0.80,
    borderline_threshold=0.73,
)
```

## Anti-patterns

- **Tuning thresholds before the schema.** A bad schema description costs
  more recall than any threshold change will recover. Fix the schema first.
- **Bumping `--review-threshold` above 0.9** to "make the pipeline look
  confident." This just hides errors; it doesn't fix them.
- **Lowering `--merge-threshold` below 0.7** without enabling the LLM
  tie-break. You'll silently merge unrelated entities and not catch it
  until production.

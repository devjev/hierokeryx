# Cross-document resolution

The tutorial covered the end-to-end pipeline. This page goes deeper into
the cross-document phase: what it does, how to run it standalone, and how
to inspect its output.

## What it does

After per-document coreference, every entity has a `canonical` form and
a within-doc confidence. The cross-document phase decides which entities
across different documents refer to the same real-world thing.

The algorithm, blocked by entity type:

1. **Embed** each entity using a sentence-transformer over its canonical
   form plus a few context windows from its mentions.
2. **Threshold cluster**: union-find any pair with cosine similarity
   ≥ `merge_threshold` (default `0.82`).
3. **Tie-break**: for any leftover entity whose nearest cluster centroid
   is between `borderline_threshold` and `merge_threshold`, ask the LLM
   to adjudicate.
4. **Singleton** anything still unassigned.

The result is an [`EntityRegistry`][hierokeryx.models.EntityRegistry]
mapping `cluster_id → {canonical, type, member_entity_ids}` and an
updated list of [`ExtractionResult`][hierokeryx.models.ExtractionResult]s
where each entity now carries a `cluster_id`.

See [`resolve_crossdoc`][hierokeryx.resolve.crossdoc.resolve_crossdoc] for
the entry point and
[`cluster_by_type`][hierokeryx.resolve.cluster.cluster_by_type] for the
algorithm.

## Run it standalone

If you've already extracted entities and just want to re-cluster:

```bash
hkx resolve workdir/                        # uses defaults
hkx resolve workdir/ --threshold 0.85       # tighter clusters
hkx resolve workdir/ --no-llm               # skip LLM tie-break
```

This expects `workdir/extractions/*.json` and `workdir/schema.yaml` to
exist (the outputs of `hkx extract` or `hkx pipeline`). It writes
`workdir/registry.json` and updates each extraction in place with the
new `cluster_id`.

## Library API

```python
from hierokeryx import pipeline
from hierokeryx.resolve.crossdoc import resolve_crossdoc
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.llm.anthropic_client import AnthropicClient
from hierokeryx.schema import load_schema

schema = load_schema("workdir/schema.yaml")
results = pipeline.load_extractions_dir("workdir/extractions")

updated_results, registry = resolve_crossdoc(
    results,
    schema,
    llm_client=AnthropicClient(),
    embedder=SentenceTransformerEmbedder(),
    merge_threshold=0.82,
    borderline_threshold=0.75,
)

print(f"{len(registry.clusters)} cross-doc cluster(s)")
for cluster_id, cluster in registry.clusters.items():
    canonical = registry.canonical_by_cluster[cluster_id]
    print(f"  {cluster_id} ({len(cluster)}) — {canonical}")
```

## Incremental resolution against an existing workdir

Cross-doc resolution is batch by default — every run re-embeds and
re-clusters every entity. For a growing corpus, this is wasteful and
also rewrites cluster ids as the data shifts. Incremental mode assigns
new entities to existing clusters (or creates new ones) without
touching the old corpus.

The mechanism is a small sidecar file alongside `registry.json`:

```text
workdir/
├── registry.json
├── registry_embeddings.npz        # one centroid per cluster
└── registry_embeddings.meta.json  # embedder id + dim
```

Every `hkx resolve` and `hkx pipeline` writes this sidecar
automatically. To resolve a new batch of documents against it:

```bash
# First batch — produces wd1/ with registry + sidecar
hkx pipeline docs/batch1 -s schema.yaml -o wd1/

# Second batch — assigns into wd1's clusters or creates new ones,
# writing the merged result to wd2/.
hkx pipeline docs/batch2 -s schema.yaml -o wd2/ --against wd1/

# `hkx resolve` accepts the same flag if you've already extracted.
hkx resolve wd2/ --against wd1/
```

The decision per new entity, blocked by entity type:

1. If similarity to the nearest existing centroid is at or above
   `merge_threshold` → assign to that existing cluster.
2. If it's between `borderline_threshold` and `merge_threshold` and an
   LLM client is available → tie-break against the nearest existing
   cluster.
3. Otherwise → either join a new-batch cluster formed by an earlier
   entity in the same run (if their similarity is above the merge
   threshold), or become its own new singleton cluster.

The sidecar's `embedder_id` is recorded so swapping the embedding
model between runs is rejected loudly rather than silently producing
garbage similarities.

### Library API

```python
from hierokeryx import pipeline
from hierokeryx.resolve.crossdoc import resolve_incremental
from hierokeryx.resolve.embed import SentenceTransformerEmbedder

existing_registry = pipeline.load_registry("wd1/registry.json")
existing_centroids = pipeline.load_centroids("wd1/")
new_results = pipeline.load_extractions_dir("wd2/extractions")

updated, merged_registry, merged_centroids = resolve_incremental(
    new_results,
    schema,
    existing_registry=existing_registry,
    existing_centroids=existing_centroids,
    embedder=SentenceTransformerEmbedder(),
    llm_client=...,  # optional; required for borderline tie-break
)
pipeline.save_registry(merged_registry, "wd2/registry.json")
pipeline.save_centroids("wd2/", merged_centroids)
```

### Retrofitting older workdirs

Workdirs created before the centroid sidecar existed can be upgraded
in place — the recomputation reads only the existing extractions, no
LLM calls:

```bash
hkx resolve-centroids-rebuild path/to/old-workdir
```

After this, the workdir is a valid target for `--against`.

### When NOT to use incremental

- **You changed the schema.** New entity types or substantially edited
  type descriptions mean the LLM's coref decisions on the old corpus
  may no longer match. Re-batch.
- **You changed the embedder.** The centroid sidecar will refuse the
  merge — rebuild from scratch (or `resolve-centroids-rebuild` first).
- **Drift across many incremental runs is unacceptable.** Each
  incremental run is greedy and order-dependent; over many batches
  cluster shapes can diverge from what a single batch run would
  produce. Periodically re-run a full batch resolve as a baseline —
  the [eval harness](evaluating-resolution.md) makes the comparison
  measurable.

## Inspecting the registry

```python
registry = pipeline.load_registry("workdir/registry.json")

# What entities are in cluster X?
cluster_id = "pers_abc"
member_entity_ids = registry.clusters[cluster_id]
canonical = registry.canonical_by_cluster[cluster_id]
entity_type = registry.type_by_cluster[cluster_id]

# Which documents does cluster X appear in?
results = pipeline.load_extractions_dir("workdir/extractions")
doc_ids = sorted({
    r.document.id
    for r in results
    for e in r.entities
    if e.cluster_id == cluster_id
})
```

## When to skip the LLM tie-break

The LLM tie-break is the most expensive cross-doc operation by far — for
a corpus with N borderline candidates, it's one extra LLM call per
candidate. Skip it when:

- You're iterating on schema design and need fast feedback.
- Your corpus has very distinctive names (gene symbols, ISBNs, exact
  product codes) where embedding similarity is already decisive.
- You're running offline and don't want to spend tokens.

```bash
hkx resolve workdir/ --no-llm
```

Without the LLM, borderline candidates stay as singletons. You can
always re-run with `hkx resolve workdir/` (no flag) to do a second pass
later.

## Memory and scale

The clustering step is O(N²) in entities-per-type because it computes a
full pairwise similarity matrix in a NumPy block. At 10k entities per
type, that's a 400 MB float32 matrix — fine. At 100k, expect 40 GB,
which is not fine.

Workarounds for large corpora:

- **Type-block your data** — if you have 100k Person entities but only
  1k Organization entities, the bottleneck is just the Persons. Often
  you can split into sub-runs by document subset.
- **Use a vector index** (FAISS, hnswlib) over the embeddings and approx-
  cluster. Out of scope for v0.1; the
  [`cluster_by_type`][hierokeryx.resolve.cluster.cluster_by_type] function
  is small enough to fork.

For most corpora, this isn't a real issue — the pipeline is designed
around <10k documents per run.

## When the result looks wrong

Common failure modes and what to do:

| Symptom                          | Likely cause                          | Fix                                                   |
|----------------------------------|---------------------------------------|-------------------------------------------------------|
| Different people merged          | Embedding similarity overstates match | Raise `--merge-threshold` to `0.85+`                  |
| Same person split across docs    | Canonical forms vary too much         | Lower `--merge-threshold` to `0.78`, keep LLM enabled |
| LLM merges entities you can't link | Hallucinated cluster id              | File a bug with the JSONL — the linter should catch it |
| Clusters of one                  | LLM tie-break disabled / no neighbours | Re-run without `--no-llm`                             |

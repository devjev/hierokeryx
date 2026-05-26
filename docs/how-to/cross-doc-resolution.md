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

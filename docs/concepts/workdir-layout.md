# Workdir layout

Every CLI run that produces artifacts writes them under a single
**workdir** — a directory you control, with a fixed layout the rest of
the tooling assumes.

## The canonical layout

```text
workdir/
├── schema.yaml                    # the EntitySchema used for this run
├── manifest.json                  # run metadata
├── extractions/
│   ├── doc_01.json                # one ExtractionResult per document
│   ├── doc_02.json
│   └── ...
├── registry.json                  # cross-doc EntityRegistry
├── registry_embeddings.npz        # cluster centroid sidecar (for incremental)
├── registry_embeddings.meta.json  # embedder id + sidecar metadata
└── review/
    ├── doc_05.jsonl               # flagged entities, one document per file
    └── ...
```

Every file is self-describing — you can `cat` any of them and understand
the contents without consulting the rest of the workdir.

## File-by-file

### `schema.yaml`

A verbatim copy of the [`EntitySchema`][hierokeryx.models.EntitySchema]
the pipeline ran against. Saved here so the workdir is self-contained:
you can move it to another machine and `hkx inspect` will still work.

### `manifest.json`

Run-level metadata produced by
[`pipeline.run`][hierokeryx.pipeline.run]:

```json
{
  "created_at": "2026-05-26T14:08:00.123456",
  "schema_fingerprint": "a1b2c3d4...",
  "n_documents": 3,
  "n_entities": 7,
  "n_flagged": 1,
  "model_versions": {
    "gliner": "urchade/gliner_large-v2.5",
    "embedder": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

Useful for: telling whether two workdirs came from the same code path,
debugging "why does this run look different from last week's".

### `extractions/<doc_id>.json`

One [`ExtractionResult`][hierokeryx.models.ExtractionResult] per
document. Contains the document text, the entities with their mentions,
and the schema version. Round-trippable via
[`load_extraction`][hierokeryx.pipeline.load_extraction] and
[`save_extraction`][hierokeryx.pipeline.save_extraction].

The filename is the document's `id`, sanitized (private
`_safe_filename` helper in `hierokeryx.review.jsonl`) — slashes and
unprintable characters become underscores so the file always exists on
disk.

### `registry.json`

The cross-doc [`EntityRegistry`][hierokeryx.models.EntityRegistry]:
which entities ended up in which clusters, with canonical forms and
types per cluster.

By default the CLI rewrites the registry from scratch on each
`hkx resolve` / `hkx pipeline`. To add new documents to an existing
registry without re-clustering everything, pass
`--against <existing-workdir>` — see
[Cross-document resolution](../how-to/cross-doc-resolution.md#incremental-resolution-against-an-existing-workdir).

### `registry_embeddings.npz` + `registry_embeddings.meta.json`

A compact NumPy sidecar of cluster centroids (one L2-normalized vector
per cluster, plus member counts for weighted running-mean updates) and
a small JSON manifest recording the `embedder_id` that produced them.

This file is written automatically by every `hkx resolve` / `hkx
pipeline` run and is the input for incremental resolution via
`--against`. It carries the embedder identity so future incremental
runs refuse to merge centroids from a different embedding model
(silent embedder swaps produce garbage similarities). Workdirs created
before this sidecar existed can be retrofitted with
`hkx resolve-centroids-rebuild <workdir>`.

### `review/<doc_id>.jsonl`

One JSONL file per document with at least one flagged entity (or all
documents, if you passed `--all-for-review`). See
[Why JSONL HITL](why-jsonl-hitl.md) for the format.

By default this directory only contains files for documents that have
something flagged — a clean run with no flagged entities produces an
empty `review/` directory. Pass `--all-for-review` to write every
document regardless.

## Lifecycle

A workdir progresses through three phases:

1. **After `hkx extract`** — `schema.yaml`, `extractions/*.json`. No
   registry, no review. Useful for inspecting GLiNER + within-doc coref
   output before running cross-doc resolution.
2. **After `hkx resolve`** — adds `registry.json` plus the
   `registry_embeddings.npz` + `.meta.json` centroid sidecar. The
   extractions are rewritten in place to carry `cluster_id`. Run this
   any time you want to recluster (different threshold, different LLM
   tie-break setting). Pass `--against <existing-workdir>` to resolve
   incrementally instead of from scratch.
3. **After `hkx pipeline`** — fully populated, including `review/*.jsonl`
   and `manifest.json`.

`hkx review import` mutates `extractions/*.json` to reflect human edits;
it does not touch `registry.json` (cluster assignments are not
re-derived on import, by design — a human override should be sticky).

## Working with multiple workdirs

Common patterns:

- **One workdir per dataset.** `workdir-curie/`, `workdir-einstein/`.
- **Versioned workdirs for experiments.** `workdir-v1/`,
  `workdir-v2-tightcluster/`. Diff `manifest.json` to see what changed.
- **Workdir as PR artifact.** Commit the workdir to a separate
  `data/` repo. The git history then includes both the model output
  and any human edits.

Whatever you do, treat the workdir as opaque to anything except `hkx`
and the documented library helpers. The internal JSON structure is
stable across patch releases (see `schema_version` in each file) but
not pinned forever.

## What's *not* in the workdir

- **The source documents.** You bring those. The workdir only stores
  what was extracted from them. The `text_sha` on each review file is
  enough to detect drift but not to reconstruct the document.
- **LLM transcripts.** No prompts or responses are persisted. If you
  need them for audit, hook into
  [`AnthropicClient`][hierokeryx.llm.anthropic_client.AnthropicClient]
  via Python's `logging` and capture from there.
- **Model weights.** GLiNER and the sentence-transformer cache under
  `HF_HOME` (default `~/.cache/huggingface/`); the workdir only
  references model *names*.

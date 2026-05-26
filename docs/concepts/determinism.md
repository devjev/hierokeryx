# Determinism and reproducibility

`hierokeryx` makes a few deliberate choices to stay reproducible even when
an LLM is in the loop. This page lists what is stable, what is not, and
what to do if you need stronger guarantees.

## What is deterministic

- **GLiNER outputs.** Given the same model weights, threshold, and input
  text, GLiNER returns the same spans in the same order.
- **Embedding outputs.** Sentence-transformer encodings are deterministic
  per device (CUDA / MPS / CPU). They are not bit-identical across
  devices, but cosine similarities agree to ~6 decimal places.
- **Threshold clustering.** Union-find over pairwise similarities with a
  fixed threshold is order-independent.
- **Canonical mention/entity IDs.** Derived from content hashes — not
  random UUIDs. Same document + same schema + same model versions
  yields the same `mention_id` and `entity_id` across runs.

[`make_mention_id`][hierokeryx.models.make_mention_id],
[`make_entity_id`][hierokeryx.models.make_entity_id], and
[`make_cluster_id`][hierokeryx.models.make_cluster_id] are the helpers
that compute these — all three are public so external tooling can
re-derive them.

## What is not deterministic

- **LLM outputs.** Claude (and any other modern LLM) is sampled at
  `temperature > 0` by default. Two pipeline runs over the same input
  will likely produce slightly different cluster assignments and
  canonical forms.
- **Cross-doc tie-break decisions.** Same source as above — the borderline
  candidates land on an LLM, which is non-deterministic.

The two non-deterministic outputs cascade into anything downstream that
depends on them, so the full pipeline run-to-run is *similar* but not
*identical*.

## How to detect drift

Two artifacts make drift visible:

### The schema fingerprint

[`EntitySchema.fingerprint()`][hierokeryx.models.EntitySchema] is a
stable hash of the schema content. Every extraction stamps the
fingerprint into its `schema_version` field. If you re-run and the
fingerprint changes, the schema changed too.

### `text_sha`

Each review JSONL file's header carries a 16-character SHA-256 prefix of
the source document. The linter refuses to apply edits when the hash
disagrees with the current document — i.e., the document was edited
between export and import.

Together they answer the two most common drift questions: *did the
schema change?* and *did the underlying text change?*

## Recipes for reproducibility

If you need bit-stable runs (e.g., for compliance), there are a few
moves:

### 1. Pin the model versions

```python
GLiNERExtractor(model_id="urchade/gliner_large-v2.5@<commit_sha>")
SentenceTransformerEmbedder(model_id="sentence-transformers/all-MiniLM-L6-v2@<commit_sha>")
```

The HuggingFace `@<sha>` syntax pins to an exact commit. If you don't
pin, the next upstream weight reshuffle will silently change your
outputs.

### 2. Pin the Claude model

Don't use a "smart" alias like `claude-sonnet-latest`. Pin the explicit
versioned id:

```python
AnthropicClient(coref_model="claude-sonnet-4-6", tiebreak_model="claude-sonnet-4-6")
```

Anthropic publishes deprecation timelines for specific versioned ids.

### 3. Lower temperature where supported

The `AnthropicClient` default temperature is the SDK default (`1.0`). For
clustering tasks lower is generally better:

```python
# In a custom backend, pass temperature=0 to the underlying provider.
```

`temperature=0` doesn't guarantee determinism (most providers reserve
the right to vary at zero) but reduces drift substantially in practice.

### 4. Capture and replay with VCR

`tests/integration/test_pipeline_smoke.py` uses
[`vcrpy`](https://github.com/kevin1024/vcrpy) cassettes to record real
API responses and replay them on subsequent runs. The same trick works
for production audits — wrap your `AnthropicClient` in a recording
session and rerun the pipeline against the cassette to get bit-stable
output.

### 5. Commit the workdir

A workdir is the cheapest reproducibility checkpoint. If you commit
`workdir/` to git, anyone can `hkx inspect` and `hkx review` against
exactly the same artifacts you saw, regardless of what the model does
tomorrow.

## What we won't promise

`hierokeryx` does not (and at v0.1 will not) attempt to make Claude
deterministic. We treat the LLM's output as a sample from a
distribution. The pipeline's job is to:

- Re-check load-bearing invariants (e.g., mention spans quote the source
  verbatim).
- Score and route uncertain decisions to humans.
- Make the inputs and the metadata reproducible enough that you can
  audit the *delta* between two runs.

That's what's tractable today. Full bit-stability would require running
a self-hosted LLM in greedy decoding mode — feasible (via a custom
[`LLMClient`][hierokeryx.llm.protocol.LLMClient] backend) but out of
scope for the default pipeline.

## Further reading

- [Workdir layout](workdir-layout.md) — what gets persisted and where.
- [Why JSONL HITL](why-jsonl-hitl.md) — `text_sha`'s role in catching
  stale reviews.
- [LLM safety](llm-safety.md) — adjacent concerns about *what* the LLM
  sees, not just *what* it returns.

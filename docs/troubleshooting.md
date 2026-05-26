# Troubleshooting

Common symptoms, root causes, and fixes. If the issue you're hitting
isn't here,
[open an issue](https://github.com/jevtarassov/hierokeryx/issues/new) —
this page grows from real reports.

## Installation and environment

### `ImportError: libstdc++.so.6` or similar on NixOS

PyPI's ML wheels (`torch`, `transformers`) are linked against the host
distro's C++ runtime. NixOS doesn't put those on the loader path.

**Fix**: use the project's Nix devshell.

```bash
nix develop          # or direnv allow
uv sync --all-groups
```

The shell sets `LD_LIBRARY_PATH` to the right libs. See
[Installation › NixOS](installation.md#nixos-nix-devshell).

### `RuntimeError: PyPI ruff binary not executable` on NixOS

The PyPI `ruff` is a dynamically linked standalone binary that NixOS
can't run unpatched.

**Fix**: use the Nix-provided ruff from the flake. `nix develop` puts it
on PATH. Don't `uv add ruff`.

### `hkx: command not found` after install

`uv add hierokeryx` doesn't put `hkx` on your global PATH — it installs
into the project's `.venv`.

**Fix**: `uv run hkx ...` or `uvx --from hierokeryx hkx ...`. To install
globally, use `pipx install hierokeryx` or `uv tool install hierokeryx`.

## First-run problems

### The CLI looks frozen for a minute on the first run

GLiNER is downloading its model (~1.7 GB to `HF_HOME`). There is no
progress indicator in `hkx` v0.1.

**Fix**: run the download with progress before your first pipeline:

```bash
huggingface-cli download urchade/gliner_large-v2.5
```

Subsequent runs use the cache and are fast.

### `LLMError: AuthenticationError` from Claude

`ANTHROPIC_API_KEY` is missing, expired, or for the wrong workspace.

**Fix**: `export ANTHROPIC_API_KEY=sk-ant-...`. Verify with:

```bash
curl -X POST https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
```

### `LLMError: rate_limit_error` on a large batch

Anthropic's tier-1 rate limits are easy to hit on a 1000-doc batch.

**Fix**: the client retries with exponential backoff, but on persistent
limits you either:

- Upgrade your Anthropic tier.
- Run the pipeline with `--no-llm-tiebreak` to halve the call count.
- Throttle the batch externally (split into 100-doc chunks).

## Extraction issues

### "Span misaligned" / Pydantic validation error on `_spans_align`

A mention's declared `text` doesn't match `document.text[start:end]`.
This is the load-bearing invariant of the pipeline and it's enforced
strictly.

**Causes**:

- A human-edited review JSONL where `start`/`end` were nudged but
  `text` was not (or vice versa).
- Document text was edited between extract and import (review's
  `text_sha` should catch this; if it didn't, the linter wasn't run).
- A custom `LLMClient` produced mentions with offsets it invented
  (don't do this — see [Why GLiNER + LLM](concepts/why-gliner-plus-llm.md)).

**Fix**: run `hkx review lint workdir/review --workdir workdir/`. It
shows the exact line and the diff between expected and actual text.

### GLiNER misses entities you can see

Recall is too low. Possibilities:

- `--threshold` too high (default `0.4`).
- The schema description is vague — both GLiNER and the LLM read it.
- The entity type has zero examples and a generic name.

**Fix**, in order of effort:

1. Lower `--threshold` to `0.3` or `0.25`.
2. Improve the entity description: "A named individual human being"
   is better than "person".
3. Add 3–5 schema `examples` from your actual domain.

### GLiNER hits entities you don't want

Recall is too high.

**Fix**:

1. Raise `--threshold` to `0.5` or `0.6`.
2. Narrow the schema description.
3. As a last resort, post-filter — entities are typed, so a
   one-liner can drop unwanted types.

## Resolution issues

### Different people merged into one cluster

Embedding similarity overshot.

**Fix**: raise `--merge-threshold` to `0.85` or higher. Keep the LLM
tie-break enabled so borderline cases get a second look.

### The same person split across documents

Embedding similarity undershot, usually because canonical forms
varied a lot ("Marie Curie" vs "M. Curie" vs "Skłodowska-Curie").

**Fix**:

- Lower `--merge-threshold` to `0.78`–`0.80` and keep the LLM
  tie-break on (it costs tokens but is the right call here).
- Review the flagged entities — borderline merges that the LLM
  adjudicated should appear in the review file with reason
  `ambiguous_merge`.

### LLM returns a cluster id that doesn't exist

The model invented an id instead of picking from the input. This is
rare but happens.

**Fix**: it's a bug. The pipeline rejects the call and retries; if
retries are exhausted you'll see `LLMError`. File an issue with the
input documents (or a redacted reproducer).

## Review and import issues

### `hkx review lint` reports "text_sha mismatch"

The source document was edited between export and import.

**Fix**: don't edit the source document during a review cycle. Either:

- Discard the in-flight review and re-export against the new text.
- Restore the document to its pre-edit state.

The lint failure is a feature — it prevents applying stale edits to a
changed document.

### `hkx review lint` reports "unknown type"

A `op: add` line or `op: edit` line introduced a `type` that isn't in
`workdir/schema.yaml`.

**Fix**: use a declared type, or extend the schema and re-run
extraction.

### `hkx review import` rejects a line

Most often the mention `start`/`end` don't quote `document.text`
verbatim. The error message includes the offsets and the actual vs
expected text — usually a one-character drift.

**Fix**: edit the mention `start`/`end` to match, or revert that
mention's changes and add the entity afresh with `op: add`.

## Performance

### One document takes 10+ seconds

The first document of any run pays the model-load cost (~10s for
GLiNER + ~5s for the embedder, on CPU). Subsequent documents in the
same process should be <2s each.

**Fix**: if this is in a service, load the extractor once and reuse.
See [Library mode](how-to/library-mode.md#reusing-the-extractor-across-calls).

### Memory blows up on a large cross-doc run

The cross-doc clustering allocates an N² float32 matrix per entity
type. At 50k entities per type that's ~10 GB.

**Fix**:

- Split the run by document subset and merge registries afterward.
- Fork
  [`cluster_by_type`][hierokeryx.resolve.cluster.cluster_by_type] to
  use a vector index (FAISS / hnswlib) for the nearest-neighbour
  search.

See [Cross-document resolution › Memory and scale](how-to/cross-doc-resolution.md#memory-and-scale).

## Debugging tips

- **Turn on verbose logging.** `hkx pipeline ... --verbose` enables
  INFO-level logs from every stage.
- **Inspect the workdir.** `hkx inspect workdir/` prints schema,
  manifest, and a table of top entities. Often enough to spot the
  failure mode.
- **Look at the raw `extractions/*.json`.** They're plain JSON. `jq` is
  your friend.
- **Replay against VCR cassettes** for deterministic debugging — see
  [Determinism](concepts/determinism.md).

# Changelog

All notable changes to `hierokeryx` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Incremental cross-document resolution.** `pipeline.run` and `hkx resolve`
  / `hkx pipeline` now accept `--against <existing-workdir>` to assign new
  entities to an existing `EntityRegistry` without re-clustering the entire
  corpus. Backed by a new `registry_embeddings.npz` centroid sidecar
  persisted next to `registry.json`. `hkx resolve-centroids-rebuild` retrofits
  the sidecar onto workdirs created before this release. New library entry
  point: `hierokeryx.resolve.crossdoc.resolve_incremental`.
- **Evaluation harness.** New `hierokeryx.eval` module with pairwise and
  BCubed P/R/F1 metrics, JSONL gold-cluster format, and a threshold-sweep
  helper. New CLI: `hkx eval --workdir <path> --gold <path> [--sweep]
  [--json-out <file>]`.
- **OpenAI-compatible gateway LLM client** (`StandardGatewayClient`) for
  Azure-OpenAI-shaped gateways that proxy Claude on Bedrock or other
  backends. CLI commands gained `--provider {gateway,anthropic}`
  (`gateway` is now the default). `openai>=1.50` ships as an optional
  `[gateway]` extra.

### Changed

- `pipeline.run` now always persists a centroid sidecar
  (`registry_embeddings.npz` + `.meta.json`) alongside `registry.json`,
  recording the embedder id so future incremental runs can refuse to
  merge centroids across embedder changes.

## [0.1.0] — 2026-05-26

The initial alpha release. Full extract → resolve → review pipeline end-to-end.

### Added

- **Span extraction** via GLiNER 0.2.x with token-snapped, de-overlapped
  character offsets — no LLM-invented offsets.
- **Within-document coreference** via an LLM (Anthropic Claude by default,
  swappable via the `LLMClient` Protocol).
- **Cross-document entity resolution** via sentence-transformer embeddings,
  threshold-based union-find clustering, and LLM tie-break for borderline
  pairs.
- **File-based HITL review**: export low-confidence entities as one-record-
  per-line JSONL, edit in any editor, lint against a JSON Schema, and replay
  edits back into the workdir.
- **Pydantic-typed domain models**: `Document`, `Span`, `Mention`, `Entity`,
  `EntitySchema`, `EntityRegistry`, etc.
- **Confidence math**: ensemble within-doc + LLM self-report scores, plus a
  margin-aware cross-doc score that routes uncertain entities to review.
- **`hkx` CLI**: `schema init`/`validate`, `extract`, `resolve`, `pipeline`,
  `review export`/`lint`/`import`, `inspect`.
- **Nix devshell** providing Python 3.13, uv, ruff, and the C++ runtime libs
  ML wheels need on NixOS.

[Unreleased]: https://github.com/jevtarassov/hierokeryx/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jevtarassov/hierokeryx/releases/tag/v0.1.0

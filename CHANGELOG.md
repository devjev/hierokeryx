# Changelog

All notable changes to `hierokeryx` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

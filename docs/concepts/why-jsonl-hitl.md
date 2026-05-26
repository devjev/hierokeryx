# Why JSONL HITL

The H in HITL — Human In The Loop — is a design feature, not a fallback.
`hierokeryx` is built around the assumption that on real corpora some
percentage of entities will need a human eye, and the tooling for that
should be excellent.

## What JSONL gets you

The review files are
[`JSON Lines`](https://jsonlines.org/) — one JSON object per line, with
the first line being a header carrying `$schema`, `doc_id`, `text_sha`,
and `schema_version`.

This format is a deliberate choice, not the most obvious one (a single
JSON document, or a CSV, or a database row would all work). What JSONL
gets you:

- **Editor-agnostic.** VS Code, neovim, JetBrains, Sublime, plain
  `nano` — they all open `.jsonl` cleanly. Anyone on your team can
  participate in review without installing an annotation UI.
- **Inline JSON Schema validation.** The `$schema` field on the header
  line is a stable URI. Any JSON-Schema-aware editor will validate the
  rest of the lines against `src/hierokeryx/review/schema_v1.json` as
  the reviewer types — bad field types and missing required fields
  light up immediately.
- **Git-diffable.** One entity per line means a PR touching three
  entities shows exactly three changed lines. Compare to a wrapping
  JSON array: any small edit reformats the whole file.
- **Stream-processable.** A CI job can lint thousands of files line by
  line without loading any of them fully into memory.
- **Survives editor crashes.** A half-saved JSONL still has its first
  N lines valid. A half-saved nested JSON document is broken in the
  middle.

## What we considered and rejected

- **Single JSON file per doc.** Bad diffs, no streaming, no inline
  validation past the file's outer object.
- **CSV.** No nested mentions list. Quoting nightmare for the inevitable
  comma in a canonical name.
- **YAML.** Loose enough that hand edits introduce subtle parse errors.
  No mainstream inline-validation tooling.
- **Custom annotation UI.** Real cost: every change to the wire format
  requires a UI release. Discounted benefit: people don't actually want
  to open another tab to edit a few entities a week.

## The `op` field

Each line carries `op`: `keep`, `reject`, `edit`, or `add`. This is the
review's vocabulary:

- `keep` (default on export) — accept the entity as-is.
- `reject` — drop it entirely. Useful for false positives.
- `edit` — replace mutable fields (canonical, type, mentions, etc.).
- `add` — introduce a new human-curated entity. `id` must start with
  `human_` so it's distinguishable from model-generated ids in audit
  logs.

The semantics are deliberately small — there's no `merge` or `split`
op. Reviewers express merges by editing one line and rejecting another,
which keeps the diff explicit instead of hiding multi-entity changes in
one cryptic op.

## What the linter checks

[`lint_review_dir`][hierokeryx.review.lint.lint_review_dir] runs before
any review touches your workdir:

- Every mention's `text` field quotes the source document verbatim at
  the declared offsets.
- Every type is declared in the schema.
- `add` ops use `human_*` ids.
- `text_sha` on the header matches the current document text (catches
  stale reviews against an edited document).
- Required fields are present, types are sane, confidence is in `[0, 1]`.

The linter is your gate. If you wire it into CI, no malformed review
can corrupt a workdir.

## Why `text_sha`

It's a 16-character SHA-256 prefix of the document text. If the document
is edited between export and import, the hashes diverge and the linter
refuses to apply the review. This catches the worst silent failure mode
of any HITL system: reviewing an old version of the data and applying
the edits to the new version.

## When to skip the JSONL workflow

You don't have to use the JSONL loop. The cheapest alternative:

- Set `--review-threshold 0.0` so nothing is flagged, then post-process
  the workdir's `extractions/` files however you want.

Or build your own UI on top of
[`hierokeryx.review.jsonl`][hierokeryx.review.jsonl] (see
[Roundtrip JSONL programmatically](../how-to/programmatic-jsonl.md)).

The JSONL format is stable across patch releases — the `$schema` marker
on the header line carries the wire-format version. Major-version
changes will follow a documented migration path.

## Further reading

- [Determinism](determinism.md) — how `text_sha` and the schema
  fingerprint give the workdir its self-describing property.
- [Workdir layout](workdir-layout.md) — where review files live in the
  workdir and how they relate to extractions.

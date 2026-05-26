# Roundtrip JSONL programmatically

The `hkx review` CLI is the path most users take, but the underlying
helpers are public — wire them into your own tooling, an annotation UI,
or a CI gate.

## The public surface

From [`hierokeryx.review.jsonl`](../reference/api/review.md):

- [`write_review`][hierokeryx.review.jsonl.write_review] — write one
  document's JSONL file.
- [`write_review_dir`][hierokeryx.review.jsonl.write_review_dir] — write a
  whole directory, one file per document.
- [`read_review`][hierokeryx.review.jsonl.read_review] — parse one file.
- [`read_review_dir`][hierokeryx.review.jsonl.read_review_dir] — parse a
  directory keyed by `doc_id`.

From [`hierokeryx.review.lint`](../reference/api/review.md):

- [`lint_review_file`][hierokeryx.review.lint.lint_review_file] — validate
  one file against optional document / schema / original extraction.
- [`lint_review_dir`][hierokeryx.review.lint.lint_review_dir] — validate a
  whole directory.

From [`hierokeryx.review.apply`](../reference/api/review.md):

- [`apply_review`][hierokeryx.review.apply.apply_review] — replay edits
  onto an [`ExtractionResult`][hierokeryx.models.ExtractionResult].

## Export programmatically

```python
from hierokeryx import pipeline
from hierokeryx.confidence import route_for_review
from hierokeryx.review.jsonl import write_review_dir

results = pipeline.load_extractions_dir("workdir/extractions")
all_entities = [e for r in results for e in r.entities]
flagged = route_for_review(all_entities, cluster_threshold=0.7)

paths = write_review_dir(
    results,
    directory="review/",
    flagged=flagged,
    only_flagged=True,
)
print(f"Wrote {len(paths)} review file(s)")
```

`flagged` is a list of [`ReviewItem`][hierokeryx.models.ReviewItem]s
carrying `(doc_id, entity_id, reason)` triples. The reason ends up as
metadata on the exported line and helps the reviewer prioritise.

## Custom flagging rules

You don't have to use `route_for_review`. Build your own
`list[ReviewItem]`:

```python
from hierokeryx.models import ReviewItem

flagged = [
    ReviewItem(
        doc_id=result.document.id,
        entity_id=entity.id,
        reason="low_within_doc_confidence",
    )
    for result in results
    for entity in result.entities
    if entity.type == "Person" and entity.confidence < 0.8
]
```

Use this when, for example, you want a per-type policy ("review every
Person, ignore Organizations") or when you have an external signal
(domain-specific blocklist, prior labels) that should override the
default routing.

## Lint in CI

```python
from hierokeryx.review.lint import lint_review_dir
from hierokeryx.schema import load_schema
from hierokeryx import pipeline

schema = load_schema("workdir/schema.yaml")
extractions = {
    r.document.id: r
    for r in pipeline.load_extractions_dir("workdir/extractions")
}
documents = {doc_id: r.document for doc_id, r in extractions.items()}

errors_by_doc = lint_review_dir(
    "review/",
    documents=documents,
    schema=schema,
    extractions=extractions,
)

failed = {k: v for k, v in errors_by_doc.items() if v}
if failed:
    for doc_id, errs in failed.items():
        for err in errs:
            print(f"{doc_id}: {err}")
    sys.exit(1)
```

The linter catches:

- Mention spans that don't quote the document verbatim.
- Unknown entity types (not in the schema).
- Missing required fields, bad `op` values.
- Stale reviews against an edited document (header `text_sha` mismatch).
- `add` ops with non-`human_*` ids.

Wire this into a pre-commit hook or a GitHub Actions step on the PR that
modifies review files.

## Apply edits in-process

If you want to apply edits without going through the workdir round-trip:

```python
from hierokeryx.review.apply import apply_review
from hierokeryx.review.jsonl import read_review

header, lines = read_review("review/curie_2.jsonl")
original = pipeline.load_extraction("workdir/extractions/curie_2.json")
edited = apply_review(original, header, lines)
pipeline.save_extraction(edited, "workdir/extractions/curie_2.json")
```

`apply_review` re-validates every mention span against the original
document text — a bad offset raises immediately. This is the same check
the CLI's `hkx review import` runs.

## Annotation UI integration

The JSONL format is intentionally editor-agnostic, but it's also
straightforward to build a web UI on top:

1. List flagged files: `read_review_dir("review/")`.
2. For each file, present the entities with their mentions highlighted in
   the original document text.
3. On save, write the user's choices back through
   [`write_review`][hierokeryx.review.jsonl.write_review].
4. On commit, lint with
   [`lint_review_dir`][hierokeryx.review.lint.lint_review_dir] before
   accepting.

The wire format is stable across patch releases — the `$schema` field
on the JSONL header line is the stable URI consumers should check.
Major-version bumps will follow a documented migration path.

# Tutorial

This tutorial gets you from zero to a reviewed entity extraction in about
**15 minutes**. By the end you will have:

1. Defined an entity schema (Person / Organization).
2. Run the full pipeline on a small set of documents.
3. Inspected the results and round-tripped a low-confidence entity through
   a JSONL human review.

We assume you have already finished [Installation](../installation.md) — the
`hkx` command is on your PATH, `ANTHROPIC_API_KEY` is set, and you have a
working `uv` or `pip` environment.

!!! warning "First run downloads a model (~1.7 GB)"
    The first time GLiNER loads, it downloads the `urchade/gliner_large-v2.5`
    weights to your HuggingFace cache. This happens once and there is **no
    progress bar in `hkx` v0.1** — the CLI will look frozen for 30–90 seconds
    on a typical connection. If you want a progress indicator, run
    `huggingface-cli download urchade/gliner_large-v2.5` first.

## The three steps

<div class="grid cards" markdown>

- :material-numeric-1-circle:{ .lg } **[Define a schema](01-define-schema.md)**

    Declare the entity types you care about — `EntityType(name, description,
    examples)`. Schemas are per-call, so any domain works.

- :material-numeric-2-circle:{ .lg } **[Run the pipeline](02-run-pipeline.md)**

    `hkx pipeline <input> --schema schema.yaml --out workdir/` runs GLiNER,
    LLM coreference, and cross-document resolution end-to-end.

- :material-numeric-3-circle:{ .lg } **[Inspect and review](03-inspect-and-review.md)**

    Look at the results with `hkx inspect`, hand-edit a flagged entity in
    JSONL, lint, and import the edit back into the workdir.

</div>

## What you'll be working with

The example uses three short documents about the same scientist — Marie
Curie — across different angles. The pipeline should produce:

- Multiple mentions per entity within each document ("Curie", "she", "Marie Curie").
- One cross-document cluster identifying all three documents as referring to
  the same Marie Curie.
- Possibly a low-confidence cluster for an ambiguous mention, depending on
  the documents you supply.

If you want to follow along with the same fixtures the test suite uses, the
files are at `tests/fixtures/docs/doc_curie*.txt` in the repo.

[Start with the schema :material-arrow-right:](01-define-schema.md){ .md-button .md-button--primary }

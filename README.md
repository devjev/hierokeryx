# hierokeryx

Robust entity extraction and resolution from text. Built for cases where you need accurate character spans, within-document coreference, cross-document clustering, and a human in the loop for the cases the model misses.

The Python package is `hierokeryx`; the CLI command is `hkx`.

## Design

- **Spans**: a small zero-shot NER model (GLiNER) extracts mentions with exact character offsets. We don't ask an LLM to invent offsets.
- **Resolution**: an LLM clusters mentions into entities, picks canonical forms, and decides cross-document merges.
- **Schema**: user-defined per call. `EntityType(name, description, examples)` — any domain.
- **Human-in-the-loop**: file-based JSONL round-trip. Export low-confidence items, edit in any editor, re-import.
- **LLM provider-agnostic**: a small `LLMClient` Protocol; Anthropic Claude is the default and only v1 backend.

Output, per document: `ExtractionResult(document, entities)` where each `Entity` carries a list of `Mention`s with character `Span`s, a `canonical` form, an optional cross-doc `cluster_id`, and a `confidence`.

## Quick start

```bash
uv sync
uv run hkx schema init --out schema.yaml
# edit schema.yaml to declare your entity types

export ANTHROPIC_API_KEY=...
uv run hkx pipeline path/to/docs --schema schema.yaml --out workdir/

uv run hkx review export workdir/ --only-flagged --out review/
# hand-edit any review/<doc_id>.jsonl line
uv run hkx review lint   review/
uv run hkx review import review/ --workdir workdir/
uv run hkx inspect workdir/
```

## CLI

```
hkx schema init                       [--out schema.yaml]
hkx schema validate <schema.yaml>

hkx extract <input>  --schema schema.yaml --out workdir/
hkx resolve <workdir>                 [--threshold 0.75]
hkx pipeline <input> --schema schema.yaml --out workdir/ [--review-threshold 0.7]

hkx review export <workdir> --out review/ [--only-flagged]
hkx review lint   <path>
hkx review import <review/> --workdir <workdir>

hkx inspect <workdir>
```

`<input>` accepts a file, glob, or directory.

## Library

```python
from hierokeryx import pipeline, EntitySchema, EntityType, Document

schema = EntitySchema(types=[
    EntityType(name="Person",       description="A named individual human"),
    EntityType(name="Organization", description="A company, agency, institution"),
])

result = pipeline.run_one(
    Document(id="d1", text="Marie Curie won the Nobel Prize. Curie also..."),
    schema=schema,
)

for entity in result.entities:
    print(entity.canonical, entity.type, [m.span.text for m in entity.mentions])
```

## Status

Alpha. v1 ships full pipeline end-to-end with GLiNER + Claude. See `/home/jev/.claude/plans/start-a-project-for-robust-truffle.md` for the design document.

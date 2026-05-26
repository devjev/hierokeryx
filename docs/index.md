---
hide:
  - navigation
---

# hierokeryx

> Robust entity extraction and resolution from text, with a human in the loop
> for the cases the model misses.

`hierokeryx` is a small Python library and CLI (`hkx`) that turns unstructured
text into typed entities you can trust. It pairs a zero-shot NER model
([GLiNER](https://github.com/urchade/GLiNER)) for exact character spans with
an LLM ([Anthropic Claude](https://www.anthropic.com/claude) by default) for
within-document coreference and cross-document resolution, then routes any
low-confidence entity to a file-based JSONL review you can edit in any editor.

[Get started in 15 minutes :material-arrow-right:](tutorial/index.md){ .md-button .md-button--primary }
[See the API reference](reference/api/index.md){ .md-button }

## Design philosophy

- **Spans are facts; LLM-invented offsets are not.** A small zero-shot NER
  model produces character-aligned mentions. The LLM never invents offsets.
- **Resolution is judgment.** Clustering mentions into entities, picking
  canonical forms, and deciding cross-document merges requires reasoning —
  that's what the LLM is for.
- **Humans review what the model is unsure about.** Confidence-routed JSONL
  is editor-agnostic, diffs cleanly, and survives every IDE you'll ever use.
- **Provider-agnostic.** The LLM lives behind a tiny `LLMClient` Protocol.
  Bring your own provider; the rest of the pipeline doesn't care.
- **Schema-per-call.** `EntityType(name, description, examples)`. Any domain —
  people, drugs, parts, parties — without retraining.

## At a glance

```python
from hierokeryx import pipeline, Document, EntitySchema, EntityType
from hierokeryx.llm.anthropic_client import AnthropicClient

schema = EntitySchema(types=[
    EntityType(name="Person",       description="A named individual human"),
    EntityType(name="Organization", description="A company, agency, institution"),
])

result = pipeline.run_one(
    Document(id="d1", text="Marie Curie won the Nobel Prize. Curie also..."),
    schema=schema,
    llm_client=AnthropicClient(),
)

for entity in result.entities:
    print(entity.canonical, entity.type, [m.span.text for m in entity.mentions])
```

Output, per document, is an
[`ExtractionResult`][hierokeryx.models.ExtractionResult]: a document plus a
list of [`Entity`][hierokeryx.models.Entity] objects, each carrying its
[`Mention`][hierokeryx.models.Mention]s (with character
[`Span`][hierokeryx.models.Span]s), a canonical form, a cross-doc
`cluster_id`, and a `confidence`.

## When to use it

`hierokeryx` is the right tool when you need **any of** the following:

- Exact character offsets you can highlight in your UI.
- Domain-specific entity types (the schema is user-defined per call).
- Within-document coreference (resolving "she" / "Curie" / "the chemist" to
  Marie Curie).
- Cross-document clustering ("Marie Curie" in doc A and "M. Curie" in doc B
  are the same person).
- A human review loop without writing a custom annotation UI.

It is **not** the right tool if you need:

- Streaming, sub-100ms-per-document extraction. The LLM round-trip dominates.
- Open-vocabulary extraction without a schema. Define one.
- Relations or events between entities. Out of scope at v0.1.

[Compare to spaCy, GLiNER raw, and LangChain extraction :material-arrow-right:](comparison.md){ .md-button }

## Status

Alpha (v0.1.0). The library and CLI ship end-to-end with GLiNER + Claude.
The public API may still change before v0.2; see
[`CHANGELOG.md`](changelog.md). Feedback and bug reports very welcome.

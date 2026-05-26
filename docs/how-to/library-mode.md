# Library mode (no CLI)

The CLI is a thin wrapper around the library. If you're building a service
or wiring `hierokeryx` into an existing app, talk to the library directly.

## Single document, no workdir

The simplest case — extract and resolve one document, get a result back:

```python
from hierokeryx import pipeline, Document, EntitySchema, EntityType
from hierokeryx.llm.anthropic_client import AnthropicClient

schema = EntitySchema(types=[
    EntityType(name="Person",       description="A named individual human being."),
    EntityType(name="Organization", description="A company, agency, institution."),
])

result = pipeline.run_one(
    Document(id="d1", text="Marie Curie won the Nobel Prize. Curie also..."),
    schema=schema,
    llm_client=AnthropicClient(),
)

for entity in result.entities:
    print(entity.canonical, entity.type,
          [m.span.text for m in entity.mentions])
```

`run_one` runs GLiNER extraction and within-document coreference. It
does **not** do cross-document resolution and does **not** write
anything to disk — perfect for stateless services.

## Many documents, no workdir

```python
results = [
    pipeline.run_one(doc, schema=schema, llm_client=llm)
    for doc in docs
]

from hierokeryx.resolve.crossdoc import resolve_crossdoc
from hierokeryx.resolve.embed import SentenceTransformerEmbedder

resolved, registry = resolve_crossdoc(
    results,
    schema,
    llm_client=llm,
    embedder=SentenceTransformerEmbedder(),
)
```

`resolved` is the updated list of
[`ExtractionResult`][hierokeryx.models.ExtractionResult]s with
`cluster_id` populated; `registry` is the
[`EntityRegistry`][hierokeryx.models.EntityRegistry].

## Many documents, with workdir

Identical to what the CLI does:

```python
from pathlib import Path
from hierokeryx import pipeline

run = pipeline.run(
    documents=docs,
    schema=schema,
    workdir=Path("workdir"),
    llm_client=llm,
    review_threshold=0.7,
    merge_threshold=0.82,
)

# Full PipelineRun:
print(run.workdir)
print(len(run.extraction_results))
print(len(run.registry.clusters))
print(len(run.flagged))
print(run.review_paths)
```

[`PipelineRun`][hierokeryx.pipeline.PipelineRun] is a frozen dataclass
carrying every artifact path the run produced, so you can stream them
back to a service or commit them to a git repo.

## Reusing the extractor across calls

The GLiNER model takes ~10 seconds to load. In a long-running service,
load once and reuse:

```python
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.resolve.embed import SentenceTransformerEmbedder

# At service startup:
extractor = GLiNERExtractor(device="cuda")
embedder  = SentenceTransformerEmbedder(device="cuda")
llm       = AnthropicClient()

# Per request:
def extract_one(doc: Document, schema: EntitySchema):
    return pipeline.run_one(
        doc,
        schema=schema,
        extractor=extractor,
        llm_client=llm,
    )
```

`GLiNERExtractor` is intentionally cheap to construct (the model load is
lazy — first `.extract()` call triggers it). Once loaded, it's
thread-safe for concurrent `.extract()` calls.

## Loading a workdir produced elsewhere

If a separate process did the extraction:

```python
from hierokeryx import pipeline

results  = pipeline.load_extractions_dir("workdir/extractions")
registry = pipeline.load_registry("workdir/registry.json")

# Or a single doc:
result = pipeline.load_extraction("workdir/extractions/curie_1.json")
```

All loaders use the same Pydantic models the rest of the API uses, so
your code doesn't care whether a result came from an in-process call or
a serialized JSON file.

## What about async?

The Anthropic SDK has an async client; `hierokeryx`'s default backend
uses the sync one. If you need async, write a custom backend per
[Use a custom LLM backend](custom-llm-backend.md) and call it via
`asyncio.to_thread` from your async handler. There's no async pipeline
API at v0.1.

## Logging

The library uses the standard `logging` module under the `hierokeryx`
namespace. Configure it however your app does:

```python
import logging
logging.getLogger("hierokeryx").setLevel(logging.DEBUG)
```

Useful loggers:

- `hierokeryx.pipeline` — high-level stage transitions.
- `hierokeryx.extract.gliner_runner` — model load and per-doc mention counts.
- `hierokeryx.resolve.cluster` — borderline candidates and tie-break decisions.
- `hierokeryx.llm.anthropic_client` — prompt cache hits and retries.

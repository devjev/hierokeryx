# API reference

The public Python API of `hierokeryx`. Everything documented here is
covered by [SemVer](https://semver.org/) — backwards-compatible across
patch and minor releases, breaking changes only in majors. The package
is currently v0.1 (alpha), so the public surface may still shift before
v1.0.

## Top-level modules

<div class="grid cards" markdown>

- :material-cube-outline:{ .lg } **[hierokeryx.models](models.md)**

    Domain types: `Document`, `Span`, `Mention`, `Entity`,
    `EntitySchema`, `ExtractionResult`, `EntityRegistry`, and the HITL
    helpers.

- :material-pipe:{ .lg } **[hierokeryx.pipeline](pipeline.md)**

    Orchestrator: `run`, `run_one`, `import_reviewed`, plus the
    workdir persistence helpers.

- :material-script-text-outline:{ .lg } **[hierokeryx.schema](schema.md)**

    Load and save `EntitySchema` from YAML or JSON.

- :material-percent-outline:{ .lg } **[hierokeryx.confidence](confidence.md)**

    Within-doc and cross-doc confidence ensembling, plus the review
    router.

</div>

## Subpackages

<div class="grid cards" markdown>

- :material-magnify:{ .lg } **[hierokeryx.extract](extract.md)**

    GLiNER-backed span extraction and token-alignment helpers.

- :material-vector-link:{ .lg } **[hierokeryx.resolve](resolve.md)**

    Within-doc coref, cross-doc clustering, and the embedder.

- :material-robot-outline:{ .lg } **[hierokeryx.llm](llm.md)**

    LLM Protocol, Anthropic Claude implementation, prompts and tool
    schemas.

- :material-account-edit:{ .lg } **[hierokeryx.review](review.md)**

    JSONL HITL: read/write, lint, apply.

</div>

## Top-level re-exports

For convenience, the core types are re-exported at the package root:

```python
from hierokeryx import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    Entity,
    EntityRegistry,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    MergeDecision,
    ReviewItem,
    Span,
)
```

Functions and protocols stay in their submodule — `pipeline.run`,
`hierokeryx.llm.protocol.LLMClient`, etc.

"""Basic library usage of hierokeryx.

Run inside the project's nix dev shell:

    nix develop
    export ANTHROPIC_API_KEY=...
    uv run python examples/basic_usage.py
"""

from __future__ import annotations

from hierokeryx import Document, EntitySchema, EntityType, pipeline
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.anthropic_client import AnthropicClient


def main() -> None:
    schema = EntitySchema(
        types=[
            EntityType(name="Person", description="A named individual human"),
            EntityType(name="Organization", description="A company, agency, institution"),
            EntityType(name="Location", description="A geographic place"),
        ]
    )

    doc = Document(
        id="curie-001",
        text=(
            "Marie Curie was a Polish-French physicist. Curie discovered radium "
            "with her husband Pierre in Paris. She won two Nobel Prizes."
        ),
    )

    # Use the small GLiNER variant for a fast demo; default is the large one.
    extractor = GLiNERExtractor(model_id="urchade/gliner_small-v2.1")
    llm = AnthropicClient()  # reads ANTHROPIC_API_KEY from env

    result = pipeline.run_one(doc, schema, extractor=extractor, llm_client=llm)

    print(f"\nDocument: {result.document.id}")
    print(f"Schema fingerprint: {result.schema_version}")
    print(f"Entities: {len(result.entities)}\n")
    for entity in result.entities:
        spans = ", ".join(
            f"{m.span.text!r}@({m.span.start},{m.span.end})" for m in entity.mentions
        )
        print(f"  [{entity.type}] {entity.canonical} (conf={entity.confidence:.2f})")
        print(f"    mentions: {spans}")


if __name__ == "__main__":
    main()

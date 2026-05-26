"""Basic library usage of hierokeryx.

Extracts entities from a single document with GLiNER spans + LLM coref.
Uses the OpenAI-compatible gateway client by default — swap in
`AnthropicClient` if you'd rather talk to the Anthropic API directly.

Run inside the project's nix dev shell:

    nix develop

    # Default: OpenAI-compatible gateway (e.g. Azure-shaped, Bedrock-backed)
    export GATEWAY_BASE_URL=https://org.example.net
    export GATEWAY_API_KEY=...
    export GATEWAY_API_VERSION=2024-10-21
    uv run python examples/basic_usage.py

    # Or talk to Anthropic directly: edit `make_llm()` below.
"""

from __future__ import annotations

from hierokeryx import Document, EntitySchema, EntityType, pipeline
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.protocol import LLMClient
from hierokeryx.llm.standard_gateway_client import StandardGatewayClient


def make_llm() -> LLMClient:
    """Default to the gateway client. Swap to AnthropicClient if you prefer.

    The pipeline only depends on the LLMClient Protocol, so either is fine:

        from hierokeryx.llm.anthropic_client import AnthropicClient
        return AnthropicClient()  # reads ANTHROPIC_API_KEY
    """
    return StandardGatewayClient()  # reads GATEWAY_BASE_URL / _API_KEY / _API_VERSION


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
    llm = make_llm()

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

    print(
        "\nNext steps:"
        "\n  - examples/incremental_pipeline.py — resolve new docs against an existing registry"
        "\n  - examples/evaluate_with_gold.py   — score a workdir against JSONL gold labels"
    )


if __name__ == "__main__":
    main()

# Use a custom LLM backend

The default backend is Anthropic Claude. Swapping providers is a matter of
implementing the [`LLMClient`][hierokeryx.llm.protocol.LLMClient] Protocol
— two methods, no inheritance — and passing your client into the pipeline.

## The protocol

```python
from typing import Protocol, runtime_checkable
from hierokeryx.models import (
    CorefCluster, CrossDocCandidate, Document,
    EntitySchema, Mention, MergeDecision,
)

@runtime_checkable
class LLMClient(Protocol):
    def cluster_mentions(
        self,
        document: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> list[CorefCluster]: ...

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]: ...
```

Any class that exposes these two methods qualifies. Subclassing is not
required — duck typing is enough.

## Minimal example: OpenAI tool-use

```python title="my_openai_client.py"
from __future__ import annotations

import json
import os

from openai import OpenAI
from hierokeryx.llm.prompts import (
    COREF_SYSTEM_PROMPT,
    CROSSDOC_SYSTEM_PROMPT,
    render_schema_block,
)
from hierokeryx.llm.tools import (
    RECORD_CLUSTERS_TOOL,
    RECORD_MERGE_DECISIONS_TOOL,
)
from hierokeryx.llm.protocol import LLMError
from hierokeryx.models import (
    CorefCluster, CrossDocCandidate, Document,
    EntitySchema, MergeDecision, Mention,
)


def _to_openai_tool(t: dict) -> dict:
    """Convert hierokeryx's Anthropic-shaped tool spec to OpenAI's shape."""
    return {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }


class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = model

    def cluster_mentions(
        self,
        document: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> list[CorefCluster]:
        user_msg = json.dumps({
            "document": document.text,
            "mentions": [
                {"id": m.id, "text": m.span.text, "type": m.type,
                 "start": m.span.start, "end": m.span.end}
                for m in mentions
            ],
        })
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system",
                 "content": COREF_SYSTEM_PROMPT + "\n\n" + render_schema_block(schema)},
                {"role": "user", "content": user_msg},
            ],
            tools=[_to_openai_tool(RECORD_CLUSTERS_TOOL)],
            tool_choice={"type": "function",
                         "function": {"name": "record_clusters"}},
        )
        try:
            payload = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
        except (IndexError, KeyError, json.JSONDecodeError) as exc:
            raise LLMError(f"OpenAI returned no usable tool call: {exc}") from exc

        return [
            CorefCluster(
                mention_ids=c["mention_ids"],
                canonical=c["canonical"],
                type=c["type"],
                confidence=c["confidence"],
                rationale=c.get("rationale"),
            )
            for c in payload["clusters"]
        ]

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        # Symmetric to cluster_mentions; omitted for brevity.
        ...
```

Use it:

```python
from hierokeryx import pipeline, Document, EntitySchema, EntityType

schema = EntitySchema(types=[EntityType(name="Person", description="...")])
docs = [Document(id="d1", text="...")]

run = pipeline.run(
    documents=docs,
    schema=schema,
    workdir="workdir/",
    llm_client=OpenAIClient(),
)
```

## What to reuse from `hierokeryx.llm`

You don't have to rebuild prompts or tool schemas — they're public.
From [`hierokeryx.llm.prompts`](../reference/api/llm.md):

- `COREF_SYSTEM_PROMPT` — within-doc system prompt with the hard rules
  the resolver depends on.
- `CROSSDOC_SYSTEM_PROMPT` — cross-doc tie-break system prompt.
- [`render_schema_block`][hierokeryx.llm.prompts.render_schema_block] —
  emits a stable JSON block to append to the system prompt.

From [`hierokeryx.llm.tools`](../reference/api/llm.md):

- `RECORD_CLUSTERS_TOOL` and `RECORD_MERGE_DECISIONS_TOOL` —
  JSON-Schema tool definitions in Anthropic shape; trivially
  convertible to OpenAI's function-calling shape.

These live in the cache prefix — anything you change here invalidates
prompt caching on providers that support it.

## When NOT to write a custom backend

If your provider has a stable Anthropic-shaped tool-use API (most modern
LLM gateways and a few self-hosted servers like
[llama.cpp's `--tools`](https://github.com/ggerganov/llama.cpp) endpoint),
you can usually just point
[`AnthropicClient`][hierokeryx.llm.anthropic_client.AnthropicClient] at
the gateway via its `api_key` / base-URL parameter and skip the custom
class entirely.

## Testing your backend

Reuse the test pattern in `tests/unit/test_anthropic_client.py`:

1. Build a small `Document` + `Mention` fixture.
2. Call `cluster_mentions` and assert the returned
   `CorefCluster.mention_ids` partition the input ids.
3. Use [`vcrpy`](https://github.com/kevin1024/vcrpy) to record real API
   responses once, then replay them on subsequent runs.

The full pipeline integration test
(`tests/integration/test_pipeline_smoke.py`) is provider-agnostic — drop
in your client and run `pytest -m integration`.

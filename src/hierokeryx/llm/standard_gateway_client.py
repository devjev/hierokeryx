"""OpenAI-compatible gateway implementation of LLMClient.

Targets Azure-OpenAI-shaped gateways (URL pattern
`{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=...`)
that may route to any backing model — including Anthropic Claude on Bedrock
via cross-region inference profile IDs like
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`.

The model string passed in is used both as the OpenAI request's `model` field
and as the URL deployment segment (Azure SDK behavior). Whether the gateway
routes on the path or on the body is gateway-specific; passing the same value
in both places works for both styles.

Tool-use schemas are translated from the Anthropic shape used elsewhere in
this package into OpenAI's `tools` / `tool_choice` shape on the fly.
Anthropic-style `cache_control` markers are not sent — the chat-completions
wire format does not carry them, and prompt caching (if any) is whatever the
gateway/backend does automatically.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from hierokeryx.llm.prompts import (
    COREF_SYSTEM_PROMPT,
    CROSSDOC_SYSTEM_PROMPT,
    render_schema_block,
)
from hierokeryx.llm.protocol import LLMError
from hierokeryx.llm.tools import RECORD_CLUSTERS_TOOL, RECORD_MERGE_DECISIONS_TOOL
from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    Mention,
    MergeDecision,
)

logger = logging.getLogger(__name__)

DEFAULT_COREF_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_TIEBREAK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_MAX_TOKENS = 4096


class StandardGatewayClient:
    """LLMClient backed by an OpenAI-compatible (Azure-shaped) gateway.

    Config is read from environment variables by default:
      - `GATEWAY_BASE_URL`   e.g. `https://org.net`
      - `GATEWAY_API_KEY`    sent as the Azure-style `api-key` header
      - `GATEWAY_API_VERSION` e.g. `2024-10-21`
    """

    def __init__(
        self,
        coref_model: str = DEFAULT_COREF_MODEL,
        tiebreak_model: str = DEFAULT_TIEBREAK_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        max_attempts: int = 2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.coref_model = coref_model
        self.tiebreak_model = tiebreak_model
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self._base_url = base_url or os.environ.get("GATEWAY_BASE_URL")
        self._api_key = api_key or os.environ.get("GATEWAY_API_KEY")
        self._api_version = api_version or os.environ.get("GATEWAY_API_VERSION")
        self._client: Any = None
        self._cache_stats = {"read": 0, "input": 0}

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import AzureOpenAI

            missing = [
                name for name, val in (
                    ("GATEWAY_BASE_URL", self._base_url),
                    ("GATEWAY_API_KEY", self._api_key),
                    ("GATEWAY_API_VERSION", self._api_version),
                ) if not val
            ]
            if missing:
                raise LLMError(
                    f"StandardGatewayClient missing config: {', '.join(missing)}. "
                    "Export the env vars or pass them to the constructor."
                )
            self._client = AzureOpenAI(
                azure_endpoint=self._base_url,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def cache_hit_ratio(self) -> float | None:
        """Fraction of input tokens reported as cached by the gateway, if any."""
        total = self._cache_stats["input"]
        return (self._cache_stats["read"] / total) if total else None

    def cluster_mentions(
        self,
        document: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> list[CorefCluster]:
        if not mentions:
            return []

        mention_payload = [
            {
                "id": m.id,
                "type": m.type,
                "start": m.span.start,
                "end": m.span.end,
                "text": m.span.text,
                "score": round(m.score, 3),
            }
            for m in mentions
        ]
        user_block = (
            "<document>\n"
            f"<id>{document.id}</id>\n"
            f"<text>{document.text}</text>\n"
            "</document>\n\n"
            "<mentions>\n"
            f"{json.dumps(mention_payload, ensure_ascii=False, indent=2)}\n"
            "</mentions>"
        )

        raw = self._call_tool(
            model=self.coref_model,
            system_prompt=COREF_SYSTEM_PROMPT,
            schema=schema,
            tool=RECORD_CLUSTERS_TOOL,
            user_text=user_block,
        )
        clusters_raw = raw.get("clusters", [])
        if not isinstance(clusters_raw, list):
            raise LLMError(f"record_clusters returned non-list 'clusters': {raw!r}")

        valid_mention_ids = {m.id for m in mentions}
        valid_types = set(schema.type_names)
        clusters: list[CorefCluster] = []
        for raw_cluster in clusters_raw:
            try:
                cluster = CorefCluster.model_validate(raw_cluster)
            except ValidationError as e:
                raise LLMError(f"Invalid cluster from LLM: {raw_cluster!r} ({e})") from e
            unknown_ids = set(cluster.mention_ids) - valid_mention_ids
            if unknown_ids:
                raise LLMError(
                    f"Cluster references unknown mention ids: {sorted(unknown_ids)}"
                )
            if cluster.type not in valid_types:
                raise LLMError(
                    f"Cluster type {cluster.type!r} not in schema {sorted(valid_types)}"
                )
            clusters.append(cluster)

        seen: set[str] = set()
        for c in clusters:
            dupes = seen.intersection(c.mention_ids)
            if dupes:
                raise LLMError(f"Mention(s) {sorted(dupes)} appear in multiple clusters")
            seen.update(c.mention_ids)
        missing = valid_mention_ids - seen
        if missing:
            raise LLMError(
                f"LLM did not cluster all mentions; missing: {sorted(missing)}"
            )
        return clusters

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        if not candidates:
            return []

        payload = [c.model_dump() for c in candidates]
        user_block = (
            "<candidates>\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
            "</candidates>"
        )
        raw = self._call_tool(
            model=self.tiebreak_model,
            system_prompt=CROSSDOC_SYSTEM_PROMPT,
            schema=schema,
            tool=RECORD_MERGE_DECISIONS_TOOL,
            user_text=user_block,
        )
        decisions_raw = raw.get("decisions", [])
        if not isinstance(decisions_raw, list):
            raise LLMError(f"record_merge_decisions returned non-list: {raw!r}")

        valid_ids = {c.entity_id for c in candidates}
        valid_clusters = {c.nearest_cluster_id for c in candidates if c.nearest_cluster_id}
        decisions: list[MergeDecision] = []
        for raw_d in decisions_raw:
            try:
                d = MergeDecision.model_validate(raw_d)
            except ValidationError as e:
                raise LLMError(f"Invalid decision: {raw_d!r} ({e})") from e
            if d.entity_id not in valid_ids:
                raise LLMError(f"Decision for unknown entity_id={d.entity_id!r}")
            if d.target_cluster_id is not None and d.target_cluster_id not in valid_clusters:
                raise LLMError(
                    f"Decision references unknown cluster_id={d.target_cluster_id!r}"
                )
            decisions.append(d)
        decided_ids = {d.entity_id for d in decisions}
        missing = valid_ids - decided_ids
        if missing:
            raise LLMError(f"Missing decisions for entity_ids: {sorted(missing)}")
        return decisions

    def _call_tool(
        self,
        *,
        model: str,
        system_prompt: str,
        schema: EntitySchema,
        tool: dict[str, Any],
        user_text: str,
    ) -> dict[str, Any]:
        """Force a single function-call response, retrying on transient errors."""
        system_text = f"{system_prompt}\n\n{render_schema_block(schema)}"
        openai_tool = _to_openai_tool(tool)
        tool_choice = {"type": "function", "function": {"name": tool["name"]}}

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    tools=[openai_tool],
                    tool_choice=tool_choice,
                )
            except Exception as e:
                last_error = e
                logger.warning("Gateway call failed (attempt %d): %s", attempt, e)
                continue

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._cache_stats["input"] += getattr(usage, "prompt_tokens", 0) or 0
                details = getattr(usage, "prompt_tokens_details", None)
                if details is not None:
                    self._cache_stats["read"] += getattr(details, "cached_tokens", 0) or 0

            choice = response.choices[0] if response.choices else None
            message = getattr(choice, "message", None) if choice else None
            tool_calls = getattr(message, "tool_calls", None) if message else None
            if tool_calls:
                for call in tool_calls:
                    fn = getattr(call, "function", None)
                    if fn and getattr(fn, "name", None) == tool["name"]:
                        try:
                            return dict(json.loads(fn.arguments))
                        except json.JSONDecodeError as e:
                            last_error = LLMError(
                                f"Tool arguments for {tool['name']} were not valid JSON: {e}"
                            )
                            break
                else:
                    last_error = LLMError(
                        f"No tool_call for {tool['name']}; "
                        f"got: {[getattr(c.function, 'name', None) for c in tool_calls]}"
                    )
            else:
                last_error = LLMError(
                    "Response contained no tool_calls; "
                    f"finish_reason={getattr(choice, 'finish_reason', None)!r}"
                )

        raise LLMError(
            f"Gateway call failed after {self.max_attempts} attempt(s): {last_error}"
        ) from last_error


def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic-shaped tool spec to OpenAI's `tools` entry shape."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }

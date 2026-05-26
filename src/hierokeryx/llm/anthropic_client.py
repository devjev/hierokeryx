"""Anthropic Claude implementation of LLMClient.

Uses tool-use for structured output and prompt caching on the stable prefix
(system prompt + schema + tool definitions). Per-document content is sent
after the cache breakpoint so a multi-doc batch reuses the cached prefix on
each call.
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

DEFAULT_COREF_MODEL = "claude-haiku-4-5"
DEFAULT_TIEBREAK_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 4096


class AnthropicClient:
    """Anthropic Claude implementation of the LLMClient protocol."""

    def __init__(
        self,
        coref_model: str = DEFAULT_COREF_MODEL,
        tiebreak_model: str = DEFAULT_TIEBREAK_MODEL,
        api_key: str | None = None,
        max_attempts: int = 2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.coref_model = coref_model
        self.tiebreak_model = tiebreak_model
        self.max_attempts = max_attempts
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client: Any = None
        self._cache_stats = {"read": 0, "creation": 0, "input": 0}

    @property
    def client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            if not self._api_key:
                raise LLMError(
                    "ANTHROPIC_API_KEY is not set. Export it or pass api_key="
                    "to AnthropicClient."
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def cache_hit_ratio(self) -> float | None:
        """Fraction of input tokens served from cache across the client's lifetime."""
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
        """Send a tool-use request, retrying on transient JSON/validation errors."""
        system_blocks = [
            {"type": "text", "text": system_prompt},
            {
                "type": "text",
                "text": render_schema_block(schema),
                "cache_control": {"type": "ephemeral"},
            },
        ]
        tools_param = [tool]
        tool_choice = {"type": "tool", "name": tool["name"]}

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    temperature=0.0,
                    system=system_blocks,
                    tools=tools_param,
                    tool_choice=tool_choice,
                    messages=[{"role": "user", "content": user_text}],
                )
            except Exception as e:
                last_error = e
                logger.warning("Anthropic call failed (attempt %d): %s", attempt, e)
                continue

            usage = getattr(response, "usage", None)
            if usage is not None:
                self._cache_stats["input"] += getattr(usage, "input_tokens", 0) or 0
                self._cache_stats["read"] += (
                    getattr(usage, "cache_read_input_tokens", 0) or 0
                )
                self._cache_stats["creation"] += (
                    getattr(usage, "cache_creation_input_tokens", 0) or 0
                )

            for block in response.content:
                if getattr(block, "type", None) == "tool_use" and block.name == tool["name"]:
                    return dict(block.input)
            last_error = LLMError(
                f"Response did not contain a tool_use for {tool['name']}; "
                f"got blocks: {[getattr(b, 'type', None) for b in response.content]}"
            )

        raise LLMError(
            f"LLM call failed after {self.max_attempts} attempt(s): {last_error}"
        ) from last_error

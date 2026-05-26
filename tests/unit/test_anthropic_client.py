"""Tests for AnthropicClient — validation, error paths, cache stats.

We never hit the real Anthropic API. Instead, we replace `client.messages.create`
with a stub that returns a canned response object whose .content carries a
tool_use block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from hierokeryx.llm.anthropic_client import AnthropicClient
from hierokeryx.llm.protocol import LLMError
from hierokeryx.models import (
    CrossDocCandidate,
    Document,
    EntitySchema,
    EntityType,
    Mention,
    Span,
)


@dataclass
class _ToolUseBlock:
    type: str
    name: str
    input: dict[str, Any]


@dataclass
class _Usage:
    input_tokens: int = 100
    cache_read_input_tokens: int = 60
    cache_creation_input_tokens: int = 0


@dataclass
class _FakeResponse:
    content: list[Any]
    usage: _Usage


def _fake_messages_create_factory(tool_name: str, tool_input: dict[str, Any]):
    def _fake(**kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            content=[_ToolUseBlock(type="tool_use", name=tool_name, input=tool_input)],
            usage=_Usage(),
        )

    return _fake


@pytest.fixture
def schema() -> EntitySchema:
    return EntitySchema(
        types=[
            EntityType(name="Person", description="A human individual"),
            EntityType(name="Organization", description="A formal group"),
        ]
    )


@pytest.fixture
def doc() -> Document:
    return Document(id="d1", text="Marie Curie won the Nobel. Curie was Polish.")


@pytest.fixture
def mentions() -> list[Mention]:
    return [
        Mention(id="m1", span=Span(start=0, end=11, text="Marie Curie"), type="Person", score=0.95),
        Mention(id="m2", span=Span(start=27, end=32, text="Curie"), type="Person", score=0.85),
    ]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> AnthropicClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-testing")
    return AnthropicClient()


def _install_fake(monkeypatch: pytest.MonkeyPatch, client: AnthropicClient,
                  tool_name: str, tool_input: dict[str, Any]) -> None:
    """Replace the underlying Anthropic client with a stub returning tool_input."""

    class _StubMessages:
        def create(self, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse(
                content=[_ToolUseBlock(type="tool_use", name=tool_name, input=tool_input)],
                usage=_Usage(),
            )

    class _StubAnthropic:
        messages = _StubMessages()

    client._client = _StubAnthropic()


class TestClusterMentions:
    def test_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {
                        "mention_ids": ["m1", "m2"],
                        "canonical": "Marie Curie",
                        "type": "Person",
                        "confidence": 0.92,
                        "rationale": "Both refer to Marie Curie.",
                    }
                ]
            },
        )
        clusters = client.cluster_mentions(doc, mentions, schema)
        assert len(clusters) == 1
        assert clusters[0].canonical == "Marie Curie"
        assert clusters[0].mention_ids == ["m1", "m2"]

    def test_empty_mentions_returns_empty(
        self,
        client: AnthropicClient,
        doc: Document,
        schema: EntitySchema,
    ) -> None:
        # Should not make any API call when there are no mentions.
        assert client.cluster_mentions(doc, [], schema) == []

    def test_unknown_mention_id_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {
                        "mention_ids": ["m1", "ghost"],
                        "canonical": "X",
                        "type": "Person",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        with pytest.raises(LLMError, match="unknown mention ids"):
            client.cluster_mentions(doc, mentions, schema)

    def test_unknown_type_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {
                        "mention_ids": ["m1", "m2"],
                        "canonical": "X",
                        "type": "Alien",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        with pytest.raises(LLMError, match="not in schema"):
            client.cluster_mentions(doc, mentions, schema)

    def test_overlapping_clusters_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {"mention_ids": ["m1"], "canonical": "A", "type": "Person", "confidence": 0.9},
                    {"mention_ids": ["m1", "m2"], "canonical": "B", "type": "Person", "confidence": 0.9},
                ]
            },
        )
        with pytest.raises(LLMError, match="multiple clusters"):
            client.cluster_mentions(doc, mentions, schema)

    def test_missing_mentions_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {"mention_ids": ["m1"], "canonical": "A", "type": "Person", "confidence": 0.9},
                ]
            },
        )
        with pytest.raises(LLMError, match="did not cluster all mentions"):
            client.cluster_mentions(doc, mentions, schema)

    def test_cache_stats_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        doc: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> None:
        _install_fake(
            monkeypatch,
            client,
            "record_clusters",
            {
                "clusters": [
                    {
                        "mention_ids": ["m1", "m2"],
                        "canonical": "Marie Curie",
                        "type": "Person",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        client.cluster_mentions(doc, mentions, schema)
        # Default fake: 60 cache_read / 100 input
        assert client.cache_hit_ratio() == 0.6


class TestResolveCrossdoc:
    def test_happy_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        schema: EntitySchema,
    ) -> None:
        candidates = [
            CrossDocCandidate(
                entity_id="e1",
                doc_id="d1",
                type="Person",
                canonical="Marie Curie",
                contexts=["Marie Curie won the Nobel."],
                nearest_cluster_id="c1",
                nearest_similarity=0.82,
            ),
        ]
        _install_fake(
            monkeypatch,
            client,
            "record_merge_decisions",
            {
                "decisions": [
                    {
                        "entity_id": "e1",
                        "target_cluster_id": "c1",
                        "confidence": 0.88,
                    }
                ]
            },
        )
        decisions = client.resolve_crossdoc(candidates, schema)
        assert len(decisions) == 1
        assert decisions[0].target_cluster_id == "c1"

    def test_invented_cluster_id_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        client: AnthropicClient,
        schema: EntitySchema,
    ) -> None:
        candidates = [
            CrossDocCandidate(
                entity_id="e1",
                doc_id="d1",
                type="Person",
                canonical="X",
                contexts=[],
                nearest_cluster_id="c1",
                nearest_similarity=0.5,
            ),
        ]
        _install_fake(
            monkeypatch,
            client,
            "record_merge_decisions",
            {
                "decisions": [
                    {
                        "entity_id": "e1",
                        "target_cluster_id": "c-invented",
                        "confidence": 0.9,
                    }
                ]
            },
        )
        with pytest.raises(LLMError, match="unknown cluster_id"):
            client.resolve_crossdoc(candidates, schema)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = AnthropicClient(api_key=None)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        _ = c.client

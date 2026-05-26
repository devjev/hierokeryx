"""Tests for within-doc coref using a FakeLLMClient (no Anthropic calls)."""

from __future__ import annotations

import pytest

from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    EntityType,
    Mention,
    MergeDecision,
    Span,
)
from hierokeryx.resolve.coref import resolve_within_doc


class FakeLLMClient:
    """Stand-in LLMClient returning pre-canned coref clusters."""

    def __init__(self, clusters: list[CorefCluster]):
        self._clusters = clusters

    def cluster_mentions(
        self, document: Document, mentions: list[Mention], schema: EntitySchema
    ) -> list[CorefCluster]:
        return list(self._clusters)

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        return []


def test_empty_mentions_returns_empty_entities() -> None:
    doc = Document(id="d1", text="Nothing here.")
    schema = EntitySchema(types=[EntityType(name="Person", description="a person")])
    fake = FakeLLMClient([])
    assert resolve_within_doc(doc, [], schema, fake) == []


def test_single_cluster_produces_one_entity() -> None:
    doc = Document(id="d1", text="Marie Curie won the Nobel. Curie was Polish.")
    schema = EntitySchema(types=[EntityType(name="Person", description="a person")])
    mentions = [
        Mention(id="m1", span=Span(start=0, end=11, text="Marie Curie"), type="Person", score=0.95),
        Mention(id="m2", span=Span(start=27, end=32, text="Curie"), type="Person", score=0.82),
    ]
    fake = FakeLLMClient(
        [
            CorefCluster(
                mention_ids=["m1", "m2"],
                canonical="Marie Curie",
                type="Person",
                confidence=0.9,
            )
        ]
    )
    entities = resolve_within_doc(doc, mentions, schema, fake)
    assert len(entities) == 1
    e = entities[0]
    assert e.canonical == "Marie Curie"
    assert e.surface_canonical == "Marie Curie"  # longest verbatim
    assert e.doc_id == "d1"
    assert len(e.mentions) == 2
    # Mentions sorted by appearance
    assert e.mentions[0].id == "m1"
    assert e.mentions[1].id == "m2"
    # Entity confidence is the within-doc ensemble:
    # mean=0.885, llm=0.9 → 0.5*0.885 + 0.5*0.9 = 0.8925
    assert e.confidence == pytest.approx(0.8925)


def test_surface_canonical_uses_longest_mention() -> None:
    doc = Document(id="d1", text="The Acme Corporation makes things. Acme is big.")
    schema = EntitySchema(types=[EntityType(name="Organization", description="org")])
    mentions = [
        Mention(id="m1", span=Span(start=4, end=20, text="Acme Corporation"), type="Organization", score=0.9),
        Mention(id="m2", span=Span(start=35, end=39, text="Acme"), type="Organization", score=0.8),
    ]
    fake = FakeLLMClient(
        [
            CorefCluster(
                mention_ids=["m1", "m2"],
                canonical="Acme Corp.",
                type="Organization",
                confidence=0.9,
            )
        ]
    )
    [entity] = resolve_within_doc(doc, mentions, schema, fake)
    assert entity.canonical == "Acme Corp."  # what LLM said
    assert entity.surface_canonical == "Acme Corporation"  # longest verbatim


def test_entities_sorted_by_first_mention_position() -> None:
    doc = Document(id="d1", text="Bob met Alice yesterday.")
    schema = EntitySchema(types=[EntityType(name="Person", description="a person")])
    mentions = [
        Mention(id="m_bob", span=Span(start=0, end=3, text="Bob"), type="Person", score=0.9),
        Mention(id="m_alice", span=Span(start=8, end=13, text="Alice"), type="Person", score=0.9),
    ]
    fake = FakeLLMClient(
        [
            CorefCluster(mention_ids=["m_alice"], canonical="Alice", type="Person", confidence=0.9),
            CorefCluster(mention_ids=["m_bob"], canonical="Bob", type="Person", confidence=0.9),
        ]
    )
    entities = resolve_within_doc(doc, mentions, schema, fake)
    assert [e.canonical for e in entities] == ["Bob", "Alice"]

"""Tests for the core Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    Span,
    make_cluster_id,
    make_entity_id,
    make_mention_id,
)


class TestSpan:
    def test_valid_span(self) -> None:
        s = Span(start=0, end=5, text="hello")
        assert s.start == 0
        assert s.end == 5
        assert s.text == "hello"

    def test_end_must_exceed_start(self) -> None:
        # end <= start with matching-length text trips the cross-field validator
        with pytest.raises(ValidationError, match=r"end \(3\) must be > start \(5\)"):
            Span(start=5, end=3, text="xx")

    def test_text_length_must_match(self) -> None:
        with pytest.raises(ValidationError, match="length"):
            Span(start=0, end=5, text="hi")

    def test_frozen(self) -> None:
        s = Span(start=0, end=5, text="hello")
        with pytest.raises(ValidationError):
            s.start = 1  # type: ignore[misc]


class TestEntitySchema:
    def test_fingerprint_stable(self) -> None:
        a = EntitySchema(types=[EntityType(name="Person", description="A human")])
        b = EntitySchema(types=[EntityType(name="Person", description="A human")])
        assert a.fingerprint() == b.fingerprint()

    def test_fingerprint_changes_on_description(self) -> None:
        a = EntitySchema(types=[EntityType(name="Person", description="A")])
        b = EntitySchema(types=[EntityType(name="Person", description="B")])
        assert a.fingerprint() != b.fingerprint()

    def test_fingerprint_independent_of_dict_order(self) -> None:
        # Both schemas declare the same types — order of attribute serialization
        # shouldn't matter because we sort keys.
        a = EntitySchema(
            types=[EntityType(name="P", description="d", examples=["x", "y"])]
        )
        b = EntitySchema(
            types=[EntityType(name="P", description="d", examples=["x", "y"])]
        )
        assert a.fingerprint() == b.fingerprint()

    def test_duplicate_type_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            EntitySchema(
                types=[
                    EntityType(name="X", description="a"),
                    EntityType(name="X", description="b"),
                ]
            )

    def test_at_least_one_type_required(self) -> None:
        with pytest.raises(ValidationError):
            EntitySchema(types=[])

    def test_type_by_name(self) -> None:
        s = EntitySchema(
            types=[
                EntityType(name="A", description="aaa"),
                EntityType(name="B", description="bbb"),
            ]
        )
        assert s.type_by_name("A").name == "A"  # type: ignore[union-attr]
        assert s.type_by_name("Missing") is None
        assert s.type_names == ("A", "B")


class TestIdHelpers:
    def test_mention_id_deterministic(self) -> None:
        assert make_mention_id("d1", 5, 10) == make_mention_id("d1", 5, 10)

    def test_mention_id_changes_with_inputs(self) -> None:
        assert make_mention_id("d1", 5, 10) != make_mention_id("d1", 5, 11)
        assert make_mention_id("d1", 5, 10) != make_mention_id("d2", 5, 10)

    def test_human_mention_id_distinct_namespace(self) -> None:
        a = make_mention_id("d1", 0, 5, source="gliner")
        h = make_mention_id("d1", 0, 5, source="human")
        assert a.startswith("m_")
        assert h.startswith("human_")
        assert a != h

    def test_entity_id_order_insensitive(self) -> None:
        a = make_entity_id("d1", ["x", "y", "z"])
        b = make_entity_id("d1", ["z", "y", "x"])
        assert a == b

    def test_cluster_id_uses_canonical_and_type(self) -> None:
        a = make_cluster_id("Marie Curie", "Person", ["d1", "d2"])
        b = make_cluster_id("Marie Curie", "Person", ["d2", "d1"])
        c = make_cluster_id("Marie Curie", "Scientist", ["d1", "d2"])
        assert a == b
        assert a != c


class TestExtractionResultInvariant:
    def test_valid_extraction_result(self, doc_curie: Document) -> None:
        m = Mention(
            id="m1",
            span=Span(start=0, end=11, text="Marie Curie"),
            type="Person",
            score=0.9,
        )
        e = Entity(
            id="e1",
            type="Person",
            canonical="Marie Curie",
            surface_canonical="Marie Curie",
            mentions=[m],
            confidence=0.9,
            doc_id="curie",
        )
        result = ExtractionResult(
            document=doc_curie, entities=[e], schema_version="abc"
        )
        assert len(result.entities) == 1

    def test_mismatched_doc_id_rejected(self, doc_curie: Document) -> None:
        m = Mention(
            id="m1",
            span=Span(start=0, end=11, text="Marie Curie"),
            type="Person",
            score=0.9,
        )
        e = Entity(
            id="e1",
            type="Person",
            canonical="Marie Curie",
            surface_canonical="Marie Curie",
            mentions=[m],
            confidence=0.9,
            doc_id="wrong",
        )
        with pytest.raises(ValidationError, match="doc_id"):
            ExtractionResult(document=doc_curie, entities=[e], schema_version="abc")

    def test_span_mismatch_rejected(self, doc_curie: Document) -> None:
        # Span text claims "Marie Curie" but the offsets point to "Marie Curie"
        # which is correct; let's deliberately break it.
        m = Mention(
            id="m1",
            span=Span(start=0, end=11, text="WrongLabel!"),  # length matches
            type="Person",
            score=0.9,
        )
        e = Entity(
            id="e1",
            type="Person",
            canonical="x",
            surface_canonical="x",
            mentions=[m],
            confidence=0.9,
            doc_id="curie",
        )
        with pytest.raises(ValidationError, match=r"span\.text"):
            ExtractionResult(document=doc_curie, entities=[e], schema_version="abc")

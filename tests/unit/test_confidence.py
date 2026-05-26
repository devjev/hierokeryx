"""Tests for confidence ensembling and HITL routing."""

from __future__ import annotations

import pytest

from hierokeryx.confidence import (
    crossdoc_confidence,
    route_for_review,
    within_doc_confidence,
)
from hierokeryx.models import Entity, Mention, Span


def _mk_mention(start: int, end: int, score: float, idx: int = 0) -> Mention:
    return Mention(
        id=f"m{idx}",
        span=Span(start=start, end=end, text="x" * (end - start)),
        type="Person",
        score=score,
    )


def _mk_entity(
    confidence: float,
    mention_score: float = 0.9,
    cluster_id: str | None = None,
) -> Entity:
    return Entity(
        id="e1",
        type="Person",
        canonical="C",
        surface_canonical="C",
        mentions=[_mk_mention(0, 5, mention_score, 1)],
        confidence=confidence,
        doc_id="d1",
        cluster_id=cluster_id,
    )


class TestWithinDocConfidence:
    def test_perfect_inputs(self) -> None:
        m = [_mk_mention(0, 5, 1.0, 1)]
        assert within_doc_confidence(m, 1.0) == 1.0

    def test_zero_inputs(self) -> None:
        m = [_mk_mention(0, 5, 0.0, 1)]
        assert within_doc_confidence(m, 0.0) == 0.0

    def test_half_and_half(self) -> None:
        m = [_mk_mention(0, 5, 0.6, 1), _mk_mention(10, 15, 0.4, 2)]
        # mean score = 0.5; llm = 0.8; combined = 0.5*0.5 + 0.5*0.8 = 0.65
        assert within_doc_confidence(m, 0.8) == pytest.approx(0.65, abs=1e-6)

    def test_clamps_llm_overshoot(self) -> None:
        m = [_mk_mention(0, 5, 1.0, 1)]
        assert within_doc_confidence(m, 1.5) == 1.0

    def test_empty_mentions_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            within_doc_confidence([], 0.9)


class TestCrossdocConfidence:
    def test_perfect_inputs(self) -> None:
        assert crossdoc_confidence(1.0, 1.0, 1.0, 0.0) == 1.0

    def test_zero_margin_caps_at_eighty_pct(self) -> None:
        assert crossdoc_confidence(1.0, 1.0, 0.0, 0.0) == pytest.approx(0.8)

    def test_negative_margin_treated_as_zero(self) -> None:
        # second_sim > top_sim should not give negative margin
        assert crossdoc_confidence(1.0, 1.0, 0.4, 0.7) == pytest.approx(0.8)


class TestRouting:
    def test_high_confidence_not_flagged(self) -> None:
        e = _mk_entity(confidence=0.9, mention_score=0.9)
        assert route_for_review([e]) == []

    def test_low_cluster_conf_flagged(self) -> None:
        e = _mk_entity(confidence=0.5, mention_score=0.9)
        items = route_for_review([e])
        assert len(items) == 1
        assert items[0].reason == "low_cluster_conf"
        assert items[0].entity_id == "e1"

    def test_low_span_score_flagged_even_when_cluster_ok(self) -> None:
        e = _mk_entity(confidence=0.9, mention_score=0.3)
        items = route_for_review([e])
        assert items[0].reason == "low_span_conf"

    def test_crossdoc_assignment_with_low_conf_is_ambiguous_merge(self) -> None:
        e = _mk_entity(confidence=0.5, cluster_id="c1")
        items = route_for_review([e])
        assert items[0].reason == "ambiguous_merge"

    def test_threshold_is_configurable(self) -> None:
        e = _mk_entity(confidence=0.85, mention_score=0.85)
        assert route_for_review([e], cluster_threshold=0.9) != []
        assert route_for_review([e], cluster_threshold=0.8) == []

"""Tests for mention normalization (overlap dedup, multi-label preservation)."""

from __future__ import annotations

from hierokeryx.extract.gliner_runner import _overlaps, normalize_mentions
from hierokeryx.models import Mention, Span


def _make(start: int, end: int, type_: str, score: float, idx: int = 0) -> Mention:
    return Mention(
        id=f"m{idx}",
        span=Span(start=start, end=end, text="x" * (end - start)),
        type=type_,
        score=score,
    )


def test_overlaps_helper() -> None:
    a = Span(start=0, end=5, text="aaaaa")
    b = Span(start=3, end=8, text="bbbbb")
    c = Span(start=5, end=10, text="ccccc")
    assert _overlaps(a, b)
    assert not _overlaps(a, c)  # touching but not overlapping


def test_exact_duplicates_collapsed_keeping_higher_score() -> None:
    m1 = _make(0, 5, "Person", 0.7, 1)
    m2 = _make(0, 5, "Person", 0.9, 2)
    out = normalize_mentions([m1, m2])
    assert len(out) == 1
    assert out[0].score == 0.9


def test_overlapping_same_type_keeps_highest_score() -> None:
    m1 = _make(0, 10, "Person", 0.8, 1)  # longer
    m2 = _make(3, 7, "Person", 0.95, 2)  # nested, higher score
    out = normalize_mentions([m1, m2])
    assert len(out) == 1
    assert out[0].score == 0.95


def test_different_types_can_overlap() -> None:
    m_org = _make(0, 15, "Organization", 0.9, 1)  # "Bank of America"
    m_loc = _make(8, 15, "Location", 0.7, 2)  # "America"
    out = normalize_mentions([m_org, m_loc])
    assert len(out) == 2
    types = {m.type for m in out}
    assert types == {"Organization", "Location"}


def test_non_overlapping_same_type_all_kept() -> None:
    m1 = _make(0, 5, "Person", 0.9, 1)
    m2 = _make(10, 15, "Person", 0.8, 2)
    m3 = _make(20, 25, "Person", 0.7, 3)
    out = normalize_mentions([m1, m2, m3])
    assert len(out) == 3
    assert [m.span.start for m in out] == [0, 10, 20]


def test_output_is_sorted_by_span_then_type() -> None:
    m1 = _make(10, 15, "Z", 0.9, 1)
    m2 = _make(0, 5, "A", 0.9, 2)
    m3 = _make(0, 5, "B", 0.9, 3)
    out = normalize_mentions([m1, m2, m3])
    assert [(m.span.start, m.type) for m in out] == [(0, "A"), (0, "B"), (10, "Z")]

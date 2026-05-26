"""Tests for the tokenizer and span-snapping helper."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from hierokeryx.extract.tokenize import snap_to_token_boundary, tokenize


def test_tokenize_simple_words() -> None:
    text = "Hello world!"
    tokens = tokenize(text)
    assert [text[s:e] for s, e in tokens] == ["Hello", "world", "!"]


def test_tokenize_apostrophes_and_hyphens() -> None:
    text = "Don't quit — it's a state-of-the-art system."
    tokens = tokenize(text)
    rendered = [text[s:e] for s, e in tokens]
    # "Don't", "quit", em-dash, "it's", "a", "state-of-the-art", "system", "."
    assert "Don't" in rendered
    assert "it's" in rendered
    assert "state-of-the-art" in rendered
    assert "." in rendered


def test_tokenize_unicode() -> None:
    text = "Naïve café — résumé"
    tokens = tokenize(text)
    rendered = [text[s:e] for s, e in tokens]
    assert "Naïve" in rendered
    assert "café" in rendered
    assert "résumé" in rendered


def test_snap_already_aligned_is_noop() -> None:
    text = "Marie Curie was a physicist."
    # "Marie Curie" is chars 0..11
    assert snap_to_token_boundary(text, 0, 11) == (0, 11)


def test_snap_expands_partial_token_at_start() -> None:
    text = "Marie Curie was a physicist."
    # "rie Curie" starts at 2 (partial through "Marie") → should snap to 0
    snapped = snap_to_token_boundary(text, 2, 11)
    assert snapped == (0, 11)


def test_snap_expands_partial_token_at_end() -> None:
    text = "Marie Curie was a physicist."
    # "Marie Cur" ends mid-token at 9 → snap to 11
    snapped = snap_to_token_boundary(text, 0, 9)
    assert snapped == (0, 11)


def test_snap_handles_both_edges() -> None:
    text = "abcd efgh ijkl"
    # "bcd efg" — both edges mid-token → snap to "abcd efgh"
    snapped = snap_to_token_boundary(text, 1, 7)
    assert snapped == (0, 9)


def test_snap_trims_whitespace_after_expansion() -> None:
    text = "  hello  "
    # Range starts inside the leading whitespace
    snapped = snap_to_token_boundary(text, 0, 8)
    s, e = snapped
    assert text[s:e].strip() == text[s:e], f"whitespace boundary: {text[s:e]!r}"


@given(
    text=st.text(
        alphabet=st.characters(blacklist_categories=("Cc", "Cs"), min_codepoint=32),
        min_size=1,
        max_size=200,
    ),
)
def test_snap_never_produces_invalid_range(text: str) -> None:
    """For any input text and any in-bounds range, snap must return a valid
    (start, end) where 0 <= start <= end <= len(text) and the slice has no
    leading/trailing whitespace (or is empty/equal)."""
    if len(text) < 2:
        return
    start = len(text) // 4
    end = max(start + 1, 3 * len(text) // 4)
    s, e = snap_to_token_boundary(text, start, end)
    assert 0 <= s <= e <= len(text)
    if s < e:
        snippet = text[s:e]
        assert snippet == snippet.strip() or not snippet.strip()

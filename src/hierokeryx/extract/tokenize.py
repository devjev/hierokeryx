"""Regex tokenizer used to snap GLiNER spans to token boundaries.

GLiNER occasionally returns spans that bisect a token (e.g. "Cur" inside "Curie"
when the subword model breaks awkwardly). Snapping to the surrounding token
boundary repairs these without rejecting the mention.
"""

from __future__ import annotations

import re

# Unicode-aware word: letters, marks, numbers, plus internal apostrophes/hyphens.
# Falls through to single non-whitespace chars (punctuation, symbols) so every
# character lands in exactly one token span.
_TOKEN_RE = re.compile(
    r"[\w](?:[\w'\-’]*[\w])?|\S",
    re.UNICODE,
)


def tokenize(text: str) -> list[tuple[int, int]]:
    """Return [(start, end), ...] character offsets for tokens in `text`."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def snap_to_token_boundary(
    text: str, start: int, end: int, tokens: list[tuple[int, int]] | None = None
) -> tuple[int, int]:
    """Expand (start, end) outward to the nearest enclosing token boundaries.

    If the input span already aligns with tokens, returns it unchanged.
    If it crosses partial tokens at either edge, extends to include the full token.
    Whitespace-only inputs (no enclosing token) are returned unchanged.
    """
    if start >= end or start < 0 or end > len(text):
        return (start, end)
    if tokens is None:
        tokens = tokenize(text)
    if not tokens:
        return (start, end)

    new_start = start
    new_end = end
    for tok_start, tok_end in tokens:
        if tok_start < start < tok_end:
            new_start = min(new_start, tok_start)
        if tok_start < end < tok_end:
            new_end = max(new_end, tok_end)
        if tok_start >= end:
            break

    # Trim leading/trailing whitespace introduced by snapping (rare).
    while new_start < new_end and text[new_start].isspace():
        new_start += 1
    while new_end > new_start and text[new_end - 1].isspace():
        new_end -= 1
    return (new_start, new_end)

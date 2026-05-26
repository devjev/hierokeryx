"""Load-bearing integration test: GLiNER spans must quote the document verbatim.

Uses the small v2.1 GLiNER variant (~150 MB download on first run) to keep CI
turnaround reasonable. Overrides via HIEROKERYX_TEST_GLINER_MODEL.
"""

from __future__ import annotations

import os

import pytest

from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.models import Document, EntitySchema, EntityType

MODEL_ID = os.environ.get(
    "HIEROKERYX_TEST_GLINER_MODEL", "urchade/gliner_small-v2.1"
)


@pytest.fixture(scope="module")
def extractor() -> GLiNERExtractor:
    return GLiNERExtractor(model_id=MODEL_ID, threshold=0.3)


@pytest.fixture
def schema() -> EntitySchema:
    return EntitySchema(
        types=[
            EntityType(name="Person", description="A named individual human"),
            EntityType(name="Organization", description="A company, agency, institution"),
            EntityType(name="Location", description="A geographic place"),
        ]
    )


@pytest.mark.integration
def test_spans_quote_document_verbatim(
    extractor: GLiNERExtractor, schema: EntitySchema
) -> None:
    doc = Document(
        id="t1",
        text=(
            "Marie Curie was born in Warsaw and later worked in Paris. "
            "She co-founded the Curie Institute with her husband Pierre."
        ),
    )
    mentions = extractor.extract(doc, schema)

    # Load-bearing invariant: every span's stored text must equal the slice.
    for m in mentions:
        actual = doc.text[m.span.start : m.span.end]
        assert actual == m.span.text, (
            f"Span text mismatch for mention {m.id}: "
            f"stored {m.span.text!r} != doc slice {actual!r}"
        )


@pytest.mark.integration
def test_finds_at_least_one_person(
    extractor: GLiNERExtractor, schema: EntitySchema
) -> None:
    doc = Document(
        id="t2",
        text="Marie Curie won the Nobel Prize in Physics in 1903.",
    )
    mentions = extractor.extract(doc, schema)
    person_mentions = [m for m in mentions if m.type == "Person"]
    assert person_mentions, f"Expected at least one Person mention, got: {mentions}"
    # The Person mention should overlap with "Marie Curie"
    marie_start = doc.text.find("Marie")
    marie_end = doc.text.find("Curie") + len("Curie")
    assert any(
        m.span.start < marie_end and m.span.end > marie_start for m in person_mentions
    ), f"Person mentions: {person_mentions}"


@pytest.mark.integration
def test_normalize_collapses_overlaps(
    extractor: GLiNERExtractor, schema: EntitySchema
) -> None:
    doc = Document(id="t3", text="Apple Inc. was founded by Steve Jobs in California.")
    mentions = extractor.extract(doc, schema)

    # Within each type, no two mentions should overlap (normalize_mentions invariant).
    by_type: dict[str, list] = {}
    for m in mentions:
        by_type.setdefault(m.type, []).append(m)
    for type_name, ms in by_type.items():
        for i, a in enumerate(ms):
            for b in ms[i + 1 :]:
                overlap = a.span.start < b.span.end and b.span.start < a.span.end
                assert not overlap, (
                    f"Overlapping {type_name} mentions: "
                    f"{a.span.text!r}@({a.span.start},{a.span.end}) "
                    f"and {b.span.text!r}@({b.span.start},{b.span.end})"
                )

"""Tests for the JSONL review round-trip: export → edit → import → apply."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    ReviewItem,
    Span,
)
from hierokeryx.review.apply import apply_review
from hierokeryx.review.jsonl import (
    SCHEMA_MARKER,
    read_review,
    write_review,
    write_review_dir,
)
from hierokeryx.review.lint import lint_review_file


def _mk_extraction() -> ExtractionResult:
    text = "Marie Curie discovered radium. Curie also won a Nobel Prize."
    doc = Document(id="doc1", text=text)
    mentions = [
        Mention(id="m1", span=Span(start=0, end=11, text="Marie Curie"), type="Person", score=0.95),
        Mention(id="m2", span=Span(start=31, end=36, text="Curie"), type="Person", score=0.82),
    ]
    entity = Entity(
        id="e1",
        type="Person",
        canonical="Marie Curie",
        surface_canonical="Marie Curie",
        mentions=mentions,
        confidence=0.85,
        doc_id="doc1",
    )
    return ExtractionResult(document=doc, entities=[entity], schema_version="abc123")


def _schema() -> EntitySchema:
    return EntitySchema(types=[EntityType(name="Person", description="A human individual")])


def test_export_emits_schema_header_then_one_line_per_entity(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)
    raw = out.read_text(encoding="utf-8").splitlines()
    assert len(raw) == 2
    header = json.loads(raw[0])
    assert header["$schema"] == SCHEMA_MARKER
    assert header["doc_id"] == "doc1"
    line = json.loads(raw[1])
    assert line["op"] == "keep"
    assert line["id"] == "e1"
    assert line["canonical"] == "Marie Curie"


def test_keep_op_roundtrip_preserves_extraction(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)
    header, lines = read_review(out)
    applied = apply_review(result, header, lines)
    assert applied.entities == result.entities


def test_reject_op_drops_entity(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)

    raw = out.read_text(encoding="utf-8").splitlines()
    line = json.loads(raw[1])
    line["op"] = "reject"
    out.write_text(raw[0] + "\n" + json.dumps(line) + "\n")

    header, lines = read_review(out)
    applied = apply_review(result, header, lines)
    assert applied.entities == []


def test_edit_op_replaces_canonical(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)

    raw = out.read_text(encoding="utf-8").splitlines()
    line = json.loads(raw[1])
    line["op"] = "edit"
    line["canonical"] = "Marie Skłodowska-Curie"
    out.write_text(raw[0] + "\n" + json.dumps(line) + "\n")

    header, lines = read_review(out)
    applied = apply_review(result, header, lines)
    assert applied.entities[0].canonical == "Marie Skłodowska-Curie"
    # Surface canonical untouched — it still reflects the verbatim text
    assert applied.entities[0].surface_canonical == "Marie Curie"


def test_add_op_inserts_new_entity(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)

    raw = out.read_text(encoding="utf-8").splitlines()
    added_line = {
        "op": "add",
        "id": "human_added1",
        "type": "Person",
        "canonical": "Radium personification",
        "surface_canonical": "radium",
        "confidence": 1.0,
        "mentions": [{"start": 23, "end": 29, "text": "radium", "score": 1.0, "source": "human"}],
    }
    out.write_text(raw[0] + "\n" + raw[1] + "\n" + json.dumps(added_line) + "\n")

    header, lines = read_review(out)
    applied = apply_review(result, header, lines)
    assert len(applied.entities) == 2
    added = next(e for e in applied.entities if e.id == "human_added1")
    assert added.canonical == "Radium personification"
    assert added.mentions[0].source == "human"


def test_add_op_rejects_non_human_id(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)
    raw = out.read_text(encoding="utf-8").splitlines()
    added_line = {
        "op": "add",
        "id": "e_not_human",
        "type": "Person",
        "canonical": "x",
        "surface_canonical": "x",
        "confidence": 1.0,
        "mentions": [{"start": 23, "end": 29, "text": "radium", "score": 1.0, "source": "human"}],
    }
    out.write_text(raw[0] + "\n" + raw[1] + "\n" + json.dumps(added_line) + "\n")
    header, lines = read_review(out)
    with pytest.raises(ValueError, match="human_"):
        apply_review(result, header, lines)


def test_lint_catches_text_mismatch(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)

    # Manually corrupt: change the mention text without changing the offsets.
    raw = out.read_text(encoding="utf-8").splitlines()
    line = json.loads(raw[1])
    line["mentions"][0]["text"] = "Wrong text!"
    out.write_text(raw[0] + "\n" + json.dumps(line) + "\n")

    errors = lint_review_file(out, document=result.document, schema=_schema(), original=result)
    assert any("text mismatch" in e.message for e in errors)


def test_lint_catches_unknown_type(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)
    raw = out.read_text(encoding="utf-8").splitlines()
    line = json.loads(raw[1])
    line["type"] = "Alien"
    out.write_text(raw[0] + "\n" + json.dumps(line) + "\n")

    errors = lint_review_file(out, document=result.document, schema=_schema(), original=result)
    assert any("not in schema" in e.message for e in errors)


def test_lint_catches_text_sha_drift(tmp_path: Path) -> None:
    result = _mk_extraction()
    out = tmp_path / "doc1.jsonl"
    write_review(result, out)
    # Pretend the document was later modified.
    new_doc = Document(id="doc1", text=result.document.text + " EXTRA.")
    errors = lint_review_file(out, document=new_doc, schema=_schema())
    assert any("text_sha" in e.message for e in errors)


def test_write_dir_only_flagged_skips_clean_docs(tmp_path: Path) -> None:
    result = _mk_extraction()
    # No flagged items → only_flagged should produce no files
    written = write_review_dir([result], tmp_path / "review", only_flagged=True)
    assert written == []
    # With a flagged item it does produce a file
    flagged = [ReviewItem(doc_id="doc1", entity_id="e1", reason="low_cluster_conf", current=result.entities[0])]
    written = write_review_dir([result], tmp_path / "review", flagged=flagged, only_flagged=True)
    assert len(written) == 1

"""Serialize and parse hierokeryx review JSONL files.

Format: one document per file, one entity per line, sorted by
`(first_mention_start, entity_id)` for diff stability. Line 1 is a `$schema`
header so JSON-Schema-aware editors validate inline as the user types.

Mentions are stored flat (no nested `span` object) to keep the lines short
and editable by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from hierokeryx.models import (
    Entity,
    ExtractionResult,
    Mention,
    MentionSource,
    ReviewItem,
    ReviewOp,
    ReviewReason,
    Span,
    make_mention_id,
)

SCHEMA_MARKER = "hierokeryx/review/v1"
"""Stable URI written as the `$schema` field of the JSONL header line.

JSON-Schema-aware editors use it to auto-validate each line as the user
types. Bump the trailing version segment on any breaking change to the
wire format.
"""


class ReviewHeader(BaseModel):
    """First line of a review JSONL file.

    Carries the doc identity (`doc_id`), a short content hash of the source
    text (`text_sha`) so the linter can detect a stale review against an
    edited document, and the `schema_version` the entities were extracted
    against.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_marker: Literal["hierokeryx/review/v1"] = Field(
        default=SCHEMA_MARKER, alias="$schema"
    )
    doc_id: str
    text_sha: str
    schema_version: str


class ReviewMention(BaseModel):
    """Flat mention representation in the JSONL wire format.

    The nested [`Span`][hierokeryx.models.Span] of a core
    [`Mention`][hierokeryx.models.Mention] is hoisted into top-level
    `start`/`end`/`text` fields to keep one-line edits short and readable.
    Human-added mentions may omit `id`; one will be derived from
    `(doc_id, start, end, source)` on import.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = None  # auto-generated for human-added mentions
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    source: MentionSource = "gliner"


class ReviewEntityLine(BaseModel):
    """One entity as it appears on a line of a review JSONL file.

    The `op` field tells [`apply_review`][hierokeryx.review.apply.apply_review]
    what to do with the line on import: `keep` accepts the current state,
    `edit` replaces the original entity's mutable fields (including its
    mentions), `reject` drops the entity, and `add` introduces a new
    human-curated entity.

    `reason` and `candidates` are informational annotations populated on
    export and ignored on import.
    """

    model_config = ConfigDict(extra="forbid")

    op: ReviewOp = "keep"
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    surface_canonical: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    mentions: list[ReviewMention] = Field(min_length=1)
    cluster_id: str | None = None

    # Informational on export — ignored on import.
    reason: ReviewReason | None = None
    candidates: list[str] = Field(default_factory=list)


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _mention_to_review(mention: Mention) -> ReviewMention:
    return ReviewMention(
        id=mention.id,
        start=mention.span.start,
        end=mention.span.end,
        text=mention.span.text,
        score=mention.score,
        source=mention.source,
    )


def _entity_to_line(
    entity: Entity,
    op: ReviewOp = "keep",
    reason: ReviewReason | None = None,
    candidates: list[str] | None = None,
) -> ReviewEntityLine:
    return ReviewEntityLine(
        op=op,
        id=entity.id,
        type=entity.type,
        canonical=entity.canonical,
        surface_canonical=entity.surface_canonical,
        confidence=entity.confidence,
        mentions=[_mention_to_review(m) for m in entity.mentions],
        cluster_id=entity.cluster_id,
        reason=reason,
        candidates=list(candidates or []),
    )


def write_review(
    result: ExtractionResult,
    path: str | Path,
    *,
    flagged_ids: dict[str, ReviewReason] | None = None,
    only_flagged: bool = False,
) -> None:
    """Write one document's entities to a review JSONL file.

    If `flagged_ids` is supplied, each id maps to a `reason` annotation. If
    `only_flagged` is True, only flagged entities are written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    header = ReviewHeader(
        doc_id=result.document.id,
        text_sha=_text_sha(result.document.text),
        schema_version=result.schema_version,
    )

    flagged_ids = flagged_ids or {}
    entities_to_write = [
        e for e in result.entities if (e.id in flagged_ids or not only_flagged)
    ]
    sorted_entities = sorted(
        entities_to_write,
        key=lambda e: (e.mentions[0].span.start if e.mentions else 0, e.id),
    )

    with p.open("w", encoding="utf-8") as f:
        f.write(header.model_dump_json(by_alias=True) + "\n")
        for entity in sorted_entities:
            line = _entity_to_line(entity, reason=flagged_ids.get(entity.id))
            f.write(line.model_dump_json(exclude_none=True) + "\n")


def write_review_dir(
    results: list[ExtractionResult],
    directory: str | Path,
    *,
    flagged: list[ReviewItem] | None = None,
    only_flagged: bool = False,
) -> list[Path]:
    """Write one JSONL file per document under `directory/`."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    by_doc: dict[str, dict[str, ReviewReason]] = {}
    for item in flagged or []:
        by_doc.setdefault(item.doc_id, {})[item.entity_id] = item.reason

    written: list[Path] = []
    for result in results:
        doc_id = result.document.id
        flagged_for_doc = by_doc.get(doc_id, {})
        if only_flagged and not flagged_for_doc:
            continue
        path = d / f"{_safe_filename(doc_id)}.jsonl"
        write_review(
            result,
            path,
            flagged_ids=flagged_for_doc,
            only_flagged=only_flagged,
        )
        written.append(path)
    return written


def read_review(path: str | Path) -> tuple[ReviewHeader, list[ReviewEntityLine]]:
    """Parse one review JSONL file into its header and entity lines."""
    p = Path(path)
    lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty review file: {p}")
    header_data: dict[str, Any] = json.loads(lines[0])
    header = ReviewHeader.model_validate(header_data)
    entity_lines = [ReviewEntityLine.model_validate_json(ln) for ln in lines[1:]]
    return header, entity_lines


def read_review_dir(
    directory: str | Path,
) -> dict[str, tuple[ReviewHeader, list[ReviewEntityLine]]]:
    """Parse every `*.jsonl` review file under `directory`, keyed by `doc_id`."""
    d = Path(directory)
    out: dict[str, tuple[ReviewHeader, list[ReviewEntityLine]]] = {}
    for path in sorted(d.glob("*.jsonl")):
        header, entries = read_review(path)
        out[header.doc_id] = (header, entries)
    return out


def review_mention_to_mention(rm: ReviewMention, doc_id: str, entity_type: str) -> Mention:
    """Convert a wire-format ReviewMention back into a core Mention model."""
    mention_id = rm.id or make_mention_id(doc_id, rm.start, rm.end, source=rm.source)
    return Mention(
        id=mention_id,
        span=Span(start=rm.start, end=rm.end, text=rm.text),
        type=entity_type,
        score=rm.score,
        source=rm.source,
    )


def _safe_filename(doc_id: str) -> str:
    """Return a filesystem-safe filename derived from a doc_id."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)

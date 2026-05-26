"""Replay edited review JSONL files into a new ExtractionResult.

Op semantics:
- `keep` (default): use the original entity unchanged. If the line's fields
  differ from the original, the line wins (treated as a quiet edit).
- `reject`: drop the entity.
- `edit`: replace the original entity with the line's content.
- `add`: append a new entity. Its `id` must start with `human_`.

Apply runs `ExtractionResult.model_validate` at the end, so every replayed
mention is checked against the document text via the load-bearing invariant.
"""

from __future__ import annotations

from hierokeryx.models import Entity, ExtractionResult, Mention
from hierokeryx.review.jsonl import (
    ReviewEntityLine,
    ReviewHeader,
    review_mention_to_mention,
)


def apply_review(
    original: ExtractionResult,
    header: ReviewHeader,
    lines: list[ReviewEntityLine],
) -> ExtractionResult:
    """Apply a parsed review file to its source ExtractionResult."""
    if header.doc_id != original.document.id:
        raise ValueError(
            f"Review header doc_id={header.doc_id!r} does not match "
            f"original document id={original.document.id!r}"
        )

    original_by_id = {e.id: e for e in original.entities}
    new_entities: list[Entity] = []
    seen_ids: set[str] = set()

    for line in lines:
        if line.op == "reject":
            seen_ids.add(line.id)
            continue
        if line.op == "add":
            if not line.id.startswith("human_"):
                raise ValueError(
                    f"Added entity must have id starting with 'human_'; got {line.id!r}"
                )
            new_entities.append(_line_to_entity(line, original.document.id))
            continue
        # keep / edit
        if line.id not in original_by_id:
            raise ValueError(
                f"op={line.op!r} references unknown entity id {line.id!r}"
            )
        seen_ids.add(line.id)
        new_entities.append(_line_to_entity(line, original.document.id))

    # Entities not mentioned in the review file are preserved as-is.
    for original_id, entity in original_by_id.items():
        if original_id not in seen_ids:
            new_entities.append(entity)

    new_entities.sort(
        key=lambda e: (e.mentions[0].span.start if e.mentions else 0, e.id)
    )

    return ExtractionResult(
        document=original.document,
        entities=new_entities,
        schema_version=original.schema_version,
        model_versions=original.model_versions,
    )


def _line_to_entity(line: ReviewEntityLine, doc_id: str) -> Entity:
    mentions: list[Mention] = [
        review_mention_to_mention(rm, doc_id, line.type) for rm in line.mentions
    ]
    mentions.sort(key=lambda m: (m.span.start, m.span.end))
    return Entity(
        id=line.id,
        type=line.type,
        canonical=line.canonical,
        surface_canonical=line.surface_canonical,
        mentions=mentions,
        confidence=line.confidence,
        doc_id=doc_id,
        cluster_id=line.cluster_id,
    )

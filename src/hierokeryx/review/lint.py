"""Semantic validation of edited review files.

Catches the failure modes JSON Schema can't: span offsets that don't match
the original document text, types not in the schema, duplicate entity ids,
references to mentions that don't exist, etc.
"""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from hierokeryx.models import Document, EntitySchema, ExtractionResult
from hierokeryx.review.jsonl import (
    ReviewEntityLine,
    ReviewHeader,
    _text_sha,
    read_review,
    read_review_dir,
)


@dataclass(frozen=True)
class LintError:
    file: Path | None
    line: int  # 1-indexed; 0 = file-level
    message: str

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}" if self.file else f"line {self.line}"
        return f"{loc}: {self.message}"


def lint_review_file(
    path: str | Path,
    *,
    document: Document | None = None,
    schema: EntitySchema | None = None,
    original: ExtractionResult | None = None,
) -> list[LintError]:
    p = Path(path)
    try:
        header, entries = read_review(p)
    except Exception as e:  # pragma: no cover - reraised as a structured error
        return [LintError(p, 0, f"parse error: {e}")]

    errors: list[LintError] = []
    errors.extend(_lint_header(p, header, document=document, original=original))
    errors.extend(_lint_entries(p, header, entries, document=document, schema=schema, original=original))
    return errors


def lint_review_dir(
    directory: str | Path,
    *,
    documents: dict[str, Document] | None = None,
    schema: EntitySchema | None = None,
    extractions: dict[str, ExtractionResult] | None = None,
) -> dict[str, list[LintError]]:
    d = Path(directory)
    parsed = read_review_dir(d)
    out: dict[str, list[LintError]] = {}
    documents = documents or {}
    extractions = extractions or {}
    for doc_id, (_header, _entries) in parsed.items():
        file_path = d / f"{doc_id}.jsonl"
        out[doc_id] = lint_review_file(
            file_path,
            document=documents.get(doc_id),
            schema=schema,
            original=extractions.get(doc_id),
        )
    return out


def _lint_header(
    path: Path,
    header: ReviewHeader,
    *,
    document: Document | None,
    original: ExtractionResult | None,
) -> list[LintError]:
    errors: list[LintError] = []
    if document is not None:
        if document.id != header.doc_id:
            errors.append(
                LintError(path, 1, f"doc_id mismatch: header={header.doc_id!r} doc={document.id!r}")
            )
        if header.text_sha != _text_sha(document.text):
            errors.append(
                LintError(path, 1, "text_sha mismatch — document changed since export")
            )
    if original is not None and original.schema_version != header.schema_version:
        errors.append(
            LintError(
                path,
                1,
                f"schema_version mismatch: header={header.schema_version!r} "
                f"original={original.schema_version!r}",
            )
        )
    return errors


def _lint_entries(
    path: Path,
    header: ReviewHeader,
    entries: list[ReviewEntityLine],
    *,
    document: Document | None,
    schema: EntitySchema | None,
    original: ExtractionResult | None,
) -> list[LintError]:
    errors: list[LintError] = []
    id_counts = Counter(e.id for e in entries)
    valid_types = set(schema.type_names) if schema else None
    original_ids = {e.id for e in original.entities} if original is not None else None

    for idx, entry in enumerate(entries, start=2):  # line 1 is header
        if id_counts[entry.id] > 1:
            errors.append(LintError(path, idx, f"duplicate entity id: {entry.id!r}"))
        if valid_types is not None and entry.type not in valid_types:
            errors.append(
                LintError(path, idx, f"type {entry.type!r} not in schema {sorted(valid_types)}")
            )
        if entry.op in {"edit", "keep", "reject"} and original_ids is not None and entry.id not in original_ids:
            errors.append(
                LintError(
                    path, idx,
                    f"op={entry.op!r} references unknown entity id {entry.id!r} "
                    f"(use op='add' for new entities)",
                )
            )
        if entry.op == "add" and not entry.id.startswith("human_"):
            errors.append(
                LintError(path, idx, f"added entity must have id starting with 'human_'; got {entry.id!r}")
            )
        if document is not None:
            for mention in entry.mentions:
                if mention.end > len(document.text) or mention.start < 0:
                    errors.append(
                        LintError(path, idx, f"mention span out of bounds: ({mention.start}, {mention.end})")
                    )
                    continue
                actual = document.text[mention.start : mention.end]
                if actual != mention.text:
                    errors.append(
                        LintError(
                            path, idx,
                            f"span text mismatch: stored {mention.text!r} != "
                            f"doc[{mention.start}:{mention.end}]={actual!r}",
                        )
                    )
        # Within-entity span overlap check
        spans = sorted((m.start, m.end) for m in entry.mentions)
        for (a_s, a_e), (b_s, b_e) in itertools.pairwise(spans):
            if a_s < b_e and b_s < a_e:
                errors.append(
                    LintError(path, idx, f"overlapping mentions in same entity: ({a_s},{a_e}) and ({b_s},{b_e})")
                )

    return errors

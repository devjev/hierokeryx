"""JSONL gold cluster assignments for eval.

One record per labeled entity:

    {"doc_id": "doc_curie", "entity_id": "e_abc123", "gold_cluster_id": "g_marie"}

Gold cluster ids are opaque labels — only consistency across records matters.
Entities not present in the gold file are excluded from metric computation.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class GoldEntry(BaseModel):
    """One labeled entity in a gold file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    gold_cluster_id: str = Field(min_length=1)


def load_gold(path: str | Path) -> list[GoldEntry]:
    """Read a JSONL gold file, one `GoldEntry` per non-blank line."""
    entries: list[GoldEntry] = []
    seen: set[tuple[str, str]] = set()
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry = GoldEntry.model_validate(json.loads(line))
        key = (entry.doc_id, entry.entity_id)
        if key in seen:
            raise ValueError(
                f"{path}:{lineno}: duplicate (doc_id, entity_id)={key}"
            )
        seen.add(key)
        entries.append(entry)
    return entries


def save_gold(entries: list[GoldEntry], path: str | Path) -> None:
    """Write gold entries as JSONL (one record per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [e.model_dump_json() for e in entries]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

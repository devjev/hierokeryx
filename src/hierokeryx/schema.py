"""Load and save user-declared entity schemas from YAML/JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from hierokeryx.models import EntitySchema, EntityType


def load_schema(path: str | Path) -> EntitySchema:
    """Load an EntitySchema from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Schema file not found: {p}")
    raw = p.read_text(encoding="utf-8")
    data: Any
    if p.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
    elif p.suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(
            f"Unsupported schema file extension {p.suffix!r} (use .yaml, .yml, or .json)"
        )
    if not isinstance(data, dict):
        raise ValueError(f"Schema root must be a mapping; got {type(data).__name__}")
    return EntitySchema.model_validate(data)


def save_schema(schema: EntitySchema, path: str | Path) -> None:
    """Save an EntitySchema to YAML or JSON, format chosen by extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = schema.model_dump()
    if p.suffix in {".yaml", ".yml"}:
        p.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    elif p.suffix == ".json":
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported schema file extension {p.suffix!r} (use .yaml, .yml, or .json)"
        )


DEFAULT_SCHEMA = EntitySchema(
    types=[
        EntityType(
            name="Person",
            description="A named individual human being",
            examples=["Marie Curie", "Albert Einstein", "Ada Lovelace"],
        ),
        EntityType(
            name="Organization",
            description="A company, agency, institution, or other formal group",
            examples=["NASA", "Apple Inc.", "the United Nations"],
        ),
        EntityType(
            name="Location",
            description="A geographic place: country, city, region, landmark, address",
            examples=["Paris", "Mount Everest", "1600 Pennsylvania Avenue"],
        ),
    ],
    version="1",
)

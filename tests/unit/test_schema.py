"""Tests for schema load/save."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hierokeryx.models import EntitySchema, EntityType
from hierokeryx.schema import DEFAULT_SCHEMA, load_schema, save_schema


def test_default_schema_valid() -> None:
    assert len(DEFAULT_SCHEMA.types) == 3
    assert DEFAULT_SCHEMA.type_names == ("Person", "Organization", "Location")


def test_load_save_yaml_roundtrip(tmp_path: Path) -> None:
    schema = EntitySchema(
        types=[
            EntityType(name="Drug", description="A pharmaceutical compound", examples=["aspirin"]),
            EntityType(name="Dose", description="Quantity of a drug administered"),
        ]
    )
    p = tmp_path / "schema.yaml"
    save_schema(schema, p)
    loaded = load_schema(p)
    assert loaded == schema
    assert loaded.fingerprint() == schema.fingerprint()


def test_load_save_json_roundtrip(tmp_path: Path) -> None:
    schema = EntitySchema(
        types=[EntityType(name="X", description="x")],
        version="2",
    )
    p = tmp_path / "schema.json"
    save_schema(schema, p)
    loaded = load_schema(p)
    assert loaded == schema


def test_load_unknown_extension_rejected(tmp_path: Path) -> None:
    p = tmp_path / "schema.toml"
    p.write_text("x = 1")
    with pytest.raises(ValueError, match="extension"):
        load_schema(p)


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_schema(tmp_path / "nope.yaml")


def test_load_non_mapping_rejected(tmp_path: Path) -> None:
    p = tmp_path / "schema.json"
    p.write_text(json.dumps(["not", "a", "mapping"]))
    with pytest.raises(ValueError, match="mapping"):
        load_schema(p)

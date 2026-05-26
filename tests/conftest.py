"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from hierokeryx.models import Document, EntitySchema, EntityType


@pytest.fixture
def schema() -> EntitySchema:
    return EntitySchema(
        types=[
            EntityType(name="Person", description="A named individual human"),
            EntityType(name="Organization", description="A company, agency, or institution"),
        ]
    )


@pytest.fixture
def doc_curie() -> Document:
    return Document(
        id="curie",
        text=(
            "Marie Curie was a Polish physicist and chemist. "
            "Curie was the first woman to win a Nobel Prize."
        ),
    )


@pytest.fixture
def fixtures_root() -> Path:
    return Path(__file__).parent / "fixtures"

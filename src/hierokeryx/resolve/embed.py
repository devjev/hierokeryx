"""Sentence-transformer embeddings for cross-document entity comparison.

Embeddings are L2-normalized so dot product equals cosine similarity. The
model is loaded lazily so importing this module is cheap.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from hierokeryx.models import Document, Entity, ExtractionResult

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Default embedder — small, fast, CPU-friendly."""

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformer: %s", self.model_id)
            self._model = SentenceTransformer(self.model_id, device=self.device)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)


def build_entity_repr(
    entity: Entity,
    document: Document | None = None,
    *,
    max_contexts: int = 2,
    context_window: int = 80,
) -> str:
    """Build a representation string for an entity: canonical + short contexts."""
    parts = [entity.canonical]
    if document is not None:
        for mention in entity.mentions[:max_contexts]:
            ctx = _context_window(document.text, mention.span.start, mention.span.end, context_window)
            parts.append(ctx)
    return " | ".join(parts)


def _context_window(text: str, start: int, end: int, half: int) -> str:
    a = max(0, start - half)
    b = min(len(text), end + half)
    return text[a:b].replace("\n", " ").strip()


def encode_extraction_results(
    results: list[ExtractionResult],
    embedder: SentenceTransformerEmbedder,
) -> tuple[list[tuple[ExtractionResult, Entity]], np.ndarray]:
    """Embed every entity across `results`. Returns (entity_index, embeddings).

    `entity_index[i]` corresponds to row `i` of `embeddings`.
    """
    entity_index: list[tuple[ExtractionResult, Entity]] = []
    texts: list[str] = []
    for result in results:
        for entity in result.entities:
            entity_index.append((result, entity))
            texts.append(build_entity_repr(entity, result.document))
    embeddings = embedder.encode(texts)
    return entity_index, embeddings

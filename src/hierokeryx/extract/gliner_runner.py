"""GLiNER-based span extraction. Lazy-loads the model on first `extract` call."""

from __future__ import annotations

import logging
from typing import Any

from hierokeryx.extract.tokenize import snap_to_token_boundary, tokenize
from hierokeryx.models import (
    Document,
    EntitySchema,
    Mention,
    Span,
    make_mention_id,
)

logger = logging.getLogger(__name__)


class GLiNERExtractor:
    """Extract character-indexed mentions for a user-defined schema.

    The model is loaded lazily so importing this module is cheap. Call
    `extract(document, schema)` on each document; the same extractor instance
    can be reused across documents and across schemas (labels are passed per
    call).
    """

    def __init__(
        self,
        model_id: str = "urchade/gliner_large-v2.5",
        threshold: float = 0.4,
        flat_ner: bool = False,
        multi_label: bool = True,
        label_template: str = "{name}",
        device: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.flat_ner = flat_ner
        self.multi_label = multi_label
        self.label_template = label_template
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from gliner import GLiNER

            logger.info("Loading GLiNER model: %s", self.model_id)
            self._model = GLiNER.from_pretrained(self.model_id)
            if self.device:
                self._model = self._model.to(self.device)
        return self._model

    def extract(self, document: Document, schema: EntitySchema) -> list[Mention]:
        """Return mentions for `document` under `schema`. Spans are
        token-snapped and de-overlapped within each type.
        """
        labels = [
            self.label_template.format(name=t.name, description=t.description)
            for t in schema.types
        ]
        label_to_type = {label: t.name for label, t in zip(labels, schema.types, strict=True)}

        raw_predictions = self.model.predict_entities(
            document.text,
            labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            multi_label=self.multi_label,
        )

        tokens = tokenize(document.text)
        mentions: list[Mention] = []
        for pred in raw_predictions:
            type_name = label_to_type.get(pred["label"])
            if type_name is None:
                logger.warning("Unmapped GLiNER label: %r", pred["label"])
                continue

            raw_start, raw_end = int(pred["start"]), int(pred["end"])
            start, end = snap_to_token_boundary(document.text, raw_start, raw_end, tokens)
            if end <= start:
                continue
            text = document.text[start:end]

            mentions.append(
                Mention(
                    id=make_mention_id(document.id, start, end),
                    span=Span(start=start, end=end, text=text),
                    type=type_name,
                    score=float(pred["score"]),
                    source="gliner",
                )
            )

        return normalize_mentions(mentions)


def normalize_mentions(mentions: list[Mention]) -> list[Mention]:
    """Within each entity type, greedily keep the highest-scoring non-overlapping
    spans. Different-type overlaps are preserved (multi-label is meaningful).
    """
    by_type: dict[str, list[Mention]] = {}
    for m in mentions:
        by_type.setdefault(m.type, []).append(m)

    accepted: list[Mention] = []
    for type_mentions in by_type.values():
        ranked = sorted(
            type_mentions,
            key=lambda m: (-m.score, m.span.start, -(m.span.end - m.span.start)),
        )
        kept: list[Mention] = []
        for cand in ranked:
            if any(_overlaps(cand.span, prev.span) for prev in kept):
                continue
            kept.append(cand)
        accepted.extend(kept)

    return sorted(accepted, key=lambda m: (m.span.start, m.span.end, m.type))


def _overlaps(a: Span, b: Span) -> bool:
    return a.start < b.end and b.start < a.end

"""Core Pydantic v2 data models for hierokeryx.

All models are frozen and forbid extra fields. JSONL keys map 1:1 to Pydantic
field names so the review-file serializer in `hierokeryx.review.jsonl` is a
thin wrapper around `model_dump_json` / `model_validate_json`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MentionSource = Literal["gliner", "llm", "human"]
ReviewReason = Literal[
    "low_span_conf",
    "low_cluster_conf",
    "ambiguous_merge",
    "type_uncertain",
]
ReviewOp = Literal["keep", "reject", "edit", "add"]
# v1 ships keep / reject / edit / add. "Split" is expressed as one `reject` plus
# two or more `add` lines; "merge" is one `add` carrying the unioned mentions
# plus `reject` on the originals. Dedicated split/merge ops may return in v2 if
# the UUID-grouping ergonomics earn their keep on real reviewer workflows.


class EntityType(BaseModel):
    """One declared entity type. The description is fed to GLiNER as part of
    the label string; richer descriptions materially improve zero-shot recall.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)


class EntitySchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    types: list[EntityType] = Field(min_length=1)
    version: str = "1"

    @model_validator(mode="after")
    def _unique_names(self) -> EntitySchema:
        names = [t.name for t in self.types]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate type names in schema: {names}")
        return self

    def fingerprint(self) -> str:
        """Stable content hash; busts LLM prompt caches and tags artifacts."""
        payload = {
            "version": self.version,
            "types": [
                {"name": t.name, "description": t.description, "examples": list(t.examples)}
                for t in self.types
            ],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def type_by_name(self, name: str) -> EntityType | None:
        return next((t for t in self.types if t.name == name), None)

    @property
    def type_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.types)


class Span(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Span:
        if self.end <= self.start:
            raise ValueError(f"Span end ({self.end}) must be > start ({self.start})")
        if len(self.text) != self.end - self.start:
            raise ValueError(
                f"Span text length {len(self.text)} != end-start "
                f"{self.end - self.start} (text={self.text!r})"
            )
        return self


class Mention(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    span: Span
    type: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    source: MentionSource = "gliner"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    canonical: str = Field(min_length=1)
    surface_canonical: str = Field(min_length=1)
    mentions: list[Mention] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    doc_id: str = Field(min_length=1)
    cluster_id: str | None = None


class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    document: Document
    entities: list[Entity]
    schema_version: str
    model_versions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spans_align(self) -> ExtractionResult:
        # Load-bearing invariant: every mention span quotes the document verbatim.
        for entity in self.entities:
            if entity.doc_id != self.document.id:
                raise ValueError(
                    f"Entity {entity.id} has doc_id={entity.doc_id!r}, "
                    f"expected {self.document.id!r}"
                )
            for mention in entity.mentions:
                actual = self.document.text[mention.span.start : mention.span.end]
                if actual != mention.span.text:
                    raise ValueError(
                        f"Mention {mention.id}: span.text={mention.span.text!r} "
                        f"!= doc.text[{mention.span.start}:{mention.span.end}]={actual!r}"
                    )
        return self


class EntityRegistry(BaseModel):
    """Cross-document canonical entity registry.

    `clusters[cluster_id] = [f"{doc_id}/{entity_id}", ...]` so an entity is
    addressable across documents without needing to reload them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clusters: dict[str, list[str]] = Field(default_factory=dict)
    canonical_by_cluster: dict[str, str] = Field(default_factory=dict)
    type_by_cluster: dict[str, str] = Field(default_factory=dict)
    schema_version: str


class CorefCluster(BaseModel):
    """LLM-returned cluster of within-doc mentions referring to one entity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mention_ids: list[str] = Field(min_length=1)
    canonical: str = Field(min_length=1)
    type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class CrossDocCandidate(BaseModel):
    """An entity offered to the cross-doc resolver alongside its nearest cluster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str
    doc_id: str
    type: str
    canonical: str
    contexts: list[str] = Field(default_factory=list)
    nearest_cluster_id: str | None = None
    nearest_similarity: float = 0.0


class MergeDecision(BaseModel):
    """LLM tie-break decision: assign a candidate to a cluster, or keep singleton."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str
    target_cluster_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class ReviewItem(BaseModel):
    """A flagged entity routed to the human-in-the-loop queue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    entity_id: str
    reason: ReviewReason
    current: Entity
    candidates: list[Entity] = Field(default_factory=list)


def make_mention_id(doc_id: str, start: int, end: int, source: str = "gliner") -> str:
    """Stable mention id from (doc_id, start, end). `source` lets human-added
    mentions land in a different id namespace so reruns don't collide with them.
    """
    if source == "human":
        return f"human_{hashlib.sha1(f'{doc_id}:{start}:{end}'.encode()).hexdigest()[:10]}"
    return f"m_{hashlib.sha1(f'{doc_id}:{start}:{end}:{source}'.encode()).hexdigest()[:10]}"


def make_entity_id(doc_id: str, mention_ids: list[str]) -> str:
    """Entity id derived from sorted mention ids — stable across re-runs."""
    payload = doc_id + "|" + "|".join(sorted(mention_ids))
    return f"e_{hashlib.sha1(payload.encode()).hexdigest()[:10]}"


def make_cluster_id(canonical: str, type_name: str, member_doc_ids: list[str]) -> str:
    """Cluster id derived from canonical + type + sorted member doc ids."""
    payload = f"{type_name}|{canonical}|" + "|".join(sorted(set(member_doc_ids)))
    return f"c_{hashlib.sha1(payload.encode()).hexdigest()[:10]}"

"""Tests for cross-doc clustering using a fake embedder (no model download)."""

from __future__ import annotations

import numpy as np

from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    Span,
)
from hierokeryx.resolve.crossdoc import build_registry, resolve_crossdoc


class FakeEmbedder:
    """Returns deterministic L2-normalized embeddings keyed by the text prefix.

    Texts starting with the same letter get nearby embeddings so we can craft
    expected clusters precisely.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for t in texts:
            first = t.strip()[:1] or "_"
            # Project the first char into a 4-d direction; near-identical texts
            # get the same vector. We add a tiny per-text perturbation so HDBSCAN
            # has more than one point per cluster.
            base = np.zeros(4, dtype=np.float32)
            base[ord(first) % 4] = 1.0
            raw = np.frombuffer(t.encode("utf-8")[:4].ljust(4, b"\0"), dtype=np.uint8)
            noise = (raw.astype(np.float32) - 64) / 1000
            vec = base + noise
            vec = vec / max(np.linalg.norm(vec), 1e-9)
            vectors.append(vec)
        return np.asarray(vectors, dtype=np.float32)


def _mk_entity(eid: str, doc_id: str, canonical: str, type_: str = "Person") -> Entity:
    return Entity(
        id=eid,
        type=type_,
        canonical=canonical,
        surface_canonical=canonical,
        mentions=[
            Mention(
                id=f"m_{eid}",
                span=Span(start=0, end=len(canonical), text=canonical),
                type=type_,
                score=0.9,
            )
        ],
        confidence=0.9,
        doc_id=doc_id,
    )


def _mk_result(doc_id: str, text: str, entities: list[Entity]) -> ExtractionResult:
    doc = Document(id=doc_id, text=text)
    return ExtractionResult(
        document=doc,
        entities=entities,
        schema_version="test",
    )


def test_empty_inputs_return_empty_registry() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    updated, registry = resolve_crossdoc([], schema)
    assert updated == []
    assert registry.clusters == {}


def test_singletons_get_unique_cluster_ids() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    r1 = _mk_result("d1", "Alice was here.", [_mk_entity("e1", "d1", "Alice")])
    r2 = _mk_result("d2", "Bob was there.", [_mk_entity("e2", "d2", "Bob")])
    updated, registry = resolve_crossdoc([r1, r2], schema, embedder=FakeEmbedder())
    cluster_ids = {e.cluster_id for r in updated for e in r.entities}
    assert None not in cluster_ids
    assert len(cluster_ids) == 2
    # Registry tracks every clustered entity
    assert sum(len(v) for v in registry.clusters.values()) == 2


def test_similar_entities_cluster_together() -> None:
    """Entities with same canonical text start with the same character —
    FakeEmbedder gives them nearly identical embeddings so HDBSCAN merges them.
    """
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    results = [
        _mk_result(f"d{i}", f"Alice in doc {i}.", [_mk_entity(f"e{i}", f"d{i}", "Alice")])
        for i in range(3)
    ]
    updated, registry = resolve_crossdoc(results, schema, embedder=FakeEmbedder())
    # All three Alice entities should share one cluster
    alice_clusters = {e.cluster_id for r in updated for e in r.entities}
    assert len(alice_clusters) == 1, f"Expected one cluster, got {alice_clusters}"
    [the_cluster] = list(registry.clusters)
    assert len(registry.clusters[the_cluster]) == 3
    assert registry.canonical_by_cluster[the_cluster] == "Alice"
    assert registry.type_by_cluster[the_cluster] == "Person"


def test_different_types_never_share_cluster() -> None:
    schema = EntitySchema(
        types=[
            EntityType(name="Person", description="x"),
            EntityType(name="Organization", description="y"),
        ]
    )
    person = _mk_entity("e1", "d1", "Alice", type_="Person")
    org = _mk_entity("e2", "d2", "Alice", type_="Organization")
    r1 = _mk_result("d1", "Alice", [person])
    r2 = _mk_result("d2", "Alice", [org])
    updated, _registry = resolve_crossdoc([r1, r2], schema, embedder=FakeEmbedder())
    clusters = {e.cluster_id for r in updated for e in r.entities}
    assert len(clusters) == 2


def test_build_registry_from_pre_assigned() -> None:
    """build_registry can be called directly on results that already carry cluster_ids."""
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    e1 = _mk_entity("e1", "d1", "Marie Curie").model_copy(update={"cluster_id": "c_x"})
    e2 = _mk_entity("e2", "d2", "M. Curie").model_copy(update={"cluster_id": "c_x"})
    e3 = _mk_entity("e3", "d3", "Pierre Curie").model_copy(update={"cluster_id": "c_y"})
    results = [
        _mk_result("d1", "Marie Curie", [e1]),
        _mk_result("d2", "M. Curie", [e2]),
        _mk_result("d3", "Pierre Curie", [e3]),
    ]
    registry = build_registry(results, schema)
    assert set(registry.clusters) == {"c_x", "c_y"}
    assert sorted(registry.clusters["c_x"]) == ["d1/e1", "d2/e2"]
    # Longest surface_canonical wins
    assert registry.canonical_by_cluster["c_x"] == "Marie Curie"

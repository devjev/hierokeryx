"""Tests for incremental cross-doc resolution + centroid sidecar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    Span,
)
from hierokeryx.resolve.centroids import (
    RegistryCentroids,
    compute_centroids,
    load_centroids,
    save_centroids,
    update_centroids,
)
from hierokeryx.resolve.crossdoc import resolve_crossdoc, resolve_incremental


class FakeEmbedder:
    """Deterministic char-prefix embedder — same shape as the one in test_crossdoc."""

    model_id = "fake-mini"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for t in texts:
            first = t.strip()[:1] or "_"
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
    return ExtractionResult(
        document=Document(id=doc_id, text=text),
        entities=entities,
        schema_version="test",
    )


# ---- centroid sidecar persistence ---------------------------------------

def test_compute_centroids_and_save_load_roundtrip(tmp_path: Path) -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    results = [
        _mk_result(f"d{i}", "Alice", [_mk_entity(f"e{i}", f"d{i}", "Alice")])
        for i in range(3)
    ]
    embedder = FakeEmbedder()
    resolved, _ = resolve_crossdoc(results, schema, embedder=embedder)  # type: ignore[arg-type]
    centroids = compute_centroids(resolved, embedder)  # type: ignore[arg-type]
    assert len(centroids.cluster_ids) >= 1
    assert centroids.embedder_id == "fake-mini"

    save_centroids(tmp_path, centroids)
    loaded = load_centroids(tmp_path)
    assert loaded.cluster_ids == centroids.cluster_ids
    assert loaded.embedder_id == centroids.embedder_id
    np.testing.assert_allclose(loaded.centroids, centroids.centroids, rtol=1e-5)
    np.testing.assert_array_equal(loaded.sizes, centroids.sizes)


def test_update_centroids_running_mean() -> None:
    existing = RegistryCentroids(
        cluster_ids=("c1",),
        centroids=np.array([[1.0, 0.0]], dtype=np.float32),
        sizes=np.array([2], dtype=np.int32),
        embedder_id="fake",
    )
    new_vec = np.array([[0.0, 1.0]], dtype=np.float32)
    out = update_centroids(
        existing,
        additions={"c1": new_vec},
        new_clusters={},
        embedder_id="fake",
    )
    # (1,0)*2 + (0,1)*1 → (2, 1)/3 = (0.667, 0.333), then L2-normalized.
    expected = np.array([2.0, 1.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(out.centroids[0], expected, rtol=1e-5)
    assert int(out.sizes[0]) == 3


def test_update_centroids_appends_new_cluster() -> None:
    existing = RegistryCentroids(
        cluster_ids=("c1",),
        centroids=np.array([[1.0, 0.0]], dtype=np.float32),
        sizes=np.array([1], dtype=np.int32),
        embedder_id="fake",
    )
    out = update_centroids(
        existing,
        additions={},
        new_clusters={"c2": np.array([[0.0, 1.0]], dtype=np.float32)},
        embedder_id="fake",
    )
    assert out.cluster_ids == ("c1", "c2")
    np.testing.assert_allclose(out.centroids[1], np.array([0.0, 1.0]), rtol=1e-5)


def test_update_centroids_rejects_embedder_mismatch() -> None:
    existing = RegistryCentroids(
        cluster_ids=(),
        centroids=np.empty((0, 0), dtype=np.float32),
        sizes=np.empty((0,), dtype=np.int32),
        embedder_id="fake-A",
    )
    with pytest.raises(ValueError, match="embedder mismatch"):
        update_centroids(existing, additions={}, new_clusters={}, embedder_id="fake-B")


# ---- resolve_incremental ------------------------------------------------

def test_incremental_assigns_new_entity_to_existing_cluster() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    # Batch on two Alice docs first.
    batch_results = [
        _mk_result(f"d{i}", "Alice", [_mk_entity(f"e{i}", f"d{i}", "Alice")])
        for i in range(2)
    ]
    embedder = FakeEmbedder()
    resolved, registry = resolve_crossdoc(batch_results, schema, embedder=embedder)  # type: ignore[arg-type]
    centroids = compute_centroids(resolved, embedder)  # type: ignore[arg-type]
    assert len(registry.clusters) == 1
    [existing_cluster] = list(registry.clusters)

    # Incremental: a new Alice doc should join the existing cluster.
    new_results = [_mk_result("d99", "Alice", [_mk_entity("e99", "d99", "Alice")])]
    updated, new_registry, new_centroids = resolve_incremental(
        new_results,
        schema,
        existing_registry=registry,
        existing_centroids=centroids,
        embedder=embedder,  # type: ignore[arg-type]
    )
    [e99_entity] = updated[0].entities
    assert e99_entity.cluster_id == existing_cluster
    # Registry size unchanged (joined an existing cluster).
    assert set(new_registry.clusters) == {existing_cluster}
    # Centroid size incremented.
    pos = new_centroids.cluster_ids.index(existing_cluster)
    assert int(new_centroids.sizes[pos]) == 3


def test_incremental_creates_new_cluster_for_unseen_entity() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    batch_results = [
        _mk_result("d1", "Alice", [_mk_entity("e1", "d1", "Alice")]),
        _mk_result("d2", "Alice", [_mk_entity("e2", "d2", "Alice")]),
    ]
    embedder = FakeEmbedder()
    resolved, registry = resolve_crossdoc(batch_results, schema, embedder=embedder)  # type: ignore[arg-type]
    centroids = compute_centroids(resolved, embedder)  # type: ignore[arg-type]
    existing_clusters = set(registry.clusters)

    # Bob starts with a different first letter → different fake embedding.
    new_results = [_mk_result("d3", "Bob", [_mk_entity("e3", "d3", "Bob")])]
    updated, new_registry, _ = resolve_incremental(
        new_results,
        schema,
        existing_registry=registry,
        existing_centroids=centroids,
        embedder=embedder,  # type: ignore[arg-type]
    )
    [e3_entity] = updated[0].entities
    assert e3_entity.cluster_id is not None
    assert e3_entity.cluster_id not in existing_clusters
    assert set(new_registry.clusters) == existing_clusters | {e3_entity.cluster_id}


def test_incremental_two_new_docs_with_same_unseen_entity_share_cluster() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    batch_results = [_mk_result("d1", "Alice", [_mk_entity("e1", "d1", "Alice")])]
    embedder = FakeEmbedder()
    resolved, registry = resolve_crossdoc(batch_results, schema, embedder=embedder)  # type: ignore[arg-type]
    centroids = compute_centroids(resolved, embedder)  # type: ignore[arg-type]

    # Two new Bob docs — should form ONE new cluster, not two singletons.
    new_results = [
        _mk_result(f"d{i}", "Bob", [_mk_entity(f"e{i}", f"d{i}", "Bob")])
        for i in (10, 11)
    ]
    updated, new_registry, _ = resolve_incremental(
        new_results,
        schema,
        existing_registry=registry,
        existing_centroids=centroids,
        embedder=embedder,  # type: ignore[arg-type]
    )
    bob_clusters = {r.entities[0].cluster_id for r in updated}
    assert len(bob_clusters) == 1
    [bob_cid] = bob_clusters
    assert bob_cid not in registry.clusters
    assert len(new_registry.clusters[bob_cid]) == 2


def test_incremental_no_cross_type_merges() -> None:
    schema = EntitySchema(
        types=[
            EntityType(name="Person", description="x"),
            EntityType(name="Organization", description="y"),
        ]
    )
    batch_results = [
        _mk_result("d1", "Alice", [_mk_entity("e1", "d1", "Alice", type_="Person")])
    ]
    embedder = FakeEmbedder()
    resolved, registry = resolve_crossdoc(batch_results, schema, embedder=embedder)  # type: ignore[arg-type]
    centroids = compute_centroids(resolved, embedder)  # type: ignore[arg-type]
    [person_cid] = list(registry.clusters)

    new_results = [_mk_result("d2", "Alice", [_mk_entity("e2", "d2", "Alice", type_="Organization")])]
    updated, _new_registry, _ = resolve_incremental(
        new_results,
        schema,
        existing_registry=registry,
        existing_centroids=centroids,
        embedder=embedder,  # type: ignore[arg-type]
    )
    [org_entity] = updated[0].entities
    # Same canonical, same embedding direction (same first char), but different
    # type → must NOT merge into the Person cluster.
    assert org_entity.cluster_id != person_cid

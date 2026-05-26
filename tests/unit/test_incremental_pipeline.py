"""Pipeline-level smoke for incremental resolution via `pipeline.run`.

Uses an in-memory fake extractor + fake LLM + fake embedder so the test
costs nothing — but still exercises the file-system persistence layer
(extractions, registry, centroid sidecar) and the `incremental_from` wiring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hierokeryx import pipeline
from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    EntityType,
    Mention,
    MergeDecision,
    Span,
    make_mention_id,
)


class WordExtractor:
    """Pretends to be GLiNER: emits one Mention per whitespace-separated word."""

    model_id = "word-extractor"

    def extract(self, document: Document, schema: EntitySchema) -> list[Mention]:
        mentions: list[Mention] = []
        cursor = 0
        for word in document.text.split():
            start = document.text.find(word, cursor)
            end = start + len(word)
            cursor = end
            mentions.append(
                Mention(
                    id=make_mention_id(document.id, start, end),
                    span=Span(start=start, end=end, text=word),
                    type=schema.type_names[0],
                    score=0.9,
                )
            )
        return mentions


class CanonicalLLM:
    coref_model = "fake-coref"
    tiebreak_model = "fake-tiebreak"

    def cluster_mentions(
        self, document: Document, mentions: list[Mention], schema: EntitySchema
    ) -> list[CorefCluster]:
        groups: dict[str, list[Mention]] = {}
        for m in mentions:
            groups.setdefault(m.span.text.lower(), []).append(m)
        return [
            CorefCluster(
                mention_ids=[m.id for m in ms],
                canonical=ms[0].span.text,
                type=ms[0].type,
                confidence=0.9,
            )
            for ms in groups.values()
        ]

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        return [
            MergeDecision(
                entity_id=c.entity_id,
                target_cluster_id=c.nearest_cluster_id,
                confidence=0.8,
            )
            for c in candidates
        ]


class FirstCharEmbedder:
    model_id = "first-char"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vecs = []
        for t in texts:
            ch = (t.strip()[:1] or "_").lower()
            v = np.zeros(26, dtype=np.float32)
            idx = (ord(ch) - ord("a")) % 26
            v[idx] = 1.0
            vecs.append(v)
        return np.asarray(vecs, dtype=np.float32)


def _schema() -> EntitySchema:
    return EntitySchema(types=[EntityType(name="Person", description="x")])


def test_pipeline_incremental_merges_into_existing_workdir(tmp_path: Path) -> None:
    schema = _schema()
    docs_batch1 = [
        Document(id="d1", text="Alice"),
        Document(id="d2", text="Alice"),
    ]
    docs_batch2 = [Document(id="d3", text="Alice"), Document(id="d4", text="Bob")]

    # Batch 1
    wd1 = tmp_path / "wd1"
    run1 = pipeline.run(
        documents=docs_batch1,
        schema=schema,
        workdir=wd1,
        extractor=WordExtractor(),  # type: ignore[arg-type]
        llm_client=CanonicalLLM(),
        embedder=FirstCharEmbedder(),  # type: ignore[arg-type]
    )
    assert (wd1 / "registry.json").exists()
    assert (wd1 / "registry_embeddings.npz").exists()
    assert (wd1 / "registry_embeddings.meta.json").exists()
    assert len(run1.registry.clusters) == 1, "expected one Alice cluster after batch 1"
    [alice_cid] = list(run1.registry.clusters)

    # Batch 2: incremental against wd1
    wd2 = tmp_path / "wd2"
    run2 = pipeline.run(
        documents=docs_batch2,
        schema=schema,
        workdir=wd2,
        extractor=WordExtractor(),  # type: ignore[arg-type]
        llm_client=CanonicalLLM(),
        embedder=FirstCharEmbedder(),  # type: ignore[arg-type]
        incremental_from=wd1,
    )
    # d3's Alice joined the existing cluster; d4's Bob is a new cluster.
    cluster_ids = {e.cluster_id for r in run2.extraction_results for e in r.entities}
    assert alice_cid in cluster_ids
    new_cluster_ids = cluster_ids - {alice_cid}
    assert len(new_cluster_ids) == 1, f"expected one new cluster, got {cluster_ids}"

    # Registry now has both clusters, with Alice spanning all three Alice docs.
    assert alice_cid in run2.registry.clusters
    alice_members = run2.registry.clusters[alice_cid]
    assert {ref.split("/", 1)[0] for ref in alice_members} == {"d1", "d2", "d3"}

    # Centroid sidecar persisted with both clusters.
    centroids = pipeline.load_centroids(wd2)
    assert set(centroids.cluster_ids) >= {alice_cid, *new_cluster_ids}


def test_pipeline_batch_writes_centroid_sidecar(tmp_path: Path) -> None:
    schema = _schema()
    docs = [Document(id=f"d{i}", text="Alice") for i in range(2)]
    wd = tmp_path / "wd"
    pipeline.run(
        documents=docs,
        schema=schema,
        workdir=wd,
        extractor=WordExtractor(),  # type: ignore[arg-type]
        llm_client=CanonicalLLM(),
        embedder=FirstCharEmbedder(),  # type: ignore[arg-type]
    )
    centroids = pipeline.load_centroids(wd)
    assert len(centroids.cluster_ids) >= 1
    assert centroids.embedder_id == "first-char"


def test_pipeline_resolve_centroids_rebuild_creates_sidecar(tmp_path: Path) -> None:
    # Simulate a pre-existing workdir by running a batch then deleting the sidecar.
    schema = _schema()
    docs = [Document(id="d1", text="Alice"), Document(id="d2", text="Alice")]
    wd = tmp_path / "wd"
    pipeline.run(
        documents=docs,
        schema=schema,
        workdir=wd,
        extractor=WordExtractor(),  # type: ignore[arg-type]
        llm_client=CanonicalLLM(),
        embedder=FirstCharEmbedder(),  # type: ignore[arg-type]
    )
    # Wipe the sidecar to mimic a legacy workdir.
    (wd / "registry_embeddings.npz").unlink()
    (wd / "registry_embeddings.meta.json").unlink()

    # Recompute via compute_centroids + save_centroids (what the CLI command does).
    results = pipeline.load_extractions_dir(wd / "extractions")
    centroids = pipeline.compute_centroids(results, FirstCharEmbedder())  # type: ignore[arg-type]
    pipeline.save_centroids(wd, centroids)

    loaded = pipeline.load_centroids(wd)
    assert len(loaded.cluster_ids) == len(centroids.cluster_ids)

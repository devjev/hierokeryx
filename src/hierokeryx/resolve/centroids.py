"""Cluster-centroid sidecar for incremental cross-document resolution.

Stored alongside `EntityRegistry` so new documents can be matched against
existing clusters without re-embedding the old corpus. Saved as a single
`.npz` file plus a tiny JSON manifest sibling:

    workdir/
      registry.json
      registry_embeddings.npz       # cluster_ids, centroids, sizes
      registry_embeddings.meta.json # embedder_id, schema_version, dim

The `embedder_id` is recorded so we can refuse to merge across embedder
changes — mixing centroids from different embedding models silently produces
garbage similarities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hierokeryx.models import ExtractionResult
from hierokeryx.resolve.embed import (
    SentenceTransformerEmbedder,
    encode_extraction_results,
)


@dataclass(frozen=True)
class RegistryCentroids:
    """One L2-normalized centroid per cluster, plus member-count for weighted
    running-mean updates and the `embedder_id` that produced them.
    """

    cluster_ids: tuple[str, ...]
    centroids: np.ndarray  # shape (n_clusters, dim), float32, L2-normalized
    sizes: np.ndarray  # shape (n_clusters,), int32, member counts
    embedder_id: str

    def __post_init__(self) -> None:
        n = len(self.cluster_ids)
        if self.centroids.shape[0] != n:
            raise ValueError(
                f"centroids rows {self.centroids.shape[0]} != cluster_ids {n}"
            )
        if self.sizes.shape[0] != n:
            raise ValueError(f"sizes len {self.sizes.shape[0]} != cluster_ids {n}")

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1]) if self.centroids.size else 0

    def index_of(self, cluster_id: str) -> int | None:
        try:
            return self.cluster_ids.index(cluster_id)
        except ValueError:
            return None


def compute_centroids(
    extraction_results: list[ExtractionResult],
    embedder: SentenceTransformerEmbedder,
) -> RegistryCentroids:
    """Embed every entity and average per cluster_id to derive centroids.

    Used to retrofit a sidecar onto a workdir that pre-dates centroid
    persistence, and as the canonical builder after a batch resolve run.
    """
    entity_index, embeddings = encode_extraction_results(extraction_results, embedder)
    if embeddings.size == 0:
        return RegistryCentroids(
            cluster_ids=(),
            centroids=np.empty((0, 0), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.int32),
            embedder_id=embedder.model_id,
        )

    rows_by_cluster: dict[str, list[int]] = {}
    for idx, (_, entity) in enumerate(entity_index):
        if entity.cluster_id is None:
            continue
        rows_by_cluster.setdefault(entity.cluster_id, []).append(idx)

    cluster_ids = tuple(sorted(rows_by_cluster.keys()))
    if not cluster_ids:
        return RegistryCentroids(
            cluster_ids=(),
            centroids=np.empty((0, embeddings.shape[1]), dtype=np.float32),
            sizes=np.empty((0,), dtype=np.int32),
            embedder_id=embedder.model_id,
        )

    centroids = np.zeros((len(cluster_ids), embeddings.shape[1]), dtype=np.float32)
    sizes = np.zeros((len(cluster_ids),), dtype=np.int32)
    for i, cid in enumerate(cluster_ids):
        rows = rows_by_cluster[cid]
        c = embeddings[rows].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-9)
        centroids[i] = c.astype(np.float32)
        sizes[i] = len(rows)
    return RegistryCentroids(
        cluster_ids=cluster_ids,
        centroids=centroids,
        sizes=sizes,
        embedder_id=embedder.model_id,
    )


def update_centroids(
    existing: RegistryCentroids,
    *,
    additions: dict[str, np.ndarray],  # cluster_id -> (k_i, dim) new vectors
    new_clusters: dict[str, np.ndarray],  # cluster_id -> (k_i, dim) new vectors
    embedder_id: str,
) -> RegistryCentroids:
    """Return centroids with `additions` merged into existing clusters
    (running mean weighted by current size) and `new_clusters` appended.

    All input vectors are assumed L2-normalized; outputs are re-normalized.
    """
    if existing.embedder_id and embedder_id != existing.embedder_id:
        raise ValueError(
            f"embedder mismatch: existing={existing.embedder_id!r} new={embedder_id!r}. "
            "Refusing to merge centroids from different embedding models."
        )

    cluster_ids = list(existing.cluster_ids)
    centroids = (
        existing.centroids.copy()
        if existing.centroids.size
        else np.empty((0, 0), dtype=np.float32)
    )
    sizes = existing.sizes.copy()

    for cid, vecs in additions.items():
        idx = cluster_ids.index(cid) if cid in cluster_ids else None
        if idx is None:
            raise KeyError(f"addition for unknown cluster {cid!r}")
        old_n = int(sizes[idx])
        new_n = vecs.shape[0]
        if new_n == 0:
            continue
        summed = centroids[idx] * old_n + vecs.sum(axis=0)
        total = old_n + new_n
        merged = summed / total
        merged = merged / max(float(np.linalg.norm(merged)), 1e-9)
        centroids[idx] = merged.astype(np.float32)
        sizes[idx] = total

    for cid, vecs in new_clusters.items():
        if cid in cluster_ids:
            raise ValueError(f"cluster id {cid!r} already exists in registry")
        if vecs.shape[0] == 0:
            continue
        c = vecs.mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-9)
        c = c.astype(np.float32)
        centroids = (
            c.reshape(1, -1)
            if centroids.size == 0
            else np.vstack([centroids, c.reshape(1, -1)])
        )
        sizes = np.concatenate([sizes, np.array([vecs.shape[0]], dtype=np.int32)])
        cluster_ids.append(cid)

    return RegistryCentroids(
        cluster_ids=tuple(cluster_ids),
        centroids=centroids,
        sizes=sizes,
        embedder_id=embedder_id,
    )


# Persistence ---------------------------------------------------------------

def centroids_paths(workdir: str | Path) -> tuple[Path, Path]:
    """Return `(npz_path, meta_path)` for a workdir."""
    base = Path(workdir)
    return base / "registry_embeddings.npz", base / "registry_embeddings.meta.json"


def save_centroids(workdir: str | Path, centroids: RegistryCentroids) -> None:
    npz_path, meta_path = centroids_paths(workdir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        cluster_ids=np.array(centroids.cluster_ids, dtype=object),
        centroids=centroids.centroids,
        sizes=centroids.sizes,
    )
    meta_path.write_text(
        json.dumps(
            {
                "embedder_id": centroids.embedder_id,
                "n_clusters": len(centroids.cluster_ids),
                "dim": centroids.dim,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_centroids(workdir: str | Path) -> RegistryCentroids:
    npz_path, meta_path = centroids_paths(workdir)
    if not npz_path.exists():
        raise FileNotFoundError(f"No centroid sidecar at {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"No centroid metadata at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=True) as data:
        cluster_ids = tuple(str(x) for x in data["cluster_ids"].tolist())
        centroids = np.asarray(data["centroids"], dtype=np.float32)
        sizes = np.asarray(data["sizes"], dtype=np.int32)
    return RegistryCentroids(
        cluster_ids=cluster_ids,
        centroids=centroids,
        sizes=sizes,
        embedder_id=str(meta["embedder_id"]),
    )

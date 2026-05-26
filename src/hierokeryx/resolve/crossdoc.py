"""Cross-document resolution entry point: embed entities, cluster, register.

Two entry points:

- `resolve_crossdoc` is the batch path — embed and cluster all entities
  from scratch. Used by `hkx resolve` / `hkx pipeline` without `--against`.

- `resolve_incremental` adds new documents against a pre-existing
  `EntityRegistry` and its `RegistryCentroids` sidecar without re-embedding
  the old corpus. New entities are matched greedily (per-entity, type-blocked)
  against existing centroids and any new-cluster centroids built up during
  the run. Borderline matches are batched and adjudicated by the LLM after
  the greedy pass.
"""

from __future__ import annotations

import logging

import numpy as np

from hierokeryx.llm.protocol import LLMClient
from hierokeryx.models import (
    CrossDocCandidate,
    EntityRegistry,
    EntitySchema,
    ExtractionResult,
    make_cluster_id,
)
from hierokeryx.resolve.centroids import RegistryCentroids, update_centroids
from hierokeryx.resolve.cluster import (
    ClusterAssignment,
    apply_assignments,
    cluster_by_type,
)
from hierokeryx.resolve.embed import (
    SentenceTransformerEmbedder,
    encode_extraction_results,
)

logger = logging.getLogger(__name__)


def resolve_crossdoc(
    extraction_results: list[ExtractionResult],
    schema: EntitySchema,
    *,
    llm_client: LLMClient | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    merge_threshold: float = 0.82,
    borderline_threshold: float = 0.75,
) -> tuple[list[ExtractionResult], EntityRegistry]:
    """Cluster entities across documents, tag each with `cluster_id`, and
    return the updated extraction results plus a corpus-level registry.
    """
    if not extraction_results:
        return ([], EntityRegistry(schema_version=schema.fingerprint()))

    embedder = embedder or SentenceTransformerEmbedder()
    entity_index, embeddings = encode_extraction_results(extraction_results, embedder)
    if embeddings.size == 0:
        return (
            list(extraction_results),
            EntityRegistry(schema_version=schema.fingerprint()),
        )

    assignments = cluster_by_type(
        entity_index,
        embeddings,
        schema,
        llm_client=llm_client,
        merge_threshold=merge_threshold,
        borderline_threshold=borderline_threshold,
    )

    updated = apply_assignments(extraction_results, assignments)
    registry = build_registry(updated, schema, assignments=assignments)
    return updated, registry


def build_registry(
    extraction_results: list[ExtractionResult],
    schema: EntitySchema,
    *,
    assignments: dict[str, ClusterAssignment] | None = None,
) -> EntityRegistry:
    """Construct an EntityRegistry from per-doc results that already carry cluster_ids."""
    clusters: dict[str, list[str]] = {}
    canonical_by_cluster: dict[str, str] = {}
    type_by_cluster: dict[str, str] = {}

    for result in extraction_results:
        for entity in result.entities:
            if entity.cluster_id is None:
                continue
            members = clusters.setdefault(entity.cluster_id, [])
            members.append(f"{result.document.id}/{entity.id}")
            # Take the longest surface canonical seen for this cluster
            current = canonical_by_cluster.get(entity.cluster_id, "")
            if len(entity.surface_canonical) > len(current):
                canonical_by_cluster[entity.cluster_id] = entity.surface_canonical
            type_by_cluster[entity.cluster_id] = entity.type

    return EntityRegistry(
        clusters=clusters,
        canonical_by_cluster=canonical_by_cluster,
        type_by_cluster=type_by_cluster,
        schema_version=schema.fingerprint(),
    )


# Incremental ---------------------------------------------------------------


def resolve_incremental(
    new_results: list[ExtractionResult],
    schema: EntitySchema,
    *,
    existing_registry: EntityRegistry,
    existing_centroids: RegistryCentroids,
    llm_client: LLMClient | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    merge_threshold: float = 0.82,
    borderline_threshold: float = 0.75,
) -> tuple[list[ExtractionResult], EntityRegistry, RegistryCentroids]:
    """Resolve new docs against an existing registry without re-embedding it.

    For each new entity (type-blocked, in document/extraction order):

    1. Compute similarity to every existing-cluster centroid of the same type
       and to every new-cluster centroid built up so far in this run.
    2. If the best similarity is at or above `merge_threshold`, assign to
       that cluster (existing wins ties over new). Otherwise:
       - if it's at or above `borderline_threshold` AND `llm_client` is
         provided, hold for LLM tie-break against the nearest existing
         cluster;
       - else create a new singleton cluster and continue (its centroid
         becomes a candidate target for subsequent entities in this run).
    3. After the greedy pass, run `llm_client.resolve_crossdoc` once with
       all borderline candidates and apply the decisions; any candidate
       the LLM rejects stays as a new singleton.

    Returns updated results (with `cluster_id` tagged), a merged registry,
    and the updated centroid sidecar.
    """
    if not new_results:
        return (
            list(new_results),
            existing_registry,
            existing_centroids,
        )
    if existing_centroids.embedder_id and embedder is not None and \
            embedder.model_id != existing_centroids.embedder_id:
        raise ValueError(
            f"embedder mismatch: existing={existing_centroids.embedder_id!r}, "
            f"new={embedder.model_id!r}. Refusing to merge centroids from "
            "different embedding models."
        )

    embedder = embedder or SentenceTransformerEmbedder(
        model_id=existing_centroids.embedder_id or "sentence-transformers/all-MiniLM-L6-v2"
    )
    entity_index, embeddings = encode_extraction_results(new_results, embedder)
    if embeddings.size == 0:
        return list(new_results), existing_registry, existing_centroids

    # Per-type bookkeeping for existing centroids: row indices into the
    # existing_centroids.centroids matrix, filtered to this type.
    existing_rows_by_type: dict[str, list[int]] = {}
    for i, cid in enumerate(existing_centroids.cluster_ids):
        t = existing_registry.type_by_cluster.get(cid)
        if t is None:
            continue
        existing_rows_by_type.setdefault(t, []).append(i)

    # Per-type running state for new clusters created during this run.
    new_cluster_ids_by_type: dict[str, list[str]] = {}
    new_centroids_by_type: dict[str, list[np.ndarray]] = {}
    new_sizes_by_type: dict[str, list[int]] = {}
    # cluster_id -> list of new vectors that landed in it (for sidecar update)
    additions_to_existing: dict[str, list[np.ndarray]] = {}
    new_cluster_vectors: dict[str, list[np.ndarray]] = {}
    # cluster_id -> canonical (for new clusters, to seed registry)
    new_cluster_canonicals: dict[str, str] = {}
    new_cluster_types: dict[str, str] = {}

    assignments: dict[str, ClusterAssignment] = {}
    borderline: list[tuple[int, str, float, float]] = []  # (entity_idx, nearest_existing_cid, top_sim, second_sim)

    for entity_idx, (result, entity) in enumerate(entity_index):
        vec = embeddings[entity_idx]
        t = entity.type

        ex_rows = existing_rows_by_type.get(t, [])
        ex_sims = (
            existing_centroids.centroids[ex_rows] @ vec
            if ex_rows
            else np.empty(0, dtype=np.float32)
        )
        new_centroids = new_centroids_by_type.get(t, [])
        new_sims = (
            np.stack(new_centroids) @ vec
            if new_centroids
            else np.empty(0, dtype=np.float32)
        )

        # Best existing, best new, and the runner-up across both.
        best_ex_pos = int(np.argmax(ex_sims)) if ex_sims.size else -1
        best_ex_sim = float(ex_sims[best_ex_pos]) if best_ex_pos >= 0 else 0.0
        best_new_pos = int(np.argmax(new_sims)) if new_sims.size else -1
        best_new_sim = float(new_sims[best_new_pos]) if best_new_pos >= 0 else 0.0
        all_sims = np.concatenate([ex_sims, new_sims]) if (ex_sims.size or new_sims.size) else np.empty(0)
        second_sim = float(np.sort(all_sims)[-2]) if all_sims.size >= 2 else 0.0

        nearest_existing_cid: str | None = (
            existing_centroids.cluster_ids[ex_rows[best_ex_pos]] if best_ex_pos >= 0 else None
        )

        # Decision: existing wins ties over new.
        if best_ex_sim >= merge_threshold and best_ex_sim >= best_new_sim:
            assert nearest_existing_cid is not None
            existing_cid: str = nearest_existing_cid
            assignments[entity.id] = ClusterAssignment(
                cluster_id=existing_cid,
                top_similarity=best_ex_sim,
                second_similarity=second_sim,
                source="threshold",
            )
            additions_to_existing.setdefault(existing_cid, []).append(vec)
        elif best_new_sim >= merge_threshold:
            assert best_new_pos >= 0
            cid = new_cluster_ids_by_type[t][best_new_pos]
            assignments[entity.id] = ClusterAssignment(
                cluster_id=cid,
                top_similarity=best_new_sim,
                second_similarity=second_sim,
                source="threshold",
            )
            new_cluster_vectors[cid].append(vec)
            _update_running_centroid(new_centroids_by_type[t], new_sizes_by_type[t], best_new_pos, vec)
        elif (
            llm_client is not None
            and nearest_existing_cid is not None
            and best_ex_sim >= borderline_threshold
        ):
            # Defer: defaults to new singleton, may be reassigned after tie-break.
            cid = _register_new_cluster(
                entity.canonical,
                entity.type,
                result.document.id,
                vec,
                new_cluster_ids_by_type,
                new_centroids_by_type,
                new_sizes_by_type,
                new_cluster_vectors,
                new_cluster_canonicals,
                new_cluster_types,
            )
            assignments[entity.id] = ClusterAssignment(
                cluster_id=cid,
                top_similarity=best_ex_sim,
                second_similarity=second_sim,
                source="singleton",
            )
            borderline.append((entity_idx, nearest_existing_cid, best_ex_sim, second_sim))
        else:
            cid = _register_new_cluster(
                entity.canonical,
                entity.type,
                result.document.id,
                vec,
                new_cluster_ids_by_type,
                new_centroids_by_type,
                new_sizes_by_type,
                new_cluster_vectors,
                new_cluster_canonicals,
                new_cluster_types,
            )
            assignments[entity.id] = ClusterAssignment(
                cluster_id=cid,
                top_similarity=1.0,
                second_similarity=second_sim,
                source="singleton",
            )

    # LLM tie-break on borderline candidates -------------------------------
    if borderline and llm_client is not None:
        candidates: list[CrossDocCandidate] = []
        for entity_idx, nearest_cid, top_sim, _second in borderline:
            result, entity = entity_index[entity_idx]
            contexts: list[str] = []
            for m in entity.mentions[:3]:
                a = max(0, m.span.start - 60)
                b = min(len(result.document.text), m.span.end + 60)
                contexts.append(result.document.text[a:b].replace("\n", " ").strip())
            candidates.append(
                CrossDocCandidate(
                    entity_id=entity.id,
                    doc_id=entity.doc_id,
                    type=entity.type,
                    canonical=entity.canonical,
                    contexts=contexts,
                    nearest_cluster_id=nearest_cid,
                    nearest_similarity=top_sim,
                )
            )
        decisions = {d.entity_id: d for d in llm_client.resolve_crossdoc(candidates, schema)}

        for entity_idx, nearest_cid, top_sim, second_sim in borderline:
            _, entity = entity_index[entity_idx]
            decision = decisions.get(entity.id)
            if decision is None or decision.target_cluster_id is None:
                continue  # keep the singleton placeholder
            if decision.target_cluster_id != nearest_cid:
                # LLM picked an unfamiliar cluster id; reject defensively.
                logger.warning(
                    "LLM proposed unknown target cluster %r for entity %s; keeping singleton.",
                    decision.target_cluster_id, entity.id,
                )
                continue
            # Reassign from the placeholder singleton to the existing cluster.
            placeholder_cid = assignments[entity.id].cluster_id
            assignments[entity.id] = ClusterAssignment(
                cluster_id=nearest_cid,
                top_similarity=top_sim,
                second_similarity=second_sim,
                source="llm_tiebreak",
            )
            additions_to_existing.setdefault(nearest_cid, []).append(embeddings[entity_idx])
            _drop_singleton(
                placeholder_cid,
                new_cluster_ids_by_type,
                new_centroids_by_type,
                new_sizes_by_type,
                new_cluster_vectors,
                new_cluster_canonicals,
                new_cluster_types,
            )

    # Apply, build registry & centroids -------------------------------------
    updated_results = apply_assignments(new_results, assignments)
    merged_registry = _merge_registry(
        existing_registry,
        updated_results,
        new_cluster_canonicals=new_cluster_canonicals,
        new_cluster_types=new_cluster_types,
        schema=schema,
    )
    merged_centroids = update_centroids(
        existing_centroids,
        additions={
            cid: np.stack(vecs).astype(np.float32) for cid, vecs in additions_to_existing.items()
        },
        new_clusters={
            cid: np.stack(vecs).astype(np.float32) for cid, vecs in new_cluster_vectors.items()
        },
        embedder_id=embedder.model_id,
    )
    return updated_results, merged_registry, merged_centroids


def _register_new_cluster(
    canonical: str,
    type_name: str,
    doc_id: str,
    vec: np.ndarray,
    ids_by_type: dict[str, list[str]],
    centroids_by_type: dict[str, list[np.ndarray]],
    sizes_by_type: dict[str, list[int]],
    vectors_by_cluster: dict[str, list[np.ndarray]],
    canonicals: dict[str, str],
    types: dict[str, str],
) -> str:
    cid = make_cluster_id(canonical, type_name, [doc_id])
    # Disambiguate against a same-id new cluster created earlier this run.
    suffix = 0
    base = cid
    while cid in vectors_by_cluster:
        suffix += 1
        cid = f"{base}_{suffix}"
    ids_by_type.setdefault(type_name, []).append(cid)
    centroids_by_type.setdefault(type_name, []).append(vec.astype(np.float32))
    sizes_by_type.setdefault(type_name, []).append(1)
    vectors_by_cluster[cid] = [vec]
    canonicals[cid] = canonical
    types[cid] = type_name
    return cid


def _update_running_centroid(
    centroids: list[np.ndarray],
    sizes: list[int],
    pos: int,
    new_vec: np.ndarray,
) -> None:
    old_n = sizes[pos]
    merged = (centroids[pos] * old_n + new_vec) / (old_n + 1)
    norm = float(np.linalg.norm(merged))
    centroids[pos] = (merged / max(norm, 1e-9)).astype(np.float32)
    sizes[pos] = old_n + 1


def _drop_singleton(
    cid: str,
    ids_by_type: dict[str, list[str]],
    centroids_by_type: dict[str, list[np.ndarray]],
    sizes_by_type: dict[str, list[int]],
    vectors_by_cluster: dict[str, list[np.ndarray]],
    canonicals: dict[str, str],
    types: dict[str, str],
) -> None:
    type_name = types.get(cid)
    if type_name is None:
        return
    ids = ids_by_type.get(type_name, [])
    if cid not in ids:
        return
    pos = ids.index(cid)
    ids.pop(pos)
    centroids_by_type[type_name].pop(pos)
    sizes_by_type[type_name].pop(pos)
    vectors_by_cluster.pop(cid, None)
    canonicals.pop(cid, None)
    types.pop(cid, None)


def _merge_registry(
    existing: EntityRegistry,
    updated_results: list[ExtractionResult],
    *,
    new_cluster_canonicals: dict[str, str],
    new_cluster_types: dict[str, str],
    schema: EntitySchema,
) -> EntityRegistry:
    clusters = {cid: list(members) for cid, members in existing.clusters.items()}
    canonical_by_cluster = dict(existing.canonical_by_cluster)
    type_by_cluster = dict(existing.type_by_cluster)

    for result in updated_results:
        for entity in result.entities:
            cid = entity.cluster_id
            if cid is None:
                continue
            members = clusters.setdefault(cid, [])
            members.append(f"{result.document.id}/{entity.id}")
            current = canonical_by_cluster.get(cid, "")
            if len(entity.surface_canonical) > len(current):
                canonical_by_cluster[cid] = entity.surface_canonical
            type_by_cluster[cid] = entity.type

    # Make sure new clusters have entries even if their canonical/type wasn't
    # overwritten by the loop above.
    for cid, canonical in new_cluster_canonicals.items():
        canonical_by_cluster.setdefault(cid, canonical)
    for cid, type_name in new_cluster_types.items():
        type_by_cluster.setdefault(cid, type_name)

    return EntityRegistry(
        clusters=clusters,
        canonical_by_cluster=canonical_by_cluster,
        type_by_cluster=type_by_cluster,
        schema_version=schema.fingerprint(),
    )

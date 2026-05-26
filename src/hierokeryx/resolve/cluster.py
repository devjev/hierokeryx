"""Threshold-based union-find clustering of L2-normalized entity embeddings,
blocked by entity type.

For each type-block:
  1. Compute pairwise cosine similarity over all entities in the block.
  2. Union any pair whose similarity is at or above `merge_threshold`.
  3. Singletons (no neighbour above threshold) are tagged as noise (-1) and
     handled by step 4.
  4. For each noise entity, find the nearest cluster centroid. If the
     similarity is at or above `borderline_threshold` AND an LLMClient was
     supplied, emit a CrossDocCandidate for tie-break. Otherwise the entity
     becomes its own singleton cluster.

HDBSCAN was an obvious candidate but its small-block behaviour is brittle
(three near-identical points get labelled as noise with `eom` selection;
two dissimilar points get spuriously merged with `allow_single_cluster=True`).
A simple cosine threshold over L2-normalised vectors is more predictable for
entity-resolution-style clusters that are tight and well-separated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from hierokeryx.confidence import crossdoc_confidence
from hierokeryx.llm.protocol import LLMClient
from hierokeryx.models import (
    CrossDocCandidate,
    Entity,
    EntitySchema,
    ExtractionResult,
    make_cluster_id,
)

logger = logging.getLogger(__name__)


@dataclass
class ClusterAssignment:
    """The cluster a given entity ended up in, plus the supporting similarity."""

    cluster_id: str
    top_similarity: float = 1.0
    second_similarity: float = 0.0
    source: str = "hdbscan"  # one of: "hdbscan", "llm_tiebreak", "singleton"


def cluster_by_type(
    entity_index: list[tuple[ExtractionResult, Entity]],
    embeddings: np.ndarray,
    schema: EntitySchema,
    *,
    llm_client: LLMClient | None = None,
    merge_threshold: float = 0.82,
    borderline_threshold: float = 0.75,
) -> dict[str, ClusterAssignment]:
    """Cluster entities across documents, blocked by entity type.

    Returns a mapping from entity_id -> ClusterAssignment. Every entity in
    `entity_index` is assigned to exactly one cluster (possibly a singleton).
    """
    assignments: dict[str, ClusterAssignment] = {}

    indices_by_type: dict[str, list[int]] = {}
    for idx, (_, entity) in enumerate(entity_index):
        indices_by_type.setdefault(entity.type, []).append(idx)

    for type_name, indices in indices_by_type.items():
        block_entities = [entity_index[i] for i in indices]
        block_embeddings = embeddings[indices]
        block_assignments = _cluster_block(
            type_name=type_name,
            block_entities=block_entities,
            block_embeddings=block_embeddings,
            schema=schema,
            llm_client=llm_client,
            merge_threshold=merge_threshold,
            borderline_threshold=borderline_threshold,
        )
        assignments.update(block_assignments)

    return assignments


def _cluster_block(
    *,
    type_name: str,
    block_entities: list[tuple[ExtractionResult, Entity]],
    block_embeddings: np.ndarray,
    schema: EntitySchema,
    llm_client: LLMClient | None,
    merge_threshold: float,
    borderline_threshold: float,
) -> dict[str, ClusterAssignment]:
    n = len(block_entities)
    assignments: dict[str, ClusterAssignment] = {}

    if n == 1:
        _, entity = block_entities[0]
        assignments[entity.id] = _singleton(entity, type_name)
        return assignments

    labels = _threshold_cluster(block_embeddings, threshold=merge_threshold)

    cluster_label_to_id: dict[int, str] = {}
    unique_labels = sorted({lbl for lbl in labels if lbl != -1})
    for lbl in unique_labels:
        member_indices = [i for i, x in enumerate(labels) if x == lbl]
        member_entities = [block_entities[i][1] for i in member_indices]
        member_doc_ids = [block_entities[i][0].document.id for i in member_indices]
        canonical_choice = _pick_cluster_canonical(member_entities)
        cluster_id = make_cluster_id(canonical_choice, type_name, member_doc_ids)
        cluster_label_to_id[lbl] = cluster_id

    # First pass: assign non-noise entities to their HDBSCAN clusters.
    for i, (_, entity) in enumerate(block_entities):
        if labels[i] == -1:
            continue
        cluster_id = cluster_label_to_id[labels[i]]
        top_sim, second_sim = _similarity_to_cluster(
            block_embeddings, i, labels, target_label=labels[i]
        )
        assignments[entity.id] = ClusterAssignment(
            cluster_id=cluster_id,
            top_similarity=top_sim,
            second_similarity=second_sim,
            source="hdbscan",
        )

    # Second pass: noise entities — borderline tie-break, otherwise singleton.
    noise_indices = [i for i, lbl in enumerate(labels) if lbl == -1]
    borderline: list[tuple[int, str, float, float]] = []  # (block_idx, cluster_id, top, second)
    for i in noise_indices:
        if not cluster_label_to_id:
            continue
        nearest_cluster_label, top_sim, second_sim = _nearest_cluster(
            block_embeddings, i, labels, cluster_label_to_id
        )
        if nearest_cluster_label is None:
            continue
        if top_sim >= borderline_threshold and llm_client is not None:
            borderline.append(
                (i, cluster_label_to_id[nearest_cluster_label], top_sim, second_sim)
            )

    if borderline and llm_client is not None:
        decisions_by_entity = _run_llm_tiebreak(
            borderline_indices=borderline,
            block_entities=block_entities,
            llm_client=llm_client,
            schema=schema,
        )
        for i, _target_cluster_id, top_sim, second_sim in borderline:
            _, entity = block_entities[i]
            decision = decisions_by_entity.get(entity.id)
            if decision is not None and decision.target_cluster_id is not None:
                assignments[entity.id] = ClusterAssignment(
                    cluster_id=decision.target_cluster_id,
                    top_similarity=top_sim,
                    second_similarity=second_sim,
                    source="llm_tiebreak",
                )

    # Remaining noise entities become singletons.
    for i in noise_indices:
        _, entity = block_entities[i]
        if entity.id not in assignments:
            assignments[entity.id] = _singleton(entity, type_name)

    return assignments


def _singleton(entity: Entity, type_name: str) -> ClusterAssignment:
    cluster_id = make_cluster_id(entity.canonical, type_name, [entity.doc_id])
    return ClusterAssignment(
        cluster_id=cluster_id,
        top_similarity=1.0,
        second_similarity=0.0,
        source="singleton",
    )


def _threshold_cluster(embeddings: np.ndarray, *, threshold: float) -> np.ndarray:
    """Greedy union-find: merge any pair with cosine >= threshold.

    Inputs are assumed L2-normalised, so cosine similarity equals the dot product.
    Returns one label per row. Singletons (no neighbour above threshold) get -1.
    """
    n = len(embeddings)
    if n == 0:
        return np.empty(0, dtype=int)
    if n == 1:
        return np.array([-1], dtype=int)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    similarities = embeddings @ embeddings.T
    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] >= threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1

    unique_roots = sorted({r for r in roots if sizes[r] >= 2})
    root_to_label = {r: idx for idx, r in enumerate(unique_roots)}
    labels = np.array(
        [root_to_label[r] if sizes[r] >= 2 else -1 for r in roots], dtype=int
    )
    return labels


def _similarity_to_cluster(
    embeddings: np.ndarray,
    idx: int,
    labels: np.ndarray,
    *,
    target_label: int,
) -> tuple[float, float]:
    """Cosine similarity to the centroid of the target cluster and the
    next-best non-target cluster. Inputs are assumed L2-normalised.
    """
    vec = embeddings[idx]
    cluster_indices = [i for i, lbl in enumerate(labels) if lbl == target_label and i != idx]
    if not cluster_indices:
        return (1.0, 0.0)
    target_centroid = embeddings[cluster_indices].mean(axis=0)
    target_centroid /= max(np.linalg.norm(target_centroid), 1e-9)
    top_sim = float(np.dot(vec, target_centroid))

    other_labels = {lbl for lbl in labels if lbl not in (target_label, -1)}
    second_sim = 0.0
    for lbl in other_labels:
        member_indices = [i for i, x in enumerate(labels) if x == lbl]
        centroid = embeddings[member_indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        sim = float(np.dot(vec, centroid))
        second_sim = max(second_sim, sim)

    return (top_sim, second_sim)


def _nearest_cluster(
    embeddings: np.ndarray,
    idx: int,
    labels: np.ndarray,
    cluster_label_to_id: dict[int, str],
) -> tuple[int | None, float, float]:
    """For a noise entity, find the nearest cluster centroid."""
    vec = embeddings[idx]
    sims: list[tuple[int, float]] = []
    for lbl in cluster_label_to_id:
        member_indices = [i for i, x in enumerate(labels) if x == lbl]
        centroid = embeddings[member_indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        sims.append((lbl, float(np.dot(vec, centroid))))
    if not sims:
        return (None, 0.0, 0.0)
    sims.sort(key=lambda x: -x[1])
    top_label, top_sim = sims[0]
    second_sim = sims[1][1] if len(sims) > 1 else 0.0
    return (top_label, top_sim, second_sim)


def _pick_cluster_canonical(entities: list[Entity]) -> str:
    """Choose a canonical name for an HDBSCAN cluster: longest surface_canonical."""
    return max(entities, key=lambda e: len(e.surface_canonical)).surface_canonical


def _run_llm_tiebreak(
    *,
    borderline_indices: list[tuple[int, str, float, float]],
    block_entities: list[tuple[ExtractionResult, Entity]],
    llm_client: LLMClient,
    schema: EntitySchema,
) -> dict[str, object]:
    """Ask the LLM to confirm/reject each borderline merge."""
    candidates: list[CrossDocCandidate] = []
    for i, nearest_cluster_id, top_sim, _second_sim in borderline_indices:
        result, entity = block_entities[i]
        contexts = []
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
                nearest_cluster_id=nearest_cluster_id,
                nearest_similarity=top_sim,
            )
        )
    decisions = llm_client.resolve_crossdoc(candidates, schema)
    return {d.entity_id: d for d in decisions}


def apply_assignments(
    extraction_results: list[ExtractionResult],
    assignments: dict[str, ClusterAssignment],
    *,
    llm_decision_confidence: float = 0.9,
) -> list[ExtractionResult]:
    """Return new ExtractionResults with each Entity tagged with its cluster_id
    and confidence updated to incorporate the cross-doc margin.
    """
    new_results: list[ExtractionResult] = []
    for result in extraction_results:
        new_entities = []
        for entity in result.entities:
            assignment = assignments.get(entity.id)
            if assignment is None:
                new_entities.append(entity)
                continue
            new_conf = crossdoc_confidence(
                base_confidence=entity.confidence,
                llm_decision_confidence=(
                    llm_decision_confidence if assignment.source == "llm_tiebreak" else 1.0
                ),
                top_similarity=assignment.top_similarity,
                second_similarity=assignment.second_similarity,
            )
            new_entities.append(
                entity.model_copy(
                    update={"cluster_id": assignment.cluster_id, "confidence": new_conf}
                )
            )
        new_results.append(result.model_copy(update={"entities": new_entities}))
    return new_results

"""Cross-document resolution entry point: embed entities, cluster, register."""

from __future__ import annotations

import logging

from hierokeryx.llm.protocol import LLMClient
from hierokeryx.models import EntityRegistry, EntitySchema, ExtractionResult
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

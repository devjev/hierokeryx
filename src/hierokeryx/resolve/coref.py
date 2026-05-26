"""Within-document coreference: cluster GLiNER mentions into Entities via the LLM."""

from __future__ import annotations

import logging

from hierokeryx.confidence import within_doc_confidence
from hierokeryx.llm.protocol import LLMClient
from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    Mention,
    make_entity_id,
)

logger = logging.getLogger(__name__)


def resolve_within_doc(
    document: Document,
    mentions: list[Mention],
    schema: EntitySchema,
    llm_client: LLMClient,
) -> list[Entity]:
    """Cluster `mentions` from `document` into within-doc Entities.

    Returns an empty list when `mentions` is empty. Entities are sorted by the
    character position of their earliest mention so the output is stable.
    """
    if not mentions:
        return []

    clusters = llm_client.cluster_mentions(document, mentions, schema)
    mentions_by_id = {m.id: m for m in mentions}

    entities: list[Entity] = []
    for cluster in clusters:
        cluster_mentions = sorted(
            (mentions_by_id[mid] for mid in cluster.mention_ids),
            key=lambda m: (m.span.start, m.span.end),
        )
        # surface_canonical: longest verbatim mention in the cluster — used in
        # HITL display so reviewers always see what's actually in the text,
        # regardless of how the LLM normalised the `canonical` field.
        surface_canonical = max(cluster_mentions, key=lambda m: len(m.span.text)).span.text

        entity = Entity(
            id=make_entity_id(document.id, [m.id for m in cluster_mentions]),
            type=cluster.type,
            canonical=cluster.canonical,
            surface_canonical=surface_canonical,
            mentions=cluster_mentions,
            confidence=within_doc_confidence(cluster_mentions, cluster.confidence),
            doc_id=document.id,
            cluster_id=None,
        )
        entities.append(entity)

    return sorted(entities, key=lambda e: (e.mentions[0].span.start, e.id))

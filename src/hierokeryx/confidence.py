"""Confidence ensembling and human-in-the-loop routing.

Two scoring contexts:

- **Within-doc coref**: combine the mention-score mean (GLiNER) with the LLM's
  self-reported cluster confidence. No cross-doc margin is available yet, so
  the formula is a 50/50 ensemble.

- **Cross-doc refinement**: blend the within-doc confidence, the LLM's merge
  decision confidence, and the embedding margin between the top and second-best
  candidate clusters. A small margin signals a borderline merge that should
  carry less confidence.

The routing step in this module turns those confidences into a `ReviewItem`
queue, with a reason tag the JSONL exporter exposes to the human reviewer.
"""

from __future__ import annotations

from hierokeryx.models import Entity, Mention, ReviewItem, ReviewReason

WITHIN_DOC_MENTION_WEIGHT = 0.5
WITHIN_DOC_LLM_WEIGHT = 0.5

CROSSDOC_BASE_WEIGHT = 0.4
CROSSDOC_LLM_WEIGHT = 0.4
CROSSDOC_MARGIN_WEIGHT = 0.2


def within_doc_confidence(
    mentions: list[Mention],
    llm_cluster_confidence: float,
) -> float:
    """Ensemble within-doc confidence: mean mention score and LLM self-report."""
    if not mentions:
        raise ValueError("within_doc_confidence: mentions must be non-empty")
    mean_score = sum(m.score for m in mentions) / len(mentions)
    score = (
        WITHIN_DOC_MENTION_WEIGHT * mean_score
        + WITHIN_DOC_LLM_WEIGHT * _clamp(llm_cluster_confidence)
    )
    return _clamp(score)


def crossdoc_confidence(
    base_confidence: float,
    llm_decision_confidence: float,
    top_similarity: float,
    second_similarity: float = 0.0,
) -> float:
    """Refine confidence after cross-doc cluster assignment.

    `base_confidence` is the within-doc confidence carried over.
    `top_similarity` is the cosine distance to the assigned cluster.
    `second_similarity` is the cosine distance to the next-best cluster (or 0
    if there is no rival).
    """
    margin = max(0.0, top_similarity - second_similarity)
    score = (
        CROSSDOC_BASE_WEIGHT * _clamp(base_confidence)
        + CROSSDOC_LLM_WEIGHT * _clamp(llm_decision_confidence)
        + CROSSDOC_MARGIN_WEIGHT * _clamp(margin)
    )
    return _clamp(score)


def route_for_review(
    entities: list[Entity],
    *,
    cluster_threshold: float = 0.7,
    span_threshold: float = 0.5,
) -> list[ReviewItem]:
    """Flag entities for HITL review. One ReviewItem per flagged entity.

    Reason precedence (highest first):
    - `ambiguous_merge` — entity was cross-doc-merged with low confidence
    - `low_cluster_conf` — within-doc entity has low overall confidence
    - `low_span_conf` — at least one mention scored below `span_threshold`
    """
    items: list[ReviewItem] = []
    for entity in entities:
        reason = _classify(
            entity,
            cluster_threshold=cluster_threshold,
            span_threshold=span_threshold,
        )
        if reason is None:
            continue
        items.append(
            ReviewItem(
                doc_id=entity.doc_id,
                entity_id=entity.id,
                reason=reason,
                current=entity,
            )
        )
    return items


def _classify(
    entity: Entity,
    *,
    cluster_threshold: float,
    span_threshold: float,
) -> ReviewReason | None:
    if entity.confidence < cluster_threshold:
        # If a cross-doc cluster has been assigned the low confidence reflects
        # an uncertain merge; otherwise it's a within-doc coref problem.
        return "ambiguous_merge" if entity.cluster_id is not None else "low_cluster_conf"
    min_mention_score = min(m.score for m in entity.mentions)
    if min_mention_score < span_threshold:
        return "low_span_conf"
    return None


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))

"""LLMClient Protocol — minimal interface that resolvers depend on.

Implementations need only `cluster_mentions` (within-doc coref) and
`resolve_crossdoc` (cross-doc tie-break). Anything else stays inside the
implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    Mention,
    MergeDecision,
)


class LLMError(RuntimeError):
    """Raised when an LLM call fails after all retries."""


@runtime_checkable
class LLMClient(Protocol):
    """Abstract entity-resolution LLM client.

    Any class implementing both methods below can be passed to
    [`pipeline.run`][hierokeryx.pipeline.run] or
    [`resolve_within_doc`][hierokeryx.resolve.coref.resolve_within_doc] as a
    drop-in replacement for the default Anthropic backend. See
    [`AnthropicClient`][hierokeryx.llm.anthropic_client.AnthropicClient]
    for the reference implementation.
    """

    def cluster_mentions(
        self,
        document: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> list[CorefCluster]:
        """Group within-document mentions into coreferent clusters.

        Args:
            document: The source document the mentions came from. The full text
                is passed so the model can resolve pronouns and short forms.
            mentions: Extracted mentions, typed and span-aligned. Every mention
                id MUST appear in exactly one returned cluster.
            schema: User-defined entity schema. Cluster types must come from
                `schema.types`.

        Returns:
            One [`CorefCluster`][hierokeryx.models.CorefCluster] per real-world
            entity referred to in the document. The implementation is expected
            to set a calibrated `confidence` so low-confidence clusters can be
            routed to human review.

        Raises:
            LLMError: when the underlying provider call fails after retries,
                or returns a payload that cannot be coerced into the schema.
        """
        ...

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        """Adjudicate borderline cross-document merges.

        Called only for entities whose embedding similarity to their nearest
        existing cluster sits in the "borderline" band — clear matches and
        clear mismatches are handled by thresholds without the LLM.

        Args:
            candidates: One entry per borderline entity, with a few contexts
                from the source document and the id of the nearest existing
                cluster by embedding similarity.
            schema: User-defined entity schema.

        Returns:
            One [`MergeDecision`][hierokeryx.models.MergeDecision] per
            candidate. `target_cluster_id` must be either an id present in
            the input or `None` (keep as new singleton) — never an invented id.

        Raises:
            LLMError: when the underlying provider call fails after retries.
        """
        ...

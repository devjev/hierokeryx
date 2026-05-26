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
    """Abstract entity-resolution LLM client."""

    def cluster_mentions(
        self,
        document: Document,
        mentions: list[Mention],
        schema: EntitySchema,
    ) -> list[CorefCluster]:
        """Group `mentions` from `document` into within-doc coreferent clusters."""
        ...

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        """For each candidate, decide whether to merge into its nearest cluster
        or keep as a new singleton.
        """
        ...

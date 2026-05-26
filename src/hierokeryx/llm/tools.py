"""JSON-schema tool definitions for Anthropic tool-use.

These are hand-written rather than derived from Pydantic so we control the
schema's user-facing description text, which the model reads.
"""

from __future__ import annotations

from typing import Any

RECORD_CLUSTERS_TOOL: dict[str, Any] = {
    "name": "record_clusters",
    "description": (
        "Record your final within-document coreference clustering. Call this "
        "tool exactly once after analyzing all provided mentions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "description": (
                    "One entry per real-world entity. Every input mention id "
                    "must appear in exactly one cluster's mention_ids list."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "mention_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": (
                                "Mention ids that all refer to the same entity."
                            ),
                        },
                        "canonical": {
                            "type": "string",
                            "description": (
                                "Canonical name for the entity. Prefer the most "
                                "complete, unambiguous surface form that appears "
                                "verbatim in the document."
                            ),
                        },
                        "type": {
                            "type": "string",
                            "description": (
                                "Entity type — must match a name declared in the "
                                "schema you were given."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": (
                                "Your confidence (0..1) that these mentions truly "
                                "co-refer. Lower confidence triggers human review."
                            ),
                        },
                        "rationale": {
                            "type": "string",
                            "description": (
                                "One short sentence explaining the grouping decision."
                            ),
                        },
                    },
                    "required": ["mention_ids", "canonical", "type", "confidence"],
                },
            },
        },
        "required": ["clusters"],
    },
}


RECORD_MERGE_DECISIONS_TOOL: dict[str, Any] = {
    "name": "record_merge_decisions",
    "description": (
        "Record your cross-document merge decisions. Call this tool exactly "
        "once with one decision per candidate."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "string",
                            "description": "The candidate entity id from the input.",
                        },
                        "target_cluster_id": {
                            "type": ["string", "null"],
                            "description": (
                                "Cluster id to merge into, or null to keep as a "
                                "new singleton cluster. Must be an exact id from "
                                "the input or null — do not invent ids."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "rationale": {
                            "type": "string",
                            "description": "One short sentence justifying the decision.",
                        },
                    },
                    "required": ["entity_id", "target_cluster_id", "confidence"],
                },
            },
        },
        "required": ["decisions"],
    },
}

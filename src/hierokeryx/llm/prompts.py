"""Stable prompt prefixes. Everything in this module is part of the prompt
cache prefix — kept invariant across many per-document calls.
"""

from __future__ import annotations

import json

from hierokeryx.models import EntitySchema

COREF_SYSTEM_PROMPT = """\
You are an entity-resolution assistant. Given a document and a list of entity
mentions extracted from it, you must group those mentions into entities — one
cluster per real-world entity referred to in the document.

Hard rules:
- Group mentions ONLY based on what the document text says. Do not use any
  external knowledge to merge or split.
- Every input mention id must end up in exactly one cluster.
- Each cluster gets exactly one type, drawn from the schema you were given.
- Pick a canonical form that appears verbatim in the document where possible;
  prefer the most complete, unambiguous surface form (e.g. "Marie Curie" over
  "she" or just "Curie").
- A pronoun, short form, or definite description belongs in the same cluster
  as its referent only if the text unambiguously supports the link. When in
  doubt, leave it as its own cluster.
- Set confidence low when you are not sure. Low-confidence clusters will be
  routed to a human for review — that is the right outcome, do not inflate.

Call the `record_clusters` tool exactly once with your final clustering.
"""


CROSSDOC_SYSTEM_PROMPT = """\
You are an entity-resolution assistant deciding which entities across multiple
documents refer to the same real-world entity.

You will receive candidate entities, each with:
- entity_id, doc_id, type, canonical
- contexts: short verbatim excerpts where the entity is mentioned
- nearest_cluster_id and nearest_similarity: the most similar already-known
  cluster, by embedding similarity (0..1)

For each candidate, decide:
- Return nearest_cluster_id (or any other cluster id from the input) to merge
  the candidate into that cluster, OR
- Return null to keep the candidate as a new singleton cluster.

Hard rules:
- Merge only when the contexts unambiguously support the same real-world entity.
- Different people who happen to share a name are NOT the same entity.
- Do not invent cluster ids. Use only ids that appear in the input, or null.
- Use a low confidence for ambiguous merges so they are reviewed by a human.

Call the `record_merge_decisions` tool exactly once with one decision per
candidate.
"""


def render_schema_block(schema: EntitySchema) -> str:
    """Render the schema as a stable JSON block for cache key consistency."""
    payload = {
        "version": schema.version,
        "types": [
            {
                "name": t.name,
                "description": t.description,
                "examples": list(t.examples),
            }
            for t in schema.types
        ],
    }
    return (
        "<entity_schema>\n"
        + json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n</entity_schema>"
    )

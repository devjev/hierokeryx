"""End-to-end smoke test: real GLiNER + fake LLM + real pipeline + JSONL HITL.

The Anthropic API is stubbed out so this test can run offline. A second test,
gated on ANTHROPIC_API_KEY, exercises the real Anthropic path when available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from hierokeryx import pipeline
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.llm.anthropic_client import AnthropicClient
from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    Mention,
    MergeDecision,
)
from hierokeryx.schema import load_schema

FIXTURES = Path(__file__).parent.parent / "fixtures"


class CanonicalGroupingLLM:
    """Fake LLM that clusters mentions sharing case-insensitive canonical text.

    Groups any mention whose text starts with the same word — coarse but
    sufficient for the smoke test, since we want to verify *that the pipeline
    wires the LLM into coref*, not the LLM's quality.
    """

    coref_model = "fake-canonical-coref"
    tiebreak_model = "fake-tiebreak"

    def cluster_mentions(
        self, document: Document, mentions: list[Mention], schema: EntitySchema
    ) -> list[CorefCluster]:
        if not mentions:
            return []
        groups: dict[str, list[Mention]] = {}
        for m in mentions:
            key = (m.type, m.span.text.lower().split()[0] if m.span.text else "")
            groups.setdefault(f"{key[0]}|{key[1]}", []).append(m)
        clusters: list[CorefCluster] = []
        for key, ms in groups.items():
            type_name, _word = key.split("|", 1)
            ms.sort(key=lambda m: (-len(m.span.text), m.span.start))
            canonical = ms[0].span.text
            clusters.append(
                CorefCluster(
                    mention_ids=[m.id for m in ms],
                    canonical=canonical,
                    type=type_name,
                    confidence=0.85,
                )
            )
        return clusters

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        return [
            MergeDecision(
                entity_id=c.entity_id,
                target_cluster_id=c.nearest_cluster_id,
                confidence=0.8,
            )
            for c in candidates
        ]


class HashedEmbedder:
    """Deterministic L2-normalised embedder that produces nearby vectors for
    strings sharing the same first word. Lets us test cross-doc clustering
    without downloading sentence-transformers weights.
    """

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for t in texts:
            first_word = (t.strip().split()[0] if t.strip() else "_").lower()
            base = np.zeros(32, dtype=np.float32)
            for i, c in enumerate(first_word.encode("utf-8")[:32]):
                base[i] = (c % 13) / 5.0
            v = base / max(np.linalg.norm(base), 1e-9)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture(scope="module")
def small_gliner() -> GLiNERExtractor:
    return GLiNERExtractor(
        model_id=os.environ.get("HIEROKERYX_TEST_GLINER_MODEL", "urchade/gliner_small-v2.1"),
        threshold=0.3,
    )


@pytest.mark.integration
def test_end_to_end_with_fake_llm(tmp_path: Path, small_gliner: GLiNERExtractor) -> None:
    schema = load_schema(FIXTURES / "schemas" / "people_orgs.yaml")
    docs = [
        Document(
            id=p.stem,
            text=p.read_text(encoding="utf-8"),
            source=str(p),
        )
        for p in sorted((FIXTURES / "docs").glob("*.txt"))
    ]
    assert docs, "expected fixture documents to exist"

    workdir = tmp_path / "workdir"
    run = pipeline.run(
        documents=docs,
        schema=schema,
        workdir=workdir,
        extractor=small_gliner,
        llm_client=CanonicalGroupingLLM(),
        embedder=HashedEmbedder(),
        review_threshold=0.7,
        only_flagged_review=True,
    )

    # Workdir artifacts present
    assert (workdir / "schema.yaml").exists()
    assert (workdir / "registry.json").exists()
    assert (workdir / "manifest.json").exists()
    assert (workdir / "extractions").is_dir()
    extraction_files = sorted((workdir / "extractions").glob("*.json"))
    assert len(extraction_files) == len(docs)

    # The load-bearing invariant — each entity span quotes the document.
    for r in run.extraction_results:
        for e in r.entities:
            for m in e.mentions:
                actual = r.document.text[m.span.start : m.span.end]
                assert actual == m.span.text, (
                    f"span mismatch in {r.document.id}/{e.id}: "
                    f"stored {m.span.text!r} != doc {actual!r}"
                )

    # Cross-doc clustering should detect the Curie alias across the two Curie docs.
    if "doc_curie" in [r.document.id for r in run.extraction_results] and \
       "doc_curie_short" in [r.document.id for r in run.extraction_results]:
        # Both docs should contribute to at least one shared cluster
        cluster_doc_counts = {
            cid: {ref.split("/", 1)[0] for ref in refs}
            for cid, refs in run.registry.clusters.items()
        }
        multi_doc_clusters = [c for c, docs_in in cluster_doc_counts.items() if len(docs_in) >= 2]
        assert multi_doc_clusters, (
            "expected at least one cluster spanning multiple docs given the "
            f"shared 'Curie' alias; got registry={run.registry.clusters}"
        )


@pytest.mark.integration
def test_pipeline_roundtrip_through_review(
    tmp_path: Path, small_gliner: GLiNERExtractor
) -> None:
    schema = load_schema(FIXTURES / "schemas" / "people_orgs.yaml")
    doc = Document(
        id="curie",
        text=(FIXTURES / "docs" / "doc_curie_short.txt").read_text(encoding="utf-8"),
    )

    workdir = tmp_path / "workdir"
    run = pipeline.run(
        documents=[doc],
        schema=schema,
        workdir=workdir,
        extractor=small_gliner,
        llm_client=CanonicalGroupingLLM(),
        embedder=HashedEmbedder(),
        review_threshold=1.0,  # flag every entity so we have review files
        only_flagged_review=False,
    )

    review_files = sorted((workdir / "review").glob("*.jsonl"))
    assert review_files, "expected at least one review file"
    review_file = review_files[0]

    # Reject the first non-header entity to verify replay works.
    lines = review_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2
    header_line = lines[0]
    edited_lines = [header_line]
    for body in lines[1:]:
        entry = json.loads(body)
        entry["op"] = "reject"
        edited_lines.append(json.dumps(entry))
        # only reject the first one; keep the rest as-is
        for rest in lines[len(edited_lines) :]:
            edited_lines.append(rest)
        break
    review_file.write_text("\n".join(edited_lines) + "\n", encoding="utf-8")

    updated = pipeline.import_reviewed(workdir)
    assert len(updated) == 1
    # At least one entity was rejected
    assert sum(len(r.entities) for r in updated) < sum(
        len(r.entities) for r in run.extraction_results
    )


@pytest.mark.integration
def test_anthropic_path_when_key_available(
    tmp_path: Path, small_gliner: GLiNERExtractor
) -> None:
    """Run the real Anthropic path. Skipped without ANTHROPIC_API_KEY."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    schema = load_schema(FIXTURES / "schemas" / "people_orgs.yaml")
    doc = Document(
        id="curie",
        text=(FIXTURES / "docs" / "doc_curie_short.txt").read_text(encoding="utf-8"),
    )
    result = pipeline.run_one(
        doc, schema, extractor=small_gliner, llm_client=AnthropicClient()
    )
    assert result.entities, "Anthropic coref returned no entities"
    # Every span still quotes the doc verbatim
    for e in result.entities:
        for m in e.mentions:
            assert doc.text[m.span.start : m.span.end] == m.span.text

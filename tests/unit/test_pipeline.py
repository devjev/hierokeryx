"""Tests for the pipeline orchestrator using fake extractor + fake LLM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hierokeryx import pipeline
from hierokeryx.models import (
    CorefCluster,
    CrossDocCandidate,
    Document,
    EntitySchema,
    EntityType,
    Mention,
    MergeDecision,
    Span,
)


class FakeExtractor:
    """Returns one mention per occurrence of a fixed keyword in the document."""

    model_id = "fake/gliner"

    def __init__(self, keyword: str, type_name: str = "Person", score: float = 0.9):
        self.keyword = keyword
        self.type_name = type_name
        self.score = score

    def extract(self, document: Document, schema: EntitySchema) -> list[Mention]:
        from hierokeryx.models import make_mention_id

        mentions = []
        start = 0
        while True:
            idx = document.text.find(self.keyword, start)
            if idx < 0:
                break
            end = idx + len(self.keyword)
            mentions.append(
                Mention(
                    id=make_mention_id(document.id, idx, end),
                    span=Span(start=idx, end=end, text=self.keyword),
                    type=self.type_name,
                    score=self.score,
                )
            )
            start = end
        return mentions


class FakeLLM:
    """Returns one cluster containing all input mentions. coref_model is read by the pipeline."""

    coref_model = "fake-coref-v1"
    tiebreak_model = "fake-tiebreak-v1"

    def cluster_mentions(
        self, document: Document, mentions: list[Mention], schema: EntitySchema
    ) -> list[CorefCluster]:
        if not mentions:
            return []
        return [
            CorefCluster(
                mention_ids=[m.id for m in mentions],
                canonical=mentions[0].span.text,
                type=mentions[0].type,
                confidence=0.9,
            )
        ]

    def resolve_crossdoc(
        self,
        candidates: list[CrossDocCandidate],
        schema: EntitySchema,
    ) -> list[MergeDecision]:
        return [
            MergeDecision(
                entity_id=c.entity_id,
                target_cluster_id=c.nearest_cluster_id,
                confidence=0.85,
            )
            for c in candidates
        ]


class FakeEmbedder:
    """Same-keyword entities get nearly identical embeddings; different ones diverge."""

    model_id = "fake/embedder"

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = []
        for t in texts:
            first = t.strip()[:1] or "_"
            base = np.zeros(8, dtype=np.float32)
            base[ord(first) % 8] = 1.0
            noise = np.zeros(8, dtype=np.float32)
            for i, c in enumerate(t.encode("utf-8")[:8]):
                noise[i] = (c % 13) / 1000.0
            v = base + noise
            v /= max(np.linalg.norm(v), 1e-9)
            vectors.append(v)
        return np.asarray(vectors, dtype=np.float32)


def test_pipeline_writes_expected_workdir_structure(tmp_path: Path) -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="A human")])
    docs = [
        Document(id="d1", text="Alice met Alice again in Alice's office."),
        Document(id="d2", text="Bob saw Bob at Bob's place."),
    ]
    workdir = tmp_path / "wd"
    run = pipeline.run(
        documents=docs,
        schema=schema,
        workdir=workdir,
        extractor=FakeExtractor(keyword="Alice"),  # only matches d1
        llm_client=FakeLLM(),
        embedder=FakeEmbedder(),
        review_threshold=0.0,  # nothing flagged
    )

    assert (workdir / "schema.yaml").exists()
    assert (workdir / "manifest.json").exists()
    assert (workdir / "registry.json").exists()
    assert (workdir / "extractions" / "d1.json").exists()
    assert (workdir / "extractions" / "d2.json").exists()

    manifest = json.loads((workdir / "manifest.json").read_text())
    assert manifest["n_documents"] == 2

    [d1_result] = [r for r in run.extraction_results if r.document.id == "d1"]
    assert len(d1_result.entities) == 1
    [d2_result] = [r for r in run.extraction_results if r.document.id == "d2"]
    assert len(d2_result.entities) == 0  # FakeExtractor didn't match "Bob"


def test_pipeline_routes_low_confidence_entities_to_review(tmp_path: Path) -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="A human")])
    doc = Document(id="d1", text="Alice walked. Alice ran. Alice slept.")
    workdir = tmp_path / "wd"
    run = pipeline.run(
        documents=[doc],
        schema=schema,
        workdir=workdir,
        extractor=FakeExtractor(keyword="Alice", score=0.4),  # low GLiNER score
        llm_client=FakeLLM(),
        embedder=FakeEmbedder(),
        review_threshold=0.9,
        only_flagged_review=True,
    )
    assert run.flagged, "expected low-confidence Alice entity to be flagged"
    assert len(run.review_paths) == 1
    assert run.review_paths[0].name == "d1.jsonl"


def test_pipeline_import_reviewed_replays_edits(tmp_path: Path) -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="A human")])
    doc = Document(id="d1", text="Alice waited.")
    workdir = tmp_path / "wd"
    pipeline.run(
        documents=[doc],
        schema=schema,
        workdir=workdir,
        extractor=FakeExtractor(keyword="Alice"),
        llm_client=FakeLLM(),
        embedder=FakeEmbedder(),
        review_threshold=1.0,  # flag everything
        only_flagged_review=False,
    )

    review_file = workdir / "review" / "d1.jsonl"
    assert review_file.exists()

    # Reject the only entity by editing the file in place.
    raw_lines = review_file.read_text(encoding="utf-8").splitlines()
    header = raw_lines[0]
    entity_line = json.loads(raw_lines[1])
    entity_line["op"] = "reject"
    review_file.write_text(header + "\n" + json.dumps(entity_line) + "\n")

    [updated] = pipeline.import_reviewed(workdir)
    assert updated.entities == []
    # Reload from disk to verify the file was rewritten.
    reloaded = pipeline.load_extraction(workdir / "extractions" / "d1.json")
    assert reloaded.entities == []

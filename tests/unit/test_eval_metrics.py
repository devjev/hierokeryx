"""Pairwise + BCubed metrics on hand-built cases, plus the gold JSONL loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from hierokeryx.eval.gold import GoldEntry, load_gold, save_gold
from hierokeryx.eval.metrics import bcubed_prf, pairwise_prf
from hierokeryx.eval.report import evaluate, sweep_thresholds
from hierokeryx.models import (
    Document,
    Entity,
    EntitySchema,
    EntityType,
    ExtractionResult,
    Mention,
    Span,
)

# ---- pairwise -------------------------------------------------------------

def test_pairwise_perfect_match() -> None:
    sys_ = {"a": "c1", "b": "c1", "c": "c2"}
    gold = {"a": "g1", "b": "g1", "c": "g2"}  # cluster labels are opaque
    p, r, f = pairwise_prf(sys_, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_pairwise_all_singletons_vs_one_cluster() -> None:
    sys_ = {"a": "c1", "b": "c2", "c": "c3"}
    gold = {"a": "g1", "b": "g1", "c": "g1"}
    # System has 0 co-clustered pairs, gold has 3 → P=1 (no false positives), R=0
    p, r, f = pairwise_prf(sys_, gold)
    assert p == 1.0
    assert r == 0.0
    assert f == 0.0


def test_pairwise_one_split_cluster() -> None:
    # Gold: {a,b,c} all together. System: {a,b} and {c}.
    # Pairs: (a,b), (a,c), (b,c). System co-clusters only (a,b). Gold all 3.
    # TP=1, FP=0, FN=2 → P=1, R=1/3, F=0.5
    sys_ = {"a": "c1", "b": "c1", "c": "c2"}
    gold = {"a": "g1", "b": "g1", "c": "g1"}
    p, r, f = pairwise_prf(sys_, gold)
    assert p == 1.0
    assert r == pytest.approx(1 / 3)
    assert f == pytest.approx(0.5)


def test_pairwise_intersects_keys() -> None:
    # Extra system key not in gold should be ignored entirely.
    sys_ = {"a": "c1", "b": "c1", "extra": "c1"}
    gold = {"a": "g1", "b": "g1"}
    p, r, f = pairwise_prf(sys_, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)


# ---- BCubed --------------------------------------------------------------

def test_bcubed_perfect_match() -> None:
    sys_ = {"a": "c1", "b": "c1", "c": "c2"}
    gold = {"a": "g1", "b": "g1", "c": "g2"}
    p, r, f = bcubed_prf(sys_, gold)
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_bcubed_split_cluster() -> None:
    # Gold: {a,b,c}. System: {a,b}, {c}.
    # a: sys_cluster={a,b}, gold={a,b,c}, overlap=2 → P=2/2=1, R=2/3
    # b: same as a → P=1, R=2/3
    # c: sys={c}, gold={a,b,c}, overlap=1 → P=1, R=1/3
    # Mean: P = 1, R = (2/3 + 2/3 + 1/3)/3 = (5/3)/3 = 5/9
    sys_ = {"a": "c1", "b": "c1", "c": "c2"}
    gold = {"a": "g1", "b": "g1", "c": "g1"}
    p, r, _f = bcubed_prf(sys_, gold)
    assert p == 1.0
    assert r == pytest.approx(5 / 9)


def test_bcubed_one_big_cluster_vs_singletons() -> None:
    # System lumps everyone together; gold says everyone is a singleton.
    # Each entity: sys={a,b,c}, gold={x}, overlap=1 → P=1/3, R=1/1=1
    sys_ = {"a": "c1", "b": "c1", "c": "c1"}
    gold = {"a": "g1", "b": "g2", "c": "g3"}
    p, r, _f = bcubed_prf(sys_, gold)
    assert p == pytest.approx(1 / 3)
    assert r == 1.0


# ---- gold JSONL roundtrip -----------------------------------------------

def test_gold_jsonl_roundtrip(tmp_path: Path) -> None:
    entries = [
        GoldEntry(doc_id="d1", entity_id="e1", gold_cluster_id="g_a"),
        GoldEntry(doc_id="d1", entity_id="e2", gold_cluster_id="g_b"),
        GoldEntry(doc_id="d2", entity_id="e1", gold_cluster_id="g_a"),
    ]
    path = tmp_path / "gold.jsonl"
    save_gold(entries, path)
    loaded = load_gold(path)
    assert loaded == entries


def test_gold_jsonl_rejects_duplicate_entity(tmp_path: Path) -> None:
    path = tmp_path / "dupe.jsonl"
    path.write_text(
        '{"doc_id": "d1", "entity_id": "e1", "gold_cluster_id": "g_a"}\n'
        '{"doc_id": "d1", "entity_id": "e1", "gold_cluster_id": "g_b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_gold(path)


def test_gold_jsonl_skips_blank_and_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "g.jsonl"
    path.write_text(
        "# header comment\n"
        "\n"
        '{"doc_id": "d1", "entity_id": "e1", "gold_cluster_id": "g_a"}\n'
        "\n",
        encoding="utf-8",
    )
    assert len(load_gold(path)) == 1


# ---- evaluate(extractions, gold) -----------------------------------------

def _mk_result(doc_id: str, specs: list[tuple[str, str | None]]) -> ExtractionResult:
    """Build an ExtractionResult with entities laid out sequentially in the doc text.

    `specs` is a list of `(entity_id, cluster_id)` pairs. Each entity is a
    single mention whose span text is the entity_id itself, separated by
    spaces — so the per-mention span invariant holds.
    """
    parts: list[str] = []
    starts: list[int] = []
    cursor = 0
    for eid, _ in specs:
        if cursor > 0:
            parts.append(" ")
            cursor += 1
        starts.append(cursor)
        parts.append(eid)
        cursor += len(eid)
    text = "".join(parts) or "x"
    entities: list[Entity] = []
    for (eid, cid), start in zip(specs, starts, strict=True):
        end = start + len(eid)
        entities.append(
            Entity(
                id=eid,
                type="Person",
                canonical=eid,
                surface_canonical=eid,
                mentions=[
                    Mention(
                        id=f"m_{eid}",
                        span=Span(start=start, end=end, text=eid),
                        type="Person",
                        score=0.9,
                    )
                ],
                confidence=0.9,
                doc_id=doc_id,
                cluster_id=cid,
            )
        )
    return ExtractionResult(
        document=Document(id=doc_id, text=text),
        entities=entities,
        schema_version="t",
    )


def test_evaluate_end_to_end() -> None:
    # Two docs, three entities; gold groups e1 and e2 together.
    results = [
        _mk_result("d1", [("e1", "sys_a")]),
        _mk_result("d2", [("e2", "sys_a"), ("e3", "sys_b")]),
    ]
    gold = [
        GoldEntry(doc_id="d1", entity_id="e1", gold_cluster_id="g_x"),
        GoldEntry(doc_id="d2", entity_id="e2", gold_cluster_id="g_x"),
        GoldEntry(doc_id="d2", entity_id="e3", gold_cluster_id="g_y"),
    ]
    report = evaluate(results, gold)
    assert report.n_entities_scored == 3
    assert report.n_clusters_system == 2
    assert report.n_clusters_gold == 2
    assert report.pairwise.f1 == 1.0
    assert report.bcubed.f1 == 1.0


def test_evaluate_partial_gold_only_scores_intersection() -> None:
    results = [_mk_result("d1", [("e1", "sys_a"), ("e2", "sys_a"), ("e3", "sys_b")])]
    gold = [GoldEntry(doc_id="d1", entity_id="e1", gold_cluster_id="g_x")]
    report = evaluate(results, gold)
    # Only e1 overlaps — 1 entity, degenerate case (no pairs to score).
    assert report.n_entities_scored == 1


# ---- sweep_thresholds (smoke) -------------------------------------------

class _FakeEmbedder:
    """Tiny embedder so the sweep can run without downloading a model."""

    model_id = "fake-mini"

    def encode(self, texts: list[str]):  # type: ignore[no-untyped-def]
        import numpy as np

        vecs = []
        for t in texts:
            ch = t.strip()[:1] or "_"
            v = np.zeros(4, dtype="float32")
            v[ord(ch) % 4] = 1.0
            vecs.append(v / max(float(np.linalg.norm(v)), 1e-9))
        return np.asarray(vecs, dtype="float32")


def test_sweep_thresholds_returns_one_report_per_grid_point() -> None:
    schema = EntitySchema(types=[EntityType(name="Person", description="x")])
    results = [_mk_result("d1", [("e1", None)]), _mk_result("d2", [("e2", None)])]
    gold = [
        GoldEntry(doc_id="d1", entity_id="e1", gold_cluster_id="g_x"),
        GoldEntry(doc_id="d2", entity_id="e2", gold_cluster_id="g_x"),
    ]
    reports = sweep_thresholds(
        results,
        gold,
        schema,
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        merge_grid=(0.82, 0.95),
        borderline_grid=(0.70, 0.80),
    )
    # 2x2 grid, but borderline > merge pairs are skipped → 2 * 2 - 0 = 4 valid
    # (0.70 ≤ 0.82, 0.80 ≤ 0.82, 0.70 ≤ 0.95, 0.80 ≤ 0.95)
    assert len(reports) == 4
    for r in reports:
        assert "merge_threshold" in r.config
        assert "borderline_threshold" in r.config

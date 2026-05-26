"""Pairwise and BCubed precision / recall / F1 for entity-resolution clusters.

Both metrics operate on `dict[entity_key, cluster_id]` mappings (the
`entity_key` is opaque — typically `f"{doc_id}/{entity_id}"`). They compute
over the intersection of keys present in both `system` and `gold`, so a
partial gold set is supported.

The implementations are intentionally short and dependency-free; correctness
matters more than micro-performance for eval sets that fit in memory.
"""

from __future__ import annotations

from collections import defaultdict

Assignment = dict[str, str]
PRF_tuple = tuple[float, float, float]


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _restrict(system: Assignment, gold: Assignment) -> tuple[Assignment, Assignment]:
    common = system.keys() & gold.keys()
    return {k: system[k] for k in common}, {k: gold[k] for k in common}


def pairwise_prf(system: Assignment, gold: Assignment) -> PRF_tuple:
    """Standard pairwise precision / recall / F1.

    Considers every unordered pair of entities in the intersection.
    - TP: pair is co-clustered in both system and gold.
    - FP: pair is co-clustered in system but not in gold.
    - FN: pair is co-clustered in gold but not in system.
    Precision = TP / (TP + FP), Recall = TP / (TP + FN).
    """
    sys_r, gold_r = _restrict(system, gold)
    keys = sorted(sys_r.keys())
    n = len(keys)
    if n < 2:
        return (1.0, 1.0, 1.0) if n == 1 else (0.0, 0.0, 0.0)

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            ki, kj = keys[i], keys[j]
            sys_same = sys_r[ki] == sys_r[kj]
            gold_same = gold_r[ki] == gold_r[kj]
            if sys_same and gold_same:
                tp += 1
            elif sys_same and not gold_same:
                fp += 1
            elif gold_same and not sys_same:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return (precision, recall, _f1(precision, recall))


def bcubed_prf(system: Assignment, gold: Assignment) -> PRF_tuple:
    """BCubed precision / recall / F1.

    For each entity:
      precision = |system_cluster ∩ gold_cluster| / |system_cluster|
      recall    = |system_cluster ∩ gold_cluster| / |gold_cluster|
    Final scores are means over all entities. More robust than pairwise to
    skewed cluster sizes; standard in coref / ER evaluation.
    """
    sys_r, gold_r = _restrict(system, gold)
    if not sys_r:
        return (0.0, 0.0, 0.0)

    sys_members: dict[str, list[str]] = defaultdict(list)
    gold_members: dict[str, list[str]] = defaultdict(list)
    for k, c in sys_r.items():
        sys_members[c].append(k)
    for k, c in gold_r.items():
        gold_members[c].append(k)

    total_p = total_r = 0.0
    for k in sys_r:
        sys_set = set(sys_members[sys_r[k]])
        gold_set = set(gold_members[gold_r[k]])
        overlap = len(sys_set & gold_set)
        total_p += overlap / len(sys_set)
        total_r += overlap / len(gold_set)
    n = len(sys_r)
    precision = total_p / n
    recall = total_r / n
    return (precision, recall, _f1(precision, recall))

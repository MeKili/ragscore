"""Ranking metrics. ``ranked`` is a list of document ids, best first."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError("k must be >= 1")


def hit_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if any relevant document appears in the top ``k``."""
    _check_k(k)
    return 1.0 if any(doc in relevant for doc in ranked[:k]) else 0.0


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the top ``k`` slots that hold a relevant document."""
    _check_k(k)
    return sum(1 for doc in ranked[:k] if doc in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant documents found in the top ``k``."""
    _check_k(k)
    if not relevant:
        raise ValueError("recall is undefined without relevant documents")
    return len({doc for doc in ranked[:k] if doc in relevant}) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """1 / rank of the first relevant document, or 0 if none is retrieved."""
    for position, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / position
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Collection[str]) -> float:
    """Mean of precision values at each rank where a relevant document appears."""
    if not relevant:
        raise ValueError("average precision is undefined without relevant documents")
    hits = 0
    total = 0.0
    seen: set[str] = set()
    for position, doc in enumerate(ranked, start=1):
        if doc in relevant and doc not in seen:
            seen.add(doc)
            hits += 1
            total += hits / position
    return total / len(set(relevant))


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance."""
    _check_k(k)
    if not gains:
        raise ValueError("nDCG is undefined without relevant documents")

    def dcg(values: Sequence[float]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(values))

    actual = dcg([gains.get(doc, 0.0) for doc in ranked[:k]])
    ideal = dcg(sorted(gains.values(), reverse=True)[:k])
    return 0.0 if ideal == 0.0 else actual / ideal

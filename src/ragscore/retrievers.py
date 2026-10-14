"""Reference retrievers: a from-scratch BM25 and a naive term-overlap baseline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

from ragscore.answer_metrics import tokenize
from ragscore.dataset import Document


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[str]:
        """Return up to ``k`` document ids, best first."""
        ...


class BM25:
    """Okapi BM25 with the usual ``k1`` and ``b`` parameters."""

    name = "bm25"

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        self.k1, self.b = k1, b
        self._ids = [d.id for d in docs]
        self._tf = [Counter(tokenize(d.text)) for d in docs]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._len) / len(self._len) or 1.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(docs)
        self._idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score(self, index: int, terms: Sequence[str]) -> float:
        tf, length = self._tf[index], self._len[index]
        score = 0.0
        for term in terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            norm = freq + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            score += self._idf.get(term, 0.0) * freq * (self.k1 + 1) / norm
        return score

    def search(self, query: str, k: int) -> list[str]:
        terms = tokenize(query)
        scored = [(self._score(i, terms), doc_id) for i, doc_id in enumerate(self._ids)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for score, doc_id in scored[:k] if score > 0.0]


class TermOverlap:
    """Counts distinct query terms that appear in the document. A deliberately weak baseline."""

    name = "term_overlap"

    def __init__(self, docs: Sequence[Document]) -> None:
        if not docs:
            raise ValueError("cannot index an empty corpus")
        self._ids = [d.id for d in docs]
        self._terms = [set(tokenize(d.text)) for d in docs]

    def search(self, query: str, k: int) -> list[str]:
        q = set(tokenize(query))
        scored = [
            (len(q & terms), doc_id) for terms, doc_id in zip(self._terms, self._ids, strict=True)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [doc_id for score, doc_id in scored[:k] if score > 0]


def build_retriever(name: str, docs: Sequence[Document]) -> Retriever:
    if name == "bm25":
        return BM25(docs)
    if name == "term_overlap":
        return TermOverlap(docs)
    raise ValueError(f"unknown retriever {name!r}; choose bm25 or term_overlap")

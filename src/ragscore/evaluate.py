"""Run a retriever over a query set and collect per-query metric values."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ragscore import answer_metrics as am
from ragscore import retrieval_metrics as rm
from ragscore.dataset import Document, Query
from ragscore.retrievers import Retriever


@dataclass
class SystemResult:
    """Per-query metric values for one system. Lists are aligned with ``query_ids``."""

    name: str
    query_ids: list[str]
    per_query: dict[str, list[float]]

    def mean(self, metric: str) -> float:
        values = self.per_query[metric]
        return sum(values) / len(values)

    def metrics(self) -> list[str]:
        return list(self.per_query)

    def save(self, path: Path) -> None:
        payload = {"name": self.name, "query_ids": self.query_ids, "per_query": self.per_query}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SystemResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["name"], list(data["query_ids"]), dict(data["per_query"]))


def evaluate_retriever(
    retriever: Retriever, queries: Sequence[Query], ks: Sequence[int] = (1, 3, 5, 10)
) -> SystemResult:
    """Compute hit, recall, nDCG at each k plus MRR and average precision."""
    if not queries:
        raise ValueError("no queries to evaluate")
    depth = max(ks)
    per_query: dict[str, list[float]] = {"mrr": [], "map": []}
    for k in ks:
        for family in ("hit", "recall", "ndcg"):
            per_query[f"{family}@{k}"] = []
    for query in queries:
        ranked = retriever.search(query.text, depth)
        relevant = query.relevant
        per_query["mrr"].append(rm.reciprocal_rank(ranked, relevant))
        per_query["map"].append(rm.average_precision(ranked, relevant))
        for k in ks:
            per_query[f"hit@{k}"].append(rm.hit_at_k(ranked, relevant, k))
            per_query[f"recall@{k}"].append(rm.recall_at_k(ranked, relevant, k))
            per_query[f"ndcg@{k}"].append(rm.ndcg_at_k(ranked, query.gains, k))
    return SystemResult(retriever.name, [q.id for q in queries], per_query)


def evaluate_answers(
    name: str,
    queries: Sequence[Query],
    answers: Mapping[str, str],
    corpus: Sequence[Document],
    retrieved: Mapping[str, Sequence[str]],
) -> SystemResult:
    """Score generated answers.

    ``answers`` maps query id to the answer text and ``retrieved`` maps query id to the ids of the
    passages that were given to the generator. Exact match and F1 are only computed for queries
    that have a reference answer, so those lists can be shorter than the others.
    """
    text_of = {d.id: d.text for d in corpus}
    per_query: dict[str, list[float]] = {
        "groundedness": [],
        "citation_precision": [],
        "citation_recall": [],
        "exact_match": [],
        "token_f1": [],
    }
    ids: list[str] = []
    for query in queries:
        if query.id not in answers:
            raise ValueError(f"no answer provided for query {query.id!r}")
        answer = answers[query.id]
        context = [text_of[doc] for doc in retrieved.get(query.id, []) if doc in text_of]
        ids.append(query.id)
        per_query["groundedness"].append(am.groundedness(answer, context))
        per_query["citation_precision"].append(am.citation_precision(answer, query.relevant))
        per_query["citation_recall"].append(am.citation_recall(answer, query.relevant))
        if query.reference is not None:
            per_query["exact_match"].append(am.exact_match(answer, query.reference))
            per_query["token_f1"].append(am.token_f1(answer, query.reference))
    if not per_query["exact_match"]:
        del per_query["exact_match"], per_query["token_f1"]
    return SystemResult(name, ids, per_query)

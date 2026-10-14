"""Corpus and query files in JSON Lines format."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    gains: dict[str, float] = field(default_factory=dict)
    reference: str | None = None

    @property
    def relevant(self) -> set[str]:
        return {doc for doc, gain in self.gains.items() if gain > 0}


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield record


def load_corpus(path: Path) -> list[Document]:
    """Each line: ``{"id": "d1", "text": "..."}``. Ids must be unique."""
    docs: list[Document] = []
    seen: set[str] = set()
    for record in _read_jsonl(path):
        doc_id, text = str(record["id"]), str(record["text"])
        if doc_id in seen:
            raise ValueError(f"duplicate document id {doc_id!r}")
        seen.add(doc_id)
        docs.append(Document(doc_id, text))
    if not docs:
        raise ValueError(f"{path} contains no documents")
    return docs


def load_queries(path: Path) -> list[Query]:
    """Each line: ``{"id", "text", "relevant": ["d1"] or {"d1": 2}, "reference": "optional"}``."""
    queries: list[Query] = []
    for record in _read_jsonl(path):
        relevant = record["relevant"]
        gains = (
            {str(k): float(v) for k, v in relevant.items()}
            if isinstance(relevant, dict)
            else {str(doc): 1.0 for doc in relevant}
        )
        query = Query(
            id=str(record["id"]),
            text=str(record["text"]),
            gains=gains,
            reference=record.get("reference"),
        )
        if not query.relevant:
            raise ValueError(f"query {query.id!r} has no relevant documents")
        queries.append(query)
    if not queries:
        raise ValueError(f"{path} contains no queries")
    return queries

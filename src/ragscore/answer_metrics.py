"""Answer-level metrics: overlap with a reference, grounding in context, and citation quality."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Iterable

_ARTICLES = {"a", "an", "the"}
_STOPWORDS = _ARTICLES | {
    "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for", "and", "or",
    "it", "its", "this", "that", "with", "as", "by", "at", "from", "which", "has", "have",
}  # fmt: skip
_TOKEN = re.compile(r"[a-z0-9]+")
_CITATION = re.compile(r"\[([^\[\]]+)\]")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens."""
    return _TOKEN.findall(text.lower())


def strip_citations(text: str) -> str:
    """Remove ``[d3]`` style markers so they are not scored as answer words."""
    return _CITATION.sub(" ", text)


def normalize_answer(text: str) -> list[str]:
    """Tokens with citations and articles removed, as in SQuAD-style evaluation."""
    return [t for t in tokenize(strip_citations(text)) if t not in _ARTICLES]


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    """Harmonic mean of token precision and recall against the reference answer."""
    pred = Counter(normalize_answer(prediction))
    ref = Counter(normalize_answer(reference))
    if not pred and not ref:
        return 1.0
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def groundedness(answer: str, contexts: Iterable[str]) -> float:
    """Share of the answer's content words that occur somewhere in the retrieved context.

    A cheap lexical proxy for faithfulness. It cannot detect a wrong claim built from context
    words, so treat it as a smoke test rather than a verdict.
    """
    context_tokens: set[str] = set()
    for text in contexts:
        context_tokens.update(tokenize(text))
    words = tokenize(strip_citations(answer))
    content = [t for t in words if t not in _STOPWORDS and not t.isdigit()]
    if not content:
        return 1.0
    return sum(1 for t in content if t in context_tokens) / len(content)


def cited_ids(answer: str) -> list[str]:
    """Document ids cited like ``[d3]`` or ``[d3, d7]`` in the answer, in order, without repeats."""
    found: list[str] = []
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found


def citation_precision(answer: str, relevant: Collection[str]) -> float:
    """Share of cited documents that are relevant. An answer without citations scores 0."""
    cited = cited_ids(answer)
    if not cited:
        return 0.0
    return sum(1 for doc in cited if doc in relevant) / len(cited)


def citation_recall(answer: str, relevant: Collection[str]) -> float:
    """Share of relevant documents that the answer cites."""
    if not relevant:
        raise ValueError("citation recall is undefined without relevant documents")
    cited = set(cited_ids(answer))
    return len(cited & set(relevant)) / len(set(relevant))

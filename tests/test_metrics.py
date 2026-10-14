import math

import pytest

from ragscore import answer_metrics as am
from ragscore import retrieval_metrics as rm


def test_hit_precision_recall() -> None:
    ranked = ["a", "b", "c", "d"]
    assert rm.hit_at_k(ranked, {"c"}, 2) == 0.0
    assert rm.hit_at_k(ranked, {"c"}, 3) == 1.0
    assert rm.precision_at_k(ranked, {"a", "c"}, 4) == 0.5
    assert rm.recall_at_k(ranked, {"a", "z"}, 4) == 0.5


def test_reciprocal_rank_and_average_precision() -> None:
    assert rm.reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert rm.reciprocal_rank(["x"], {"a"}) == 0.0
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert rm.average_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx((1 + 2 / 3) / 2)


def test_average_precision_ignores_duplicate_hits() -> None:
    assert rm.average_precision(["a", "a", "a"], {"a"}) == pytest.approx(1.0)


def test_ndcg_perfect_and_imperfect() -> None:
    gains = {"a": 3.0, "b": 1.0}
    assert rm.ndcg_at_k(["a", "b"], gains, 2) == pytest.approx(1.0)
    swapped = rm.ndcg_at_k(["b", "a"], gains, 2)
    expected = (1 / math.log2(2) + 3 / math.log2(3)) / (3 / math.log2(2) + 1 / math.log2(3))
    assert swapped == pytest.approx(expected)
    assert rm.ndcg_at_k(["x", "y"], gains, 2) == 0.0


@pytest.mark.parametrize("bad", [0, -1])
def test_k_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError):
        rm.hit_at_k(["a"], {"a"}, bad)


def test_undefined_without_relevant() -> None:
    with pytest.raises(ValueError):
        rm.recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError):
        rm.ndcg_at_k(["a"], {}, 1)


def test_exact_match_ignores_case_articles_and_punctuation() -> None:
    assert am.exact_match("The Bloom filter!", "bloom filter") == 1.0
    assert am.exact_match("bloom", "bloom filter") == 0.0


def test_token_f1() -> None:
    assert am.token_f1("x y z", "y z w") == pytest.approx(2 / 3)
    assert am.token_f1("the cat [d1]", "cat") == 1.0
    assert am.token_f1("", "") == 1.0
    assert am.token_f1("x", "y") == 0.0


def test_groundedness_detects_unsupported_words() -> None:
    context = ["BM25 scores documents by term frequency"]
    assert am.groundedness("BM25 scores documents", context) == 1.0
    assert am.groundedness("BM25 uses quantum entanglement", context) < 0.5
    assert am.groundedness("is the of", context) == 1.0


def test_citations() -> None:
    answer = "BM25 ranks text [d3]. Fusion merges lists [d5, d3]."
    assert am.cited_ids(answer) == ["d3", "d5"]
    assert am.citation_precision(answer, {"d3"}) == 0.5
    assert am.citation_recall(answer, {"d3", "d9"}) == 0.5
    assert am.citation_precision("no citations here", {"d3"}) == 0.0

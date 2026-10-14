import json
from pathlib import Path

import pytest

from ragscore.cli import main
from ragscore.dataset import Document, load_corpus, load_queries
from ragscore.evaluate import SystemResult, evaluate_answers, evaluate_retriever
from ragscore.report import metrics_table
from ragscore.retrievers import BM25, TermOverlap, build_retriever
from ragscore.stats import compare_paired

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_bm25_ranks_the_matching_document_first() -> None:
    docs = [
        Document("a", "cats purr and sleep all day"),
        Document("b", "bm25 ranks documents by term statistics"),
        Document("c", "dogs bark at night"),
    ]
    assert BM25(docs).search("how does bm25 rank documents", 3)[0] == "b"


def test_bm25_returns_nothing_for_unknown_terms() -> None:
    assert BM25([Document("a", "hello world")]).search("zzz", 3) == []


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError):
        BM25([])
    with pytest.raises(ValueError):
        TermOverlap([])
    with pytest.raises(ValueError):
        build_retriever("nope", [Document("a", "x")])


def test_bm25_beats_term_overlap_on_example_data() -> None:
    corpus = load_corpus(EXAMPLES / "corpus.jsonl")
    queries = load_queries(EXAMPLES / "queries.jsonl")
    bm25 = evaluate_retriever(BM25(corpus), queries, ks=(1, 3))
    weak = evaluate_retriever(TermOverlap(corpus), queries, ks=(1, 3))
    assert bm25.mean("ndcg@3") >= weak.mean("ndcg@3")
    assert bm25.mean("hit@3") >= 0.8


def test_result_round_trips_through_json(tmp_path: Path) -> None:
    corpus = load_corpus(EXAMPLES / "corpus.jsonl")
    queries = load_queries(EXAMPLES / "queries.jsonl")
    result = evaluate_retriever(BM25(corpus), queries, ks=(1,))
    path = tmp_path / "r.json"
    result.save(path)
    loaded = SystemResult.load(path)
    assert loaded.query_ids == result.query_ids
    assert loaded.per_query == result.per_query
    assert "| bm25 |" in metrics_table([loaded])


def test_compare_paired_detects_a_clear_difference() -> None:
    a = [0.9, 0.8, 0.85, 0.95, 0.9, 0.8, 0.9, 0.85]
    b = [0.5, 0.4, 0.45, 0.55, 0.5, 0.4, 0.5, 0.45]
    c = compare_paired("ndcg", a, b, resamples=2000, seed=1)
    assert c.significant
    assert c.diff == pytest.approx(0.4, abs=1e-9)
    assert c.p_value < 0.05


def test_compare_paired_is_not_significant_for_identical_systems() -> None:
    a = [0.5, 0.7, 0.6, 0.8, 0.4]
    c = compare_paired("ndcg", a, list(a), resamples=1000, seed=2)
    assert not c.significant
    assert c.diff == 0.0


def test_compare_paired_validates_input() -> None:
    with pytest.raises(ValueError):
        compare_paired("m", [1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        compare_paired("m", [1.0], [1.0])


def test_dataset_validation(tmp_path: Path) -> None:
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"id": "a", "text": "x"}\n{"id": "a", "text": "y"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_corpus(dup)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_corpus(bad)
    empty_rel = tmp_path / "q.jsonl"
    empty_rel.write_text('{"id": "q", "text": "t", "relevant": []}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no relevant"):
        load_queries(empty_rel)


def test_evaluate_answers_scores_citations_and_overlap() -> None:
    corpus = load_corpus(EXAMPLES / "corpus.jsonl")
    queries = load_queries(EXAMPLES / "queries.jsonl")[:2]
    answers = {"q1": "backpropagation [d2]", "q2": "It is BM25 [d3]"}
    retrieved = {"q1": ["d2", "d1"], "q2": ["d3"]}
    result = evaluate_answers("demo", queries, answers, corpus, retrieved)
    assert result.mean("citation_precision") == 1.0
    assert result.mean("exact_match") >= 0.5
    with pytest.raises(ValueError, match="no answer"):
        evaluate_answers("demo", queries, {"q1": "x"}, corpus, retrieved)


def test_cli_eval_compare_and_answers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus, queries = EXAMPLES / "corpus.jsonl", EXAMPLES / "queries.jsonl"
    out_a, out_b = tmp_path / "a.json", tmp_path / "b.json"
    base = ["eval", "--corpus", str(corpus), "--queries", str(queries), "--k", "1,3"]
    assert main([*base, "--retriever", "bm25", "--out", str(out_a)]) == 0
    assert main([*base, "--retriever", "term_overlap", "--out", str(out_b)]) == 0
    capsys.readouterr()
    assert (
        main(["compare", str(out_a), str(out_b), "--metric", "ndcg@3", "--resamples", "500"]) == 0
    )
    assert "ndcg@3" in capsys.readouterr().out

    answers = tmp_path / "answers.jsonl"
    answers.write_text(
        json.dumps({"query_id": "q1", "answer": "backpropagation [d2]", "retrieved": ["d2"]})
        + "\n",
        encoding="utf-8",
    )
    only_q1 = tmp_path / "q1.jsonl"
    only_q1.write_text(queries.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    cmd = ["answers", "--corpus", str(corpus), "--queries", str(only_q1), "--answers", str(answers)]
    assert main(cmd) == 0
    assert "citation_precision" in capsys.readouterr().out


def test_cli_compare_rejects_mismatched_queries(tmp_path: Path) -> None:
    a = SystemResult("a", ["q1", "q2"], {"m": [1.0, 0.0]})
    b = SystemResult("b", ["q1", "q3"], {"m": [1.0, 0.0]})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    a.save(pa)
    b.save(pb)
    with pytest.raises(SystemExit):
        main(["compare", str(pa), str(pb)])

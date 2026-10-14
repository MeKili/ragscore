"""Command line interface: ``ragscore eval``, ``ragscore compare`` and ``ragscore answers``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ragscore.dataset import load_corpus, load_queries
from ragscore.evaluate import SystemResult, evaluate_answers, evaluate_retriever
from ragscore.report import comparison_line, metrics_table
from ragscore.retrievers import build_retriever
from ragscore.stats import compare_paired


def _parse_ks(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def _cmd_eval(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    result = evaluate_retriever(build_retriever(args.retriever, corpus), queries, _parse_ks(args.k))
    print(metrics_table([result]))
    if args.out:
        result.save(args.out)
        print(f"\nper-query scores written to {args.out}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    first, second = SystemResult.load(args.a), SystemResult.load(args.b)
    if first.query_ids != second.query_ids:
        raise SystemExit("the two result files do not cover the same queries in the same order")
    metrics = args.metric or [m for m in first.metrics() if m in second.per_query]
    for metric in metrics:
        comparison = compare_paired(
            metric,
            first.per_query[metric],
            second.per_query[metric],
            args.resamples,
            seed=args.seed,
        )
        print(comparison_line(comparison, first.name, second.name))
    return 0


def _read_answers(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    answers: dict[str, str] = {}
    retrieved: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                answers[str(record["query_id"])] = str(record["answer"])
                retrieved[str(record["query_id"])] = [str(d) for d in record.get("retrieved", [])]
    return answers, retrieved


def _cmd_answers(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    answers, retrieved = _read_answers(args.answers)
    result = evaluate_answers(args.name, queries, answers, corpus, retrieved)
    print(metrics_table([result]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragscore", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="evaluate a reference retriever on a query set")
    ev.add_argument("--corpus", type=Path, required=True)
    ev.add_argument("--queries", type=Path, required=True)
    ev.add_argument("--retriever", default="bm25", choices=["bm25", "term_overlap"])
    ev.add_argument("--k", default="1,3,5", help="comma separated cut-offs")
    ev.add_argument("--out", type=Path, help="write per-query scores here for later comparison")
    ev.set_defaults(func=_cmd_eval)

    cmp_ = sub.add_parser("compare", help="paired significance test between two result files")
    cmp_.add_argument("a", type=Path)
    cmp_.add_argument("b", type=Path)
    cmp_.add_argument("--metric", action="append", help="metric to test (repeatable)")
    cmp_.add_argument("--resamples", type=int, default=10_000)
    cmp_.add_argument("--seed", type=int, default=0)
    cmp_.set_defaults(func=_cmd_compare)

    ans = sub.add_parser("answers", help="score generated answers")
    ans.add_argument("--corpus", type=Path, required=True)
    ans.add_argument("--queries", type=Path, required=True)
    ans.add_argument(
        "--answers", type=Path, required=True, help="JSONL: query_id, answer, retrieved"
    )
    ans.add_argument("--name", default="system")
    ans.set_defaults(func=_cmd_answers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

"""Plain-text and Markdown reports."""

from __future__ import annotations

from collections.abc import Sequence

from ragscore.evaluate import SystemResult
from ragscore.stats import Comparison


def metrics_table(results: Sequence[SystemResult]) -> str:
    """Markdown table with one row per system and one column per metric."""
    if not results:
        raise ValueError("no results to report")
    metrics = results[0].metrics()
    header = "| system | " + " | ".join(metrics) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in metrics) + " |"
    rows = [
        f"| {r.name} | " + " | ".join(f"{r.mean(m):.3f}" for m in metrics) + " |" for r in results
    ]
    return "\n".join([header, divider, *rows])


def comparison_line(c: Comparison, name_a: str, name_b: str) -> str:
    verdict = "significant" if c.significant else "not significant"
    return (
        f"{c.metric}: {name_a}={c.mean_a:.3f} {name_b}={c.mean_b:.3f} "
        f"diff={c.diff:+.3f} 95% CI [{c.ci_low:+.3f}, {c.ci_high:+.3f}] "
        f"p={c.p_value:.4f} n={c.n} ({verdict})"
    )

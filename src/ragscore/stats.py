"""Paired significance testing for comparing two systems on the same queries."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int

    @property
    def significant(self) -> bool:
        """True when the confidence interval for the difference excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def compare_paired(
    metric: str,
    a: Sequence[float],
    b: Sequence[float],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Paired bootstrap interval and sign-flip permutation p-value for ``mean(a) - mean(b)``.

    Both lists must hold the per-query scores of the two systems on the same queries, in the same
    order. Resampling queries (not scores) keeps the pairing, which is what makes the test fair.
    """
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if len(a) < 2:
        raise ValueError("need at least two paired observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    observed = _mean(diffs)

    boot = sorted(_mean([diffs[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    low, high = _percentile(boot, tail), _percentile(boot, 1.0 - tail)

    extreme = 0
    for _ in range(resamples):
        flipped = _mean([d if rng.random() < 0.5 else -d for d in diffs])
        if abs(flipped) >= abs(observed) - 1e-12:
            extreme += 1
    p_value = (extreme + 1) / (resamples + 1)

    return Comparison(metric, _mean(a), _mean(b), observed, low, high, p_value, n)

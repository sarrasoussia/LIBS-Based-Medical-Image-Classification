from __future__ import annotations

import itertools
from typing import Dict, Iterable, List

import numpy as np


def summarize_metric(values: Iterable[float]) -> Dict[str, float]:
    """Return mean/std and 95% CI (normal approximation) for a metric."""
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize empty metric list.")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    sem = std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    ci = 1.96 * sem
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - ci),
        "ci95_high": float(mean + ci),
    }


def paired_permutation_pvalue(
    baseline_values: Iterable[float],
    ga_values: Iterable[float],
    num_permutations: int = 20000,
    seed: int = 42,
) -> float:
    """Two-sided paired permutation test over mean difference.

    Null hypothesis: paired differences are centered at zero.
    """
    b = np.array(list(baseline_values), dtype=np.float64)
    g = np.array(list(ga_values), dtype=np.float64)
    if b.shape != g.shape:
        raise ValueError("baseline_values and ga_values must have same length")
    if b.size == 0:
        raise ValueError("Cannot test empty metric lists")

    diffs = g - b
    observed = float(abs(diffs.mean()))
    n = diffs.size

    if n <= 14:
        signs = list(itertools.product([-1.0, 1.0], repeat=n))
        perm_means = [abs(float((diffs * np.array(s)).mean())) for s in signs]
        count = sum(v >= observed for v in perm_means)
        return float(count / len(perm_means))

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(num_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        test_stat = abs(float((diffs * signs).mean()))
        if test_stat >= observed:
            count += 1
    return float((count + 1) / (num_permutations + 1))

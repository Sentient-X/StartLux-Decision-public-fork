"""Soft targets that say exactly as much as a label knows.

A progress reading with uncertainty sigma becomes a distribution over score levels; a reading near a decision
threshold splits its mass across the two sides. Calibration is learned from these targets, so they must not be
sharper than the label behind them.
"""
from __future__ import annotations

import math


def _cdf(x: float, mu: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def binned(value: float, edges: list[float], sigma: float) -> list[float]:
    """P(value falls in each bin) for a reading N(value, sigma) clipped to [edges[0], edges[-1]]. len(edges) = bins + 1."""
    if sigma <= 0:
        out = [0.0] * (len(edges) - 1)
        k = min(max(sum(value >= e for e in edges[1:-1]), 0), len(out) - 1)
        out[k] = 1.0
        return out
    lo, hi = edges[0], edges[-1]
    mass = []
    for i in range(len(edges) - 1):
        a = -math.inf if i == 0 else edges[i]
        b = math.inf if i == len(edges) - 2 else edges[i + 1]
        mass.append((1.0 if b == math.inf else _cdf(b, value, sigma)) - (0.0 if a == -math.inf else _cdf(a, value, sigma)))
    del lo, hi
    total = sum(mass)
    return [m / total for m in mass]


def levels(value: float, n: int, sigma: float) -> list[float]:
    """A [0, 1] value as n equal score levels, lowest first."""
    return binned(value, [i / n for i in range(n + 1)], sigma)


def three_way(delta: float, threshold: float, sigma: float) -> list[float]:
    """A change of `delta` (with reading noise sigma) as [up, flat, down] around +-threshold."""
    return binned(delta, [-1e9, -threshold, threshold, 1e9], sigma)[::-1]


def yes_probability(margin: float, scale: float) -> float:
    """P(yes) for a yes/no question decided by the sign of `margin`, soft over `scale` (0: hard)."""
    if scale <= 0:
        return 1.0 if margin > 0 else 0.0
    z = max(-60.0, min(60.0, margin / scale))
    return 1 / (1 + math.exp(-z))


def tidy(dist: dict[str, float], floor: float = 1e-4) -> dict[str, float]:
    """Drop near-zero mass and renormalise, so targets stay readable in the JSONL."""
    kept = {k: v for k, v in dist.items() if v >= floor}
    total = sum(kept.values())
    return {k: round(v / total, 4) for k, v in kept.items()}

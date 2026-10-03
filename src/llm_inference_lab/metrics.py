"""Metric helpers. Names make the mechanism visible; units stay abstract."""

from __future__ import annotations

from typing import Iterable

from .deterministic import mean, percentile


def distribution(values: Iterable[int | float], prefix: str) -> dict[str, float]:
    numbers = list(values)
    return {
        f"{prefix}_mean": round(mean(numbers), 6),
        f"{prefix}_p50": round(percentile(numbers, 50), 6),
        f"{prefix}_p95": round(percentile(numbers, 95), 6),
        f"{prefix}_p99": round(percentile(numbers, 99), 6),
        f"{prefix}_max": round(max(numbers), 6) if numbers else 0.0,
    }

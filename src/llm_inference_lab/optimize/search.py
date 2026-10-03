"""Exhaustive search over the explicitly declared scenario policy space."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable


def search_space_configurations(
    search_space: dict[str, tuple[Any, ...]], max_candidates: int | None = None
) -> list[dict[str, Any]]:
    keys = sorted(search_space)
    values = [search_space[key] for key in keys]
    count = 1
    for options in values:
        count *= len(options)
    if count == 0:
        return []
    if max_candidates is not None and count > max_candidates:
        raise ValueError(
            f"declared search space has {count} candidates, above max_candidates={max_candidates}"
        )
    return [dict(zip(keys, combination)) for combination in product(*values)]


def objective(kind: str, metrics: dict[str, Any]) -> tuple[float, ...]:
    """Return a lower-is-better tuple declared by this release."""

    if kind == "queue":
        return (
            float(metrics.get("turnaround_ticks_p95", 0.0)),
            float(metrics.get("queue_wait_ticks_p95", 0.0)),
            float(metrics.get("makespan_ticks", 0.0)),
        )
    if kind == "kv":
        stalls = int(metrics.get("admission_failures", 0)) + int(metrics.get("growth_stalls", 0))
        return (
            float(1000 * stalls + int(metrics.get("peak_internal_waste_tokens", 0)) + int(metrics.get("page_table_entries", 0))),
            float(metrics.get("peak_internal_waste_tokens", 0)),
            float(metrics.get("makespan_ticks", 0)),
        )
    if kind == "batching":
        return (
            float(metrics.get("padded_work", 0)),
            float(metrics.get("turnaround_ticks_p95", 0.0)),
            float(metrics.get("queue_wait_ticks_p95", 0.0)),
            float(metrics.get("makespan_ticks", 0)),
        )
    raise ValueError(f"unsupported objective kind: {kind}")

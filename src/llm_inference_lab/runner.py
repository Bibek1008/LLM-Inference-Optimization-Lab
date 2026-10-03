"""Dispatch and baseline configuration for the public simulator surface."""

from __future__ import annotations

from typing import Any

from .contracts import Scenario, SimulationResult
from .simulators import batching, kv_cache, queueing


def baseline_config(kind: str, scenario: Scenario) -> dict[str, Any]:
    parameters = scenario.parameters
    if kind == "queue":
        return {"policy": "fifo", "servers": parameters.get("servers", 1)}
    if kind == "kv":
        return {
            "policy": "contiguous",
            "page_size": parameters.get("page_size", 4),
            "capacity_tokens": parameters.get("capacity_tokens", 32),
        }
    if kind == "batching":
        return {
            "policy": "static",
            "batch_size": parameters.get("batch_size", 2),
            "token_budget": parameters.get("token_budget", 4),
            "slots": parameters.get("slots", parameters.get("batch_size", 2)),
            "prefill_chunk": parameters.get("prefill_chunk", 2),
        }
    raise ValueError(f"unsupported simulator kind: {kind}")


def run_scenario(kind: str, scenario: Scenario, config: dict[str, Any]) -> SimulationResult:
    if scenario.kind != kind:
        raise ValueError(f"scenario kind {scenario.kind!r} does not match {kind!r}")
    merged = baseline_config(kind, scenario)
    merged.update(config)
    if kind == "queue":
        return queueing.run(scenario, merged)
    if kind == "kv":
        return kv_cache.run(scenario, merged)
    if kind == "batching":
        return batching.run(scenario, merged)
    raise ValueError(f"unsupported simulator kind: {kind}")

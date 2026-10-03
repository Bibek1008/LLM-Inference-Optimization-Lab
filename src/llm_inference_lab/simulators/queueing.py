"""A deterministic abstract work-unit queue and admission simulator."""

from __future__ import annotations

from typing import Any

from ..contracts import Scenario, SimulationResult
from ..metrics import distribution


def _policy_key(policy: str, state: dict[str, Any], index: int, tick: int) -> tuple[Any, ...]:
    wait = tick - state["arrival"]
    if policy == "fifo":
        return (state["arrival"], index)
    if policy == "shortest":
        return (state["remaining"], state["arrival"], index)
    if policy == "interactive_first":
        class_rank = 0 if state["request"].get("class", "batch") == "interactive" else 1
        # Waiting time is a bounded, visible aging term. It cannot reorder two
        # otherwise equal requests nondeterministically because index is final.
        return (class_rank, -min(wait, 1000), state["remaining"], state["arrival"], index)
    raise ValueError(f"unknown queue policy: {policy}")


def run(scenario: Scenario, config: dict[str, Any]) -> SimulationResult:
    policy = str(config.get("policy", "fifo"))
    servers = config.get("servers", scenario.parameters.get("servers", 1))
    if not isinstance(servers, int) or servers < 1:
        raise ValueError("queue servers must be a positive integer")

    states: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(scenario.requests):
        state = {
            "request": request,
            "index": index,
            "arrival": request["arrival"],
            "remaining": request["work"],
            "emitted": 0,
            "admitted_tick": None,
            "completion_tick": None,
            "status": "waiting",
        }
        states.append(state)
        by_id[request["id"]] = state

    events: list[dict[str, Any]] = []
    arrived: set[str] = set()
    completion_order: list[str] = []
    max_queue_depth = 0
    tick = 0
    safety_limit = max(1000, sum(request["work"] for request in scenario.requests) * 4 + 100)
    while len(completion_order) < len(states):
        if tick > safety_limit:
            raise RuntimeError("queue simulator exceeded its deterministic safety bound")

        for state in states:
            request_id = state["request"]["id"]
            if request_id not in arrived and state["arrival"] <= tick:
                arrived.add(request_id)
                events.append(
                    {
                        "tick": tick,
                        "type": "request_arrived",
                        "request_id": request_id,
                        "policy": policy,
                    }
                )

        ready = [
            state
            for state in states
            if state["arrival"] <= tick and state["status"] != "completed" and state["remaining"] > 0
        ]
        max_queue_depth = max(max_queue_depth, len(ready))
        selected = sorted(
            ready,
            key=lambda state: _policy_key(policy, state, state["index"], tick),
        )[:servers]

        completed_this_tick: list[dict[str, Any]] = []
        for state in selected:
            request_id = state["request"]["id"]
            if state["admitted_tick"] is None:
                state["admitted_tick"] = tick
                state["status"] = "active"
                events.append(
                    {
                        "tick": tick,
                        "type": "admitted",
                        "request_id": request_id,
                        "policy": policy,
                    }
                )
            before = state["remaining"]
            state["remaining"] -= 1
            state["emitted"] += 1
            events.append(
                {
                    "tick": tick,
                    "type": "service_unit",
                    "request_id": request_id,
                    "unit_index": state["emitted"],
                    "remaining_before": before,
                    "policy": policy,
                }
            )
            if state["remaining"] == 0:
                state["status"] = "completed"
                state["completion_tick"] = tick + 1
                completion_order.append(request_id)
                completed_this_tick.append(state)
        for state in completed_this_tick:
            events.append(
                {
                    "tick": tick + 1,
                    "type": "completed",
                    "request_id": state["request"]["id"],
                    "policy": policy,
                }
            )
        tick += 1

    turnaround = {
        request_id: state["completion_tick"] - state["arrival"]
        for request_id, state in by_id.items()
    }
    waits = {
        request_id: turnaround[request_id] - state["request"]["work"]
        for request_id, state in by_id.items()
    }
    deadline_misses = sum(
        1
        for state in states
        if state["request"].get("deadline") is not None
        and state["completion_tick"] > state["request"]["deadline"]
    )
    total_work = sum(state["request"]["work"] for state in states)
    makespan = max(state["completion_tick"] for state in states) if states else 0
    metrics: dict[str, Any] = {
        "abstract_units": "work_units_and_ticks",
        "completed_work_units": total_work,
        "rejected_work_units": 0,
        "makespan_ticks": makespan,
        "deadline_misses": deadline_misses,
        "max_queue_depth": max_queue_depth,
        "servers": servers,
        "completion_order": completion_order,
        "throughput_work_per_tick": round(total_work / makespan, 6) if makespan else 0.0,
        "fairness_wait_gap_ticks": (max(waits.values()) - min(waits.values())) if waits else 0,
    }
    metrics.update(distribution(list(turnaround.values()), "turnaround_ticks"))
    metrics.update(distribution(list(waits.values()), "queue_wait_ticks"))
    trace = {
        "service_units_by_request": {
            request_id: [
                event["unit_index"]
                for event in events
                if event["type"] == "service_unit" and event["request_id"] == request_id
            ]
            for request_id in by_id
        }
    }
    state_summary = {
        request_id: {
            "status": state["status"],
            "emitted": state["emitted"],
            "completion_tick": state["completion_tick"],
        }
        for request_id, state in by_id.items()
    }
    return SimulationResult(
        kind="queue",
        scenario_name=scenario.name,
        scenario_hash=scenario.scenario_hash,
        policy=policy,
        config={"policy": policy, "servers": servers},
        events=events,
        metrics=metrics,
        trace=trace,
        state={"terminal": state_summary},
    )

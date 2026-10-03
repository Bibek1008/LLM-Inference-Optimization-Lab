"""Static, dynamic, and continuous prefill/decode batching models."""

from __future__ import annotations

from math import ceil
from typing import Any

from ..contracts import Scenario, SimulationResult
from ..metrics import distribution


def _request_states(scenario: Scenario) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    states: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(scenario.requests):
        state = {
            "request": request,
            "index": index,
            "prefill_remaining": request["prompt_tokens"],
            "decode_remaining": request["decode_tokens"],
            "prefill_done": 0,
            "decode_done": 0,
            "status": "waiting",
            "admitted_tick": None,
            "completion_tick": None,
        }
        states.append(state)
        by_id[request["id"]] = state
    return states, by_id


def _grouped_run(scenario: Scenario, config: dict[str, Any], wait_for_full: bool) -> SimulationResult:
    policy = str(config["policy"])
    batch_size = int(config.get("batch_size", scenario.parameters.get("batch_size", 2)))
    token_budget = int(config.get("token_budget", scenario.parameters.get("token_budget", 4)))
    slots = int(config.get("slots", scenario.parameters.get("slots", batch_size)))
    if batch_size < 1 or token_budget < 1 or slots < 1:
        raise ValueError("batch_size, token_budget, and slots must be positive")

    states, by_id = _request_states(scenario)
    events: list[dict[str, Any]] = []
    arrived: set[str] = set()
    completion_order: list[str] = []
    starts: dict[str, int] = {}
    max_queue_depth = 0
    tick = 0
    while len(completion_order) < len(states):
        for state in states:
            request_id = state["request"]["id"]
            if request_id not in arrived and state["request"]["arrival"] <= tick:
                arrived.add(request_id)
                events.append(
                    {"tick": tick, "type": "request_arrived", "request_id": request_id, "policy": policy}
                )
        ready = [state for state in states if state["status"] == "waiting" and state["request"]["arrival"] <= tick]
        max_queue_depth = max(max_queue_depth, len(ready))
        if not ready:
            tick += 1
            continue
        future_exists = any(
            state["status"] == "waiting" and state["request"]["arrival"] > tick for state in states
        )
        if wait_for_full and len(ready) < batch_size and future_exists:
            tick += 1
            continue
        group = ready[:batch_size]
        max_prompt = max(state["request"]["prompt_tokens"] for state in group)
        max_decode = max(state["request"]["decode_tokens"] for state in group)
        padded_work = len(group) * (max_prompt + max_decode) - sum(
            state["request"]["prompt_tokens"] + state["request"]["decode_tokens"] for state in group
        )
        batch_work = len(group) * (max_prompt + max_decode)
        duration = max(1, ceil(batch_work / token_budget))
        end_tick = tick + duration
        events.append(
            {
                "tick": tick,
                "type": "batch_started",
                "batch_ids": [state["request"]["id"] for state in group],
                "batch_size": len(group),
                "padded_work": padded_work,
                "policy": policy,
            }
        )
        for state in group:
            request_id = state["request"]["id"]
            state["status"] = "active"
            state["admitted_tick"] = tick
            starts[request_id] = tick
            events.append(
                {
                    "tick": tick,
                    "type": "admitted",
                    "request_id": request_id,
                    "active_count": len(group),
                    "policy": policy,
                }
            )
            prompt = state["request"]["prompt_tokens"]
            decode = state["request"]["decode_tokens"]
            state["prefill_done"] = prompt
            state["decode_done"] = decode
            events.append(
                {"tick": tick, "type": "prefill_work", "request_id": request_id, "units": prompt, "policy": policy}
            )
            events.append(
                {
                    "tick": tick,
                    "type": "decode_work",
                    "request_id": request_id,
                    "units": decode,
                    "token_indices": list(range(1, decode + 1)),
                    "policy": policy,
                }
            )
        for state in group:
            request_id = state["request"]["id"]
            state["status"] = "completed"
            state["completion_tick"] = end_tick
            completion_order.append(request_id)
            events.append(
                {"tick": end_tick, "type": "completed", "request_id": request_id, "policy": policy}
            )
        tick = end_tick

    return _finish_result(
        scenario,
        policy,
        {"policy": policy, "batch_size": batch_size, "token_budget": token_budget, "slots": slots},
        states,
        by_id,
        events,
        completion_order,
        max_queue_depth,
        padded_work_total=sum(
            event["padded_work"] for event in events if event["type"] == "batch_started"
        ),
        batch_count=sum(1 for event in events if event["type"] == "batch_started"),
        starts=starts,
    )


def _continuous_run(scenario: Scenario, config: dict[str, Any]) -> SimulationResult:
    policy = "continuous"
    slots = int(config.get("slots", scenario.parameters.get("slots", 2)))
    token_budget = int(config.get("token_budget", scenario.parameters.get("token_budget", 4)))
    prefill_chunk = int(config.get("prefill_chunk", scenario.parameters.get("prefill_chunk", 2)))
    if slots < 1 or token_budget < 1 or prefill_chunk < 1:
        raise ValueError("slots, token_budget, and prefill_chunk must be positive")
    states, by_id = _request_states(scenario)
    events: list[dict[str, Any]] = []
    arrived: set[str] = set()
    waiting: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    completion_order: list[str] = []
    max_queue_depth = 0
    max_active = 0
    tick = 0
    safety_limit = max(2000, sum(s["request"]["prompt_tokens"] + s["request"]["decode_tokens"] for s in states) * 4 + 200)
    while len(completion_order) < len(states):
        if tick > safety_limit:
            raise RuntimeError("continuous batching exceeded its deterministic safety bound")
        for state in states:
            request_id = state["request"]["id"]
            if request_id not in arrived and state["request"]["arrival"] <= tick:
                arrived.add(request_id)
                waiting.append(state)
                events.append(
                    {"tick": tick, "type": "request_arrived", "request_id": request_id, "policy": policy}
                )
        waiting.sort(key=lambda state: (state["request"]["arrival"], state["index"]))
        while waiting and len(active) < slots:
            state = waiting.pop(0)
            state["status"] = "active"
            state["admitted_tick"] = tick
            active.append(state)
            events.append(
                {
                    "tick": tick,
                    "type": "admitted",
                    "request_id": state["request"]["id"],
                    "active_count": len(active),
                    "policy": policy,
                }
            )
            if state["prefill_remaining"] == 0 and state["decode_remaining"] == 0:
                state["status"] = "completed"
                state["completion_tick"] = tick
                completion_order.append(state["request"]["id"])
                active.remove(state)
                events.append(
                    {"tick": tick, "type": "completed", "request_id": state["request"]["id"], "policy": policy}
                )
        max_queue_depth = max(max_queue_depth, len(waiting))
        max_active = max(max_active, len(active))
        budget = token_budget
        # Round-robin by stable admission order. Prefill chunks expose the
        # fairness trade-off while each decode unit remains exactly one token.
        completed_this_tick: list[dict[str, Any]] = []
        for state in list(active):
            if budget <= 0:
                break
            request_id = state["request"]["id"]
            if state["prefill_remaining"] > 0:
                units = min(prefill_chunk, state["prefill_remaining"], budget)
                state["prefill_remaining"] -= units
                state["prefill_done"] += units
                budget -= units
                events.append(
                    {"tick": tick, "type": "prefill_work", "request_id": request_id, "units": units, "policy": policy}
                )
            elif state["decode_remaining"] > 0:
                state["decode_remaining"] -= 1
                state["decode_done"] += 1
                budget -= 1
                events.append(
                    {
                        "tick": tick,
                        "type": "decode_token",
                        "request_id": request_id,
                        "token_index": state["decode_done"],
                        "policy": policy,
                    }
                )
            if state["prefill_remaining"] == 0 and state["decode_remaining"] == 0:
                state["status"] = "completing"
                state["completion_tick"] = tick + 1
                completion_order.append(request_id)
                active.remove(state)
                completed_this_tick.append(state)
        for state in completed_this_tick:
            state["status"] = "completed"
            events.append(
                {
                    "tick": tick + 1,
                    "type": "completed",
                    "request_id": state["request"]["id"],
                    "policy": policy,
                }
            )
        tick += 1

    return _finish_result(
        scenario,
        policy,
        {"policy": policy, "slots": slots, "token_budget": token_budget, "prefill_chunk": prefill_chunk},
        states,
        by_id,
        events,
        completion_order,
        max_queue_depth,
        padded_work_total=0,
        batch_count=0,
        starts={state["request"]["id"]: state["admitted_tick"] for state in states},
        max_active=max_active,
    )


def _finish_result(
    scenario: Scenario,
    policy: str,
    config: dict[str, Any],
    states: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    events: list[dict[str, Any]],
    completion_order: list[str],
    max_queue_depth: int,
    *,
    padded_work_total: int,
    batch_count: int,
    starts: dict[str, int],
    max_active: int = 0,
) -> SimulationResult:
    useful_work = sum(
        state["request"]["prompt_tokens"] + state["request"]["decode_tokens"] for state in states
    )
    waits = [starts[state["request"]["id"]] - state["request"]["arrival"] for state in states]
    turnarounds = [state["completion_tick"] - state["request"]["arrival"] for state in states]
    makespan = max((state["completion_tick"] or 0) for state in states) if states else 0
    metrics: dict[str, Any] = {
        "abstract_units": "prefill_decode_work_units_and_ticks",
        "useful_work": useful_work,
        "padded_work": padded_work_total,
        "makespan_ticks": makespan,
        "batch_count": batch_count,
        "max_queue_depth": max_queue_depth,
        "max_active_slots": max_active,
        "decode_tokens": sum(state["request"]["decode_tokens"] for state in states),
        "prefill_tokens": sum(state["request"]["prompt_tokens"] for state in states),
        "completion_order": completion_order,
        "throughput_work_per_tick": round(useful_work / makespan, 6) if makespan else 0.0,
    }
    metrics.update(distribution(waits, "queue_wait_ticks"))
    metrics.update(distribution(turnarounds, "turnaround_ticks"))
    trace = {
        "decode_tokens_by_request": {
            request_id: [
                event["token_index"]
                for event in events
                if event["type"] == "decode_token" and event["request_id"] == request_id
            ]
            or list(range(1, by_id[request_id]["request"]["decode_tokens"] + 1))
            for request_id in by_id
        }
    }
    state_summary = {
        request_id: {
            "status": state["status"],
            "prefill_done": state["prefill_done"],
            "decode_done": state["decode_done"],
            "admitted_tick": state["admitted_tick"],
            "completion_tick": state["completion_tick"],
        }
        for request_id, state in by_id.items()
    }
    return SimulationResult(
        kind="batching",
        scenario_name=scenario.name,
        scenario_hash=scenario.scenario_hash,
        policy=policy,
        config=config,
        events=events,
        metrics=metrics,
        trace=trace,
        state={"terminal": state_summary},
    )


def run(scenario: Scenario, config: dict[str, Any]) -> SimulationResult:
    policy = str(config.get("policy", "static"))
    if policy == "continuous":
        return _continuous_run(scenario, config)
    if policy == "static":
        return _grouped_run(scenario, {**config, "policy": policy}, wait_for_full=True)
    if policy == "dynamic":
        return _grouped_run(scenario, {**config, "policy": policy}, wait_for_full=False)
    raise ValueError(f"unknown batching policy: {policy}")

"""Paged KV-cache capacity and ownership simulation.

The cache stores abstract logical tokens in fixed-size pages. It deliberately
does not model bytes/second, VRAM, or a real attention kernel.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from ..contracts import Scenario, SimulationResult


def _first_free_pages(used: dict[int, str], capacity_pages: int, count: int) -> list[int] | None:
    free = [page for page in range(capacity_pages) if page not in used]
    return free[:count] if len(free) >= count else None


def _contiguous_pages(
    used: dict[int, str], capacity_pages: int, count: int
) -> list[int] | None:
    if count == 0:
        return []
    run: list[int] = []
    for page in range(capacity_pages):
        if page in used:
            run = []
        else:
            run.append(page)
            if len(run) == count:
                return run
    return None


def run(scenario: Scenario, config: dict[str, Any]) -> SimulationResult:
    policy = str(config.get("policy", "contiguous"))
    if policy not in {"contiguous", "paged"}:
        raise ValueError(f"unknown kv policy: {policy}")
    page_size = config.get("page_size", scenario.parameters.get("page_size", 4))
    capacity_tokens = config.get("capacity_tokens", scenario.parameters.get("capacity_tokens", 32))
    if not isinstance(page_size, int) or page_size < 1:
        raise ValueError("kv page_size must be a positive integer")
    if not isinstance(capacity_tokens, int) or capacity_tokens < page_size:
        raise ValueError("kv capacity_tokens must be >= page_size")
    capacity_pages = capacity_tokens // page_size

    states: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, request in enumerate(scenario.requests):
        prompt = request["prompt_tokens"]
        decode = request["decode_tokens"]
        state = {
            "request": request,
            "index": index,
            "status": "waiting",
            "emitted": 0,
            "logical_tokens": prompt,
            "pages": set(),
            "admitted_tick": None,
            "completion_tick": None,
        }
        states.append(state)
        by_id[request["id"]] = state

    used: dict[int, str] = {}
    events: list[dict[str, Any]] = []
    arrived: set[str] = set()
    completion_order: list[str] = []
    admission_stalls = 0
    admission_failures = 0
    growth_stalls = 0
    evictions = 0
    peak_resident_pages = 0
    peak_internal_waste = 0
    page_table_entries = 0
    preemptions = 0
    recomputations = 0
    max_preemptions = int(config.get("max_preemptions", scenario.parameters.get("max_preemptions", 100)))
    tick = 0
    total_requested_tokens = sum(
        request["prompt_tokens"] + request["decode_tokens"] for request in scenario.requests
    )
    safety_limit = max(2000, total_requested_tokens * 20 + 200)

    def release(state: dict[str, Any], release_tick: int, reason: str) -> None:
        nonlocal evictions
        request_id = state["request"]["id"]
        pages = sorted(state["pages"])
        for page in pages:
            owner = used.pop(page, None)
            if owner != request_id:
                raise RuntimeError(f"allocator ownership corruption for page {page}")
        state["pages"].clear()
        if reason == "evicted":
            evictions += 1
        events.append(
            {
                "tick": release_tick,
                "type": "cache_released",
                "request_id": request_id,
                "pages": pages,
                "reason": reason,
                "policy": policy,
            }
        )

    def allocate(state: dict[str, Any], count: int, allocation_tick: int, reason: str) -> bool:
        nonlocal page_table_entries
        if count == 0:
            pages: list[int] = []
        elif policy == "contiguous" and reason == "admission":
            pages = _contiguous_pages(used, capacity_pages, count) or []
            if len(pages) != count:
                return False
        else:
            pages = _first_free_pages(used, capacity_pages, count) or []
            if len(pages) != count:
                return False
        request_id = state["request"]["id"]
        for page in pages:
            if page in used:
                raise RuntimeError(f"allocator selected an occupied page: {page}")
            used[page] = request_id
            state["pages"].add(page)
        page_table_entries += len(pages)
        events.append(
            {
                "tick": allocation_tick,
                "type": "cache_allocated",
                "request_id": request_id,
                "pages": pages,
                "reason": reason,
                "policy": policy,
            }
        )
        return True

    while len(completion_order) < len(states):
        if tick > safety_limit:
            raise RuntimeError("kv simulator exceeded its deterministic safety bound")

        for state in states:
            request_id = state["request"]["id"]
            if request_id not in arrived and state["request"]["arrival"] <= tick:
                arrived.add(request_id)
                events.append(
                    {
                        "tick": tick,
                        "type": "request_arrived",
                        "request_id": request_id,
                        "policy": policy,
                    }
                )

        # Cancellation is explicit and releases all owned pages.
        for state in states:
            request = state["request"]
            if (
                state["status"] in {"waiting", "active"}
                and request["arrival"] <= tick
                and request.get("cancel_at") is not None
            ):
                if tick >= request["cancel_at"]:
                    if state["status"] == "active":
                        release(state, tick, "cancelled")
                    state["status"] = "cancelled"
                    state["completion_tick"] = tick
                    completion_order.append(request["id"])
                    events.append(
                        {
                            "tick": tick,
                            "type": "cancelled",
                            "request_id": request["id"],
                            "policy": policy,
                        }
                    )

        # Admission is FIFO. The policy difference is whether a full logical
        # sequence is reserved now or pages are acquired as decode grows.
        for state in sorted(
            [
                candidate
                for candidate in states
                if candidate["status"] == "waiting" and candidate["request"]["arrival"] <= tick
            ],
            key=lambda candidate: (candidate["request"]["arrival"], candidate["index"]),
        ):
            request = state["request"]
            request_id = request["id"]
            total_tokens = request["prompt_tokens"] + request["decode_tokens"]
            if total_tokens > capacity_tokens:
                state["status"] = "rejected"
                state["completion_tick"] = tick
                completion_order.append(request_id)
                admission_failures += 1
                events.append(
                    {
                        "tick": tick,
                        "type": "rejected",
                        "request_id": request_id,
                        "reason": "request_exceeds_capacity",
                        "policy": policy,
                    }
                )
                continue
            initial_tokens = (
                total_tokens
                if policy == "contiguous"
                else (state["logical_tokens"] if state.get("preempted") else request["prompt_tokens"])
            )
            pages_needed = ceil(initial_tokens / page_size)
            if not allocate(state, pages_needed, tick, "admission"):
                admission_stalls += 1
                events.append(
                    {
                        "tick": tick,
                        "type": "admission_wait",
                        "request_id": request_id,
                        "needed_pages": pages_needed,
                        "policy": policy,
                    }
                )
                continue
            state["status"] = "active"
            state["preempted"] = False
            state["admitted_tick"] = tick
            events.append(
                {
                    "tick": tick,
                    "type": "admitted",
                    "request_id": request_id,
                    "pages": sorted(state["pages"]),
                    "policy": policy,
                }
            )
            if request["decode_tokens"] == 0:
                state["status"] = "completed"
                state["completion_tick"] = tick
                completion_order.append(request_id)
                release(state, tick, "completed")
                events.append(
                    {"tick": tick, "type": "completed", "request_id": request_id, "policy": policy}
                )

        # One abstract decode token per active request per tick.
        completed_this_tick: list[dict[str, Any]] = []
        for state in sorted(
            [candidate for candidate in states if candidate["status"] == "active"],
            key=lambda candidate: (candidate["admitted_tick"], candidate["index"]),
        ):
            if state["status"] != "active":
                continue
            request = state["request"]
            if state["emitted"] >= request["decode_tokens"]:
                continue
            next_logical_tokens = request["prompt_tokens"] + state["emitted"] + 1
            capacity_in_pages = len(state["pages"]) * page_size
            if next_logical_tokens > capacity_in_pages:
                if not allocate(state, 1, tick, "growth"):
                    growth_stalls += 1
                    events.append(
                        {
                            "tick": tick,
                            "type": "growth_stall",
                            "request_id": request["id"],
                            "needed_logical_tokens": next_logical_tokens,
                            "policy": policy,
                        }
                    )
                    if policy == "paged" and preemptions < max_preemptions:
                        victims = [
                            candidate
                            for candidate in states
                            if candidate["status"] == "active" and candidate["pages"]
                        ]
                        if victims:
                            victim = max(
                                victims,
                                key=lambda candidate: (
                                    candidate["request"]["decode_tokens"] - candidate["emitted"],
                                    len(candidate["pages"]),
                                    -candidate["index"],
                                ),
                            )
                            release(victim, tick, "preempted")
                            victim["status"] = "waiting"
                            victim["preempted"] = True
                            preemptions += 1
                            recomputations += victim["emitted"]
                            events.append(
                                {
                                    "tick": tick,
                                    "type": "preempted",
                                    "request_id": victim["request"]["id"],
                                    "recomputed_tokens": victim["emitted"],
                                    "policy": policy,
                                }
                            )
                    continue
            state["emitted"] += 1
            state["logical_tokens"] = request["prompt_tokens"] + state["emitted"]
            events.append(
                {
                    "tick": tick,
                    "type": "decode_token",
                    "request_id": request["id"],
                    "token_index": state["emitted"],
                    "logical_tokens": state["logical_tokens"],
                    "pages": sorted(state["pages"]),
                    "policy": policy,
                }
            )
            if state["emitted"] == request["decode_tokens"]:
                state["status"] = "completing"
                state["completion_tick"] = tick + 1
                completion_order.append(request["id"])
                completed_this_tick.append(state)

        # Release after all current-tick decode events so the append-only trace
        # remains ordered and a request cannot consume a page freed in the
        # middle of the same abstract tick.
        for state in completed_this_tick:
            state["status"] = "completed"
            release(state, tick + 1, "completed")
            events.append(
                {
                    "tick": tick + 1,
                    "type": "completed",
                    "request_id": state["request"]["id"],
                    "policy": policy,
                }
            )

        resident_pages = len(used)
        peak_resident_pages = max(peak_resident_pages, resident_pages)
        internal_waste = sum(
            len(state["pages"]) * page_size - state["logical_tokens"]
            for state in states
            if state["status"] == "active"
        )
        peak_internal_waste = max(peak_internal_waste, internal_waste)
        tick += 1

    makespan = max((state["completion_tick"] or 0) for state in states) if states else 0
    metrics: dict[str, Any] = {
        "abstract_units": "logical_tokens_and_pages",
        "capacity_tokens": capacity_pages * page_size,
        "capacity_pages": capacity_pages,
        "page_size_tokens": page_size,
        "theoretical_requested_tokens": total_requested_tokens,
        "peak_resident_pages": peak_resident_pages,
        "peak_internal_waste_tokens": peak_internal_waste,
        "page_table_entries": page_table_entries,
        "admission_stalls": admission_stalls,
        "admission_failures": admission_failures,
        "growth_stalls": growth_stalls,
        "evictions": evictions,
        "recomputations": recomputations,
        "preemptions": preemptions,
        "makespan_ticks": makespan,
        "completion_order": completion_order,
    }
    trace = {
        "decode_tokens_by_request": {
            request_id: [
                event["token_index"]
                for event in events
                if event["type"] == "decode_token" and event["request_id"] == request_id
            ]
            for request_id in by_id
        }
    }
    state_summary = {
        request_id: {
            "status": state["status"],
            "emitted": state["emitted"],
            "logical_tokens": state["logical_tokens"],
            "pages_at_end": sorted(state["pages"]),
            "completion_tick": state["completion_tick"],
        }
        for request_id, state in by_id.items()
    }
    return SimulationResult(
        kind="kv",
        scenario_name=scenario.name,
        scenario_hash=scenario.scenario_hash,
        policy=policy,
        config={"policy": policy, "page_size": page_size, "capacity_tokens": capacity_pages * page_size},
        events=events,
        metrics=metrics,
        trace=trace,
        state={"terminal": state_summary, "free_pages_at_end": sorted(set(range(capacity_pages)) - set(used))},
    )

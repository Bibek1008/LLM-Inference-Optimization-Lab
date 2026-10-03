"""Independent correctness checks for simulation traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .contracts import Scenario, SimulationResult


def _check(condition: bool, checks: dict[str, bool], failures: list[str], name: str, detail: str) -> None:
    checks[name] = condition
    if not condition:
        failures.append(detail)


def _common_checks(
    scenario: Scenario, result: SimulationResult, checks: dict[str, bool], failures: list[str]
) -> dict[str, list[dict[str, Any]]]:
    expected_ids = [request["id"] for request in scenario.requests]
    expected_set = set(expected_ids)
    events_by_request: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ticks = []
    for event in result.events:
        if not isinstance(event, dict):
            failures.append("event is not an object")
            continue
        request_id = event.get("request_id")
        if request_id is None and event.get("type") == "batch_started":
            tick = event.get("tick")
            if isinstance(tick, int) and tick >= 0:
                ticks.append(tick)
            else:
                failures.append("batch_started has invalid tick")
            continue
        if request_id not in expected_set:
            failures.append(f"event has unknown request_id: {request_id!r}")
            continue
        events_by_request[request_id].append(event)
        tick = event.get("tick")
        if not isinstance(tick, int) or tick < 0:
            failures.append(f"event for {request_id} has invalid tick")
        else:
            ticks.append(tick)
    _check(
        result.scenario_hash == scenario.scenario_hash,
        checks,
        failures,
        "scenario_hash_matches",
        "result scenario hash does not match the input scenario",
    )
    _check(
        ticks == sorted(ticks),
        checks,
        failures,
        "stable_event_tick_order",
        "event ticks are not non-decreasing",
    )
    arrivals = Counter(
        request_id for request_id, events in events_by_request.items() for event in events if event.get("type") == "request_arrived"
    )
    _check(
        all(arrivals[request_id] == 1 for request_id in expected_ids),
        checks,
        failures,
        "one_arrival_each",
        "each request must have exactly one request_arrived event",
    )
    terminals = Counter(
        request_id
        for request_id, events in events_by_request.items()
        for event in events
        if event.get("type") in {"completed", "rejected", "cancelled"}
    )
    _check(
        all(terminals[request_id] == 1 for request_id in expected_ids),
        checks,
        failures,
        "one_terminal_event_each",
        "each request must have exactly one terminal event",
    )
    return events_by_request


def _validate_queue(
    scenario: Scenario,
    result: SimulationResult,
    events_by_request: dict[str, list[dict[str, Any]]],
    checks: dict[str, bool],
    failures: list[str],
) -> None:
    servers = int(result.config.get("servers", scenario.parameters.get("servers", 1)))
    by_id = {request["id"]: request for request in scenario.requests}
    service_by_tick = Counter()
    exact_work = True
    no_early_service = True
    for request_id, events in events_by_request.items():
        units = [event for event in events if event.get("type") == "service_unit"]
        indices = [event.get("unit_index") for event in units]
        expected = list(range(1, by_id[request_id]["work"] + 1))
        exact_work = exact_work and indices == expected
        no_early_service = no_early_service and all(
            event.get("tick", -1) >= by_id[request_id]["arrival"] for event in units
        )
        for event in units:
            service_by_tick[event["tick"]] += 1
    _check(exact_work, checks, failures, "exact_work_trace", "queue emitted work units are missing, duplicated, or reordered")
    _check(no_early_service, checks, failures, "no_service_before_arrival", "queue served a request before arrival")
    _check(
        all(count <= servers for count in service_by_tick.values()),
        checks,
        failures,
        "server_capacity",
        "queue served more requests than its declared server capacity",
    )


def _validate_kv(
    scenario: Scenario,
    result: SimulationResult,
    events_by_request: dict[str, list[dict[str, Any]]],
    checks: dict[str, bool],
    failures: list[str],
) -> None:
    page_size = int(result.config["page_size"])
    capacity_tokens = int(result.config["capacity_tokens"])
    capacity_pages = capacity_tokens // page_size
    by_id = {request["id"]: request for request in scenario.requests}
    owners: dict[int, str] = {}
    ownership_ok = True
    capacity_ok = True
    token_trace_ok = True
    for event in result.events:
        event_type = event.get("type")
        if event_type == "cache_allocated":
            request_id = event["request_id"]
            pages = event.get("pages", [])
            for page in pages:
                if not isinstance(page, int) or not 0 <= page < capacity_pages or page in owners:
                    ownership_ok = False
                else:
                    owners[page] = request_id
            capacity_ok = capacity_ok and len(owners) <= capacity_pages
        elif event_type == "cache_released":
            request_id = event["request_id"]
            for page in event.get("pages", []):
                if owners.get(page) != request_id:
                    ownership_ok = False
                else:
                    del owners[page]
        elif event_type == "decode_token":
            request_id = event["request_id"]
            token_index = event.get("token_index")
            token_trace_ok = token_trace_ok and event.get("logical_tokens") == by_id[request_id]["prompt_tokens"] + token_index
    for request_id, events in events_by_request.items():
        indices = [event.get("token_index") for event in events if event.get("type") == "decode_token"]
        token_trace_ok = token_trace_ok and indices == list(range(1, len(indices) + 1))
    _check(ownership_ok, checks, failures, "unique_page_ownership", "a page was shared or released by the wrong request")
    _check(capacity_ok, checks, failures, "page_capacity", "allocated pages exceeded declared capacity")
    _check(token_trace_ok, checks, failures, "ordered_decode_trace", "KV decode token order or logical length is invalid")
    _check(not owners, checks, failures, "all_pages_released", "KV pages remained allocated at the end of the run")
    for request_id, events in events_by_request.items():
        emitted = sum(1 for event in events if event.get("type") == "decode_token")
        terminal_types = {event.get("type") for event in events if event.get("type") in {"completed", "cancelled", "rejected"}}
        if terminal_types == {"completed"}:
            _check(
                emitted == by_id[request_id]["decode_tokens"],
                checks,
                failures,
                f"decode_count_{request_id}",
                f"completed KV request {request_id} emitted the wrong number of decode tokens",
            )


def _validate_batching(
    scenario: Scenario,
    result: SimulationResult,
    events_by_request: dict[str, list[dict[str, Any]]],
    checks: dict[str, bool],
    failures: list[str],
) -> None:
    slots = int(result.config.get("slots", scenario.parameters.get("slots", 1)))
    by_id = {request["id"]: request for request in scenario.requests}
    prefill_counts: Counter[str] = Counter()
    decode_counts: Counter[str] = Counter()
    token_indices: dict[str, list[int]] = defaultdict(list)
    for event in result.events:
        if event.get("type") == "admitted":
            _check(
                event.get("active_count", 0) <= slots,
                checks,
                failures,
                "batch_slot_capacity",
                "batch admitted more active requests than slots",
            )
        if event.get("type") == "prefill_work":
            prefill_counts[event["request_id"]] += event.get("units", 0)
        if event.get("type") == "decode_work":
            decode_counts[event["request_id"]] += event.get("units", 0)
            token_indices[event["request_id"]].extend(event.get("token_indices", []))
        if event.get("type") == "decode_token":
            decode_counts[event["request_id"]] += 1
            token_indices[event["request_id"]].append(event.get("token_index"))
    for request_id, request in by_id.items():
        _check(
            prefill_counts[request_id] == request["prompt_tokens"],
            checks,
            failures,
            f"prefill_count_{request_id}",
            f"request {request_id} did not process exactly its prompt work",
        )
        _check(
            decode_counts[request_id] == request["decode_tokens"],
            checks,
            failures,
            f"decode_count_{request_id}",
            f"request {request_id} did not emit exactly its decode work",
        )
        _check(
            token_indices[request_id] == list(range(1, request["decode_tokens"] + 1)),
            checks,
            failures,
            f"decode_order_{request_id}",
            f"request {request_id} decode token order is invalid",
        )
        all_work_events = [event for event in events_by_request[request_id] if event.get("type") in {"prefill_work", "decode_work", "decode_token"}]
        _check(
            all(event.get("tick", -1) >= request["arrival"] for event in all_work_events),
            checks,
            failures,
            f"no_work_before_arrival_{request_id}",
            f"request {request_id} received work before arrival",
        )


def validate_result(scenario: Scenario, result: SimulationResult) -> dict[str, Any]:
    """Return a structured report without trusting policy-reported metrics."""

    checks: dict[str, bool] = {}
    failures: list[str] = []
    events_by_request = _common_checks(scenario, result, checks, failures)
    if scenario.kind == "queue":
        _validate_queue(scenario, result, events_by_request, checks, failures)
    elif scenario.kind == "kv":
        _validate_kv(scenario, result, events_by_request, checks, failures)
    elif scenario.kind == "batching":
        _validate_batching(scenario, result, events_by_request, checks, failures)
    else:
        failures.append(f"no validator for kind {scenario.kind}")
    return {"passed": not failures, "checks": checks, "failures": failures}

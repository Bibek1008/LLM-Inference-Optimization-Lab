"""Validated JSON scenario loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contracts import Scenario
from .deterministic import sha256_file, sha256_json


class ScenarioError(ValueError):
    """Raised when a scenario violates the public fixture contract."""


_ROOT_FIELDS = {
    "schema_version",
    "kind",
    "name",
    "version",
    "seed",
    "parameters",
    "requests",
    "search_space",
    "description",
}

_REQUEST_FIELDS = {
    "queue": {"id", "arrival", "work", "class", "deadline"},
    "kv": {"id", "arrival", "prompt_tokens", "decode_tokens", "cancel_at"},
    "batching": {"id", "arrival", "prompt_tokens", "decode_tokens", "class", "deadline"},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScenarioError(message)


def _validate_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    _require(value >= minimum, f"{label} must be >= {minimum}")
    return value


def load_scenario(path: str | Path, expected_kind: str | None = None) -> Scenario:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioError(f"cannot read scenario {path}: {exc}") from exc

    _require(isinstance(document, dict), "scenario root must be an object")
    unknown = set(document) - _ROOT_FIELDS
    _require(not unknown, f"unknown scenario fields: {sorted(unknown)}")
    _require(document.get("schema_version") == 1, "schema_version must be 1")
    kind = document.get("kind")
    _require(kind in _REQUEST_FIELDS, f"unsupported kind: {kind!r}")
    if expected_kind is not None:
        _require(kind == expected_kind, f"scenario kind {kind!r} does not match {expected_kind!r}")
    name = document.get("name")
    version = document.get("version")
    _require(isinstance(name, str) and name, "name must be a non-empty string")
    _require(isinstance(version, str) and version, "version must be a non-empty string")
    seed = _validate_int(document.get("seed"), "seed")
    parameters = document.get("parameters", {})
    _require(isinstance(parameters, dict), "parameters must be an object")
    requests = document.get("requests")
    _require(isinstance(requests, list), "requests must be a list")

    seen: set[str] = set()
    normalized_requests: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        _require(isinstance(request, dict), f"request {index} must be an object")
        unknown_request_fields = set(request) - _REQUEST_FIELDS[kind]
        _require(
            not unknown_request_fields,
            f"unknown {kind} request fields at index {index}: {sorted(unknown_request_fields)}",
        )
        request_id = request.get("id")
        _require(isinstance(request_id, str) and request_id, f"request {index} id must be non-empty")
        _require(request_id not in seen, f"duplicate request id: {request_id}")
        seen.add(request_id)
        normalized = dict(request)
        _validate_int(request.get("arrival"), f"request {request_id}.arrival")
        if kind == "queue":
            _validate_int(request.get("work"), f"request {request_id}.work", minimum=1)
            if "deadline" in request:
                _validate_int(request["deadline"], f"request {request_id}.deadline")
        elif kind == "kv":
            _validate_int(request.get("prompt_tokens"), f"request {request_id}.prompt_tokens")
            _validate_int(request.get("decode_tokens"), f"request {request_id}.decode_tokens")
            if "cancel_at" in request and request["cancel_at"] is not None:
                _validate_int(request["cancel_at"], f"request {request_id}.cancel_at")
        else:
            _validate_int(request.get("prompt_tokens"), f"request {request_id}.prompt_tokens")
            _validate_int(request.get("decode_tokens"), f"request {request_id}.decode_tokens")
            if "deadline" in request:
                _validate_int(request["deadline"], f"request {request_id}.deadline")
        normalized_requests.append(normalized)

    search_space = document.get("search_space", {})
    _require(isinstance(search_space, dict), "search_space must be an object")
    normalized_search: dict[str, tuple[Any, ...]] = {}
    for field, values in search_space.items():
        _require(isinstance(field, str) and field, "search_space keys must be strings")
        _require(isinstance(values, list) and values, f"search_space[{field!r}] must be non-empty list")
        normalized_search[field] = tuple(values)

    normalized_document = dict(document)
    normalized_document["requests"] = normalized_requests
    normalized_document["search_space"] = {key: list(value) for key, value in normalized_search.items()}
    normalized_document.setdefault("parameters", {})
    normalized_document.setdefault("description", "")
    return Scenario(
        kind=kind,
        name=name,
        version=version,
        seed=seed,
        parameters=parameters,
        requests=tuple(normalized_requests),
        search_space=normalized_search,
        scenario_hash=sha256_json(normalized_document),
        source_hash=sha256_file(path),
        description=normalized_document["description"],
    )

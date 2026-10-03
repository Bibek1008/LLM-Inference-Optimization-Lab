"""Fail-closed validation for the public receipt schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..deterministic import sha256_json


def validate_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    required = {
        "schema_version",
        "receipt_type",
        "measurement_kind",
        "not_a_benchmark",
        "scenario",
        "policy",
        "baseline",
        "candidate",
        "baseline_metrics",
        "candidate_metrics",
        "correctness_result",
        "receipt_hash",
    }
    failures.extend(f"missing field: {field}" for field in sorted(required - set(receipt)))
    if receipt.get("schema_version") != 1:
        failures.append("schema_version must be 1")
    if receipt.get("receipt_type") != "simulation":
        failures.append("receipt_type must be simulation")
    if receipt.get("measurement_kind") != "simulation":
        failures.append("measurement_kind must be simulation")
    if receipt.get("not_a_benchmark") is not True:
        failures.append("simulation receipts must set not_a_benchmark=true")
    scenario = receipt.get("scenario", {})
    if not isinstance(scenario, dict) or not scenario.get("scenario_hash"):
        failures.append("scenario.scenario_hash is required")
    for label in ("baseline", "candidate"):
        record = receipt.get(label)
        if not isinstance(record, dict):
            failures.append(f"{label} must be an object")
            continue
        for field in ("policy", "config", "metrics", "correctness", "result_hash", "events"):
            if field not in record:
                failures.append(f"{label}.{field} is required")
        if not isinstance(record.get("metrics"), dict):
            failures.append(f"{label}.metrics must be an object")
        if not isinstance(record.get("correctness"), dict):
            failures.append(f"{label}.correctness must be an object")
    correctness = receipt.get("correctness_result")
    if not isinstance(correctness, dict) or not isinstance(correctness.get("passed"), bool):
        failures.append("correctness_result.passed must be boolean")
    recorded_hash = receipt.get("receipt_hash")
    if recorded_hash:
        without_hash = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        if recorded_hash != sha256_json(without_hash):
            failures.append("receipt_hash does not match receipt contents")
    return {"valid": not failures, "failures": failures}


def validate_receipt_file(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "failures": [f"cannot read receipt: {exc}"]}
    if not isinstance(value, dict):
        return {"valid": False, "failures": ["receipt root must be an object"]}
    report = validate_receipt(value)
    report["path"] = str(path)
    return report

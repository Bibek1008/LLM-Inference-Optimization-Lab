"""Write explicit, content-hashed simulation receipts."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

from ..contracts import Scenario, SimulationResult
from ..deterministic import sha256_json, utc_now


def _result_record(result: SimulationResult, correctness: dict[str, Any]) -> dict[str, Any]:
    return {
        "policy": result.policy,
        "config": result.config,
        "metrics": result.metrics,
        "correctness": correctness,
        "result_hash": result.result_hash,
        "event_count": len(result.events),
        "events": result.events,
        "trace": result.trace,
    }


def build_receipt(
    scenario: Scenario,
    baseline: SimulationResult,
    candidate: SimulationResult,
    baseline_correctness: dict[str, Any],
    candidate_correctness: dict[str, Any],
    *,
    command: list[str],
    held_out: dict[str, Any] | None = None,
    search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "simulation",
        "run_id": f"{scenario.name}-{candidate.policy}",
        "created_utc": utc_now(),
        "measurement_kind": "simulation",
        "not_a_benchmark": True,
        "claim_level": "Observed under a declared simulation model",
        "scenario": {
            "kind": scenario.kind,
            "name": scenario.name,
            "version": scenario.version,
            "seed": scenario.seed,
            "scenario_hash": scenario.scenario_hash,
            "source_file_sha256": scenario.source_hash,
        },
        "policy": {"name": candidate.policy, "config": candidate.config},
        "baseline": _result_record(baseline, baseline_correctness),
        "candidate": _result_record(candidate, candidate_correctness),
        "baseline_metrics": baseline.metrics,
        "candidate_metrics": candidate.metrics,
        "correctness_result": {
            "baseline_passed": baseline_correctness["passed"],
            "candidate_passed": candidate_correctness["passed"],
            "passed": baseline_correctness["passed"] and candidate_correctness["passed"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "command": command,
        "limitations": [
            "All values are deterministic simulation units, not GPU measurements.",
            "No real model, CUDA runtime, NVIDIA GPU, or production traffic was used.",
        ],
    }
    if held_out is not None:
        receipt["held_out"] = held_out
    if search is not None:
        receipt["search"] = search
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def write_receipt(receipt: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size:
        raise FileExistsError(f"refusing to overwrite non-empty receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

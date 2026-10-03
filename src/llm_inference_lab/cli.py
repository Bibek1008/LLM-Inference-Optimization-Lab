"""Command-line interface for running and validating the teaching lab."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .optimize.search import objective, search_space_configurations
from .receipts.validate import validate_receipt_file
from .receipts.write import build_receipt, write_receipt
from .runner import baseline_config, run_scenario
from .scenario_io import ScenarioError, load_scenario
from .validation import validate_result


KINDS = ("queue", "kv", "batching")


def _config_from_argument(raw: str | None) -> dict[str, Any]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--config-json must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--config-json must contain an object")
    return value


def _emit(receipt: dict[str, Any], out: str | None) -> None:
    if out is None or out == "-":
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return
    path = write_receipt(receipt, out)
    print(json.dumps({"written": str(path), "receipt_hash": receipt["receipt_hash"]}, sort_keys=True))


def _simulate(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, args.kind)
    base_config = baseline_config(args.kind, scenario)
    candidate_config = dict(base_config) if args.policy == "baseline" else {"policy": args.policy}
    candidate_config.update(_config_from_argument(args.config_json))
    baseline = run_scenario(args.kind, scenario, base_config)
    candidate = run_scenario(args.kind, scenario, candidate_config)
    baseline_correctness = validate_result(scenario, baseline)
    candidate_correctness = validate_result(scenario, candidate)
    receipt = build_receipt(
        scenario,
        baseline,
        candidate,
        baseline_correctness,
        candidate_correctness,
        command=sys.argv[:],
    )
    _emit(receipt, args.out)
    return 0 if receipt["correctness_result"]["passed"] else 1


def _trial_record(config: dict[str, Any], result: Any, correctness: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "config": config,
        "status": "valid" if correctness["passed"] else "rejected",
        "objective": list(objective(kind, result.metrics)) if correctness["passed"] else None,
        "metrics": result.metrics,
        "correctness": correctness,
        "result_hash": result.result_hash,
    }


def _search(args: argparse.Namespace) -> int:
    train = load_scenario(args.scenario, args.kind)
    held_out = load_scenario(args.held_out, args.kind)
    configurations = search_space_configurations(train.search_space, args.max_candidates)
    if not configurations:
        raise ValueError("scenario search_space must contain at least one candidate")

    base_config = baseline_config(args.kind, train)
    baseline = run_scenario(args.kind, train, base_config)
    baseline_correctness = validate_result(train, baseline)
    if not baseline_correctness["passed"]:
        raise RuntimeError(f"baseline correctness failed: {baseline_correctness['failures']}")

    trials: list[dict[str, Any]] = []
    valid_candidates: list[tuple[tuple[float, ...], dict[str, Any], Any, dict[str, Any]]] = []
    for config in configurations:
        candidate = run_scenario(args.kind, train, config)
        correctness = validate_result(train, candidate)
        record = _trial_record(config, candidate, correctness, args.kind)
        trials.append(record)
        if correctness["passed"]:
            valid_candidates.append((objective(args.kind, candidate.metrics), config, candidate, correctness))
    if not valid_candidates:
        raise RuntimeError("all declared candidate configurations failed independent correctness checks")
    selected_objective, selected_config, selected, selected_correctness = min(
        valid_candidates, key=lambda item: (item[0], json.dumps(item[1], sort_keys=True))
    )

    held_base_config = baseline_config(args.kind, held_out)
    held_baseline = run_scenario(args.kind, held_out, held_base_config)
    held_candidate = run_scenario(args.kind, held_out, selected_config)
    held_baseline_correctness = validate_result(held_out, held_baseline)
    held_candidate_correctness = validate_result(held_out, held_candidate)
    held_out_record = {
        "scenario": {
            "name": held_out.name,
            "seed": held_out.seed,
            "scenario_hash": held_out.scenario_hash,
            "source_file_sha256": held_out.source_hash,
        },
        "candidate_config": selected_config,
        "baseline_metrics": held_baseline.metrics,
        "candidate_metrics": held_candidate.metrics,
        "baseline_correctness": held_baseline_correctness,
        "candidate_correctness": held_candidate_correctness,
        "baseline_result_hash": held_baseline.result_hash,
        "candidate_result_hash": held_candidate.result_hash,
        "candidate_objective": list(objective(args.kind, held_candidate.metrics)),
    }
    search_record = {
        "method": "exhaustive_grid",
        "selection_objective": list(selected_objective),
        "declared_search_space": {key: list(value) for key, value in train.search_space.items()},
        "candidate_count": len(configurations),
        "valid_count": len(valid_candidates),
        "trials": trials,
        "held_out_is_not_used_for_selection": True,
    }
    receipt = build_receipt(
        train,
        baseline,
        selected,
        baseline_correctness,
        selected_correctness,
        command=sys.argv[:],
        held_out=held_out_record,
        search=search_record,
    )
    _emit(receipt, args.out)
    return 0 if held_candidate_correctness["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_inference_lab",
        description="Run deterministic CPU-only inference optimization simulations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="run one baseline/candidate comparison")
    simulate.add_argument("kind", choices=KINDS)
    simulate.add_argument("--scenario", required=True)
    simulate.add_argument("--policy", default="baseline", help="named policy, or baseline")
    simulate.add_argument("--config-json", help="JSON object of additional policy/config fields")
    simulate.add_argument("--out", help="receipt path; use - or omit for stdout")
    simulate.set_defaults(handler=_simulate)

    search = subparsers.add_parser("search", help="search a scenario's declared bounded policy space")
    search.add_argument("kind", choices=KINDS)
    search.add_argument("--scenario", required=True, help="development/training scenario")
    search.add_argument("--held-out", required=True, help="separate held-out scenario")
    search.add_argument("--max-candidates", type=int, default=1000)
    search.add_argument("--out", help="receipt path; use - or omit for stdout")
    search.set_defaults(handler=_search)

    validate = subparsers.add_parser("validate-receipt", help="validate a receipt without rerunning it")
    validate.add_argument("path")
    validate.set_defaults(handler=lambda args: _validate(args))
    return parser


def _validate(args: argparse.Namespace) -> int:
    report = validate_receipt_file(args.path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ScenarioError, ValueError, RuntimeError, FileExistsError) as exc:
        parser.error(str(exc))
        return 2

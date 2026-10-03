from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_inference_lab.deterministic import sha256_json
from llm_inference_lab.optimize.search import objective, search_space_configurations
from llm_inference_lab.receipts.validate import validate_receipt_file
from llm_inference_lab.receipts.write import build_receipt, write_receipt
from llm_inference_lab.runner import baseline_config, run_scenario
from llm_inference_lab.scenario_io import ScenarioError, load_scenario
from llm_inference_lab.validation import validate_result


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_CASES = {
    "queue": (ROOT / "scenarios/queue_train.json", ROOT / "scenarios/queue_heldout.json"),
    "kv": (ROOT / "scenarios/kv_train.json", ROOT / "scenarios/kv_heldout.json"),
    "batching": (ROOT / "scenarios/batching_train.json", ROOT / "scenarios/batching_heldout.json"),
}


class LabTests(unittest.TestCase):
    def test_all_checked_in_scenarios_load(self) -> None:
        for kind, paths in SCENARIO_CASES.items():
            for path in paths:
                scenario = load_scenario(path, kind)
                self.assertEqual(scenario.kind, kind)
                self.assertTrue(scenario.scenario_hash)
                self.assertTrue(scenario.source_hash)

    def test_unknown_scenario_fields_fail_closed(self) -> None:
        path = ROOT / "scenarios/queue_train.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        document["unexpected"] = True
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(document, handle)
            temporary = Path(handle.name)
        try:
            with self.assertRaises(ScenarioError):
                load_scenario(temporary, "queue")
        finally:
            temporary.unlink()

    def test_every_declared_candidate_is_deterministic_and_correct(self) -> None:
        for kind, (train_path, heldout_path) in SCENARIO_CASES.items():
            for path in (train_path, heldout_path):
                scenario = load_scenario(path, kind)
                configs = search_space_configurations(scenario.search_space, max_candidates=100)
                if not configs:
                    configs = [baseline_config(kind, scenario)]
                for config in configs:
                    first = run_scenario(kind, scenario, config)
                    second = run_scenario(kind, scenario, config)
                    self.assertEqual(first.as_dict(), second.as_dict(), (kind, path.name, config))
                    report = validate_result(scenario, first)
                    self.assertTrue(report["passed"], (kind, path.name, config, report))

    def test_objectives_are_declared_and_numeric(self) -> None:
        for kind, (path, _) in SCENARIO_CASES.items():
            scenario = load_scenario(path, kind)
            result = run_scenario(kind, scenario, baseline_config(kind, scenario))
            value = objective(kind, result.metrics)
            self.assertTrue(value)
            self.assertTrue(all(isinstance(item, float) for item in value))

    def test_receipt_round_trip_and_hash(self) -> None:
        scenario = load_scenario(SCENARIO_CASES["queue"][0], "queue")
        base = run_scenario("queue", scenario, baseline_config("queue", scenario))
        candidate = run_scenario("queue", scenario, {"policy": "interactive_first"})
        receipt = build_receipt(
            scenario,
            base,
            candidate,
            validate_result(scenario, base),
            validate_result(scenario, candidate),
            command=["test"],
        )
        self.assertEqual(receipt["measurement_kind"], "simulation")
        self.assertEqual(receipt["receipt_hash"], sha256_json({k: v for k, v in receipt.items() if k != "receipt_hash"}))
        with tempfile.TemporaryDirectory() as directory:
            path = write_receipt(receipt, Path(directory) / "receipt.json")
            report = validate_receipt_file(path)
            self.assertTrue(report["valid"], report)
            altered = json.loads(path.read_text(encoding="utf-8"))
            altered["candidate_metrics"]["makespan_ticks"] += 1
            path.write_text(json.dumps(altered), encoding="utf-8")
            self.assertFalse(validate_receipt_file(path)["valid"])

    def test_cli_works_from_clean_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "queue.json"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT / "src")
            command = [
                sys.executable,
                "-m",
                "llm_inference_lab",
                "simulate",
                "queue",
                "--scenario",
                str(SCENARIO_CASES["queue"][0]),
                "--policy",
                "shortest",
                "--out",
                str(output),
            ]
            completed = subprocess.run(
                command,
                cwd=directory,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(validate_receipt_file(output)["valid"])

    def test_search_space_has_bounded_count(self) -> None:
        scenario = load_scenario(SCENARIO_CASES["batching"][0], "batching")
        configurations = search_space_configurations(scenario.search_space, max_candidates=100)
        self.assertEqual(len(configurations), 18)
        with self.assertRaises(ValueError):
            search_space_configurations(scenario.search_space, max_candidates=17)


if __name__ == "__main__":
    unittest.main()

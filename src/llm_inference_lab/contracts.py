"""Value objects for scenarios and simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .deterministic import sha256_json


@dataclass(frozen=True)
class Scenario:
    kind: str
    name: str
    version: str
    seed: int
    parameters: dict[str, Any]
    requests: tuple[dict[str, Any], ...]
    search_space: dict[str, tuple[Any, ...]]
    scenario_hash: str
    source_hash: str
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "name": self.name,
            "version": self.version,
            "seed": self.seed,
            "parameters": self.parameters,
            "requests": list(self.requests),
            "search_space": {key: list(value) for key, value in self.search_space.items()},
            "description": self.description,
        }


@dataclass
class SimulationResult:
    kind: str
    scenario_name: str
    scenario_hash: str
    policy: str
    config: dict[str, Any]
    events: list[dict[str, Any]]
    metrics: dict[str, Any]
    trace: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def result_hash(self) -> str:
        return sha256_json(self.payload())

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "scenario_name": self.scenario_name,
            "scenario_hash": self.scenario_hash,
            "policy": self.policy,
            "config": self.config,
            "events": self.events,
            "metrics": self.metrics,
            "trace": self.trace,
            "state": self.state,
        }

    def as_dict(self) -> dict[str, Any]:
        result = self.payload()
        result["result_hash"] = self.result_hash
        return result

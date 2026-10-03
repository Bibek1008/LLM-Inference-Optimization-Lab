"""Deterministic, CPU-only inference-serving simulations."""

__version__ = "0.1.0"

from .contracts import Scenario, SimulationResult

__all__ = ["Scenario", "SimulationResult", "__version__"]

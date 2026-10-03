# Architecture

The package has four boundaries:

1. `scenario_io.py` loads and validates frozen JSON fixtures and computes a
   canonical scenario hash.
2. `simulators/` consumes a scenario and a pure configuration, producing an
   append-only logical event trace and derived metrics.
3. `validation.py` is independent of policy code. It recomputes cardinality,
   ordering, ownership, capacity, and terminal-state checks from the trace.
4. `receipts/` records baseline/candidate evidence and validates the receipt
   schema and content hash.

The optimizer is deliberately exhaustive and bounded. It reads only the
declared `search_space` and ranks valid candidates with a fixed lower-is-better
tuple per simulator. It does not inspect a held-out scenario while selecting.

No simulator imports a model runtime, CUDA package, HTTP client, wall-clock
timer, or subprocess. The logical tick is part of the model rather than a
measurement of the host.

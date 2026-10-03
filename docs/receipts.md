# Receipt contract

Every receipt is JSON with:

- `measurement_kind: "simulation"` and `not_a_benchmark: true`;
- the scenario name, seed, canonical scenario hash, and source-file hash;
- baseline and candidate policy/configuration, metrics, event traces, result
  hashes, and independent correctness reports;
- a top-level `correctness_result` and a content hash for fail-closed
  validation;
- environment and command provenance plus explicit limitations.

Search receipts add the declared search space, every tried configuration, the
selected training objective, and a held-out baseline/candidate comparison.
Receipt timestamps are provenance only and never enter simulator state.

These receipts support reproducibility under the same source and fixture. They
do not become benchmark receipts merely because Python ran quickly on one
machine.

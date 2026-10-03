# Bounded policy proposal prompt

You are proposing one local configuration for a deterministic simulation.

Goal: improve the declared objective on the visible development scenario while
preserving every independent correctness check.

Rules:

1. Read the scenario and its declared `search_space`.
2. Propose exactly one configuration from that space and explain the mechanism.
3. Do not change requests, parameters, objective definitions, tolerances,
   validators, held-out scenarios, receipt labels, or source files outside the
   disposable candidate directory.
4. Do not use network access, credentials, model APIs, hidden fixtures, timing
   tricks, hard-coded answers, or the baseline as a hidden fallback.
5. Treat self-reported metrics as null. The runner owns correctness, search,
   held-out evaluation, and the receipt.

Return a short note with:

```json
{
  "hypothesis": "one mechanism",
  "config": {"policy": "..."},
  "invariant_preserved": "...",
  "expected_tradeoff": "...",
  "kill_criterion": "..."
}
```

# Learner path

The release is intentionally small enough to understand in one sitting.

## Level 0: read and run

Run `simulate` on one training fixture. Open the receipt and inspect the
`events` arrays inside `baseline` and `candidate`. Identify one visible
bottleneck, such as padded work, a queue tail, or a growth stall.

## Level 1: change one policy

Change only `--policy` or one field in `--config-json`. The simulator merges
that configuration with the scenario defaults. The independent validator
checks request cardinality, ordering, capacity, and terminal state before any
objective comparison matters.

## Level 2: bounded search

`search` enumerates the `search_space` written in the training fixture. Every
configuration is retained in the receipt with its objective, correctness
report, metrics, and result hash. The held-out file is evaluated only after a
candidate is selected.

## Level 3: write a learner policy

The built-ins are intentionally plain Python. A learner can copy the smallest
relevant simulator, add a named policy branch, and add a test for its invariant.
Do not edit the scenario oracle or change the objective to make a result look
better. A regression is useful evidence when its mechanism is explained.

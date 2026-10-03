# Safe AI-assisted optimization scaffold

The repository supports a human-started proposal loop, not an autonomous
agent service:

```text
freeze scenario + validator + objective
  -> propose one config in the declared search space
  -> run the local evaluator
  -> check correctness before ranking
  -> select on development data
  -> evaluate once on held-out data
  -> review the simulation receipt
```

The runner remains the owner of the scenario hash, oracle, metrics, held-out
fixture, and evidence label. A proposal agent may write a hypothesis or a
configuration in a disposable local folder. It must not edit fixtures,
validators, receipts, thresholds, prior candidates, or the publication
workflow. Candidate-reported scores are comments, not evidence.

This first release intentionally provides only a prompt scaffold. It makes no
network calls, does not read credentials, does not invoke an LLM API, and has
no autonomous publishing or account mutation logic. Evaluating untrusted code
would require a stronger sandbox than the normal Python process isolation in
this educational repository.

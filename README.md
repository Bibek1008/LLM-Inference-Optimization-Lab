# LLM Inference Optimization Lab

A small, runnable companion repository for learning inference-serving ideas
without CUDA, a model download, or a GPU. The core package uses only the
Python 3.10+ standard library and advances an abstract logical clock. A
simulated tick, token, page, or work unit is an explanatory unit; it is not a
GPU millisecond, tokens/second result, VRAM measurement, or cloud-cost claim.

## Quick start

From the repository root:

```bash
python3 --version                 # Python 3.10 or newer
python3 -m unittest discover -s tests -v

PYTHONPATH=src python3 -m llm_inference_lab simulate queue \
  --scenario scenarios/queue_train.json \
  --policy interactive_first \
  --out /tmp/llm-lab-queue.json

PYTHONPATH=src python3 -m llm_inference_lab search batching \
  --scenario scenarios/batching_train.json \
  --held-out scenarios/batching_heldout.json \
  --out /tmp/llm-lab-batching-search.json

PYTHONPATH=src python3 -m llm_inference_lab validate-receipt \
  /tmp/llm-lab-batching-search.json
```

The commands are deterministic. The receipt records the scenario hash, policy
configuration, baseline and candidate summaries, event/result hashes,
correctness checks, and `measurement_kind: "simulation"`.

## Learner path

1. Run one fixture and inspect the JSON receipt and event trace.
2. State one bottleneck hypothesis before changing a policy.
3. Run a named candidate policy against the same scenario.
4. Use `search` for the small, declared grid in the scenario file.
5. Validate the receipt and compare the selected candidate on the separate
   held-out scenario.
6. Keep regressions and rejected candidates as learning evidence. A lower
   abstract objective is useful only when the independent correctness checks
   also pass.

The first release contains three simulations:

- `queue`: deterministic multi-server admission and scheduling (`fifo`,
  `shortest`, and `interactive_first`).
- `kv`: page-based KV-cache growth with a contiguous reservation baseline and
  a lazy paged candidate.
- `batching`: static padded batches, dynamic groups, and continuous
  prefill/decode admission.

Run `PYTHONPATH=src python3 -m llm_inference_lab --help` for the complete CLI.
The JSON scenario files document the fields that are legal to search. The
optimizer enumerates those fields only; it cannot change the workload,
objective, or correctness contract.

## What the receipts mean

Receipts are durable simulation evidence, not benchmarks. Baseline and
candidate receive equal frozen inputs. The search selects on the training
fixture, then evaluates the frozen configuration once on the held-out fixture.
Metrics such as `p95_turnaround_ticks`, `padded_work`, `growth_stalls`, and
`throughput_work_per_tick` are named to expose mechanisms, but remain abstract.

The receipt validator is intentionally fail-closed for its declared schema and
hashes. It checks that the receipt is labelled as a simulation, contains both
policies and metrics, records correctness, and has not been altered after
writing. It is a teaching integrity check, not a hostile-code sandbox.

## AI-assisted optimization boundary

[`docs/ai-optimization-harness.md`](docs/ai-optimization-harness.md) and
[`prompts/propose-policy.md`](prompts/propose-policy.md) provide a safe,
human-started scaffold. An AI may suggest one configuration inside the frozen
search space and explain a hypothesis. The local evaluator owns the oracle,
fixtures, correctness checks, held-out run, and receipt. The scaffold makes no
network calls, reads no credentials, invokes no model API, and has no
autonomous publishing logic.

## What requires real NVIDIA validation

These simulations can establish policy behavior under their declared rules:
queue ordering, page ownership/capacity arithmetic, padding waste, phase
interleaving, and deterministic trade-offs. They cannot establish real TTFT,
inter-token latency, throughput, GPU utilization, kernel efficiency, HBM/VRAM
use, CUDA graph or compilation behavior, NCCL communication, model quality,
energy, or dollars.

Those claims require a separately frozen workload and correctness oracle on a
named model revision, tokenizer, serving runtime, CUDA/driver configuration,
NVIDIA GPU, repeated paired runs, and hardware telemetry. A CPU or Metal run
would be useful observed evidence for that backend, but must not be relabelled
as NVIDIA evidence.

## Development notes

There are no required third-party packages. The optional `pyproject.toml`
describes the package for a later editable install, but the supported teaching
path runs directly from a clean checkout with `PYTHONPATH=src`.

The project intentionally does not copy course files or planning documents.
It is the hands-on simulation layer, with the course remaining the narrative
source of truth.

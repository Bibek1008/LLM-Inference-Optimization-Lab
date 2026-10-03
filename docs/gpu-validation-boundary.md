# NVIDIA validation boundary

The lab can validate conceptual rules: admission order, exact page ownership,
logical capacity, padding arithmetic, phase scheduling, and deterministic
trade-offs under a frozen cost model.

It cannot validate NVIDIA performance. A real claim about TTFT, inter-token
latency, throughput, GPU utilization, kernel choice, HBM/VRAM use, compilation,
NCCL, model quality, energy, or dollars needs a separate protocol with:

- a pinned model and tokenizer revision;
- a named serving runtime, CUDA/driver version, flags, and NVIDIA GPU;
- identical baseline/candidate requests and output/correctness checks;
- warmups, repeated paired measurements, tail statistics, and raw samples;
- hardware/resource telemetry and a held-out workload;
- a receipt that labels the result `observed_gpu`, not `simulation`.

A CPU or Metal observation can be useful for that backend, but it is not
NVIDIA evidence and should not be combined with these simulation numbers.

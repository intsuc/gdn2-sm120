# SM120 benchmark notes

This file records warm steady-state latency from the repository benchmark CLI.
It is intentionally a latency comparison of the kernel primitives, not a
claim about complete model throughput.

## Reproducibility contract

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`sm_120`)
- driver: 595.71.05 (CUDA Driver API 13.2)
- system toolkit: CUDA 13.3.1 (`nvcc` 13.3.73)
- Python 3.12.14, PyTorch 2.13.0+cu130, CuTe DSL 4.7.0
- shape specialization: `K=V=128`
- tensors: BF16 except FP32 `g`, initial state, and final state
- Q/K are L2-normalized once outside the timed calls
- scale: `1/sqrt(128) = 0.125`
- initial state and returned final state enabled for both implementations
- fused Q/K L2 normalization and gate activation disabled for both
- deterministic input seed: `20260824`; upstream-gradient seed: `20260825`
- 5 warmup calls and 50 CUDA-event samples unless noted
- compilation and Triton autotuning occur during warmup and are excluded
- compared at each implementation's public Python call boundary
- backward uses `torch.autograd.grad(..., retain_graph=True)` for both
- an untimed equivalence check runs before either timing loop
- official GDN2 commit: `95709fc250357c2dd109361c353192f2aa5913f9`
- Flash Linear Attention compatibility commit:
  `4b02d15d6a68700181b180235be62a9fb95d2a38`

The benchmark reports the median and minimum of synchronized per-call CUDA
event samples. GPU clocks, thermals, display load, and other CUDA processes can
still move small-kernel latency, so rerun on the deployment machine.

## Results (2026-08-24)

All numbers are microseconds. Speedup is official median divided by CuTe
median. `max diff` is the largest absolute difference from the untimed
CuTe-versus-official comparison.

| Path | B | T | H | Warmup / samples | CuTe median | Official median | Speedup | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk forward | 1 | 16 | 16 | 50 / 200 | 43.5 | 189.7 | **4.36x** | 1.55e-3 |
| chunk forward | 1 | 64 | 16 | 50 / 200 | 63.8 | 184.9 | **2.90x** | 1.10e-3 |
| chunk forward | 1 | 256 | 16 | 50 / 200 | 147.9 | 183.1 | **1.24x** | 1.15e-3 |
| chunk forward | 1 | 512 | 16 | 30 / 100 | 260.5 | 183.9 | 0.71x | 1.14e-3 |
| chunk backward | 1 | 16 | 16 | 50 / 200 | 112.1 | 277.1 | **2.47x** | 1.95e-3 |
| chunk backward | 1 | 64 | 16 | 30 / 100 | 220.5 | 278.1 | **1.26x** | 1.95e-3 |
| chunk backward | 1 | 128 | 16 | 20 / 100 | 510.0 | 278.7 | 0.55x | 1.95e-3 |
| token forward | 1 | 1 | 32 | 50 / 200 | 21.0 | 25.2 | **1.20x** | 2.98e-8 |
| token forward | 1 | 16 | 16 | 50 / 200 | 35.3 | 35.3 | 1.00x | 1.91e-6 |

The three bold representative rows meet the initial goal. They do not imply a
win at every sequence length: forward crosses over after T=256 in this B1/H16
sweep, while backward crosses between T=64 and T=128. Backward is capped at
T=128 for numerical correctness; a checkpoint/WY redesign is required before
claiming long-training support.

The exact values behind this table and the README figure are tracked in
[`data/benchmark-results-sm120.json`](data/benchmark-results-sm120.json). The
plotter validates the current benchmark schema, consistent environments,
positive finite timings, stored speedups, and duplicate logical shapes before
rendering:

```bash
uv run --group visualization gdn2-sm120-plot \
  docs/data/benchmark-results-sm120.json \
  --output docs/assets/benchmark-results-sm120.png
```

The plot uses medians only. A minimum is not a dispersion estimate or confidence
interval, so it is deliberately not drawn as an error bar. Token points with
different head counts are separate bars rather than a connected scaling curve.

## Commands

```bash
uv sync --locked --all-groups

uv run gdn2-sm120-bench \
  --mode chunk-forward --batch 1 --time 16 --heads 16 --dtype bf16 \
  --warmup 5 --repeats 50 --official-repo /tmp/GatedDeltaNet-2

uv run gdn2-sm120-bench \
  --mode chunk-backward --batch 1 --time 16 --heads 16 --dtype bf16 \
  --warmup 5 --repeats 50 --official-repo /tmp/GatedDeltaNet-2

uv run gdn2-sm120-bench \
  --mode token-forward --batch 1 --time 1 --heads 32 --dtype bf16 \
  --warmup 5 --repeats 50 --official-repo /tmp/GatedDeltaNet-2
```

Pass `--json benchmark-results/name.json` to persist a machine-readable result.
The output directory is ignored because measurements are host-specific. Curate
only validated publication runs into `docs/data`; plotting an entire local
output directory fails on duplicate logical shapes instead of silently choosing
between old and new measurements.

## Interpretation

The optimization goal is met when speedup is greater than one for each of the
three representative calls. Short chunks and decode latency are the first
target. Scaling to the paper's 16K-token training sweep, additional dimensions,
packed sequences, fused normalization/gates, checkpoint/WY backward, and peak
memory comparisons are separate milestones; results from the small latency
cases must not be extrapolated to those workloads.

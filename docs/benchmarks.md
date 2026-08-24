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
- explicit output scale: `0.125` for both implementations (the API default for
  K=128 would instead be `1/sqrt(128)`, approximately `0.088388`)
- initial state and returned final state enabled for both implementations
- fused Q/K L2 normalization and gate activation disabled for both
- deterministic input seed: `20260824`; upstream-gradient seed: `20260825`
- warmup/sample counts are recorded per row
- compilation and Triton autotuning occur during warmup and are excluded
- compared at each implementation's public Python call boundary
- backward-only builds each graph before timing and uses
  `torch.autograd.grad(..., retain_graph=True)` for both
- training rebuilds the forward graph inside each timed call and consumes it
  with the default one-shot `torch.autograd.grad`
- an untimed equivalence check runs before either timing loop
- official GDN2 commit: `95709fc250357c2dd109361c353192f2aa5913f9`
- Flash Linear Attention compatibility commit:
  `4b02d15d6a68700181b180235be62a9fb95d2a38`

The benchmark reports the median and minimum of synchronized per-call CUDA
event samples. GPU clocks, thermals, display load, and other CUDA processes can
still move small-kernel latency, so rerun on the deployment machine.

## Results (2026-08-25)

All numbers are microseconds. Speedup is official median divided by CuTe
median. `max diff` is the largest absolute difference from the untimed
CuTe-versus-official comparison.

| Path | B | T | H | Warmup / samples | CuTe median | Official median | Speedup | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk forward | 1 | 16 | 16 | 40 / 300 | 35.3 | 181.8 | **5.15x** | 1.32e-3 |
| chunk forward | 1 | 64 | 16 | 40 / 300 | 40.4 | 181.7 | **4.49x** | 1.36e-3 |
| chunk forward | 1 | 128 | 16 | 40 / 300 | 47.6 | 180.9 | **3.80x** | 1.60e-3 |
| chunk forward | 1 | 256 | 16 | 40 / 300 | 66.1 | 182.3 | **2.76x** | 1.39e-3 |
| chunk forward | 1 | 512 | 16 | 40 / 300 | 65.9 | 181.6 | **2.76x** | 1.43e-3 |
| chunk forward | 1 | 1024 | 16 | 40 / 300 | 102.9 | 183.4 | **1.78x** | 1.37e-3 |
| chunk forward | 1 | 2048 | 16 | 40 / 300 | 172.5 | 236.4 | **1.37x** | 1.40e-3 |
| chunk forward | 1 | 4096 | 16 | 20 / 100 | 446.1 | 476.4 | **1.07x** | 1.38e-3 |
| chunk forward | 1 | 8192 | 16 | 20 / 100 | 978.0 | 1036.4 | **1.06x** | 1.41e-3 |
| chunk forward | 1 | 16384 | 16 | 10 / 50 | 1889.5 | 2111.5 | **1.12x** | 1.50e-3 |
| chunk backward | 1 | 16 | 16 | 40 / 300 | 112.1 | 274.6 | **2.45x** | 3.91e-3 |
| chunk backward | 1 | 64 | 16 | 40 / 300 | 172.9 | 396.8 | **2.29x** | 2.44e-3 |
| chunk backward | 1 | 128 | 16 | 40 / 300 | 126.4 | 279.3 | **2.21x** | 2.08e-3 |
| chunk backward | 1 | 256 | 16 | 40 / 300 | 134.6 | 278.7 | **2.07x** | 2.44e-3 |
| chunk backward | 1 | 512 | 16 | 40 / 300 | 148.8 | 284.0 | **1.91x** | 2.93e-3 |
| chunk backward | 1 | 1024 | 16 | 40 / 300 | 299.1 | 391.1 | **1.31x** | 2.93e-3 |
| chunk backward | 1 | 2048 | 16 | 40 / 300 | 623.7 | 708.6 | **1.14x** | 3.91e-3 |
| chunk backward | 1 | 4096 | 16 | 20 / 100 | 1318.8 | 1465.4 | **1.11x** | 3.91e-3 |
| chunk backward | 1 | 8192 | 16 | 20 / 100 | 2838.6 | 3126.1 | **1.10x** | 2.93e-3 |
| chunk backward | 1 | 16384 | 16 | 10 / 50 | 5810.6 | 6353.5 | **1.09x** | 1.95e-3 |
| token forward | 1 | 1 | 32 | 50 / 200 | 21.0 | 25.8 | **1.23x** | 2.98e-8 |
| token forward | 1 | 16 | 16 | 50 / 200 | 35.3 | 35.2 | 1.00x | 1.91e-6 |

Forward and backward both stay ahead at every measured point through T=16384.
The compact-WY dispatch at T=128 reduces CuTe backward latency from 172.9 us at
T=64 to 126.4 us despite doubling the sequence length. Its advantage narrows
from 2.45x at T=16 to 1.09x at T=16384 without a measured crossover, while the
checkpointed path also removes the old T=128 correctness cap.

For full-chunk BF16 training at T>=128, forward checkpoints Y, Q-gamma,
K-tail, A-qk, and state boundaries in BF16; U and chunk decay remain FP32.
The boundary stage precomputes all independent `A_qk.T @ dO` products before
its reverse scan, stages Y/Q-gamma/K-tail with 128-bit `cp.async`, and shares
each decay value through warp shuffles. It writes BF16 `dR` and `dS`
checkpoints for the tensor-core consumers while returning the exact FP32
`dS0` separately. T=64--127, partial tails, and FP16 retain FP32 boundaries.

The local compact-WY VJP keeps gamma in FP32 but stores its persistent E and
K-bar MMA operands in the input dtype. Dual large-state and square products,
paired K16 updates, and producer epilogues reduce the full-chunk schedule to
12 ordered launches; at T>=2048 the full-chunk BF16 specialization folds the
state/gradient decay dot into an existing state-product kernel and uses 11.

The backward-only timings above reuse an already-built autograd graph, so they
do not include the checkpoint-producing forward. Use the `chunk-training`
command to measure a complete call. Compact BF16 `S`/`dS` checkpoints halve
boundary storage relative to FP32, but retaining both boundary sets plus
compact-WY workspace still makes the CuTe path use more memory than the
official implementation.

The exact values behind the per-kernel table and the README light/dark figures
are tracked in
[`data/benchmark-results-sm120.json`](data/benchmark-results-sm120.json). The
plotter validates the current benchmark schema, consistent environments,
positive finite timings, stored speedups, and duplicate logical shapes before
rendering both themes:

```bash
uv run --group visualization gdn2-sm120-plot \
  docs/data/benchmark-results-sm120.json \
  --output docs/assets/benchmark-results-sm120.png
```

`--output` is the light/fallback path; the dark-theme image is written alongside
it with `-dark` appended to the stem.

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
  --mode chunk-training --batch 1 --time 256 --heads 16 --dtype bf16 \
  --warmup 30 --repeats 100 --official-repo /tmp/GatedDeltaNet-2

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

The optimization goal is met for all three kernel families and now extends to
substantially longer chunk sequences. Both chunk forward and backward-only are
faster at every measured length through T=16384; the backward margin gradually
narrows but remains 1.09x at the longest point. These are primitive-level
latencies, not complete training-throughput measurements. Reducing
checkpoint/workspace memory, additional dimensions, packed sequences, and
fused normalization/gates remain separate milestones; the measured B1/H16
results must not be extrapolated to those workloads.

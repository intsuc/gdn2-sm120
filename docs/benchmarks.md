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
| chunk backward | 1 | 16 | 16 | 50 / 200 | 111.9 | 274.3 | **2.45x** | 3.91e-3 |
| chunk backward | 1 | 64 | 16 | 100 / 300 | 152.8 | 279.4 | **1.83x** | 2.44e-3 |
| chunk backward | 1 | 128 | 16 | 50 / 200 | 163.3 | 278.2 | **1.70x** | 2.20e-3 |
| chunk backward | 1 | 256 | 16 | 50 / 200 | 207.0 | 276.7 | **1.34x** | 3.91e-3 |
| chunk backward | 1 | 512 | 16 | 100 / 300 | 368.8 | 277.7 | 0.75x | 2.93e-3 |
| token forward | 1 | 1 | 32 | 50 / 200 | 21.0 | 25.8 | **1.23x** | 2.98e-8 |
| token forward | 1 | 16 | 16 | 50 / 200 | 35.3 | 35.2 | 1.00x | 1.91e-6 |

Forward stays ahead at every measured point through T=16384. Backward-only
stays ahead through T=256. The backward crossover moved from between T=64 and
T=128 to between T=256 and T=512, while the checkpointed path also removed the
old T=128 correctness cap.

The long backward saves FP32 state boundaries and compact-WY auxiliaries in
forward. To account for that work, the complete graph-build plus backward call
was also measured with 30 warmups and 100 samples:

| B | T | H | CuTe forward+backward | Official forward+backward | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 16 | 206.4 | 516.3 | **2.50x** |
| 1 | 256 | 16 | 263.6 | 541.2 | **2.05x** |
| 1 | 512 | 16 | 430.8 | 537.2 | **1.25x** |

Peak allocated-memory deltas for those complete calls were 35.6/69.1/132.3
MiB for CuTe at T=128/256/512, versus 19.0/36.0/70.0 MiB for the official
path. The speedup therefore comes with roughly 1.9x peak allocation in this
sweep.

These complete-call and peak-memory figures are supplemental measurements;
the tracked plot suite below contains the per-kernel rows. Use the
`chunk-training` command to reproduce the complete-call timing.

The exact values behind the per-kernel table and the README figure are tracked in
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
substantially longer chunk sequences. Forward remains faster at all measured
lengths through T=16384, while backward-only crosses between T=256 and T=512.
At T=512 the complete forward+backward call still wins because the forward gain
offsets the slower backward. The 16K point is a forward-only measurement, not a
complete training-throughput result. Reducing checkpoint/workspace memory,
additional dimensions, packed sequences, and fused normalization/gates remain
separate milestones; the measured B1/H16 results must not be extrapolated to
those workloads.

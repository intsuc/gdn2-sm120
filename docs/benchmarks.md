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
- the canonical chunk sweep fixes H=16, varies B over 1/2/4, and varies T over
  16/64/128/256/512/1024/2048/4096/8192/16384/32768; B4/T32768 backward is
  excluded because its saved state-boundary tensor exceeds CuTe's 4-GiB
  per-launch byte-address range
- the canonical token sweep fixes B=1/H=32 and varies T over powers of two;
  connected token points therefore differ only in sequence length
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

The token CLI exercises the default allocation-returning API. The reusable
`out`/`final_state_out` and explicit `inplace_final_state` modes are intended for
allocation-free serving loops and should be measured as a separate experiment;
the current publication schema does not encode allocation mode.

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
| chunk forward | 1 | 32768 | 16 | 10 / 50 | 3725.3 | 4231.9 | **1.14x** | 1.82e-3 |
| chunk forward | 2 | 16 | 16 | 40 / 300 | 35.4 | 179.8 | **5.08x** | 1.32e-3 |
| chunk forward | 2 | 64 | 16 | 40 / 300 | 44.4 | 181.4 | **4.08x** | 1.36e-3 |
| chunk forward | 2 | 128 | 16 | 40 / 300 | 56.9 | 186.0 | **3.27x** | 1.62e-3 |
| chunk forward | 2 | 256 | 16 | 40 / 300 | 82.2 | 180.5 | **2.20x** | 1.40e-3 |
| chunk forward | 2 | 512 | 16 | 40 / 300 | 97.7 | 184.4 | **1.89x** | 1.51e-3 |
| chunk forward | 2 | 1024 | 16 | 40 / 300 | 163.1 | 221.5 | **1.36x** | 1.35e-3 |
| chunk forward | 2 | 2048 | 16 | 40 / 300 | 367.7 | 433.4 | **1.18x** | 1.47e-3 |
| chunk forward | 2 | 4096 | 16 | 20 / 100 | 802.6 | 944.4 | **1.18x** | 1.37e-3 |
| chunk forward | 2 | 8192 | 16 | 20 / 100 | 1547.3 | 1956.7 | **1.26x** | 1.48e-3 |
| chunk forward | 2 | 16384 | 16 | 10 / 50 | 3043.7 | 3959.6 | **1.30x** | 2.11e-3 |
| chunk forward | 2 | 32768 | 16 | 10 / 50 | 6155.5 | 7955.2 | **1.29x** | 1.51e-3 |
| chunk forward | 4 | 16 | 16 | 40 / 300 | 40.3 | 181.7 | **4.51x** | 1.32e-3 |
| chunk forward | 4 | 64 | 16 | 40 / 300 | 59.6 | 181.7 | **3.05x** | 1.57e-3 |
| chunk forward | 4 | 128 | 16 | 40 / 300 | 85.4 | 180.8 | **2.12x** | 1.62e-3 |
| chunk forward | 4 | 256 | 16 | 40 / 300 | 136.6 | 180.7 | **1.32x** | 1.68e-3 |
| chunk forward | 4 | 512 | 16 | 40 / 300 | 144.9 | 219.2 | **1.51x** | 1.56e-3 |
| chunk forward | 4 | 1024 | 16 | 40 / 300 | 359.5 | 415.9 | **1.16x** | 1.60e-3 |
| chunk forward | 4 | 2048 | 16 | 40 / 300 | 771.3 | 920.7 | **1.19x** | 1.64e-3 |
| chunk forward | 4 | 4096 | 16 | 20 / 100 | 1493.9 | 1914.8 | **1.28x** | 1.47e-3 |
| chunk forward | 4 | 8192 | 16 | 20 / 100 | 2898.6 | 3882.8 | **1.34x** | 2.06e-3 |
| chunk forward | 4 | 16384 | 16 | 10 / 50 | 5783.9 | 7797.7 | **1.35x** | 2.28e-3 |
| chunk forward | 4 | 32768 | 16 | 10 / 50 | 11553.5 | 15661.8 | **1.36x** | 1.83e-3 |
| chunk backward | 1 | 16 | 16 | 40 / 300 | 112.1 | 274.6 | **2.45x** | 3.91e-3 |
| chunk backward | 1 | 64 | 16 | 40 / 300 | 131.6 | 279.0 | **2.12x** | 2.44e-3 |
| chunk backward | 1 | 128 | 16 | 40 / 300 | 126.6 | 281.5 | **2.22x** | 2.08e-3 |
| chunk backward | 1 | 256 | 16 | 40 / 300 | 179.4 | 398.0 | **2.22x** | 2.44e-3 |
| chunk backward | 1 | 512 | 16 | 40 / 300 | 172.1 | 329.6 | **1.91x** | 2.93e-3 |
| chunk backward | 1 | 1024 | 16 | 40 / 300 | 308.2 | 394.3 | **1.28x** | 2.93e-3 |
| chunk backward | 1 | 2048 | 16 | 40 / 300 | 639.0 | 704.5 | **1.10x** | 3.91e-3 |
| chunk backward | 1 | 4096 | 16 | 20 / 100 | 1318.8 | 1465.4 | **1.11x** | 3.91e-3 |
| chunk backward | 1 | 8192 | 16 | 20 / 100 | 2838.6 | 3126.1 | **1.10x** | 2.93e-3 |
| chunk backward | 1 | 16384 | 16 | 10 / 50 | 5810.6 | 6353.5 | **1.09x** | 1.95e-3 |
| chunk backward | 1 | 32768 | 16 | 10 / 50 | 11737.6 | 12638.0 | **1.08x** | 3.91e-3 |
| chunk backward | 2 | 16 | 16 | 40 / 300 | 119.2 | 273.6 | **2.30x** | 3.91e-3 |
| chunk backward | 2 | 64 | 16 | 40 / 300 | 132.6 | 278.0 | **2.10x** | 2.93e-3 |
| chunk backward | 2 | 128 | 16 | 40 / 300 | 134.3 | 279.5 | **2.08x** | 2.93e-3 |
| chunk backward | 2 | 256 | 16 | 40 / 300 | 188.7 | 395.0 | **2.09x** | 2.44e-3 |
| chunk backward | 2 | 512 | 16 | 40 / 300 | 283.6 | 392.3 | **1.38x** | 3.91e-3 |
| chunk backward | 2 | 1024 | 16 | 40 / 300 | 588.7 | 639.9 | **1.09x** | 2.44e-3 |
| chunk backward | 2 | 2048 | 16 | 40 / 300 | 1256.4 | 1309.6 | **1.04x** | 2.93e-3 |
| chunk backward | 2 | 4096 | 16 | 20 / 100 | 2742.8 | 2816.0 | **1.03x** | 3.91e-3 |
| chunk backward | 2 | 8192 | 16 | 20 / 100 | 5582.7 | 5675.6 | **1.02x** | 3.91e-3 |
| chunk backward | 2 | 16384 | 16 | 10 / 50 | 11298.4 | 11393.6 | **1.01x** | 3.91e-3 |
| chunk backward | 2 | 32768 | 16 | 10 / 50 | 22679.8 | 22951.8 | **1.01x** | 3.91e-3 |
| chunk backward | 4 | 16 | 16 | 40 / 300 | 149.8 | 279.6 | **1.87x** | 3.91e-3 |
| chunk backward | 4 | 64 | 16 | 40 / 300 | 142.7 | 276.9 | **1.94x** | 3.91e-3 |
| chunk backward | 4 | 128 | 16 | 40 / 300 | 148.0 | 282.7 | **1.91x** | 3.91e-3 |
| chunk backward | 4 | 256 | 16 | 40 / 300 | 271.4 | 392.1 | **1.44x** | 3.91e-3 |
| chunk backward | 4 | 512 | 16 | 40 / 300 | 577.1 | 620.1 | **1.07x** | 3.91e-3 |
| chunk backward | 4 | 1024 | 16 | 40 / 300 | 1256.1 | 1303.1 | **1.04x** | 3.91e-3 |
| chunk backward | 4 | 2048 | 16 | 40 / 300 | 2654.5 | 2716.7 | **1.02x** | 3.91e-3 |
| chunk backward | 4 | 4096 | 16 | 20 / 100 | 5444.0 | 5516.0 | **1.01x** | 3.91e-3 |
| chunk backward | 4 | 8192 | 16 | 20 / 100 | 10921.2 | 10945.9 | **1.00x** | 3.91e-3 |
| chunk backward | 4 | 16384 | 16 | 10 / 50 | 21906.1 | 22145.8 | **1.01x** | 3.91e-3 |
| token forward | 1 | 1 | 32 | 25 / 100 | 13.3 | 25.4 | **1.91x** | 2.98e-8 |
| token forward | 1 | 2 | 32 | 25 / 100 | 14.8 | 25.2 | **1.70x** | 5.96e-8 |
| token forward | 1 | 4 | 32 | 25 / 100 | 16.5 | 26.0 | **1.57x** | 5.96e-8 |
| token forward | 1 | 8 | 32 | 25 / 100 | 18.6 | 28.8 | **1.55x** | 2.38e-7 |
| token forward | 1 | 16 | 32 | 25 / 100 | 22.8 | 35.0 | **1.54x** | 3.05e-5 |
| token forward | 1 | 32 | 32 | 25 / 100 | 31.0 | 47.2 | **1.52x** | 3.05e-5 |
| token forward | 1 | 64 | 32 | 25 / 100 | 49.4 | 71.9 | **1.45x** | 6.10e-5 |
| token forward | 1 | 128 | 32 | 25 / 100 | 84.3 | 120.2 | **1.43x** | 6.10e-5 |

Forward stays ahead at every measured B1/B2/B4 point through T=32768; its
longest-sequence speedups are 1.14x, 1.29x, and 1.36x respectively. Backward is
also ahead at every supported measured point: all 11 B1 and B2 lengths and all
10 B4 lengths through T=16384. The narrowest measured margin is 1.0023x at
B4/T8192; B2/T32768 and B4/T16384 both remain 1.01x after rounding.

The canonical B1/H16 T=64 point supplies 64 chunk-head CTAs and therefore takes
the CTA-aware compact-WY parameter VJP, together with the T=64 MMA boundary
scan. T=128 additionally enables compact BF16 checkpoints and measures 126.6 us.
The checkpointed path removes the old T=128 correctness cap. B4/T32768
backward is unsupported and not benchmarked because one saved state-boundary
tensor exceeds CuTe's 4-GiB per-launch byte-address range.

The fixed B1/H32 token sweep stays ahead at all eight sampled decode lengths.
Its public-call speedup is 1.91x at T=1 and remains 1.43x at T=128; these rows
include the default output and final-state allocations on both public paths.

For full-chunk BF16 training at T>=128, forward checkpoints Y, raw Q-gamma,
K-tail, A-qk, and state boundaries in BF16; U and chunk decay remain FP32. At
T>=512, a separate compact Q-effective scratch lets training use the rearranged
long-forward identity without changing the raw Q-gamma or A-qk checkpoint bits.
The boundary stage precomputes all independent `A_qk.T @ dO` products before
its reverse scan, stages Y/Q-gamma/K-tail with 128-bit `cp.async`, and shares
each decay value through warp shuffles. The ordered scan selects V16 when
`batch * heads >= 32`, and also for the B1-sized midrange where
`16 <= batch * heads < 32` and `T <= 2048`; other shapes use V8. Both variants
retain eight K-split warps. V16 halves duplicated Y/Q-gamma/K-tail reads, while
V8 exposes twice as many CTAs for long underfilled grids. The scan writes BF16
`dR` and `dS` checkpoints for the tensor-core consumers while returning the
exact FP32 `dS0` separately. T=64--127, partial tails, and FP16 retain FP32
boundaries.

The local compact-WY VJP keeps gamma in FP32 but stores its persistent E and
K-bar MMA operands in the input dtype. Dual large-state and square products,
paired K16 updates, and producer epilogues reduce the full-chunk schedule to
12 ordered launches. The compact BF16 specialization folds the state/gradient
decay dot into an existing state-product kernel and uses 11 when
`batch * ceil(T / 16) * heads >= 2048`; for H16 this begins at B1/T2048,
B2/T1024, and B4/T512.

The backward-only timings above reuse an already-built autograd graph, so they
do not include the checkpoint-producing forward. Use the `chunk-training`
command to measure a complete call. Compact BF16 `S`/`dS` checkpoints halve
boundary storage relative to FP32, but retaining both boundary sets plus
compact-WY workspace still makes the CuTe path use more memory than the
official implementation.

## Focused optimization experiments

These CuTe-internal A/B measurements document schedule choices that are not
additional points in the canonical publication schema. Unless noted otherwise,
they use the same workstation, normalized BF16 Q/K, FP32 `g`/state, and
`scale=0.125` as the canonical suite. Medians and minima are synchronized CUDA
event microseconds.

### T=64 backward boundary and VJP dispatch

The canonical B1/T64/H16 shape has four chunks per head, so its 64 chunk-head
CTAs meet the compact-WY threshold. The focused same-session sequence below
separates the T=64 MMA boundary scan from the CTA-aware parameter-VJP dispatch.
Official timings are shown to make the run-to-run environment visible; the
implementation delta is the CuTe baseline-to-variant comparison.

| CuTe schedule | Warmup / samples | CuTe median (min) | Official median (min) | CuTe change |
|---|---:|---:|---:|---:|
| scalar boundary + chunk-parallel VJP | 30 / 200 | 153.024 (150.784) | 279.088 (267.104) | baseline |
| MMA boundary + chunk-parallel VJP | 40 / 250 | 135.648 (125.824) | 277.760 (271.232) | **11.36% lower** |
| MMA boundary + compact-WY VJP | 40 / 300 | 129.504 (120.736) | 277.008 (269.760) | **15.37% lower** |

All three untimed comparisons had `2.441e-3` maximum gradient difference. The
tracked canonical 131.616/278.992 us row is a separate 40/300 publication run;
the same-session table is the appropriate evidence for the implementation
improvement rather than a comparison against an older canonical capture.

### Zero-state T=1 token closed form

This experiment uses B1/T1/H32 with `initial_state=None`, unlike the canonical
token sweep, which enables an initial state. It therefore remains supplemental
instead of replacing or duplicating the canonical B1/T1/H32 graph point.

| Call boundary / implementation | Warmup / samples | Median | Minimum | Relative result |
|---|---:|---:|---:|---:|
| public CuTe, closed form | 25 / 100 | 13.248 | 7.200 | **1.975x vs official** |
| direct CuTe, generic recurrence | 25 / 100 | 5.776 | 5.632 | baseline |
| direct CuTe, closed form | 25 / 100 | 5.536 | 5.280 | **1.043x vs generic** |
| public official Triton | 25 / 100 | 26.160 | 25.312 | baseline |

The untimed official comparison measured `3.81e-6` maximum output difference
and zero state difference. A separate 1,000-call Nsight Systems launch sweep of
true 8/16/32-column CTA granularities measured projected launch spacings of
3.214, 3.219, and 3.189 us for one, two, and four warps respectively, versus
3.695 us for the generic direct path. The four-warp schedule won and is the only
variant retained.

### Long-forward value-tile dispatch

For T>=512, V8 supplies sixteen CTAs per batch/head and V16 supplies eight. The
focused public-forward sweep used 25 warmups and 100 samples per variant; rows
around the occupancy crossover are shown below.

| B | T | H | V8 median (min) | V16 median (min) | Winner |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 1 | 51.680 (50.528) | 54.656 (53.440) | **V8** |
| 1 | 512 | 8 | 56.832 (55.552) | 59.776 (51.680) | **V8** |
| 1 | 512 | 12 | 70.208 (69.088) | 64.848 (63.680) | **V16** |
| 1 | 512 | 16 | 72.960 (71.744) | 64.032 (63.616) | **V16** |
| 1 | 2048 | 8 | 135.744 (133.536) | 145.824 (143.712) | **V8** |
| 1 | 2048 | 12 | 183.872 (177.696) | 161.184 (159.808) | **V16** |
| 1 | 2048 | 16 | 204.128 (201.568) | 171.296 (163.424) | **V16** |
| 2 | 2048 | 32 | 1024.144 (1013.728) | 789.360 (776.320) | **V16** |

The selected rule is consequently V8 when `batch * heads < 12`, otherwise V16.
T<512 remains V8 regardless of grid size.

### Long-training Q-effective scratch

The A/B forces the legacy versus rearranged forward algebra while preserving
the same returned training auxiliaries. Return-aux forward uses 25 warmups and
100 samples; full forward-plus-backward training uses 10 warmups and 50
samples. Both are CuTe-only comparisons, not official-baseline publication
points.

| Timed path | T | Legacy median (min) | Q-effective median (min) | Speedup |
|---|---:|---:|---:|---:|
| return-aux forward | 512 | 87.616 (86.400) | 70.288 (61.888) | **1.247x** |
| return-aux forward | 1024 | 149.104 (140.352) | 113.984 (113.440) | **1.308x** |
| return-aux forward | 2048 | 306.304 (297.984) | 244.016 (240.608) | **1.255x** |
| chunk training | 512 | 207.952 (204.544) | 195.568 (192.448) | **1.063x** |
| chunk training | 1024 | 418.912 (415.840) | 385.472 (381.376) | **1.087x** |
| chunk training | 2048 | 893.344 (888.192) | 816.384 (810.240) | **1.094x** |

Across T=512/1024/2048, state output was bit-identical, every returned raw
Q-gamma/A-qk and other auxiliary was bit-identical, and maximum output
differences were `1.22e-4`, `2.44e-4`, and `2.44e-4`. A T=1024 backward-only
recheck kept all seven gradients bit-identical; its 298.016 versus 298.784 us
medians were within measurement noise, as expected because the extra
Q-effective scratch is forward-only.

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
interval, so it is deliberately not drawn as an error bar. The three panels are
stacked vertically. Within each chunk panel, fixed-H16 B1/B2/B4 observations are
dodged left-to-right around every shared log₂-spaced T tick. A short connector
pairs CuTe with Official Triton for each shape, faint solid/dashed/dotted lines
retain the within-batch trends, and the speedup rail remains centered on the
true T positions. When all token points share B and H, the token panel connects
them at log₂-spaced T positions and labels the official/CuTe speedup at every
sample. If either B or H differs, it falls back to independent grouped bars
rather than implying a scaling curve across different workloads.

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

for time in 1 2 4 8 16 32 64 128; do
  uv run gdn2-sm120-bench \
    --mode token-forward --batch 1 --time "$time" --heads 32 --dtype bf16 \
    --warmup 25 --repeats 100 --official-repo /tmp/GatedDeltaNet-2 \
    --json "benchmark-results/token-t${time}.json"
done
```

Pass `--json benchmark-results/name.json` to persist a machine-readable result.
The output directory is ignored because measurements are host-specific. Curate
only validated publication runs into `docs/data`; plotting an entire local
output directory fails on duplicate logical shapes instead of silently choosing
between old and new measurements.

## Interpretation

The optimization goal is met across every supported measured chunk point.
Chunk forward is faster at every B1/B2/B4 length through T=32768. Backward is
faster at all 11 B1 and B2 lengths and all 10 measured B4 lengths through
T=16384; B4/T32768 is outside the current CuTe per-launch address range and is
excluded. Token forward is faster at every fixed-B1/H32 point through T=128.
These are primitive-level latencies, not complete training-throughput
measurements. Reducing checkpoint/workspace memory, improving multi-batch
backward scaling, additional dimensions, packed sequences, and fused
normalization/gates remain separate milestones.

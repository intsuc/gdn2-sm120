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
- warmup/sample counts are recorded per row; after the count-based warmup,
  each implementation runs a 250 ms high-occupancy GEMM and 750 ms of the
  target call to stabilize device clocks
- process-median aggregation covers chunk backward at `T <= 2048`: normally
  three processes, five when either implementation's initial spread exceeds
  5%, and nine when fewer than four of either implementation's first five
  medians lie within 3% of its median. B2/T8 token also uses three processes;
  the JSON retains every raw per-process median and minimum
- compilation, Triton autotuning, and clock stabilization are excluded
- compared at each implementation's public Python call boundary
- the canonical chunk sweep fixes H=16, varies B over 1/2/4, and varies T over
  16/64/128/256/512/1024/2048/4096/8192/16384/32768; B4/T32768 backward is
  excluded because its saved state-boundary tensor exceeds CuTe's 4-GiB
  per-launch byte-address range
- the canonical token sweep fixes H=16, varies B over 1/2/4, and varies T over
  1/2/4/8/16/32/64/128; each connected batch series therefore differs only in
  sequence length
- backward-only builds each graph before timing and uses
  `torch.autograd.grad(..., retain_graph=True)` for both
- training rebuilds the forward graph inside each timed call and consumes it
  with the default one-shot `torch.autograd.grad`
- an untimed equivalence check runs before either timing loop
- official GDN2 commit: `95709fc250357c2dd109361c353192f2aa5913f9`
- Flash Linear Attention compatibility commit:
  `4b02d15d6a68700181b180235be62a9fb95d2a38`

The benchmark reports the median and minimum of synchronized per-call CUDA
event samples. The one-second load floor reduces the idle-clock bimodality
observed for short kernels; the independent-process aggregation above absorbs
the residual modes. Thermals, display load, and other CUDA processes can still
move latency, so rerun on the deployment machine.

The token CLI exercises the default allocation-returning API. The reusable
`out`/`final_state_out` and explicit `inplace_final_state` modes are intended for
allocation-free serving loops and should be measured as a separate experiment;
the current publication schema does not encode allocation mode.

## Results (2026-08-26)

All numbers are microseconds. Speedup is official median divided by CuTe
median. `max diff` is the largest absolute difference from the untimed
CuTe-versus-official comparison.

| Path | B | T | H | Warmup / samples | CuTe median | Official median | Speedup | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chunk forward | 1 | 16 | 16 | 40 / 300 | 22.1 | 180.6 | **8.18x** | 1.55e-3 |
| chunk forward | 1 | 64 | 16 | 40 / 300 | 35.3 | 183.7 | **5.20x** | 1.36e-3 |
| chunk forward | 1 | 128 | 16 | 40 / 300 | 39.4 | 181.9 | **4.62x** | 1.60e-3 |
| chunk forward | 1 | 256 | 16 | 40 / 300 | 49.6 | 183.6 | **3.70x** | 1.39e-3 |
| chunk forward | 1 | 512 | 16 | 40 / 300 | 59.9 | 181.6 | **3.03x** | 1.43e-3 |
| chunk forward | 1 | 1024 | 16 | 40 / 300 | 92.6 | 183.4 | **1.98x** | 1.37e-3 |
| chunk forward | 1 | 2048 | 16 | 40 / 300 | 154.2 | 236.6 | **1.53x** | 1.40e-3 |
| chunk forward | 1 | 4096 | 16 | 20 / 100 | 411.7 | 477.0 | **1.16x** | 1.38e-3 |
| chunk forward | 1 | 8192 | 16 | 20 / 100 | 897.0 | 1033.3 | **1.15x** | 1.41e-3 |
| chunk forward | 1 | 16384 | 16 | 10 / 50 | 1740.9 | 2115.5 | **1.22x** | 1.50e-3 |
| chunk forward | 1 | 32768 | 16 | 10 / 50 | 3440.5 | 4234.9 | **1.23x** | 1.82e-3 |
| chunk forward | 2 | 16 | 16 | 40 / 300 | 24.1 | 180.3 | **7.49x** | 1.55e-3 |
| chunk forward | 2 | 64 | 16 | 40 / 300 | 37.4 | 182.2 | **4.87x** | 1.36e-3 |
| chunk forward | 2 | 128 | 16 | 40 / 300 | 45.6 | 182.9 | **4.01x** | 1.62e-3 |
| chunk forward | 2 | 256 | 16 | 40 / 300 | 62.0 | 184.0 | **2.97x** | 1.40e-3 |
| chunk forward | 2 | 512 | 16 | 40 / 300 | 86.5 | 184.7 | **2.14x** | 1.51e-3 |
| chunk forward | 2 | 1024 | 16 | 40 / 300 | 143.9 | 221.4 | **1.54x** | 1.35e-3 |
| chunk forward | 2 | 2048 | 16 | 40 / 300 | 329.9 | 433.2 | **1.31x** | 1.47e-3 |
| chunk forward | 2 | 4096 | 16 | 20 / 100 | 706.3 | 948.2 | **1.34x** | 1.37e-3 |
| chunk forward | 2 | 8192 | 16 | 20 / 100 | 1369.2 | 1963.7 | **1.43x** | 1.48e-3 |
| chunk forward | 2 | 16384 | 16 | 10 / 50 | 2726.0 | 3964.4 | **1.45x** | 2.11e-3 |
| chunk forward | 2 | 32768 | 16 | 10 / 50 | 5462.1 | 7960.3 | **1.46x** | 1.51e-3 |
| chunk forward | 4 | 16 | 16 | 40 / 300 | 26.1 | 182.8 | **6.99x** | 1.55e-3 |
| chunk forward | 4 | 64 | 16 | 40 / 300 | 47.6 | 183.8 | **3.86x** | 1.57e-3 |
| chunk forward | 4 | 128 | 16 | 40 / 300 | 62.3 | 182.0 | **2.92x** | 1.62e-3 |
| chunk forward | 4 | 256 | 16 | 40 / 300 | 100.0 | 183.2 | **1.83x** | 1.68e-3 |
| chunk forward | 4 | 512 | 16 | 40 / 300 | 127.5 | 219.5 | **1.72x** | 1.56e-3 |
| chunk forward | 4 | 1024 | 16 | 40 / 300 | 321.5 | 415.8 | **1.29x** | 1.60e-3 |
| chunk forward | 4 | 2048 | 16 | 40 / 300 | 701.9 | 926.9 | **1.32x** | 1.64e-3 |
| chunk forward | 4 | 4096 | 16 | 20 / 100 | 1337.1 | 1921.0 | **1.44x** | 1.47e-3 |
| chunk forward | 4 | 8192 | 16 | 20 / 100 | 2612.2 | 3885.4 | **1.49x** | 2.06e-3 |
| chunk forward | 4 | 16384 | 16 | 10 / 50 | 5206.8 | 7807.1 | **1.50x** | 2.28e-3 |
| chunk forward | 4 | 32768 | 16 | 10 / 50 | 10333.0 | 15670.2 | **1.52x** | 1.83e-3 |
| chunk backward | 1 | 16 | 16 | 40 / 300 | 113.9 | 278.0 | **2.44x** | 1.95e-3 |
| chunk backward | 1 | 64 | 16 | 40 / 300 | 122.1 | 279.2 | **2.29x** | 2.14e-3 |
| chunk backward | 1 | 128 | 16 | 40 / 300 | 117.6 | 278.0 | **2.36x** | 2.08e-3 |
| chunk backward | 1 | 256 | 16 | 40 / 300 | 123.3 | 281.2 | **2.28x** | 2.44e-3 |
| chunk backward | 1 | 512 | 16 | 40 / 300 | 134.6 | 281.6 | **2.09x** | 2.93e-3 |
| chunk backward | 1 | 1024 | 16 | 40 / 300 | 258.1 | 396.3 | **1.54x** | 2.01e-3 |
| chunk backward | 1 | 2048 | 16 | 40 / 300 | 556.0 | 707.6 | **1.27x** | 3.91e-3 |
| chunk backward | 1 | 4096 | 16 | 20 / 100 | 1227.9 | 1466.2 | **1.19x** | 3.91e-3 |
| chunk backward | 1 | 8192 | 16 | 20 / 100 | 2581.4 | 3128.3 | **1.21x** | 1.95e-3 |
| chunk backward | 1 | 16384 | 16 | 10 / 50 | 5251.6 | 6346.0 | **1.21x** | 1.95e-3 |
| chunk backward | 1 | 32768 | 16 | 10 / 50 | 10649.0 | 12638.2 | **1.19x** | 3.91e-3 |
| chunk backward | 2 | 16 | 16 | 40 / 300 | 118.3 | 276.7 | **2.34x** | 2.44e-3 |
| chunk backward | 2 | 64 | 16 | 40 / 300 | 123.3 | 278.5 | **2.26x** | 2.14e-3 |
| chunk backward | 2 | 128 | 16 | 40 / 300 | 123.4 | 281.8 | **2.28x** | 2.93e-3 |
| chunk backward | 2 | 256 | 16 | 40 / 300 | 134.4 | 279.3 | **2.08x** | 2.44e-3 |
| chunk backward | 2 | 512 | 16 | 40 / 300 | 242.6 | 389.7 | **1.61x** | 3.91e-3 |
| chunk backward | 2 | 1024 | 16 | 40 / 300 | 520.1 | 642.0 | **1.23x** | 2.44e-3 |
| chunk backward | 2 | 2048 | 16 | 40 / 300 | 1186.8 | 1308.5 | **1.10x** | 2.93e-3 |
| chunk backward | 2 | 4096 | 16 | 20 / 100 | 2462.8 | 2825.5 | **1.15x** | 3.91e-3 |
| chunk backward | 2 | 8192 | 16 | 20 / 100 | 5043.1 | 5746.1 | **1.14x** | 3.91e-3 |
| chunk backward | 2 | 16384 | 16 | 10 / 50 | 10175.7 | 11403.6 | **1.12x** | 3.91e-3 |
| chunk backward | 2 | 32768 | 16 | 10 / 50 | 20441.3 | 23034.4 | **1.13x** | 3.91e-3 |
| chunk backward | 4 | 16 | 16 | 40 / 300 | 133.6 | 276.1 | **2.07x** | 2.44e-3 |
| chunk backward | 4 | 64 | 16 | 40 / 300 | 132.6 | 280.0 | **2.11x** | 3.91e-3 |
| chunk backward | 4 | 128 | 16 | 40 / 300 | 133.5 | 279.9 | **2.10x** | 3.91e-3 |
| chunk backward | 4 | 256 | 16 | 40 / 300 | 234.4 | 391.3 | **1.67x** | 3.91e-3 |
| chunk backward | 4 | 512 | 16 | 40 / 300 | 506.0 | 622.7 | **1.23x** | 3.91e-3 |
| chunk backward | 4 | 1024 | 16 | 40 / 300 | 1147.8 | 1251.4 | **1.09x** | 3.91e-3 |
| chunk backward | 4 | 2048 | 16 | 40 / 300 | 2377.6 | 2711.6 | **1.14x** | 3.91e-3 |
| chunk backward | 4 | 4096 | 16 | 20 / 100 | 4860.8 | 5508.0 | **1.13x** | 3.91e-3 |
| chunk backward | 4 | 8192 | 16 | 20 / 100 | 9793.4 | 10958.4 | **1.12x** | 3.91e-3 |
| chunk backward | 4 | 16384 | 16 | 10 / 50 | 19658.2 | 22164.8 | **1.13x** | 3.91e-3 |
| token forward | 1 | 1 | 16 | 25 / 100 | 13.1 | 25.0 | **1.91x** | 2.24e-8 |
| token forward | 1 | 2 | 16 | 25 / 100 | 13.1 | 24.8 | **1.90x** | 2.98e-8 |
| token forward | 1 | 4 | 16 | 25 / 100 | 14.8 | 27.0 | **1.82x** | 5.96e-8 |
| token forward | 1 | 8 | 16 | 25 / 100 | 16.9 | 29.1 | **1.72x** | 6.71e-8 |
| token forward | 1 | 16 | 16 | 25 / 100 | 21.0 | 35.1 | **1.67x** | 1.91e-6 |
| token forward | 1 | 32 | 16 | 25 / 100 | 27.2 | 46.7 | **1.72x** | 1.53e-5 |
| token forward | 1 | 64 | 16 | 25 / 100 | 43.4 | 71.8 | **1.66x** | 1.26e-3 |
| token forward | 1 | 128 | 16 | 25 / 100 | 47.4 | 120.3 | **2.54x** | 8.52e-4 |
| token forward | 2 | 1 | 16 | 25 / 100 | 13.1 | 24.6 | **1.88x** | 2.98e-8 |
| token forward | 2 | 2 | 16 | 25 / 100 | 14.8 | 24.6 | **1.66x** | 2.98e-8 |
| token forward | 2 | 4 | 16 | 25 / 100 | 14.9 | 26.7 | **1.79x** | 1.91e-6 |
| token forward | 2 | 8 | 16 | 25 / 100 | 17.0 | 29.2 | **1.71x** | 9.54e-7 |
| token forward | 2 | 16 | 16 | 25 / 100 | 23.1 | 35.3 | **1.53x** | 1.91e-6 |
| token forward | 2 | 32 | 16 | 25 / 100 | 31.3 | 47.5 | **1.52x** | 3.05e-5 |
| token forward | 2 | 64 | 16 | 25 / 100 | 45.5 | 71.2 | **1.57x** | 1.26e-3 |
| token forward | 2 | 128 | 16 | 25 / 100 | 54.6 | 120.9 | **2.22x** | 9.96e-4 |
| token forward | 4 | 1 | 16 | 25 / 100 | 14.8 | 24.7 | **1.67x** | 2.98e-8 |
| token forward | 4 | 2 | 16 | 25 / 100 | 14.9 | 24.7 | **1.65x** | 1.19e-7 |
| token forward | 4 | 4 | 16 | 25 / 100 | 16.9 | 27.9 | **1.65x** | 1.91e-6 |
| token forward | 4 | 8 | 16 | 25 / 100 | 19.1 | 30.2 | **1.59x** | 3.05e-5 |
| token forward | 4 | 16 | 16 | 25 / 100 | 25.1 | 38.1 | **1.52x** | 3.05e-5 |
| token forward | 4 | 32 | 16 | 25 / 100 | 35.3 | 50.7 | **1.44x** | 6.10e-5 |
| token forward | 4 | 64 | 16 | 25 / 100 | 55.7 | 78.0 | **1.40x** | 1.26e-3 |
| token forward | 4 | 128 | 16 | 25 / 100 | 70.7 | 132.6 | **1.88x** | 1.12e-3 |

Forward stays ahead at every measured B1/B2/B4 point through T=32768; its
longest-sequence speedups are 1.23x, 1.46x, and 1.52x respectively. Backward is
also ahead at every supported measured point: all 11 B1 and B2 lengths and all
10 B4 lengths through T=16384. The narrowest measured margin is 1.0902x at
B4/T1024.

The canonical B1/H16 T=64 point supplies 64 chunk-head CTAs and therefore takes
the CTA-aware compact-WY parameter VJP, together with the T=64 MMA boundary
scan. T=128 additionally enables compact BF16 checkpoints and measures 117.6 us.
The checkpointed path removes the old T=128 correctness cap. B4/T32768
backward is unsupported and not benchmarked because one saved state-boundary
tensor exceeds CuTe's 4-GiB per-launch byte-address range.

Token forward is ahead at all 24 measured H16 points. Its public-call speedup
at T=1 is 1.91x, 1.88x, and 1.67x for B1, B2, and B4 respectively; at T=128 it
is 2.54x, 2.22x, and 1.88x. These rows include the default output and
final-state allocations on both public paths.

For full-chunk BF16 training at T>=128, forward checkpoints Y, raw Q-gamma,
K-tail, A-qk, and state boundaries in BF16; the value auxiliary and chunk
decay remain FP32. At T>=512, a separate compact Q-effective scratch lets
training use the rearranged long-forward identity without changing the raw
Q-gamma or A-qk checkpoint bits.
At T=64 and T>=128, the training scan replaces U with FP32 R after its final
use so backward can consume the residual without another state product.
The boundary stage precomputes all independent `A_qk.T @ dO` products before
its reverse scan, stages Y/Q-gamma/K-tail with 128-bit `cp.async`, and shares
each decay value through warp shuffles. The ordered scan selects V16 when
`batch * heads >= 24`, and also for the short midrange where
`16 <= batch * heads < 24` and `T <= 512`; other shapes use V8. Both variants
retain eight K-split warps. V16 halves duplicated Y/Q-gamma/K-tail reads, while
V8 exposes twice as many CTAs for long underfilled grids. The scan writes BF16
`dR` and `dS` checkpoints for the tensor-core consumers while returning the
exact FP32 `dS0` separately. T=64--127, partial tails, and FP16 retain FP32
boundaries.

The local compact-WY VJP keeps gamma in FP32 but stores its persistent E and
K-bar MMA operands in the input dtype. Dual large-state and square products,
paired K16 updates, and producer epilogues reduce the full-chunk schedule to
12 ordered launches. The saved forward residual removes one launch at T=64 and
T>=128. Compact BF16 also folds the state/gradient decay dot into an existing
state-product kernel, leaving 10 launches for full chunks at T>=128; T=64
retains the separate dot and uses 11. The final chain shares one reciprocal
gamma across its K and decay-gradient expressions.

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

### Forward prepare and shared-memory scan

The CUDA timeline identified the ordered inter-chunk scan as the long-forward
bottleneck. At B1/T8192/H16, the baseline scan took 603.577 us (67.1% of the
899.958 us kernel sum), while independent WY preparation took 296.381 us.
V16 already beat a forced V8 schedule at this shape, so simply doubling the CTA
grid did not solve the serial-scan cost.

Preparation now evaluates only the causal score triangle, pairs low and high
rows so every full-chunk warp owns seventeen causal cells, and reuses the final
per-token exponential as the chunk decay. In the scan, Y/Q partials occupy
separate contiguous shared-memory planes, the residual tile uses an aligned
padded stride, and the last iteration no longer reloads its own Y/Q tile as a
dummy `cp.async` group. A five-capture post-change median reduced preparation
to 279.837 us and the scan to 596.246 us; the paired kernel sum median was
875.573 us (-2.7%).

The independently clock-stabilized publication rows show up to a 5.4% public
call reduction, with unchanged untimed official-comparison maxima:

| B | T | Previous CuTe | Current CuTe | CuTe change | Official | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 51.680 | 49.632 | **-4.0%** | 183.648 | **3.70x** |
| 2 | 8192 | 1437.728 | 1369.168 | **-4.8%** | 1963.728 | **1.43x** |
| 4 | 128 | 65.792 | 62.256 | **-5.4%** | 181.952 | **2.92x** |
| 4 | 32768 | 10542.496 | 10333.040 | **-2.0%** | 15670.160 | **1.52x** |

### Forward ordered-scan algebra crossover

Nsight Systems isolated the two launches of B1/T256/H16 forward. With the
original expression, the ordered inter-chunk scan took 29.888 us, or 74.4% of
the 40.160 us kernel sum; independent chunk preparation took 10.272 us. Moving
`A_qk @ R` into preparation through
`O = (Q_gamma - A_qk Y) S + A_qk U` changed preparation only to 10.464 us but
cut the ordered scan to 17.568 us (-41.2%) and the kernel sum to 28.032 us
(-30.2%).

The selector now uses this existing algebra schedule for forward-only BF16
calls from three full chunks onward, while retaining the measured 32-chunk
crossover for training because preserving raw backward operands requires an
extra Q-effective scratch. A same-process T48 sweep reduced B1/B2/B4 latency
by 6.9%, 5.3%, and 7.6%, respectively. FP16 keeps the original expression.
The then-refreshed public-call rows (before the newer shared-memory changes
above) were:

| B | T | Previous CuTe | Post-algebra CuTe | CuTe change | Official | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 39.360 | 35.296 | **-10.3%** | 182.048 | **5.16x** |
| 1 | 128 | 45.568 | 39.424 | **-13.5%** | 183.440 | **4.65x** |
| 1 | 256 | 63.040 | 51.680 | **-18.0%** | 182.016 | **3.52x** |
| 2 | 64 | 41.440 | 39.392 | **-4.9%** | 184.016 | **4.67x** |
| 2 | 128 | 53.728 | 47.584 | **-11.4%** | 182.192 | **3.83x** |
| 2 | 256 | 76.224 | 62.064 | **-18.6%** | 182.096 | **2.93x** |
| 4 | 64 | 55.808 | 49.568 | **-11.2%** | 182.688 | **3.69x** |
| 4 | 128 | 80.384 | 65.792 | **-18.2%** | 181.424 | **2.76x** |
| 4 | 256 | 129.536 | 102.912 | **-20.6%** | 182.880 | **1.78x** |

The before/after publication rows are separate clock-stabilized processes; the
Nsight capture above isolates the kernel-level cause. Untimed official
comparisons kept the same maximum differences as the prior rows. Direct
reference checks across T=48/49/63/64/65/127/128/129 observed at most
`1.22e-4` output difference between algebra schedules and unchanged final
states within the existing BF16 numerical contract.

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
former r1 canonical 132.528/280.768 us row was a separate clock-stabilized
40/300 aggregate. The current 122.096/279.232 us row additionally includes the
saved-R and reciprocal-chain changes documented below, so the same-session
table remains the appropriate evidence for the boundary/VJP dispatch itself.

### Zero-state T=1 token closed form

This experiment uses B1/T1/H16 with `initial_state=None`, unlike the canonical
B1/T1/H16 graph point, which enables an initial state. It therefore remains a
supplemental allocation-path experiment rather than another canonical point.

| Call boundary / implementation | Warmup / samples | Median | Minimum | Relative result |
|---|---:|---:|---:|---:|
| public CuTe, closed form | 25 / 100 | 12.640 | 12.064 | **1.968x vs official** |
| direct CuTe, generic recurrence | 25 / 100 | 7.424 | 4.448 | baseline |
| direct CuTe, closed form | 25 / 100 | 6.912 | 4.416 | **1.074x vs generic** |
| public official Triton | 25 / 100 | 24.880 | 18.176 | baseline |

The public CuTe/official comparison, the generic/closed-form comparison, and
the public/direct closed-form comparison were all bit-identical for both the
output and final state.

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

### Saved forward residual

The saved-R change was measured against clean `f4ed53b` on the same RTX PRO
6000, with 30 warmups and 200 CUDA-event samples at B1/T1024/H16 BF16. The
public backward median fell from 306.1--306.2 us to 279.8--280.6 us
(about 8.5%), while full forward-plus-backward training fell from
376.3--377.4 us to 364.0 us (about 3.4%). A direct compact-WY VJP A/B on the
same input measured 190.688 us for the legacy U path and 163.200 us for the
saved-R path (14.4%).

For an output-only loss at the same shape, specializing the absent terminal
VJP measured 261.9--263.2 us versus 278.6--279.5 us when a fresh state-sized
zero terminal VJP was materialized, a 5.5--6.3% reduction.

### Preparation occupancy and residual-chain reuse

Nsight Systems isolated the two forward launches at B1/T4096/H16. The original
512-thread preparation CTA used 58 registers per thread and accounted for
179.569 us of the 427.554 us kernel total. Eight warps now cover the sixteen
row products in two passes, so the same 256 solve threads fit in a 256-thread
CTA. Preparation fell to 150.629 us (-16.1%) and the two-kernel total to
400.072 us (-6.4%). Together with bypassing the autograd wrapper when no
gradient edge is needed, the public canonical row fell from 445.6 to 415.9 us.
All 33 forward rows improved; their median reduction is 7.3% and the largest is
9.4%.

For B1/T32768/H16 backward, the baseline 12-launch kernel sum was 10.597 ms.
The reverse boundary scan used 2.429 ms (22.9%), the dual dQ-gamma/dY product
2.332 ms (22.0%), the final parameter chain 1.794 ms (16.9%), and the dK-tail
product 0.897 ms (8.5%). Together these four stages accounted for 70.3%. The
boundary scan launches 256 CTAs and each CTA walks every chunk serially, while
the two large products remain the main fusion opportunity. The low-risk changes
below reduce the chain and remove a redundant state product without changing
that ordered recurrence.

The forward residual checkpoint is now written at T=64 and T>=128, rather than
only after the T=512 training-algebra crossover. The extra coalesced store
occurs after U's last read and removes `Y @ S0` plus one operand from dLower.
The same-capture GPU kernel sums were:

| Path | Legacy U | Saved R | Change |
|---|---:|---:|---:|
| B1/T64 backward | 59.494 | 50.102 | **-15.8%** |
| B1/T64 training | 77.436 | 68.437 | **-11.6%** |
| B1/T128 backward | 54.227 | 48.114 | **-11.3%** |

At the public boundary, B1/T64 backward moved from 132.5 to 122.1 us and
B4/T256 from 259.0 to 234.4 us. For the remaining long-backward bottleneck, the
final chain now computes one reciprocal gamma per token and reuses it for the K
and decay gradients. At B4/T1024 the chain fell from 218.487 to 209.393 us
(-4.2%), the 12-kernel total from 1096.805 to 1084.834 us (-1.1%), and the
published public median from 1157.2 to 1147.8 us. The tracked official
comparisons remain within `3.91e-3` maximum absolute gradient difference.

### CTA granularity and compact-backward retuning

The generic recurrent kernel was compiled side by side with its former
four-warp/V32 schedule. Each row used 25 warmups and 100 alternating CUDA-event
samples through allocation-returning `recurrent_gdn2` calls. The two-warp/V16
schedule keeps the same eight columns per warp and total warp work, but doubles
the CTA grid and reduces wave-tail underfill.

| Shape | Four-warp/V32 | Two-warp/V16 | Change |
|---|---:|---:|---:|
| B1/T32/H16 | 31.104 | 26.976 | **-13.3%** |
| B2/T32/H16 | 31.104 | 31.104 | 0.0% |
| B4/T32/H16 | 41.216 | 35.200 | **-14.6%** |
| B1/T63/H16 | 47.488 | 41.312 | **-13.0%** |
| B2/T63/H16 | 49.408 | 47.456 | **-4.0%** |
| B4/T63/H16 | 64.640 | 56.384 | **-12.8%** |

The old and current compact-backward selectors were then kept in the same
process and measured with 12 warmups plus 100 alternating samples. The current
schedule always folds the state-decay dot for compact BF16 checkpoints and
uses V8 for long boundary scans whose V16 grid cannot fill 188 SMs.

| Shape | Previous schedule | Current schedule | Change |
|---|---:|---:|---:|
| B1/T128/H16 | 128.80 | 125.52 | **-2.55%** |
| B1/T512/H16 | 140.93 | 135.84 | **-3.61%** |
| B1/T1024/H16 | 280.59 | 263.23 | **-6.19%** |
| B1/T2048/H16 | 569.63 | 552.64 | **-2.98%** |
| B2/T256/H16 | 190.61 | 179.82 | **-5.66%** |
| B4/T256/H16 | 313.23 | 291.66 | **-6.89%** |

The recurrent results were bit-identical. Across the backward A/B, the largest
gradient difference was `2.98e-7`.

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
interval, so it is deliberately not drawn as an error bar. All three panels
compare fixed-H16 B1/B2/B4 measurements with the same implementation/batch
dodging, CuTe/official pair connectors, batch-specific line styles, and
speedup rails. Each batch series is connected at log₂-spaced T positions.

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

for batch in 1 2 4; do
  for time in 1 2 4 8 16 32 64 128; do
    uv run gdn2-sm120-bench \
      --mode token-forward --batch "$batch" --time "$time" --heads 16 --dtype bf16 \
      --warmup 25 --repeats 100 --official-repo /tmp/GatedDeltaNet-2 \
      --json "benchmark-results/token-b${batch}-t${time}.json"
  done
done
```

Pass `--json benchmark-results/name.json` to persist a machine-readable result.
The output directory is ignored because measurements are host-specific. Curate
only validated publication runs into `docs/data`; plotting an entire local
output directory fails on duplicate logical shapes instead of silently choosing
between old and new measurements.

## Interpretation

The optimization goal is met across every supported measured chunk point.
Chunk forward is faster at all 33 B1/B2/B4 lengths through T=32768. Backward
is faster at all 11 B1 and B2 lengths and all 10 measured B4 lengths through
T=16384; its narrowest margin is 1.0902x at B4/T1024. B4/T32768 is outside
the current CuTe per-launch address range and is excluded. Token forward is
measured over the same fixed-H16 B1/B2/B4 batch matrix and is faster at all 24
points through T=128.
These are primitive-level latencies, not complete training-throughput
measurements. Reducing checkpoint/workspace memory, improving multi-batch
backward scaling, additional dimensions, packed sequences, and fused
normalization/gates remain separate milestones.

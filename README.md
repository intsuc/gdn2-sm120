# gdn2-sm120

SM120-specialized [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
kernels for [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), developed and
measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

The repository contains three working CUDA paths:

- BT=16 compact-WY chunkwise forward with SM120 warp MMA, an occupancy-aware
  V8/V16 scan, and a specialized algebraic pipeline for long BF16 prefill and
  training;
- a checkpointed training backward that dispatches between short recurrence,
  chunk-parallel recurrence, and compact-WY warp-MMA VJPs, including a
  CTA-aware T=64 specialization;
- a register-resident token/recurrent forward for decoding, with aligned
  128-bit state I/O, a zero-state T=1 closed form, and allocation-free serving
  options.

All three are written in CuTe DSL, compile specifically for `sm_120`, use the
active PyTorch CUDA stream, and cache TVM-FFI executors after the first JIT
compilation. A PyTorch `autograd.Function` connects the chunk forward and
backward kernels.

## Quick start

The project and its lockfile are managed with
[`uv`](https://docs.astral.sh/uv/). Python 3.12 is pinned in
`.python-version`.

```bash
uv sync --locked
uv run python -m gdn2_sm120.env
CUTE_DSL_ARCH=sm_120 uv run python -m gdn2_sm120.smoke
uv run pytest -q
```

Install the additional official-baseline dependencies with:

```bash
uv sync --locked --all-groups
```

The first call for a new static chunk/backward shape performs JIT compilation;
benchmark and production warm paths reuse an in-process compiled executor.

## API

The optimized specialization accepts contiguous tensors and benefits from
16-byte alignment.
Sequence tensors use `[batch, time, heads, 128]` layout. Q/K/V/erase/write
tensors use BF16 or FP16, log-decay `g` may use FP32, and state tensors use FP32
`[batch, heads, 128, 128]`. The token path detects aligned state buffers for
128-bit loads and stores and retains a scalar fallback for contiguous unaligned
state views. The output scale defaults to `1/sqrt(128)`.

```python
import torch

from gdn2_sm120 import chunk_gdn2, recurrent_gdn2

# Training: backward() selects the native CuTe schedule by sequence length.
output, final_state = chunk_gdn2(
    q,
    k,
    v,
    g,
    beta,
    w,
    initial_state,
    output_final_state=True,
)
loss = output.float().square().mean() + final_state.square().mean()
loss.backward()

# Decoding or short recurrent evaluation: forward-only.
output, final_state = recurrent_gdn2(q, k, v, g, beta, w, initial_state)

# Allocation-free serving: reuse both destinations.
output_buffer = torch.empty_like(v)
state_buffer = torch.empty_like(initial_state)
output, final_state = recurrent_gdn2(
    q,
    k,
    v,
    g,
    beta,
    w,
    initial_state,
    out=output_buffer,
    final_state_out=state_buffer,
)

# Stateful decoding: explicitly update the supplied state in place.
output, initial_state = recurrent_gdn2(
    q,
    k,
    v,
    g,
    beta,
    w,
    initial_state,
    out=output_buffer,
    inplace_final_state=True,
)
```

`final_state_out` and `inplace_final_state=True` are mutually exclusive. The
default remains allocation-based and never mutates `initial_state`; explicit
output buffers are validated for shape, dtype, device, contiguity, and unsafe
storage overlap.

For the common first decode token (`T=1`, `initial_state=None`), the token path
uses the exact closed form `S = outer(k, w * v)` and
`o = scale * dot(q, k) * (w * v)`. The public shape/dtype/device checks for
`g` and `beta` still apply, but this specialization does not read either tensor
because decay and erase act only on the absent previous state.

The primitive expects already-activated erase/write gates and log-decay.
Q/K L2 normalization, gate projections, grouped-value head expansion, packed
variable-length sequences, and fused gate activation are not yet part of this
first specialization. Unsupported devices and shapes fail explicitly; there
is no silent fallback.

For gradient-enabled calls of 64 tokens or more, the forward saves states at
every BT=16 boundary. BF16 full chunks at T>=128 use compact BF16 `S`/`dS`
checkpoints, while the reverse scan preserves the exact FP32 `dS0` separately;
FP16, T=64--127, and partial-tail paths retain FP32 boundaries. Backward uses
these local checkpoints instead of inverting the complete sequence from one
rounded final state. At T=64 it uses the full compact-WY tensor-core VJP when
the batch/head grid supplies at least 64 chunk-head CTAs; smaller T=64 grids
retain the chunk-local VJP. T>=128 always dispatches to compact-WY. Forward-only
calls do not allocate training checkpoints.

## Benchmark against the official Triton implementation

The benchmark loader pins the official repository to commit
`95709fc250357c2dd109361c353192f2aa5913f9` and refuses another commit unless
explicitly overridden. The compatible Flash Linear Attention dependency is
also commit-pinned in `uv.lock`.

```bash
git clone https://github.com/NVlabs/GatedDeltaNet-2.git /tmp/GatedDeltaNet-2
git -C /tmp/GatedDeltaNet-2 checkout 95709fc250357c2dd109361c353192f2aa5913f9

uv run gdn2-sm120-bench \
  --mode chunk-forward --batch 1 --time 16 --heads 16 --dtype bf16 \
  --official-repo /tmp/GatedDeltaNet-2
uv run gdn2-sm120-bench \
  --mode chunk-backward --batch 1 --time 16 --heads 16 --dtype bf16 \
  --official-repo /tmp/GatedDeltaNet-2
uv run gdn2-sm120-bench \
  --mode chunk-training --batch 1 --time 256 --heads 16 --dtype bf16 \
  --official-repo /tmp/GatedDeltaNet-2
uv run gdn2-sm120-bench \
  --mode token-forward --batch 1 --time 1 --heads 32 --dtype bf16 \
  --official-repo /tmp/GatedDeltaNet-2
```

Latency excludes compilation and Triton autotuning. Both implementations use
the same pre-normalized Q/K, FP32 state and log-decay, output scale, public
Python/autograd call boundary, warmup count, and CUDA-event measurement. An
untimed check compares outputs, state, or all gradients before timing. See
[`docs/benchmarks.md`](docs/benchmarks.md) for the measured matrix and caveats.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmark-results-sm120-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/benchmark-results-sm120.png">
  <img alt="Gated DeltaNet-2 SM120 benchmark latency comparison" src="docs/assets/benchmark-results-sm120.png">
</picture>

The light/fallback and dark-theme figures are generated from the tracked,
validated
[`docs/data/benchmark-results-sm120.json`](docs/data/benchmark-results-sm120.json)
suite. Regenerate both without installing the official benchmark stack:

```bash
uv run --group visualization gdn2-sm120-plot \
  docs/data/benchmark-results-sm120.json \
  --output docs/assets/benchmark-results-sm120.png
```

`--output` names the light/fallback figure; the plotter also writes the sibling
`docs/assets/benchmark-results-sm120-dark.png`.

When every token-forward point has the same batch and head count, its panel is
a connected sequence-length sweep with log₂-spaced T positions and a speedup
label at every sample. Mixed B/H token shapes remain independent grouped bars,
so the figure does not imply a scaling curve across different workloads.

Representative BF16 medians on the target workstation are:

| Path | Shape | CuTe SM120 | Official Triton | Speedup |
|---|---:|---:|---:|---:|
| chunk forward | B1 T16 H16 | 35.3 us | 181.8 us | **5.15x** |
| chunk forward | B1 T512 H16 | 65.9 us | 181.6 us | **2.76x** |
| chunk forward | B1 T2048 H16 | 172.5 us | 236.4 us | **1.37x** |
| chunk forward | B1 T16384 H16 | 1889.5 us | 2111.5 us | **1.12x** |
| chunk forward | B1 T32768 H16 | 3725.3 us | 4231.9 us | **1.14x** |
| chunk backward | B1 T16 H16 | 112.1 us | 274.6 us | **2.45x** |
| chunk backward | B1 T64 H16 | 175.5 us | 399.6 us | **2.28x** |
| chunk backward | B1 T512 H16 | 148.8 us | 284.0 us | **1.91x** |
| chunk backward | B1 T2048 H16 | 623.7 us | 708.6 us | **1.14x** |
| chunk backward | B1 T16384 H16 | 5810.6 us | 6353.5 us | **1.09x** |
| chunk backward | B1 T32768 H16 | 11737.6 us | 12638.0 us | **1.08x** |
| token forward | B1 T1 H32 | 13.3 us | 25.4 us | **1.91x** |
| token forward | B1 T128 H32 | 84.3 us | 120.2 us | **1.43x** |

## Why FROST is not copied directly

The cuDNN Frontend FROST GDN2 kernel is an important scheduling reference, but
its current prefill design uses 16 warps, about 205 KiB of shared memory,
Tensor Memory, and `tcgen05` instructions for the SM100/SM103 family. SM120 is
a distinct Blackwell family: the RTX PRO 6000 path instead uses register
accumulators, warp shuffles, value/key partitioning, and
`MmaF16BF16Op` warp MMA. The forward state scan and the long-sequence backward
therefore use a schedule designed for SM120 rather than a direct FROST port.

For long BF16 training, Y, raw Q-gamma, K-tail, A-qk, and the persistent
E/K-bar MMA operands use BF16, while U, chunk decay, and gamma remain FP32.
At T>=512 the forward scan consumes a temporary compact Q-effective scratch
for the rearranged output identity while preserving raw Q-gamma and A-qk
checkpoint bits for backward. Backward first precomputes every independent
`A_qk.T @ dO` product, then runs a reverse boundary scan with 128-bit
`cp.async` staging and shuffle-cached decay. It emits compact BF16 `dR` and
`dS` operands while retaining `dS0` in FP32. The chunk-local compact-WY graph
combines paired products and producer epilogues into 12 launches, or 11 at
T>=2048 on the full-chunk BF16 path where the state/gradient decay dot is folded
into a large state-product kernel.

The implementation is independently derived from the paper's equations. No
source from the official GDN2 repository (NVIDIA Source Code License-NC) is
vendored or copied. FROST and CUTLASS are referenced at fixed revisions in
[`docs/design.md`](docs/design.md).

## Project status

This is an alpha, shape-specialized kernel project rather than a drop-in
replacement for every official option. The primary production shape
`K=V=128` is implemented and numerically checked in BF16/FP16, including FP32
log-decay, optional initial state, final-state VJPs, empty recurrent sequences,
non-default CUDA streams, unaligned contiguous recurrent states, reusable
output buffers, and explicit in-place recurrent state updates. In the measured
B1/H16 BF16 sweep, both chunk forward and chunk backward remain faster than the
official path at every sampled length through T=32768. Backward speedup ranges
from 2.45x at T=16 to 1.08x at T=32768. The fixed B1/H32 token sweep is 1.91x
faster at T=1 and 1.43x faster at T=128. The checkpointed path trades memory
for speed: compact BF16 boundaries halve checkpoint bytes relative to FP32, but
the CuTe path still retains both boundary sets and compact-WY workspace and
therefore uses more memory than the official path. Additional dimensions,
packed sequences, and further reducing checkpoint memory remain optimization
work.

Licensed under Apache-2.0. See [`LICENSE`](LICENSE).

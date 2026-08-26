# gdn2-sm120

SM120-specialized [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
kernels for [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), developed and
measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

The repository contains three working CUDA paths:

- BT=16 compact-WY chunkwise forward with SM120 warp MMA, an occupancy-aware
  V8/V16 scan, and a specialized algebraic pipeline used from the first
  three-chunk BF16 inference shape and for long training sequences;
- a checkpointed training backward that dispatches between short recurrence,
  chunk-parallel recurrence, and compact-WY warp-MMA VJPs, including a
  CTA-aware T=64 specialization;
- a register-resident token/recurrent forward for decoding, with aligned
  128-bit state I/O, a zero-state T=1 closed form, and allocation-free serving
  options.

All three are written in CuTe DSL, compile specifically for `sm_120`, use the
active PyTorch CUDA stream, and cache TVM-FFI executors after the first JIT
compilation. A PyTorch `autograd.Function` connects the shape-dispatched
forward path and backward kernels.

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

# Forward-only; the wrapper selects the recurrent or chunk schedule.
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

The public APIs dispatch at their measured crossover. `chunk_gdn2` uses the
recurrent forward below `T=48`, chunk forward for the complete three-chunk
`T=48` case, recurrent forward again for `T=49--63`, and chunk forward at
`T >= 64`. `recurrent_gdn2` keeps the recurrent kernel
for `T < 64`. At `T >= 64` it uses chunk forward when every input, state, and
destination buffer is 16-byte aligned; otherwise it retains the token path.
Reusable output buffers and in-place final-state semantics are preserved across
this dispatch.

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

`g` must contain finite, non-positive log-decays. Within every BT=16 chunk,
each prefix `exp(cumsum(g))` must remain a normal positive FP32 value; keeping
the prefix sum safely above `ln(FLT_MIN)`, approximately `-87.34`, satisfies
that requirement. The SM120 reciprocal instructions flush subnormal inputs and
results, so maintaining this model-value invariant is the caller's
responsibility.

For gradient-enabled calls of 64 tokens or more, the forward saves states at
every BT=16 boundary. At `T >= 128`, sequence auxiliaries are compact even
when the sequence has a partial tail. Full-chunk BF16 sequences use compact
BF16 `S`/`dS` checkpoints, while the reverse scan preserves the exact FP32
`dS0` separately; FP16, `T=64--127`, and partial-tail paths retain FP32
boundaries.
For a partial tail, forward scans the full prefix with MMA and runs only the
last short chunk through the scalar recurrence. At `T >= 128`, reverse
processes that scalar tail first, then scans the full prefix with MMA. At T=64
the full compact-WY tensor-core VJP is used when the batch/head grid supplies
at least 64 chunk-head CTAs; smaller T=64 grids retain the chunk-local VJP.
`T >= 128` always dispatches to compact-WY. Forward-only calls do not allocate
training checkpoints and bypass the custom autograd wrapper entirely.

Below T=64, the short backward uses one-warp V8 value tiles. Its first kernel
writes final-form `dq`/`dk`/`dg`/`dbeta` FP32 partials, leaving a sum-only
reduction that does not reload K/beta/g or recompute `exp(g)`.

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
  --mode token-forward --batch 1 --time 1 --heads 16 --dtype bf16 \
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

## Why FROST is not copied directly

The cuDNN Frontend FROST GDN2 kernel is an important scheduling reference, but
its current prefill design uses 16 warps, about 205 KiB of shared memory,
Tensor Memory, and `tcgen05` instructions for the SM100/SM103 family. SM120 is
a distinct Blackwell family: the RTX PRO 6000 path instead uses register
accumulators, warp shuffles, value/key partitioning, and
`MmaF16BF16Op` warp MMA. The forward state scan and the long-sequence backward
therefore use a schedule designed for SM120 rather than a direct FROST port.

For long BF16 training, Y, raw Q-gamma, K-tail, A-qk, and the persistent
E/K-bar MMA operands use BF16, while the value auxiliary, chunk decay, and
gamma remain FP32.
The latency-bound positive-gamma inversions in forward WY preparation and the
terminal compact-WY parameter chain use the SM120 approximate reciprocal while
their surrounding factor and gradient arithmetic remain FP32. The T>1
value-tiled short path and independent chunk-local backward refine the hardware
reciprocal of `exp(g)` once with Newton's method before reconstructing a
previous state; the ordered d-state gradient recurrence still uses the FP32
decay itself, and the fused T=1 path retains precise division. Results follow
the validated BF16/FP16 tolerance contract rather than bit-exact equivalence to
precise FP32 division.
Forward-only BF16 calls use the rearranged output identity from three full
chunks onward, moving the independent A-qk products out of the ordered state
scan. The WY preparation skips unused upper-triangular products and balances
causal rows across warps, while the ordered scan uses bank-aware Y/Q and
residual shared-memory layouts and omits its redundant final prefetch. With at
least 32 full chunks and at most 16 batch-heads, each warp broadcasts its local
decay rows and overlaps the state multiply with the required residual barrier.
When training contains at least 32 full chunks, including a sequence
with a partial tail, that scan consumes a temporary compact Q-effective
scratch while preserving raw Q-gamma and A-qk checkpoint bits for backward.
At T=64 and T>=128, each training value-tile CTA safely replaces its disjoint
U columns after their last use with the FP32 residual checkpoint
`R = U - Y @ S0`; forward-only partial calls retain the tail's raw Q-gamma and
A-qk for its scalar scan. Backward first precomputes
every independent `A_qk.T @ dO` product, then runs a reverse boundary scan
with 128-bit `cp.async` staging and shuffle-cached decay. The full-chunk BF16
path emits compact BF16 `dR` and `dS` operands while retaining `dS0` in FP32;
partial tails retain FP32 boundaries and `dR`. The ordered scan uses V16 when
`batch * heads >= 24`, plus the short midrange where
`16 <= batch * heads < 24` and `T <= 512`; other shapes use V8. V16 halves
duplicated Y/Q-gamma/K-tail reads, while V8 exposes twice as many CTAs for long
underfilled grids; both retain the same eight K-split warps. The standard
chunk-local compact-WY graph combines paired products and producer epilogues
into 12 launches. The saved forward R skips its `Y @ S0` launch and reduces the
paired dLower product to `-tril(dZ @ R.T)`. T=64 therefore uses 11 launches,
while full-chunk BF16 folds the state/gradient decay dot into the shared-S0
product and uses 10 launches at `T >= 128`. The final gate chain computes one
approximate reciprocal of gamma per token and reuses it across the K and decay
gradients instead of issuing three precise divisions.

When a loss does not consume the final state, the long backward passes a
zero-terminal specialization into the boundary scan. It initializes the
terminal register state directly and avoids allocating, clearing, and reading
a `[B, H, 128, 128]` FP32 zero tensor.

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
B1/B2/B4, H16 BF16 sweeps, chunk forward remains faster than the official path
at all 33 sampled lengths through T=32768. Chunk backward remains faster for
all 11 B1 and B2 points and all 10 measured B4 points through T=16384. The
narrowest forward and backward margins are 1.36x at B1/T8192 and
1.08x at B2/T2048, respectively. B4/T32768 backward is not benchmarked
because one saved state-boundary tensor exceeds CuTe's 4-GiB per-launch
byte-address range. Token forward is measured with the same fixed H16 and
B1/B2/B4 batch matrix and remains faster at all 24 points through T=128. The
checkpointed path trades memory for speed: compact BF16
boundaries halve checkpoint bytes relative to FP32, but the CuTe path still
retains both boundary sets and compact-WY workspace and therefore uses more
memory than the official path. Additional dimensions, packed sequences, and
further reducing checkpoint memory remain optimization work.

Licensed under Apache-2.0. See [`LICENSE`](LICENSE).

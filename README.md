# gdn2-sm120

SM120-specialized [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
kernels for [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791), developed and
measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition.

The repository contains three working CUDA paths:

- BT=16 WY chunkwise forward for training/prefill;
- a value-tiled training backward (up to 128 tokens) with a fused
  single-token specialization;
- a register-resident token/recurrent forward for decoding.

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

The optimized specialization accepts contiguous sequence tensors in
`[batch, time, heads, 128]` layout. Q/K/V/erase/write tensors use BF16 or FP16,
log-decay `g` may use FP32, and state tensors use FP32
`[batch, heads, 128, 128]`.

```python
from gdn2_sm120 import chunk_gdn2, recurrent_gdn2

# Training up to 128 tokens: backward() dispatches to native CuTe.
output, final_state = chunk_gdn2(
    q,
    k,
    v,
    g,
    beta,
    w,
    initial_state,
    scale=0.125,
    output_final_state=True,
)
loss = output.float().square().mean() + final_state.square().mean()
loss.backward()

# Decoding or short recurrent evaluation: forward-only.
output, final_state = recurrent_gdn2(q, k, v, g, beta, w, initial_state, scale=0.125)
```

The primitive expects already-activated erase/write gates and log-decay.
Q/K L2 normalization, gate projections, grouped-value head expansion, packed
variable-length sequences, and fused gate activation are not yet part of this
first specialization. Unsupported devices and shapes fail explicitly; there
is no silent fallback.

The current final-state-only backward is deliberately capped at 128 tokens.
Longer training needs forward checkpoints or a full WY backward; it raises
instead of returning numerically unstable early-token gradients. Forward-only
`chunk_forward` is not subject to this cap.

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
  --mode token-forward --batch 1 --time 1 --heads 32 --dtype bf16 \
  --official-repo /tmp/GatedDeltaNet-2
```

Latency excludes compilation and Triton autotuning. Both implementations use
the same pre-normalized Q/K, FP32 state and log-decay, output scale, public
Python/autograd call boundary, warmup count, and CUDA-event measurement. An
untimed check compares outputs, state, or all gradients before timing. See
[`docs/benchmarks.md`](docs/benchmarks.md) for the measured matrix and caveats.

![Gated DeltaNet-2 SM120 benchmark latency comparison](docs/assets/benchmark-results-sm120.png)

The figure is generated from the tracked, validated
[`docs/data/benchmark-results-sm120.json`](docs/data/benchmark-results-sm120.json)
suite. Regenerate it without installing the official benchmark stack:

```bash
uv run --group visualization gdn2-sm120-plot \
  docs/data/benchmark-results-sm120.json \
  --output docs/assets/benchmark-results-sm120.png
```

Representative BF16 medians on the target workstation are:

| Path | Shape | CuTe SM120 | Official Triton | Speedup |
|---|---:|---:|---:|---:|
| chunk forward | B1 T16 H16 | 43.5 us | 189.7 us | **4.36x** |
| chunk backward | B1 T16 H16 | 112.1 us | 277.1 us | **2.47x** |
| token forward | B1 T1 H32 | 21.0 us | 25.2 us | **1.20x** |

## Why FROST is not copied directly

The cuDNN Frontend FROST GDN2 kernel is an important scheduling reference, but
its current prefill design uses 16 warps, about 205 KiB of shared memory,
Tensor Memory, and `tcgen05` instructions for the SM100/SM103 family. SM120 is
a distinct Blackwell family: the RTX PRO 6000 path instead uses register
accumulators, warp shuffles, and value-dimension partitioning. The isolated
dense products leave a clear path to SM120 warp MMA in a later specialization.

The implementation is independently derived from the paper's equations. No
source from the official GDN2 repository (NVIDIA Source Code License-NC) is
vendored or copied. FROST and CUTLASS are referenced at fixed revisions in
[`docs/design.md`](docs/design.md).

## Project status

This is an alpha, shape-specialized kernel project rather than a drop-in
replacement for every official option. The primary production shape
`K=V=128` is implemented and numerically checked in BF16/FP16, including FP32
log-decay, optional initial state, final-state VJPs, empty recurrent sequences,
and non-default CUDA streams. Chunk forward remains faster through T=256 in
the measured B1/H16 sweep, but crosses below the official implementation at
T=512. Backward crosses between T=64 and T=128 and is correctness-capped at
T=128. Long-sequence WY/checkpoint backward and additional head dimensions
remain optimization work.

Licensed under Apache-2.0. See [`LICENSE`](LICENSE).

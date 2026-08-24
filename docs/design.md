# Design notes

## Primitive and tensor contract

For one token and head, this project evaluates

```text
a = exp(g)
X = Diag(a) S_previous
e = beta * k
z = w * v
r = X.T @ e
S = X + outer(k, z - r)
o = S.T @ (scale * q)
```

`beta` is the paper's channel-wise erase gate `b`; it is named differently in
the Python API to avoid confusing it with the batch dimension. The arithmetic
that updates or differentiates the state accumulates in FP32. Sequence outputs
and sequence gradients are cast to the corresponding input dtype.

Gate activation and Q/K normalization intentionally sit outside the primitive.
This makes the numerical contract small and lets the benchmark compare the
same preprocessed tensors against the official code.

## Chunk forward

The chunk forward uses a BT=16 compact-WY decomposition:

1. one CTA per `(batch, chunk, head)` accumulates log-decay, forms asymmetric
   decay-normalized key/erase/query factors, builds the causal token products,
   and solves the unit-lower-triangular WY system;
2. CTAs partition the value dimension and walk the chunk boundaries in order,
   keeping their state fragments in registers while producing output and the
   final FP32 state. Four warps share a CTA and each warp owns one value
   column, exposing 32 CTAs per `(batch, head)` on the 188-SM target GPU.

FP32 WY auxiliaries make the current version a strong numerical baseline and
avoid the larger final-state error observed from low-precision auxiliaries.
The chunk size differs from the official Triton path's C=64 because the smaller
SM120 schedule exposes more CTAs and stays below the device's shared-memory
limit.

## Backward

The backward receives the forward final state and algebraically reconstructs
previous states in reverse with a Sherman-Morrison inverse:

```text
y = S - outer(k, z)
c = (e.T @ y) / (1 - e.T @ k)
X = y + outer(k, c)
S_previous = Diag(exp(-g)) X
```

For multiple tokens, one CTA owns a four-column value tile for each
`(batch, head)`. These CTAs emit FP32 partials for K-shaped gradients, followed
by a reduction kernel. A single-token launch uses a fused eight-warp CTA to
avoid global partials. This is the VJP for the chunk forward API, although its
present state reconstruction is token-sequential rather than the paper's full
WY backward decomposition.

The inverse requires `1 - (beta * k).T @ k` to remain nonzero. With normalized
keys and erase gates below one this is the usual delta-rule operating region.
The wrapper rejects unsupported shapes and devices, but it does not add a
per-token singularity check; maintaining this condition is the caller's
responsibility.

Reverse reconstruction cannot recover information lost when the forward state
was rounded to FP32 indefinitely. Normalized-key testing stays accurate through
T=128, while errors grow sharply by T=512. The API therefore exposes
`MAX_BACKWARD_TOKENS = 128` and rejects longer sequences. The multi-token
partial workspace is 64 KiB per token/head. A stable long-sequence path must
save FP32 chunk-boundary checkpoints during forward and recompute inside each
chunk, or implement the paper's complete WY backward.

## Token forward

Token/recurrent forward launches four value-tile CTAs per `(batch, head)`. Each
CTA has four warps and owns 32 value columns; a warp owns eight columns and a
lane owns four of the 128 key rows. Its FP32 state fragment remains in registers
through the runtime token loop. Erase and output dot products use warp-shuffle
reductions, so the kernel uses no shared memory.

The sequence-length mode is dynamic in the compiled tensor descriptor, so a
launcher is reused across different decode lengths for the same batch/head and
dtype specialization.

## SM120 architecture decision

The current FROST prefill source describes a 512-thread pipeline with
approximately 205 KiB of shared storage, 272 Tensor Memory columns, TMA stages,
and a dedicated `tcgen05` MMA warp. PTX target tables restrict the base
`tcgen05` lifecycle/MMA instructions to the SM100/SM110 families; they are not
part of SM120. NVIDIA's Blackwell GeForce CuTe example instead builds
`cute.nvgpu.warp.MmaF16BF16Op` register MMA for this family.

Accordingly, this project borrows the algorithmic decomposition and pipeline
ideas from FROST, but not its binary schedule. The SM120 implementation uses
register-resident state partitions, warp reductions, and bounded shared
storage. Its dense chunk products are isolated so a future specialization can
replace them with the SM120 warp MMA pattern used by NVIDIA's example.

## Compilation and stream semantics

Every module defaults `CUTE_DSL_ARCH` to `sm_120` before importing CuTe DSL.
An incompatible explicit value is rejected before launch, as is a GPU whose
compute capability is not 12.0. `cute.compile(..., options="--enable-tvm-ffi")`
produces an executor cached by the static shape/dtype/device specialization.
Runtime calls pass raw PyTorch tensors and the current PyTorch CUDA stream.
Compilation and launch occur in the input tensor's CUDA device context.

The cache is process-local. The first invocation includes compilation and is
not representative of steady-state latency.

Chunk-forward time/chunk counts are currently compile-time specializations and
the inter-chunk recurrence is sequential. This is excellent for short chunks,
but JIT code size and latency grow with sequence length; the measured forward
crossover is between T=256 and T=512. A long-sequence schedule needs a runtime
chunk loop plus less redundant state-factor traffic.

## Primary references

- [Gated DeltaNet-2 paper and appendices](https://arxiv.org/html/2605.22791)
- [Official Triton implementation at the pinned commit](https://github.com/NVlabs/GatedDeltaNet-2/tree/95709fc250357c2dd109361c353192f2aa5913f9)
- [cuDNN Frontend FROST GDN2 prefill](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_prefill_f16.py)
- [cuDNN Frontend FROST GDN2 backward](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py)
- [CUTLASS SM120 dense GEMM example](https://github.com/NVIDIA/cutlass/blob/7107b05535f8977f5ecb9d01ee203205b1fd9bc4/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
- [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [PTX ISA target-feature table](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes)

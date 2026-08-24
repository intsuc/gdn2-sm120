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
2. value CTAs walk chunk boundaries in a runtime loop. Eight warps split K into
   16-row fragments, keep a `16 x Vtile` FP32 state per warp in registers, and
   evaluate the dense products with `m16n8k16` warp MMA;
3. warp-local FP32 results are reduced through shared memory. The long V16
   schedule double-buffers Y/Q staging to remove two barriers per chunk, while
   T<512 uses V8 to expose more CTAs.

Normal forward-only calls store compact FP16/BF16 WY auxiliaries. A
gradient-enabled call that selects the checkpointed backward asks the same
kernel for FP32 auxiliaries and exact FP32 state boundaries. The chunk size
differs from the official Triton path's C=64 because BT=16 exposes more CTAs
and stays below the SM120 shared-memory limit.

## Backward

The short backward receives the forward final state and algebraically
reconstructs previous states in reverse with a Sherman-Morrison inverse:

```text
y = S - outer(k, z)
c = (e.T @ y) / (1 - e.T @ k)
X = y + outer(k, c)
S_previous = Diag(exp(-g)) X
```

This remains the lowest-overhead path below T=64. At T=64 and above, forward
saves every FP32 BT=16 boundary. A compact-WY boundary scan computes both the
reverse state boundaries and

```text
dR = K_tail dS_next + A_qk.T dO
```

using a K-split eight-warp MMA schedule. T=64 through T=127 use independent
chunk-local token VJPs, whose inverse reconstruction is reset from an exact
boundary after at most 16 tokens.

At T>=128, the parameter VJP uses the full compact-WY graph. For each chunk it
forms `R`, `dQ_gamma`, `dA_qk`, `dK_tail`, `dY`, `dZ`, `dE`, `dK_bar`, and the
reverse cumulative gate gradient through 18 ordered CuTe launches. The large
`16x128` and `16x16` products use warp MMA; the triangular transpose solve and
gate chain remain FP32. The boundary scan stores `dR`, avoiding two duplicate
matrix products in this stage.

The inverse requires `1 - (beta * k).T @ k` to remain nonzero. With normalized
keys and erase gates below one this is the usual delta-rule operating region.
The wrapper rejects unsupported shapes and devices, but it does not add a
per-token singularity check; maintaining this condition is the caller's
responsibility.

Reverse reconstruction cannot recover information lost when a complete long
sequence is inverted from one rounded final state. Exact boundary checkpoints
remove that instability and the public autograd path no longer has a sequence
length cap. Tensor-core operands, including FP32 auxiliaries, are narrowed to
the input dtype in shared memory, so the long VJP is a controlled
low-precision approximation rather than a bit-exact FP32 VJP. Tests through
T=512 require relative L2 error below 1% and maximum absolute error below
`5e-3`; observed maxima are about 0.5% and `3.91e-3` against the official
implementation.

For B1/T512/H16, the compact-WY stage uses about 24 MiB of lifetime-colored
sequence workspace plus 1.5 MiB of square workspace. Including saved
boundaries, auxiliaries, reverse boundaries, and outputs, the measured complete
call peaks about 132 MiB above its input baseline, versus 70 MiB for the
official path. Reducing that memory ratio and fusing the 18 stages are the next
backward targets.

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
register-resident state partitions, warp reductions, bounded shared storage,
and the `MmaF16BF16Op` warp-MMA pattern used by NVIDIA's SM120 example.

## Compilation and stream semantics

Every module defaults `CUTE_DSL_ARCH` to `sm_120` before importing CuTe DSL.
An incompatible explicit value is rejected before launch, as is a GPU whose
compute capability is not 12.0. `cute.compile(..., options="--enable-tvm-ffi")`
produces an executor cached by the static shape/dtype/device specialization.
Runtime calls pass raw PyTorch tensors and the current PyTorch CUDA stream.
Compilation and launch occur in the input tensor's CUDA device context.

The cache is process-local. The first invocation includes compilation and is
not representative of steady-state latency.

Chunk-forward tensor layouts and preparation are shape-specialized, while the
full-chunk inter-state scan uses a runtime chunk loop. The state dependency is
still sequential across chunks, but K-split MMA, V16 reuse, and double-buffered
staging move the measured forward crossover to between T=512 and T=1024.

## Primary references

- [Gated DeltaNet-2 paper and appendices](https://arxiv.org/html/2605.22791)
- [Official Triton implementation at the pinned commit](https://github.com/NVlabs/GatedDeltaNet-2/tree/95709fc250357c2dd109361c353192f2aa5913f9)
- [cuDNN Frontend FROST GDN2 prefill](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_prefill_f16.py)
- [cuDNN Frontend FROST GDN2 backward](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py)
- [CUTLASS SM120 dense GEMM example](https://github.com/NVIDIA/cutlass/blob/7107b05535f8977f5ecb9d01ee203205b1fd9bc4/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
- [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [PTX ISA target-feature table](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes)

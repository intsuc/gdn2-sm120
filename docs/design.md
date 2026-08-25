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
   and solves the unit-lower-triangular WY system. Eight warps cover the sixteen
   causal-product rows in two passes, skip the unused upper-triangular products,
   and pair low/high full-chunk rows to equalize their causal work. The final
   per-token exponential is reused as the chunk decay. The same 256 threads then
   solve K/V, each keeping its 16 FP32 values private. The 256-thread CTA permits
   more resident chunks than the former 512-thread layout while retaining the
   barrier-free private solution;
2. value CTAs walk chunk boundaries in a runtime loop. Eight warps split K into
   16-row fragments, keep a `16 x Vtile` FP32 state per warp in registers, and
   evaluate the dense products with `m16n8k16` warp MMA;
3. warp-local FP32 results are reduced through shared memory. Y/Q partials use
   separate contiguous planes and the residual MMA tile has an aligned padded
   stride, avoiding their bank-conflicted layouts. Full-chunk paths
   stage Y/Q with 128-bit `cp.async` copies and cache the 128-element decay
   vector once per CTA rather than loading it independently from every warp.
   With `ceil(T / 16) < 32` it uses an eight-column value tile (V8). From 32
   chunks onward, the launcher keeps V8 when `batch * heads < 12`, exposing 16
   CTAs per batch/head so a small grid can fill the 188 SMs. From 12
   batch-heads onward it selects V16, whose eight CTAs per batch/head halve
   duplicated scan traffic. Both schedules keep the same eight-warp K16 split
   and use one allocation with separate Y/Q planes. Algebraic V8 uses a separate
   K-tail tile because its smaller state staging allocation cannot safely hold
   16x16;
4. when a sequence ends in a partial chunk, the MMA scan handles all
   `floor(T / 16)` full chunks and materializes the prefix state. A one-chunk
   scalar scan consumes that state and handles only the final tail. Sequences
   shorter than one full chunk remain entirely on the scalar path.

For forward-only BF16 calls with at least three full chunks, including
partial-tail lengths, the output identity

```text
R = U - Y S
O = Q_gamma S + A_qk R
  = (Q_gamma - A_qk Y) S + A_qk U
```

moves the two `A_qk` products into the independent chunk-preparation CTAs and
removes `A_qk @ R` from the sequential state scan. The V8 form is profitable
from the public selector's first complete three-chunk shape; the independent
V16 crossover remains at 32 chunks. The scan pipelines the next Y/Q tile and
the current K-tail tile through 128-bit `cp.async`, reusing shared state staging
after its register load and draining the final K-tail without a redundant Y/Q
prefetch. FP16 retains the original expression to avoid overflow
in an intermediate that would cancel algebraically. BF16 training keeps its
measured 32-chunk crossover: the rearranged expression requires a temporary
compact Q-effective scratch consumed only by the state scan, while raw Q-gamma
and A-qk checkpoint bits must remain available to backward. A forward-only
partial-tail call also keeps the final chunk's raw Q-gamma and A-qk because its
scalar tail scan uses the original expression.

The BF16 specialization is validated for normalized Q/K and bounded model
activations. Values near the BF16 format limit are outside its numerical
contract because a rearranged intermediate can overflow before a later
algebraic cancellation.

Forward-only calls use compact FP16/BF16 intermediates that are not exposed. A
gradient-enabled call at `T >= 128` checkpoints Y, Q-gamma, K-tail, and A-qk
in the input dtype, including for a partial-tail sequence, because every
downstream tensor-core consumer would immediately narrow them to that dtype.
The value auxiliary and chunk decay remain FP32. At T=64 and T>=128, once the
training scan has consumed U, each value-tile CTA replaces its disjoint U
columns with `R = U - Y @ S0`; `ChunkForwardAux.u_is_residual` records the
changed contract. Full-chunk BF16 sequences also store compact BF16 state
boundaries; FP16, short, and partial-tail training calls retain FP32
boundaries. The chunk size differs from the official Triton path's C=64 because
BT=16 exposes more CTAs and stays below the SM120 shared-memory limit. With at
least 32 full BF16 chunks, Q-effective adds one
temporary input-dtype sequence scratch whose lifetime ends with forward; it is
not returned in `ChunkForwardAux`, and the raw Q-gamma/A-qk auxiliaries remain
bit-identical to the non-rearranged training path.

## Backward

The short backward receives the forward final state and algebraically
reconstructs previous states in reverse with a Sherman-Morrison inverse:

```text
y = S - outer(k, z)
c = (e.T @ y) / (1 - e.T @ k)
X = y + outer(k, c)
S_previous = Diag(exp(-g)) X
```

Except for the fused T=1 path, the short final-state-only backward assigns one
V8 value tile to each one-warp CTA. K-side inputs and gate evaluation are
shared across its two V4 subtiles. The first kernel writes final-form
`dq`/`dk`/`dg`/`dbeta` FP32 partials with the gradient channel innermost, so
the second kernel only sums value-tile contributions and does not reload
K/beta/g or recompute `exp(g)`.

This remains the lowest-overhead path below T=64. At T=64 and above, forward
saves every BT=16 boundary. At T=64 and T>=128, an eight-warp K-split MMA
boundary scan computes the reverse state boundaries together with

```text
dR = K_tail dS_next + A_qk.T dO
```

The ordered scan selects a 16-column value tile (V16) when
`batch * heads >= 24`, which launches at least 192 CTAs on the 188-SM target.
It also selects V16 when `16 <= batch * heads < 24` and `T <= 512`; other
shapes use V8. Both variants retain the same eight K-split warps. V16 halves
duplicated reads of Y, Q-gamma,
and K-tail, while V8 exposes twice as many CTAs when long serial scans would
otherwise underfill the 188-SM target. The persistent state remains FP32 in
registers; the full-chunk long BF16 path stores forward/reverse checkpoints in
BF16 and returns the exact FP32 initial-state gradient separately. T=64 through
T=127 and all partial-tail paths retain FP32 boundaries. For a partial tail at
`T >= 128`, a scalar reverse scan processes the tail first and produces the
prefix-end gradient; the ordered MMA scan then walks the full chunks in
reverse.

At T=64, the parameter VJP selects the full compact-WY graph only when
`batch * (T / 16) * heads >= 64`; this CTA-aware threshold is equivalent to
`batch * heads >= 16` at T=64. Smaller T=64 grids, and T=65--127, use
independent chunk-local token VJPs whose inverse reconstruction is reset from
an FP32 boundary after at most 16 tokens. T>=128 always uses compact-WY. For
each chunk it forms `dQ_gamma`, `dA_qk`, `dK_tail`, `dY`, `dZ`, `dE`,
`dK_bar`, and the reverse cumulative gate gradient. The standard graph uses 12
ordered CuTe launches. A saved forward R skips the backward `Y @ S0` launch
and changes the paired dLower expression to the equivalent
`-tril(dZ @ R.T)`, reducing the graph to 11 launches at T=64. The full-chunk
compact BF16 path also folds the state-decay dot into the shared-S0 product, so
T>=128 uses 10 launches. Dual-output S0/square products, paired K16 products,
and producer epilogues remove redundant reads and launch boundaries.
The large `16x128` and `16x16` products use warp MMA; the triangular transpose
solve and gate chain accumulate in FP32. The gate chain stores one reciprocal
gamma per token in registers and reuses it for the K and decay expressions,
replacing three divisions with one. The boundary scan stores `dR`,
avoiding two duplicate matrix products; compact BF16 scans round only the
returned `dR` while keeping the ordered boundary recurrence and its precompute
scratch in FP32.

If PyTorch supplies no final-state VJP, the ordered boundary scan specializes
`has_d_final_state=False`, initializes its persistent register state to zero,
and uses already-allocated scratch only as an unread tensor descriptor. This
removes the state-sized zero allocation, zero fill, and terminal-state read.

The inverse requires `1 - (beta * k).T @ k` to remain nonzero. With normalized
keys and erase gates below one this is the usual delta-rule operating region.
The wrapper rejects unsupported shapes and devices, but it does not add a
per-token singularity check; maintaining this condition is the caller's
responsibility.

Reverse reconstruction cannot recover information lost when a complete long
sequence is inverted from one rounded final state. Chunk checkpoints bound
that instability and the public autograd path no longer has a sequence-length
cap. Tensor-core operands and long BF16 checkpoints are stored or narrowed to
the input dtype, so the long VJP is a controlled low-precision approximation
rather than a bit-exact FP32 VJP. Tests require per-gradient relative L2 error
below 1% and maximum absolute error below `5e-3`; measured official comparisons
through T=32768 remain within `3.91e-3` maximum absolute error.

For B1/T512/H16, the compact-WY stage uses about 22 MiB of lifetime-colored
sequence workspace plus 1.5 MiB of square workspace.

### Multi-batch training boundary limit

Some generated CuTe compact-copy layouts use 32-bit byte offsets. Before a
gradient-enabled call, the implementation therefore checks the storage of one
`[B, ceil(T / 16) + 1, H, 128, 128]` state-boundary tensor and rejects a shape
that exceeds the 4-GiB per-launch address range instead of permitting a wrapped
address. Full-chunk BF16 training at T>=128 uses two-byte compact boundaries;
FP16, short, and partial-tail paths use four-byte FP32 boundaries for this
calculation. B4/T32768/H16 BF16 exceeds the limit and is consequently
unsupported for backward and omitted from the canonical benchmark sweep.

## Token forward

The public forward selector accounts for wrapper overhead as well as kernel
latency. `chunk_gdn2` uses recurrence below 48 tokens, chunking for the complete
three-chunk `T=48` case, recurrence for the partial-fourth-chunk `T=49--63`
window, and chunking at `T >= 64`. `recurrent_gdn2` uses
recurrence below 64 tokens. From 64 tokens onward it dispatches to chunking only
when all inputs, state, and output buffers satisfy the chunk path's 16-byte
alignment; an unaligned call remains on the token kernel. Preallocated output
and final state buffers, including explicit in-place state update, keep the
same public semantics on either path.

When `T=1` and no initial state is supplied, decay and erase multiply only the
zero previous state. The recurrence therefore reduces exactly to

```text
z = w * v
S = outer(k, z)
o = (scale * dot(q, k)) * z
```

A dedicated four-warp kernel evaluates this closed form without loading the
initial state, `g`, or `beta`; public validation of all six sequence inputs is
unchanged. Four CTAs per `(batch, head)` retain the closed form's 32-column
value tiles, while each warp produces eight output columns and, for aligned
state, two 128-bit stores per owned key row. One- and two-warp closed-form tiles
exposed more CTAs but lost slightly to the four-warp launch on the target GPU,
so they are not kept as runtime variants.

All other token/recurrent calls use the register-resident recurrence below.
Token/recurrent forward launches eight value-tile CTAs per `(batch, head)`.
Each CTA has two warps and owns 16 value columns; a warp owns eight columns and
a lane owns four of the 128 key rows. The finer CTA granularity keeps the same
total warp work while distributing underfilled decode grids across more SMs.
Its FP32 state fragment remains in registers through the runtime token loop.
Erase and output dot products use warp-shuffle reductions, so the kernel uses
no shared memory.

The warp advances its eight columns as one independent instruction-level
parallel group. It first applies decay to every resident column, interleaves the
eight erase reductions, and then interleaves the updates and output reductions.
Only lanes 0--7 load the warp's eight V/W values; indexed shuffles broadcast
those source values to the remaining lanes.

Each lane's eight contiguous state columns form two 128-bit transactions when
the state base is 16-byte aligned. Initial and final state descriptors are
specialized independently for alignment, while contiguous unaligned buffers
retain scalar state I/O. This changes only the transfer width: both paths use
the same row/column ownership and FP32 recurrence.

PyTorch may assign arbitrary strides to singleton axes of an otherwise
contiguous tensor. The JIT path canonicalizes only its compile-time descriptor
view; runtime tensors keep their original pointer and view, avoiding per-launch
normalization while preserving identical logical addresses.

Batch size and head count are compile-time launcher specializations. Sequence
length remains dynamic in the compiled tensor descriptors and in the runtime
recurrent loop, so one launcher is reused across different decode lengths for
the same batch/head, dtype, initial-state, and alignment specialization. Scale
remains a runtime FP32 value and therefore does not create a new executor for
every caller-provided scale. The zero-state T=1 closed form has its own cache;
because it does not read `g`, changing `g` between the supported sequence dtype
and FP32 does not create another executor.

The default public call allocates fresh output and final-state tensors and does
not mutate inputs. Serving code can instead provide `out` and
`final_state_out`, or request the narrowly defined
`inplace_final_state=True` mode. Exact initial/final-state identity is safe
because each state element is read and written only by its owning CTA. Partial
state aliases and output/input aliases are rejected because they could cross
CTA ownership or overwrite values that another CTA has not loaded.

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
When no gradient edge is needed, `chunk_gdn2` calls the selected executor
directly instead of entering and leaving a custom `autograd.Function`.
The common single-visible-GPU path avoids a redundant device-context guard;
multi-GPU calls compile and launch inside the input tensor's device context.
Capability/device-count queries and a bounded set of non-owning CUDA stream
wrappers are cached, while the stream handle itself remains a runtime launch
argument. Switching PyTorch streams therefore does not require recompilation or
silently send work to the stream used during JIT compilation.

The cache is process-local. The first invocation includes compilation and is
not representative of steady-state latency.

Chunk-forward tensor layouts and preparation are shape-specialized, while the
full-chunk inter-state scan uses a runtime chunk loop. The state dependency is
still sequential across chunks, but the rearranged long-BF16 pipeline removes
one product from that critical path and remains faster than the official path
through the longest measured sequence, T=32768.

## Primary references

- [Gated DeltaNet-2 paper and appendices](https://arxiv.org/html/2605.22791)
- [Official Triton implementation at the pinned commit](https://github.com/NVlabs/GatedDeltaNet-2/tree/95709fc250357c2dd109361c353192f2aa5913f9)
- [cuDNN Frontend FROST GDN2 prefill](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_prefill_f16.py)
- [cuDNN Frontend FROST GDN2 backward](https://github.com/NVIDIA/cudnn-frontend/blob/aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5/python/cudnn/linear_attention/frost/kernel/gdn2_bprop_f16.py)
- [CUTLASS SM120 dense GEMM example](https://github.com/NVIDIA/cutlass/blob/7107b05535f8977f5ecb9d01ee203205b1fd9bc4/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py)
- [CuTe DSL overview](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [PTX ISA target-feature table](https://docs.nvidia.com/cuda/parallel-thread-execution/#release-notes)

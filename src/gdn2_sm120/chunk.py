"""BT=16 chunkwise GDN2 forward kernels written in CuTe DSL.

The implementation deliberately separates work along the same boundary as the
WY algorithm:

1. every ``(batch, chunk, head)`` independently builds the asymmetric erase
   factors, solves the unit-lower-triangular system, and materializes the
   chunk-local auxiliaries;
2. for full BT=16 chunks, every ``(batch, head, value tile)`` uses eight warps
   split across K.  Each warp keeps a ``16 x 8`` or ``16 x 16`` FP32 state tile
   in registers, selected to balance grid coverage against duplicated scan
   traffic, while ``m16n8k16`` tensor-core operations evaluate all dense
   products; a deterministic shared-memory reduction combines the K partials;
3. sequences with a partial final chunk keep the full prefix on the tensor-core
   scan and run only the final short chunk through the scalar warp-reduction
   scan, avoiding a sequence-wide performance cliff.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_CHUNK_SIZE = 16
_KEY_DIM = 128
_VALUE_DIM = 128
# Eight warps evaluate the sixteen causal rows in two passes, then all 256
# threads solve one K/V coordinate.  The smaller CTA doubles residency over
# the former sixteen-warp layout without changing total arithmetic.
_PREPARE_THREADS = 256
# A PRO 6000 has 188 SMs.  One value column per warp exposes enough CTAs to
# fill the device even for B=1/H=16 instead of serializing four columns in a
# single warp.  Four warps per CTA amortize block scheduling without reducing
# the number of independently schedulable value tiles too far.
_INTER_WARPS = 4
_COLS_PER_WARP = 1
_INTER_THREADS = _INTER_WARPS * 32
_VALUE_TILE = _INTER_WARPS * _COLS_PER_WARP
# The full-chunk scan splits K across eight warps.  Each warp owns a persistent
# 16x8 or 16x16 state tile, retaining enough resident warps while replacing all
# dense products with m16n8k16 tensor-core operations.
_K_SPLIT_WARPS = _KEY_DIM // 16
_K_SPLIT_THREADS = _K_SPLIT_WARPS * 32
_K_SPLIT_VALUE_TILE = 8
_K_SPLIT_MIN_CHUNKS = 1
_K_SPLIT_FILLED_VALUE_TILE = 16
# V8 launches sixteen CTAs per (batch, head), reaching a full 188-SM wave at
# twelve heads across the batch.  From there V16 wins by halving duplicated
# scan traffic even for short sequences; below it V8 exposes the missing
# independent value tiles.  Both schedules retain the proven K16/eight-warp
# split.
_K_SPLIT_V16_MIN_BATCH_HEADS = 12
# Broadcasting each warp's local decay rows removes a CTA-wide shared-memory
# round trip and lets state decay overlap the residual publication barrier.
# It wins from four chunks through the short V16 scan for the canonical grids,
# and remains preferable for long underfilled grids.  Longer filled grids keep
# the CTA-shared vector to avoid duplicating global decay traffic.
_SHUFFLE_DECAY_MIN_CHUNKS = 4
_SHUFFLE_DECAY_SHORT_MAX_CHUNKS = 32
_SHUFFLE_DECAY_SHORT_MAX_BATCH_HEADS = 64
_SHUFFLE_DECAY_LONG_MAX_BATCH_HEADS = 16
# Some vectorized CuTe copies lower tensor byte offsets to 32-bit arithmetic.
# Reject saved boundary tensors outside the full unsigned address range before
# launching a kernel; callers must reduce B, T, or H for those training shapes.
_MAX_CUTE_TENSOR_BYTES = 1 << 32
# Moving ``A_qk @ R`` out of the ordered state scan pays off as soon as the
# public forward selector reaches its first full three-chunk shape.  Training
# retains the longer crossover: preserving raw backward operands needs an
# extra Q-effective scratch, whose cost is only recovered by a longer scan.
# The V16 algebra scan reuses its 16x16 shared state tile for K-tail; the V8
# specialization used below 32 chunks has an explicit K-tail stage.
_INFERENCE_ALGEBRA_MIN_CHUNKS = 3
_TRAINING_ALGEBRA_MIN_CHUNKS = 32
assert _K_SPLIT_FILLED_VALUE_TILE == _CHUNK_SIZE


def _select_k_split_value_tile(batch: int, heads: int) -> int:
    """Choose V8 for grid coverage or V16 to reduce duplicated scan traffic."""

    if batch * heads < _K_SPLIT_V16_MIN_BATCH_HEADS:
        return _K_SPLIT_VALUE_TILE
    return _K_SPLIT_FILLED_VALUE_TILE


def _select_shuffle_decay(batch: int, heads: int, scan_chunks: int) -> bool:
    """Use warp-broadcast decay while its saved shared traffic pays off."""

    batch_heads = batch * heads
    return scan_chunks >= _SHUFFLE_DECAY_MIN_CHUNKS and (
        batch_heads <= _SHUFFLE_DECAY_LONG_MAX_BATCH_HEADS
        or (
            scan_chunks <= _SHUFFLE_DECAY_SHORT_MAX_CHUNKS
            and batch_heads <= _SHUFFLE_DECAY_SHORT_MAX_BATCH_HEADS
        )
    )


def _state_boundary_storage_bytes(
    batch: int,
    time: int,
    heads: int,
    element_size: int,
) -> int:
    """Return bytes required by the ``[B, C + 1, H, 128, 128]`` checkpoints."""

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    return batch * (n_chunks + 1) * heads * _KEY_DIM * _VALUE_DIM * element_size


def _byte_span(tensor) -> tuple[int, int]:
    """Return the half-open byte range occupied by a contiguous tensor."""

    start = tensor.data_ptr()
    return start, start + tensor.nbytes


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < left[1] and right[0] < right[1] and left[0] < right[1] and right[0] < left[1]


@cute.jit
def _warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def _prepare_wy_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    y_out: cute.Tensor,
    u_out: cute.Tensor,
    q_gamma_out: cute.Tensor,
    q_effective_out: cute.Tensor,
    k_tail_out: cute.Tensor,
    decay_end_out: cute.Tensor,
    aqk_out: cute.Tensor,
    output: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    chunk_size: cutlass.Constexpr,
    key_dim: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
    use_algebra: cutlass.Constexpr,
    store_forward_aux: cutlass.Constexpr,
    store_partial_tail_original: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5

    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * chunk_size
    remaining = time - start
    length = cutlass.min(chunk_size, remaining)

    allocator = cutlass.utils.SmemAllocator()
    token_key_layout = cute.make_layout((chunk_size, key_dim), stride=(key_dim, 1))
    score_layout = cute.make_layout((chunk_size, chunk_size), stride=(chunk_size, 1))
    s_k_bar = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_erase_bar = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_q_gamma = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_lower = allocator.allocate_tensor(cutlass.Float32, score_layout, byte_alignment=16)
    s_aqk = allocator.allocate_tensor(cutlass.Float32, score_layout, byte_alignment=16)

    # One thread per key coordinate scans the short BT=16 gate sequence. This
    # avoids materializing cumulative gates in global memory.
    if tidx < key_dim:
        key_idx = tidx
        cumulative = cutlass.Float32(0.0)
        gamma_end = cutlass.Float32(1.0)
        for token_local in cutlass.range_constexpr(chunk_size):
            valid = token_local < length
            gamma = cutlass.Float32(1.0)
            k_value = cutlass.Float32(0.0)
            q_value = cutlass.Float32(0.0)
            beta_value = cutlass.Float32(0.0)
            if valid:
                token = start + token_local
                cumulative += g[batch, token, head, key_idx].to(cutlass.Float32)
                gamma = cute.exp(cumulative)
                gamma_end = gamma
                k_value = k[batch, token, head, key_idx].to(cutlass.Float32)
                q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
            # Precise FP32 division was the dominant instruction in this
            # otherwise latency-bound per-key preparation loop.  Gamma is
            # positive, and the SM120 reciprocal approximation stays well
            # below the validated compact FP16/BF16 numerical tolerance.
            reciprocal_gamma = cute.arch.rcp_approx(gamma)
            s_k_bar[token_local, key_idx] = k_value * reciprocal_gamma
            s_erase_bar[token_local, key_idx] = gamma * beta_value * k_value
            s_q_gamma[token_local, key_idx] = gamma * q_value * scale

        decay_end_out[batch, chunk, head, key_idx] = gamma_end
        for token_local in cutlass.range_constexpr(chunk_size):
            if token_local < length:
                token = start + token_local
                # Training exposes raw Q-gamma even when the scan consumes the
                # algebraically rearranged Q-effective scratch.
                if cutlass.const_expr(not use_algebra or store_forward_aux):
                    q_gamma_out[batch, token, head, key_idx] = s_q_gamma[token_local, key_idx].to(
                        q_gamma_out.element_type
                    )
                elif cutlass.const_expr(store_partial_tail_original):  # noqa: SIM102
                    if chunk == n_chunks - 1:
                        q_gamma_out[batch, token, head, key_idx] = s_q_gamma[
                            token_local, key_idx
                        ].to(q_gamma_out.element_type)
                k_tail_out[batch, token, head, key_idx] = (
                    gamma_end * s_k_bar[token_local, key_idx]
                ).to(k_tail_out.element_type)

    cute.arch.sync_threads()

    # Eight warps cover the sixteen rows in two passes.  This preserves the
    # one-warp reductions while halving CTA size, allowing more independent
    # chunks to reside on each SM and hide the solve/exp latency below.
    for row_group in cutlass.range_constexpr(2):
        # Pair a cheap low-triangular row with an expensive high-triangular
        # row.  For a full chunk every warp evaluates exactly seventeen
        # causal cells instead of leaving warp seven on the 24-cell critical
        # path while earlier warps wait at the following CTA barrier.
        row = warp + row_group * (_PREPARE_THREADS // 32)
        if cutlass.const_expr(row_group == 1):  # noqa: SIM102
            # Retain the original ascending assignment for a partial tail,
            # where reversing a sparse second row group can lengthen one
            # warp's path.  All preceding full chunks use the balanced map.
            if length == chunk_size:
                row = chunk_size - 1 - warp
        for col in cutlass.range_constexpr(chunk_size):
            lower_value = cutlass.Float32(0.0)
            aqk_value = cutlass.Float32(0.0)
            # Only the causal triangle is consumed.  Keeping the guard around
            # the dot products avoids evaluating and reducing the upper half
            # that is otherwise overwritten with zero below.
            if row < length and col < length and col <= row:
                for key_group in cutlass.range_constexpr(key_dim // 32):
                    key_idx = lane + key_group * 32
                    k_bar_value = s_k_bar[col, key_idx]
                    if col < row:
                        lower_value += s_erase_bar[row, key_idx] * k_bar_value
                    aqk_value += s_q_gamma[row, key_idx] * k_bar_value
                lower_value = _warp_sum(lower_value)
                aqk_value = _warp_sum(aqk_value)
            if lane == 0:
                strict_lower = lower_value if col < row else cutlass.Float32(0.0)
                causal = aqk_value if col <= row else cutlass.Float32(0.0)
                s_lower[row, col] = strict_lower
                s_aqk[row, col] = causal
                # AQK remains part of the public training checkpoint contract.
                if cutlass.const_expr(not use_algebra or store_forward_aux):
                    aqk_out[batch, chunk, head, row, col] = causal.to(aqk_out.element_type)
                elif cutlass.const_expr(store_partial_tail_original):  # noqa: SIM102
                    if chunk == n_chunks - 1:
                        aqk_out[batch, chunk, head, row, col] = causal.to(aqk_out.element_type)

    cute.arch.sync_threads()

    # Forward substitution for [Y | U] = (I + T)^-1 [E_bar | W*V].
    # Exactly 256 threads cover K+V for the production 128/128 head shape.
    if tidx < key_dim + value_dim:
        dim = tidx
        solution = cute.make_rmem_tensor((chunk_size,), cutlass.Float32)
        for row_solve in cutlass.range_constexpr(chunk_size):
            rhs = cutlass.Float32(0.0)
            if row_solve < length:
                if dim < key_dim:
                    rhs = s_erase_bar[row_solve, dim]
                else:
                    value_idx = dim - key_dim
                    token = start + row_solve
                    rhs = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
                        batch, token, head, value_idx
                    ].to(cutlass.Float32)
                for previous in cutlass.range_constexpr(row_solve):
                    rhs -= s_lower[row_solve, previous] * solution[previous]
                solution[row_solve] = rhs

        for row_store in cutlass.range_constexpr(chunk_size):
            if row_store < length:
                token = start + row_store
                if dim < key_dim:
                    y_out[batch, token, head, dim] = solution[row_store].to(y_out.element_type)
                    if cutlass.const_expr(use_algebra):
                        q_effective = (
                            s_q_gamma[row_store, dim]
                            .to(q_gamma_out.element_type)
                            .to(cutlass.Float32)
                        )
                        for previous in cutlass.range_constexpr(row_store + 1):
                            q_effective -= s_aqk[row_store, previous].to(aqk_out.element_type).to(
                                cutlass.Float32
                            ) * solution[previous].to(y_out.element_type).to(cutlass.Float32)
                        q_effective_out[batch, token, head, dim] = q_effective.to(
                            q_effective_out.element_type
                        )
                else:
                    value_idx = dim - key_dim
                    u_out[batch, token, head, value_idx] = solution[row_store].to(
                        u_out.element_type
                    )
                    if cutlass.const_expr(use_algebra):
                        output_bias = cutlass.Float32(0.0)
                        for previous in cutlass.range_constexpr(row_store + 1):
                            output_bias += s_aqk[row_store, previous].to(aqk_out.element_type).to(
                                cutlass.Float32
                            ) * solution[previous].to(u_out.element_type).to(cutlass.Float32)
                        output[batch, token, head, value_idx] = output_bias.to(output.element_type)


@cute.kernel
def _inter_chunk_kernel(
    y: cute.Tensor,
    u: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    initial_state: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    state_boundaries: cute.Tensor,
    time: cutlass.Constexpr,
    first_chunk: cutlass.Constexpr,
    scan_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_initial_state: cutlass.Constexpr,
    store_state_boundaries: cutlass.Constexpr,
    checkpoint_residual: cutlass.Constexpr,
    chunk_size: cutlass.Constexpr,
    key_dim: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
    value_tile: cutlass.Constexpr,
    cols_per_warp: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5

    value_tiles = value_dim // value_tile
    value_tile_idx = block % value_tiles
    head_block = block // value_tiles
    head = head_block % heads
    batch = head_block // heads

    for local_col in cutlass.range_constexpr(cols_per_warp):
        value_idx = value_tile_idx * value_tile + warp * cols_per_warp + local_col
        state = cute.make_rmem_tensor((key_dim // 32,), cutlass.Float32)
        residual = cute.make_rmem_tensor((chunk_size,), cutlass.Float32)

        for key_group in cutlass.range_constexpr(key_dim // 32):
            key_idx = lane + key_group * 32
            state_value = cutlass.Float32(0.0)
            if cutlass.const_expr(has_initial_state):
                state_value = initial_state[batch, head, key_idx, value_idx].to(cutlass.Float32)
            state[key_group] = state_value

        if cutlass.const_expr(store_state_boundaries and first_chunk == 0):
            for key_group in cutlass.range_constexpr(key_dim // 32):
                key_idx = lane + key_group * 32
                state_boundaries[batch, 0, head, key_idx, value_idx] = state[key_group]

        for scan_chunk in range(0, scan_chunks, 1):
            chunk = first_chunk + scan_chunk
            start = chunk * chunk_size
            length = cutlass.min(chunk_size, time - start)

            for row in cutlass.range_constexpr(chunk_size):
                y_state = cutlass.Float32(0.0)
                q_state = cutlass.Float32(0.0)
                if row < length:
                    token = start + row
                    for key_group in cutlass.range_constexpr(key_dim // 32):
                        key_idx = lane + key_group * 32
                        state_value = state[key_group]
                        y_state += y[batch, token, head, key_idx] * state_value
                        q_state += q_gamma[batch, token, head, key_idx] * state_value
                y_state = _warp_sum(y_state)
                q_state = _warp_sum(q_state)
                if row < length:
                    token = start + row
                    r_value = u[batch, token, head, value_idx] - y_state
                    residual[row] = r_value
                    if lane == 0:
                        if cutlass.const_expr(checkpoint_residual):
                            # U is partitioned by value column across CTAs, so
                            # this in-place replacement has no cross-CTA read/
                            # write hazard.  The compact-WY VJP consumes R
                            # instead of U through an equivalent dLower form.
                            u[batch, token, head, value_idx] = r_value
                        chunk_output = q_state
                        for previous in cutlass.range_constexpr(row + 1):
                            chunk_output += (
                                aqk[batch, chunk, head, row, previous] * residual[previous]
                            )
                        output[batch, token, head, value_idx] = chunk_output.to(output.element_type)

            for key_group in cutlass.range_constexpr(key_dim // 32):
                key_idx = lane + key_group * 32
                state_value = decay_end[batch, chunk, head, key_idx] * state[key_group]
                for row in cutlass.range_constexpr(chunk_size):
                    if row < length:
                        token = start + row
                        state_value += k_tail[batch, token, head, key_idx] * residual[row]
                state[key_group] = state_value
                if cutlass.const_expr(store_state_boundaries):
                    state_boundaries[batch, chunk + 1, head, key_idx, value_idx] = state_value

        for key_group in cutlass.range_constexpr(key_dim // 32):
            key_idx = lane + key_group * 32
            final_state[batch, head, key_idx, value_idx] = state[key_group]


@cute.kernel
def _inter_chunk_k_split_mma_kernel(
    y: cute.Tensor,
    u: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    initial_state: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    state_boundaries: cute.Tensor,
    time: cutlass.Int32,
    n_chunks: cutlass.Int32,
    heads: cutlass.Constexpr,
    has_initial_state: cutlass.Constexpr,
    store_state_boundaries: cutlass.Constexpr,
    use_algebra: cutlass.Constexpr,
    checkpoint_residual: cutlass.Constexpr,
    shuffle_decay: cutlass.Constexpr,
    value_tile: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_state_layout: cute.Layout,
    s_partial_layout: cute.Layout,
    s_residual_layout: cute.Layout,
    s_q_total_layout: cute.Layout,
):
    """Eight-warp scan with one persistent ``16 x value_tile`` state per warp."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_tiles = _VALUE_DIM // value_tile
    value_tile_idx = block % value_tiles
    head_block = block // value_tiles
    head = head_block % heads
    batch = head_block // heads
    value_start = value_tile_idx * value_tile
    key_start = warp * 16
    operand_type = output.element_type

    allocator = cutlass.utils.SmemAllocator()
    s_a_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_q_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_state_all = allocator.allocate_tensor(operand_type, s_state_layout, byte_alignment=1024)
    # V16 can reuse its 16x16 state staging tile after the state reaches
    # registers.  V8 only owns 16x8 per warp, so give the pipelined K-tail a
    # separate 16x16 tile instead of aliasing past the state allocation.
    if cutlass.const_expr(use_algebra and value_tile < _CHUNK_SIZE):
        s_k_tail_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    if cutlass.const_expr(use_algebra):
        s_partial_all = allocator.allocate_tensor(
            cutlass.Float32, s_partial_layout, byte_alignment=1024
        )
    else:
        s_y_partial_all = allocator.allocate_tensor(
            cutlass.Float32, s_partial_layout, byte_alignment=1024
        )
        s_q_partial_all = allocator.allocate_tensor(
            cutlass.Float32, s_partial_layout, byte_alignment=1024
        )
    s_residual = allocator.allocate_tensor(operand_type, s_residual_layout, byte_alignment=1024)
    s_q_total = allocator.allocate_tensor(cutlass.Float32, s_q_total_layout, byte_alignment=1024)
    if cutlass.const_expr(not shuffle_decay):
        s_decay = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((_KEY_DIM,), stride=(1,)), byte_alignment=16
        )
    s_a = s_a_all[warp, None, None]
    s_q = s_q_all[warp, None, None]
    s_state = s_state_all[warp, None, None]
    if cutlass.const_expr(use_algebra):
        s_y_partial = s_partial_all[warp, None, None, 0]
        s_q_partial = s_partial_all[warp, None, None, 1]
    else:
        s_y_partial = s_y_partial_all[warp, None, None]
        s_q_partial = s_q_partial_all[warp, None, None]
    s_state_as_b = cute.make_tensor(
        s_state.iterator,
        cute.make_layout((value_tile, 16), stride=(1, value_tile)),
    )
    if cutlass.const_expr(use_algebra and value_tile < _CHUNK_SIZE):
        s_k_tail_stage = s_k_tail_all[warp, None, None]
    else:
        # The BF16 state has already reached registers before the V16
        # long-path prefetch reuses it as a row-major 16x16 staging tile.
        s_k_tail_stage = cute.make_tensor(
            s_state.iterator,
            cute.make_layout((_CHUNK_SIZE, 16), stride=(16, 1)),
        )
    s_k_tail_as_a = cute.make_tensor(
        s_k_tail_stage.iterator,
        cute.make_layout((16, _CHUNK_SIZE), stride=(1, 16)),
    )

    # A single-warp tiled MMA is sliced by lane.  Warp id selects explicit
    # K-row/state views above, so Y/Q's K split never aliases state-update M.
    thr_mma = tiled_mma.get_slice(lane)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cs_state = thr_mma.partition_C(s_state)
    t_cs_y_partial = thr_mma.partition_C(s_y_partial)
    t_cs_q_partial = thr_mma.partition_C(s_q_partial)
    t_cs_q_total = thr_mma.partition_C(s_q_total)
    t_cr_a = thr_mma.make_fragment_A(t_cs_a)
    t_cr_b = thr_mma.make_fragment_B(thr_mma.partition_B(s_state_as_b))
    t_cr_y = thr_mma.make_fragment_C(t_cs_y_partial)
    t_cr_q = thr_mma.make_fragment_C(t_cs_q_partial)
    t_cr_output = thr_mma.make_fragment_C(t_cs_q_total)

    copy_atom_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    copy_atom_b_state = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type
    )
    copy_atom_b_residual = copy_atom_a
    copy_atom_a_k_tail = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type
    )
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_a_k_tail = cute.make_tiled_copy_A(copy_atom_a_k_tail, tiled_mma)
    tiled_copy_b_state = cute.make_tiled_copy_B(copy_atom_b_state, tiled_mma)
    tiled_copy_b_residual = cute.make_tiled_copy_B(copy_atom_b_residual, tiled_mma)
    copy_atom_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), operand_type, num_bits_per_copy=128
    )
    tiled_copy_g2s = cute.make_tiled_copy_tv(
        copy_atom_g2s,
        cute.make_layout((16, 2), stride=(2, 1)),
        cute.make_layout((1, 8), stride=(8, 1)),
    )
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_a_k_tail = tiled_copy_a_k_tail.get_slice(lane)
    thr_copy_b_state = tiled_copy_b_state.get_slice(lane)
    thr_copy_b_residual = tiled_copy_b_residual.get_slice(lane)
    thr_copy_g2s = tiled_copy_g2s.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_ss_a_k_tail = thr_copy_a_k_tail.partition_S(s_k_tail_as_a)
    t_ss_q_a = thr_copy_a.partition_S(s_q)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_sr_a_k_tail = thr_copy_a_k_tail.retile(t_cr_a)
    t_ss_state = thr_copy_b_state.partition_S(s_state_as_b)
    t_sr_b_state = thr_copy_b_state.retile(t_cr_b)
    t_ss_residual = thr_copy_b_residual.partition_S(s_residual)
    t_sr_b_residual = thr_copy_b_residual.retile(t_cr_b)
    t_ds_a = thr_copy_g2s.partition_D(s_a)
    t_ds_q = thr_copy_g2s.partition_D(s_q)
    t_ds_k_tail = thr_copy_g2s.partition_D(s_k_tail_stage)

    g_initial_state = cute.local_tile(
        initial_state[batch, head, None, None],
        (16, value_tile),
        (warp, value_tile_idx),
    )
    g_final_state = cute.local_tile(
        final_state[batch, head, None, None],
        (16, value_tile),
        (warp, value_tile_idx),
    )
    t_cg_initial_state = thr_mma.partition_C(g_initial_state)
    t_cg_final_state = thr_mma.partition_C(g_final_state)
    t_cr_state = thr_mma.make_fragment_C(t_cg_final_state)
    if cutlass.const_expr(has_initial_state):
        cute.autovec_copy(t_cg_initial_state, t_cr_state)
    else:
        t_cr_state.fill(0.0)
    if cutlass.const_expr(store_state_boundaries):
        g_boundary = cute.local_tile(
            state_boundaries[batch, 0, head, None, None],
            (16, value_tile),
            (warp, value_tile_idx),
        )
        t_cg_boundary = thr_mma.partition_C(g_boundary)
        if cutlass.const_expr(state_boundaries.element_type == cutlass.Float32):
            cute.autovec_copy(t_cr_state, t_cg_boundary)
        else:
            t_cr_boundary = cute.make_fragment_like(t_cr_state, state_boundaries.element_type)
            t_cr_boundary[None] = t_cr_state.load().to(state_boundaries.element_type)
            cute.autovec_copy(t_cr_boundary, t_cg_boundary)
    state_identity = cute.make_identity_tensor((16, value_tile))
    t_cp_state = thr_mma.partition_C(state_identity)

    if cutlass.const_expr(use_algebra):
        g_y_first = cute.local_tile(y[batch, None, head, None], (_CHUNK_SIZE, 16), (0, warp))
        g_q_first = cute.local_tile(q_gamma[batch, None, head, None], (_CHUNK_SIZE, 16), (0, warp))
        cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_y_first), t_ds_a)
        cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_q_first), t_ds_q)
        cute.arch.cp_async_commit_group()

    for chunk in cutlass.range(n_chunks, unroll=1):
        start = chunk * _CHUNK_SIZE

        if cutlass.const_expr(use_algebra):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

        # Convert this warp's FP32 state tile to the transposed BF16 B view.
        t_cr_state_operand = cute.make_fragment_like(t_cr_state, operand_type)
        t_cr_state_operand[None] = t_cr_state.load().to(operand_type)
        cute.autovec_copy(t_cr_state_operand, t_cs_state)
        cute.arch.sync_warp()
        cute.copy(
            tiled_copy_b_state,
            t_ss_state[None, None, 0],
            t_sr_b_state[None, None, 0],
        )

        if cutlass.const_expr(use_algebra):
            g_k_tail = cute.local_tile(
                k_tail[batch, None, head, None], (_CHUNK_SIZE, 16), (chunk, warp)
            )
            cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_k_tail), t_ds_k_tail)
            cute.arch.cp_async_commit_group()
        else:
            # Stage Y and Q together.  Their separate buffers let the same A
            # fragment consume both without a shared-memory overwrite barrier.
            if cutlass.const_expr(y.element_type == operand_type):
                # Compact forward auxiliaries already have operand_type, so a
                # 128-bit cp.async can stage them without conversion.
                g_y = cute.local_tile(y[batch, None, head, None], (_CHUNK_SIZE, 16), (chunk, warp))
                g_q = cute.local_tile(
                    q_gamma[batch, None, head, None], (_CHUNK_SIZE, 16), (chunk, warp)
                )
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_y), t_ds_a)
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_q), t_ds_q)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            else:
                # Short training specializations keep their public WY tensors
                # in FP32. cp.async cannot perform the required conversion to
                # the low-precision tensor-core operand type.
                for linear in cutlass.range(lane, _CHUNK_SIZE * 16, 32):
                    row = linear // 16
                    key_local = linear - row * 16
                    s_a[row, key_local] = y[batch, start + row, head, key_start + key_local].to(
                        operand_type
                    )
                    s_q[row, key_local] = q_gamma[
                        batch, start + row, head, key_start + key_local
                    ].to(operand_type)
            cute.arch.sync_warp()

        # Warp-local K=16 contribution to Y @ state.
        cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
        t_cr_y.fill(0.0)
        cute.gemm(
            tiled_mma,
            t_cr_y,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_y,
        )

        # Warp-local K=16 contribution to Q_gamma @ state.
        cute.copy(tiled_copy_a, t_ss_q_a[None, None, 0], t_sr_a[None, None, 0])
        t_cr_q.fill(0.0)
        cute.gemm(
            tiled_mma,
            t_cr_q,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_q,
        )
        if cutlass.const_expr(shuffle_decay):
            # Each warp needs only the sixteen decay values for its persistent
            # key rows. Broadcast a coalesced lane load instead of materializing
            # and repeatedly reading a CTA-wide shared-memory vector.
            lane_decay = decay_end[
                batch,
                chunk,
                head,
                key_start + (lane & 15),
            ].to(cutlass.Float32)
        if cutlass.const_expr(use_algebra):  # noqa: SIM102
            if chunk + 1 < n_chunks:
                next_chunk = chunk + 1
                g_y_next = cute.local_tile(
                    y[batch, None, head, None], (_CHUNK_SIZE, 16), (next_chunk, warp)
                )
                g_q_next = cute.local_tile(
                    q_gamma[batch, None, head, None],
                    (_CHUNK_SIZE, 16),
                    (next_chunk, warp),
                )
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_y_next), t_ds_a)
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_q_next), t_ds_q)
                cute.arch.cp_async_commit_group()
        cute.autovec_copy(t_cr_y, t_cs_y_partial)
        cute.autovec_copy(t_cr_q, t_cs_q_partial)
        cute.arch.sync_threads()

        # Threads reduce [row, value] coordinates in a fixed warp order.
        for output_linear in cutlass.range(tidx, _CHUNK_SIZE * value_tile, _K_SPLIT_THREADS):
            row = output_linear // value_tile
            value_local = output_linear - row * value_tile
            y_total = cutlass.Float32(0.0)
            q_total = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(_K_SPLIT_WARPS):
                if cutlass.const_expr(use_algebra):
                    y_total += s_partial_all[source_warp, row, value_local, 0]
                    q_total += s_partial_all[source_warp, row, value_local, 1]
                else:
                    y_total += s_y_partial_all[source_warp, row, value_local]
                    q_total += s_q_partial_all[source_warp, row, value_local]
            residual_value = u[batch, start + row, head, value_start + value_local] - y_total
            s_residual[value_local, row] = residual_value.to(operand_type)
            if cutlass.const_expr(checkpoint_residual):
                # Every value tile owns disjoint U/R columns.  Unlike the
                # shared Q-effective scan input, U can therefore be replaced
                # as soon as its value has reached a register without racing
                # another CTA.
                u[batch, start + row, head, value_start + value_local] = residual_value
            if cutlass.const_expr(use_algebra):
                output[batch, start + row, head, value_start + value_local] = (
                    output[batch, start + row, head, value_start + value_local].to(cutlass.Float32)
                    + q_total
                ).to(output.element_type)
            else:
                s_q_total[row, value_local] = q_total

        if cutlass.const_expr(shuffle_decay):
            # Do the independent state decay while slower warps finish
            # publishing residual elements. The following CTA barrier remains
            # necessary to make the complete tile visible to every warp.
            for state_element in cutlass.range_constexpr(cute.size(t_cr_state.shape)):
                key_local = t_cp_state[state_element][0]
                t_cr_state[state_element] *= cute.arch.shuffle_sync(lane_decay, key_local)
            cute.arch.sync_threads()
        else:
            if tidx < _KEY_DIM:
                s_decay[tidx] = decay_end[batch, chunk, head, tidx]
            cute.arch.sync_threads()
            for state_element in cutlass.range_constexpr(cute.size(t_cr_state.shape)):
                key_local = t_cp_state[state_element][0]
                t_cr_state[state_element] *= s_decay[key_start + key_local]

        # AQK @ residual is only one MMA atom.  Warp zero adds it to the
        # reduced Q partial and writes the chunk output while other warps can
        # proceed toward their independent state updates.
        if cutlass.const_expr(not use_algebra) and warp == 0:
            for linear in cutlass.range(lane, _CHUNK_SIZE * _CHUNK_SIZE, 32):
                row = linear // _CHUNK_SIZE
                previous = linear - row * _CHUNK_SIZE
                s_a[row, previous] = aqk[batch, chunk, head, row, previous].to(operand_type)
            cute.arch.sync_warp()
            cute.autovec_copy(t_cs_q_total, t_cr_output)
            cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
            cute.copy(
                tiled_copy_b_residual,
                t_ss_residual[None, None, 0],
                t_sr_b_residual[None, None, 0],
            )
            cute.gemm(
                tiled_mma,
                t_cr_output,
                t_cr_a[None, None, 0],
                t_cr_b[None, None, 0],
                t_cr_output,
            )
            g_output = cute.local_tile(
                output[batch, None, head, None],
                (_CHUNK_SIZE, value_tile),
                (chunk, value_tile_idx),
            )
            t_cg_output = thr_mma.partition_C(g_output)
            t_cr_output_cast = cute.make_fragment_like(t_cr_output, output.element_type)
            t_cr_output_cast[None] = t_cr_output.load().to(output.element_type)
            cute.autovec_copy(t_cr_output_cast, t_cg_output)

        # Each warp updates exactly the 16 key rows of its persistent state.
        cute.copy(
            tiled_copy_b_residual,
            t_ss_residual[None, None, 0],
            t_sr_b_residual[None, None, 0],
        )
        if cutlass.const_expr(use_algebra):
            # Wait for K-tail (the older group), leaving next Y/Q in flight.
            # On the final chunk there is no younger group, so drain K-tail
            # directly instead of issuing a redundant Y/Q prefetch.
            if chunk + 1 < n_chunks:
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()
            cute.copy(
                tiled_copy_a_k_tail,
                t_ss_a_k_tail[None, None, 0],
                t_sr_a_k_tail[None, None, 0],
            )
        else:
            for linear in cutlass.range(lane, 16 * _CHUNK_SIZE, 32):
                key_local = linear // _CHUNK_SIZE
                row = linear - key_local * _CHUNK_SIZE
                s_a[key_local, row] = k_tail[batch, start + row, head, key_start + key_local].to(
                    operand_type
                )
            cute.arch.sync_warp()
            cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
        cute.gemm(
            tiled_mma,
            t_cr_state,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_state,
        )
        if cutlass.const_expr(store_state_boundaries):
            g_boundary = cute.local_tile(
                state_boundaries[batch, chunk + 1, head, None, None],
                (16, value_tile),
                (warp, value_tile_idx),
            )
            t_cg_boundary = thr_mma.partition_C(g_boundary)
            if cutlass.const_expr(state_boundaries.element_type == cutlass.Float32):
                cute.autovec_copy(t_cr_state, t_cg_boundary)
            else:
                t_cr_boundary = cute.make_fragment_like(t_cr_state, state_boundaries.element_type)
                t_cr_boundary[None] = t_cr_state.load().to(state_boundaries.element_type)
                cute.autovec_copy(t_cr_boundary, t_cg_boundary)

    cute.autovec_copy(t_cr_state, t_cg_final_state)


@cute.jit
def _launch_chunk_forward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    initial_state: cute.Tensor,
    y: cute.Tensor,
    u: cute.Tensor,
    q_gamma: cute.Tensor,
    q_effective: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    state_boundaries: cute.Tensor,
    batch: cutlass.Int32,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    has_initial_state: cutlass.Constexpr,
    store_state_boundaries: cutlass.Constexpr,
    use_algebra: cutlass.Constexpr,
    store_forward_aux: cutlass.Constexpr,
    shuffle_decay: cutlass.Constexpr,
    k_split_value_tile: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    # Training no longer needs U after the scan from the compact-WY crossover
    # onward.  Preserve the allocation as an FP32 R checkpoint so backward can
    # skip Y @ S0 even before the long-forward algebra schedule takes over.
    checkpoint_residual = store_forward_aux and (time == 64 or time >= 128)
    _prepare_wy_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        y,
        u,
        q_gamma,
        q_effective,
        k_tail,
        decay_end,
        aqk,
        output,
        time,
        n_chunks,
        heads,
        scale,
        _CHUNK_SIZE,
        _KEY_DIM,
        _VALUE_DIM,
        use_algebra,
        store_forward_aux,
        time % _CHUNK_SIZE != 0,
    ).launch(
        grid=(batch * n_chunks * heads, 1, 1),
        block=(_PREPARE_THREADS, 1, 1),
        stream=stream,
    )
    full_chunks = time // _CHUNK_SIZE
    if cutlass.const_expr(full_chunks < _K_SPLIT_MIN_CHUNKS):
        _inter_chunk_kernel(
            y,
            u,
            q_effective,
            k_tail,
            decay_end,
            aqk,
            initial_state,
            output,
            final_state,
            state_boundaries,
            time,
            0,
            n_chunks,
            heads,
            has_initial_state,
            store_state_boundaries,
            checkpoint_residual,
            _CHUNK_SIZE,
            _KEY_DIM,
            _VALUE_DIM,
            _VALUE_TILE,
            _COLS_PER_WARP,
        ).launch(
            grid=(batch * heads * (_VALUE_DIM // _VALUE_TILE), 1, 1),
            block=(_INTER_THREADS, 1, 1),
            stream=stream,
        )
    else:
        value_tile = k_split_value_tile
        s_a_layout = cute.make_layout(
            (_K_SPLIT_WARPS, _CHUNK_SIZE, 16),
            stride=(_CHUNK_SIZE * 16, 16, 1),
        )
        s_state_layout = cute.make_layout(
            (_K_SPLIT_WARPS, 16, value_tile),
            stride=(16 * value_tile, value_tile, 1),
        )
        if cutlass.const_expr(use_algebra):
            # Keep Y and Q in separate contiguous planes.  Interleaving the
            # components at the innermost stride makes the V16 half-warps hit
            # the same shared-memory banks during every eight-way reduction.
            s_partial_layout = cute.make_layout(
                (_K_SPLIT_WARPS, _CHUNK_SIZE, value_tile, 2),
                stride=(
                    _CHUNK_SIZE * value_tile * 2,
                    value_tile,
                    1,
                    _CHUNK_SIZE * value_tile,
                ),
            )
        else:
            s_partial_layout = s_state_layout
        # Eight BF16 elements of padding keep every K row 16-byte aligned
        # while spreading the row-major reduction stores across twice as many
        # shared-memory bank phases as the unpadded 16-element stride.
        s_residual_layout = cute.make_layout((value_tile, _CHUNK_SIZE), stride=(_CHUNK_SIZE + 8, 1))
        s_q_total_layout = cute.make_layout((_CHUNK_SIZE, value_tile), stride=(value_tile, 1))
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(output.element_type, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, _K_SPLIT_VALUE_TILE, 16),
        )
        _inter_chunk_k_split_mma_kernel(
            y,
            u,
            q_effective,
            k_tail,
            decay_end,
            aqk,
            initial_state,
            output,
            final_state,
            state_boundaries,
            time,
            full_chunks,
            heads,
            has_initial_state,
            store_state_boundaries,
            use_algebra,
            checkpoint_residual,
            shuffle_decay,
            value_tile,
            tiled_mma,
            s_a_layout,
            s_state_layout,
            s_partial_layout,
            s_residual_layout,
            s_q_total_layout,
        ).launch(
            grid=(batch * heads * (_VALUE_DIM // value_tile), 1, 1),
            block=(_K_SPLIT_THREADS, 1, 1),
            stream=stream,
        )
        if cutlass.const_expr(time % _CHUNK_SIZE != 0):
            # The MMA launch above materializes the prefix state in
            # ``final_state``.  Stream ordering makes it a safe in-place input
            # to the one-chunk scalar tail, while saved boundaries retain their
            # global chunk indices.
            _inter_chunk_kernel(
                y,
                u,
                q_gamma,
                k_tail,
                decay_end,
                aqk,
                final_state,
                output,
                final_state,
                state_boundaries,
                time,
                full_chunks,
                1,
                heads,
                True,
                store_state_boundaries,
                checkpoint_residual,
                _CHUNK_SIZE,
                _KEY_DIM,
                _VALUE_DIM,
                _VALUE_TILE,
                _COLS_PER_WARP,
            ).launch(
                grid=(batch * heads * (_VALUE_DIM // _VALUE_TILE), 1, 1),
                block=(_INTER_THREADS, 1, 1),
                stream=stream,
            )


@dataclass(frozen=True)
class ChunkForwardAux:
    """WY tensors and state boundaries consumed by checkpointed backward.

    Calls at T>=128 checkpoint Y/Q-gamma/K-tail/A-qk in the input dtype and
    keep the value auxiliary/decay in FP32, including sequences with a partial
    tail. Training scans at T=64 and T>=128 replace U in place with
    ``R = U - Y @ S0`` after its final forward use and set ``u_is_residual``.
    Full-chunk BF16 calls also use compact BF16 state checkpoints; FP16, short,
    and partial-tail specializations keep state boundaries in FP32.
    """

    y: object
    u: object
    q_gamma: object
    k_tail: object
    decay_end: object
    aqk: object
    state_boundaries: object
    u_is_residual: bool = False


def _fake_tensor(dtype, shape):
    from cutlass.cute.runtime import make_fake_compact_tensor

    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=16,
    )


@lru_cache(maxsize=32)
def _compile_chunk_forward(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    gate_dtype,
    has_initial_state: bool,
    compact_aux: bool,
    compact_backward_aux: bool,
    compact_state_boundaries: bool,
    store_state_boundaries: bool,
    use_algebra: bool,
    store_forward_aux: bool,
    shuffle_decay: bool,
):
    """Compile once per static tensor layout; runtime calls bypass DSL tracing."""

    from cutlass.cute.runtime import make_fake_stream

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    q_shape = (batch, time, heads, _KEY_DIM)
    v_shape = (batch, time, heads, _VALUE_DIM)
    state_shape = (batch, heads, _KEY_DIM, _VALUE_DIM)
    boundary_shape = (batch, n_chunks + 1, heads, _KEY_DIM, _VALUE_DIM)
    decay_shape = (batch, n_chunks, heads, _KEY_DIM)
    aqk_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    f32 = cutlass.Float32
    aux_dtype = input_dtype if compact_aux or compact_backward_aux else f32
    u_dtype = f32 if compact_backward_aux else aux_dtype
    boundary_dtype = input_dtype if compact_state_boundaries else f32
    k_split_value_tile = _select_k_split_value_tile(batch, heads)

    return cute.compile(
        _launch_chunk_forward,
        _fake_tensor(input_dtype, q_shape),  # q
        _fake_tensor(input_dtype, q_shape),  # k
        _fake_tensor(input_dtype, v_shape),  # v
        _fake_tensor(gate_dtype, q_shape),  # g
        _fake_tensor(input_dtype, q_shape),  # beta
        _fake_tensor(input_dtype, v_shape),  # w
        _fake_tensor(f32, state_shape),  # initial state
        _fake_tensor(aux_dtype, q_shape),  # y
        _fake_tensor(u_dtype, v_shape),  # u
        _fake_tensor(aux_dtype, q_shape),  # q_gamma
        _fake_tensor(aux_dtype, q_shape),  # q_effective scan input
        _fake_tensor(aux_dtype, q_shape),  # k_tail
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, aqk_shape),
        _fake_tensor(input_dtype, v_shape),  # output
        _fake_tensor(f32, state_shape),  # final state
        _fake_tensor(
            boundary_dtype, boundary_shape if store_state_boundaries else state_shape
        ),  # optional state boundaries
        batch,
        time,
        n_chunks,
        heads,
        0.0,  # runtime scale placeholder
        has_initial_state,
        store_state_boundaries,
        use_algebra,
        store_forward_aux,
        shuffle_decay,
        k_split_value_tile,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def chunk_forward(
    q,
    k,
    v,
    g,
    beta,
    w,
    initial_state=None,
    *,
    scale: float | None = None,
    return_aux: bool = False,
    out=None,
    final_state_out=None,
):
    """Run the SM120 BT=16 chunkwise forward kernel.

    The first specialization intentionally targets the production GDN2 head
    shape ``K=V=128`` and contiguous BF16/FP16 inputs. Unsupported shapes fail
    explicitly instead of silently falling back to PyTorch.
    """

    import torch

    tensors = (q, k, v, g, beta, w)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("q, k, v, g, beta, and w must be torch.Tensor instances")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("q, k, v, g, beta, and w must be rank-4 tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have the same [B, T, H, K] shape")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v/w must match q's [B, T, H] modes")
    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if batch <= 0 or time <= 0 or heads <= 0:
        raise ValueError("batch, time, and heads must be positive")
    if key_dim != _KEY_DIM or value_dim != _VALUE_DIM:
        raise NotImplementedError("the current SM120 specialization requires K=V=128")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("q/k/v/beta/w must use bfloat16 or float16")
    if any(tensor.dtype != q.dtype for tensor in (k, v, beta, w)):
        raise TypeError("q, k, v, beta, and w must use the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must use the input dtype or float32")
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.data_ptr() % 16 != 0 for tensor in tensors):
        raise ValueError("all inputs must be 16-byte aligned")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("chunk_forward requires an SM120 CUDA device")
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("CUTE_DSL_ARCH must be sm_120 for chunk_forward")
    expected_state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor or None")
        if initial_state.shape != expected_state_shape or initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be contiguous float32 [B, H, 128, 128]")
        if not initial_state.is_cuda or not initial_state.is_contiguous():
            raise ValueError("initial_state must be a contiguous CUDA tensor")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same CUDA device as q")
        if initial_state.data_ptr() % 16 != 0:
            raise ValueError("initial_state must be 16-byte aligned")

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    full_chunks = time % _CHUNK_SIZE == 0
    compact_aux = not return_aux and time // _CHUNK_SIZE >= _K_SPLIT_MIN_CHUNKS
    compact_backward_aux = return_aux and time >= 128
    compact_state_boundaries = compact_backward_aux and full_chunks and q.dtype == torch.bfloat16
    if return_aux:
        boundary_element_size = q.element_size() if compact_state_boundaries else 4
        boundary_bytes = _state_boundary_storage_bytes(
            batch,
            time,
            heads,
            boundary_element_size,
        )
        if boundary_bytes > _MAX_CUTE_TENSOR_BYTES:
            raise ValueError(
                "state-boundary storage exceeds the CuTe 4-GiB per-launch address limit; "
                "reduce batch size, sequence length, or head count"
            )
    # The rearranged output keeps BF16's FP32-like exponent range, but could
    # overflow an FP16 intermediate that cancels in the original expression.
    # Forward-only calls have compact operands from the first full chunk and
    # recover the extra preparation work at three chunks.  Training keeps the
    # measured 32-chunk crossover because it must also preserve raw Q-gamma
    # and A-qk for backward in a separate Q-effective scratch.
    algebra_min_chunks = (
        _TRAINING_ALGEBRA_MIN_CHUNKS if return_aux else _INFERENCE_ALGEBRA_MIN_CHUNKS
    )
    use_algebra = q.dtype == torch.bfloat16 and time // _CHUNK_SIZE >= algebra_min_chunks
    store_forward_aux = return_aux
    shuffle_decay = _select_shuffle_decay(batch, heads, time // _CHUNK_SIZE)
    aux_dtype = q.dtype if compact_aux or compact_backward_aux else torch.float32
    u_dtype = torch.float32 if compact_backward_aux else aux_dtype
    if return_aux and (out is not None or final_state_out is not None):
        raise ValueError("preallocated outputs are supported only when return_aux=False")
    if out is None:
        output = torch.empty_like(v)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor or None")
        if out.shape != v.shape or out.dtype != v.dtype:
            raise ValueError("out must match the contiguous input value layout")
        if out.device != q.device or not out.is_cuda or not out.is_contiguous():
            raise ValueError("out must be a contiguous CUDA tensor on q's device")
        if out.data_ptr() % 16 != 0:
            raise ValueError("out must be 16-byte aligned for the chunk path")
        output = out
    if final_state_out is None:
        final_state = torch.empty(expected_state_shape, device=q.device, dtype=torch.float32)
    else:
        if not isinstance(final_state_out, torch.Tensor):
            raise TypeError("final_state_out must be a torch.Tensor or None")
        if final_state_out.shape != expected_state_shape or final_state_out.dtype != torch.float32:
            raise ValueError("final_state_out must be contiguous float32 [B, H, 128, 128]")
        if (
            final_state_out.device != q.device
            or not final_state_out.is_cuda
            or not final_state_out.is_contiguous()
        ):
            raise ValueError("final_state_out must be a contiguous CUDA tensor on q's device")
        if final_state_out.data_ptr() % 16 != 0:
            raise ValueError("final_state_out must be 16-byte aligned for the chunk path")
        final_state = final_state_out
    if out is not None or final_state_out is not None:
        sequence_spans = tuple(_byte_span(tensor) for tensor in tensors)
        initial_span = _byte_span(initial_state) if initial_state is not None else None
        output_span = _byte_span(output)
        final_span = _byte_span(final_state)
        if out is not None:
            if any(_spans_overlap(output_span, span) for span in sequence_spans) or (
                initial_span is not None and _spans_overlap(output_span, initial_span)
            ):
                raise ValueError("out must not overlap an input tensor")
            if _spans_overlap(output_span, final_span):
                raise ValueError("out and final_state_out must not overlap")
        if final_state_out is not None:
            if any(_spans_overlap(final_span, span) for span in sequence_spans):
                raise ValueError("final_state_out must not overlap a sequence input")
            if (
                initial_span is not None
                and final_span != initial_span
                and _spans_overlap(final_span, initial_span)
            ):
                raise ValueError(
                    "final_state_out may exactly alias initial_state but must not "
                    "partially overlap it"
                )
    # Long training prepares U in FP32 so residual subtraction preserves its
    # rounding order. The algebra scan later reuses that allocation for its
    # FP32 R checkpoint, while tensors immediately narrowed by backward MMA
    # are checkpointed in the input dtype to halve persistent traffic.
    y = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    u = torch.empty(v.shape, device=q.device, dtype=u_dtype)
    q_gamma = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    # Inference reuses its private Q-gamma allocation.  Long BF16 training
    # keeps raw Q-gamma public and gives the scan a separate compact scratch.
    q_effective = q_gamma
    if use_algebra and (store_forward_aux or not full_chunks):
        q_effective = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    k_tail = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    decay_end = torch.empty((batch, n_chunks, heads, key_dim), device=q.device, dtype=torch.float32)
    # The rearranged full-chunk inference path never publishes or reloads
    # A_qk: preparation folds A_qk @ Y and A_qk @ U into Q-effective and the
    # output bias before the ordered scan.  Alias the dead argument to the
    # already allocated output and avoid a workspace allocation that grows
    # linearly with sequence length.
    aqk_is_unused = use_algebra and not store_forward_aux and full_chunks
    if aqk_is_unused:
        aqk = output.as_strided(
            (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE),
            (
                n_chunks * heads * _CHUNK_SIZE * _CHUNK_SIZE,
                heads * _CHUNK_SIZE * _CHUNK_SIZE,
                _CHUNK_SIZE * _CHUNK_SIZE,
                _CHUNK_SIZE,
                1,
            ),
        )
    else:
        aqk = torch.empty(
            (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE),
            device=q.device,
            dtype=aux_dtype,
        )
    state_boundaries = None
    if return_aux:
        state_boundaries = torch.empty(
            (batch, n_chunks + 1, heads, key_dim, value_dim),
            device=q.device,
            dtype=q.dtype if compact_state_boundaries else torch.float32,
        )
    state_arg = initial_state if initial_state is not None else final_state
    boundary_arg = state_boundaries if state_boundaries is not None else final_state

    cutlass_input_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
    cutlass_gate_dtype = cutlass.Float32 if g.dtype == torch.float32 else cutlass_input_dtype
    device_index = q.device.index
    if device_index is None:
        raise RuntimeError("q must have a concrete CUDA device index")
    with torch.cuda.device(q.device):
        compiled = _compile_chunk_forward(
            device_index,
            batch,
            time,
            heads,
            cutlass_input_dtype,
            cutlass_gate_dtype,
            initial_state is not None,
            compact_aux,
            compact_backward_aux,
            compact_state_boundaries,
            return_aux,
            use_algebra,
            store_forward_aux,
            shuffle_decay,
        )
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        compiled(
            *tensors,
            state_arg,
            y,
            u,
            q_gamma,
            q_effective,
            k_tail,
            decay_end,
            aqk,
            output,
            final_state,
            boundary_arg,
            batch,
            output_scale,
            stream,
        )
    if not return_aux:
        return output, final_state
    return (
        output,
        final_state,
        ChunkForwardAux(
            y,
            u,
            q_gamma,
            k_tail,
            decay_end,
            aqk,
            state_boundaries,
            u_is_residual=return_aux and (time == 64 or time >= 128),
        ),
    )


__all__ = ["ChunkForwardAux", "chunk_forward"]

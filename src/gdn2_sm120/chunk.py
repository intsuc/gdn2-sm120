"""BT=16 chunkwise GDN2 forward kernels written in CuTe DSL.

The implementation deliberately separates work along the same boundary as the
WY algorithm:

1. every ``(batch, chunk, head)`` independently builds the asymmetric erase
   factors, solves the unit-lower-triangular system, and materializes the
   chunk-local auxiliaries;
2. for full BT=16 chunks, every ``(batch, head, value tile)`` uses eight warps
   split across K.  Each warp keeps a ``16 x 8`` or ``16 x 16`` FP32 state tile
   in registers while ``m16n8k16`` tensor-core operations evaluate all dense
   products; a deterministic shared-memory reduction combines the K partials;
3. sequences with a partial final chunk retain a scalar warp-reduction scan so
   arbitrary positive lengths preserve the same API and numerical semantics.
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
_PREPARE_THREADS = 512
# A PRO 6000 has 188 SMs.  One value column per warp exposes enough CTAs to
# fill the device even for B=1/H=16 instead of serializing four columns in a
# single warp.  Four warps per CTA amortize block scheduling without reducing
# the number of independently schedulable value tiles too far.
_INTER_WARPS = 4
_COLS_PER_WARP = 1
_INTER_THREADS = _INTER_WARPS * 32
_VALUE_TILE = _INTER_WARPS * _COLS_PER_WARP
# The long scan splits K across eight warps.  Each warp owns a persistent
# 16x8 (short) or 16x16 (long) state tile, retaining enough resident warps while
# replacing all dense products with m16n8k16 tensor-core operations.
_K_SPLIT_WARPS = _KEY_DIM // 16
_K_SPLIT_THREADS = _K_SPLIT_WARPS * 32
_K_SPLIT_VALUE_TILE = 8
_K_SPLIT_MIN_CHUNKS = 1
_K_SPLIT_LONG_VALUE_TILE = 16
_K_SPLIT_LONG_MIN_CHUNKS = 32


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
    k_tail_out: cute.Tensor,
    decay_end_out: cute.Tensor,
    aqk_out: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    chunk_size: cutlass.Constexpr,
    key_dim: cutlass.Constexpr,
    value_dim: cutlass.Constexpr,
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
    solution_layout = cute.make_layout(
        (chunk_size, key_dim + value_dim), stride=(key_dim + value_dim, 1)
    )
    score_layout = cute.make_layout((chunk_size, chunk_size), stride=(chunk_size, 1))
    s_k_bar = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_erase_bar = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_q_gamma = allocator.allocate_tensor(cutlass.Float32, token_key_layout, byte_alignment=16)
    s_solution = allocator.allocate_tensor(cutlass.Float32, solution_layout, byte_alignment=16)
    s_lower = allocator.allocate_tensor(cutlass.Float32, score_layout, byte_alignment=16)
    s_aqk = allocator.allocate_tensor(cutlass.Float32, score_layout, byte_alignment=16)

    # One thread per key coordinate scans the short BT=16 gate sequence. This
    # avoids materializing cumulative gates in global memory.
    if tidx < key_dim:
        key_idx = tidx
        cumulative = cutlass.Float32(0.0)
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
                k_value = k[batch, token, head, key_idx].to(cutlass.Float32)
                q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
            reciprocal_gamma = cutlass.Float32(1.0) / gamma
            s_k_bar[token_local, key_idx] = k_value * reciprocal_gamma
            s_erase_bar[token_local, key_idx] = gamma * beta_value * k_value
            s_q_gamma[token_local, key_idx] = gamma * q_value * scale

        gamma_end = cute.exp(cumulative)
        decay_end_out[batch, chunk, head, key_idx] = gamma_end
        for token_local in cutlass.range_constexpr(chunk_size):
            if token_local < length:
                token = start + token_local
                q_gamma_out[batch, token, head, key_idx] = s_q_gamma[token_local, key_idx].to(
                    q_gamma_out.element_type
                )
                k_tail_out[batch, token, head, key_idx] = (
                    gamma_end * s_k_bar[token_local, key_idx]
                ).to(k_tail_out.element_type)

    cute.arch.sync_threads()

    # Sixteen warps map one-to-one to the sixteen rows. Each lane owns four
    # key coordinates and reduces both the erase-key and query-key dot product.
    row = warp
    for col in cutlass.range_constexpr(chunk_size):
        lower_value = cutlass.Float32(0.0)
        aqk_value = cutlass.Float32(0.0)
        if row < length and col < length:
            for key_group in cutlass.range_constexpr(key_dim // 32):
                key_idx = lane + key_group * 32
                k_bar_value = s_k_bar[col, key_idx]
                lower_value += s_erase_bar[row, key_idx] * k_bar_value
                aqk_value += s_q_gamma[row, key_idx] * k_bar_value
        lower_value = _warp_sum(lower_value)
        aqk_value = _warp_sum(aqk_value)
        if lane == 0:
            strict_lower = lower_value if col < row else cutlass.Float32(0.0)
            causal = aqk_value if col <= row else cutlass.Float32(0.0)
            s_lower[row, col] = strict_lower
            s_aqk[row, col] = causal
            aqk_out[batch, chunk, head, row, col] = causal.to(aqk_out.element_type)

    cute.arch.sync_threads()

    # Forward substitution for [Y | U] = (I + T)^-1 [E_bar | W*V].
    # Exactly 256 threads cover K+V for the production 128/128 head shape.
    if tidx < key_dim + value_dim:
        dim = tidx
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
                    rhs -= s_lower[row_solve, previous] * s_solution[previous, dim]
                s_solution[row_solve, dim] = rhs
            cute.arch.sync_threads()

        for row_store in cutlass.range_constexpr(chunk_size):
            if row_store < length:
                token = start + row_store
                if dim < key_dim:
                    y_out[batch, token, head, dim] = s_solution[row_store, dim].to(
                        y_out.element_type
                    )
                else:
                    value_idx = dim - key_dim
                    u_out[batch, token, head, value_idx] = s_solution[row_store, dim].to(
                        u_out.element_type
                    )


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
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_initial_state: cutlass.Constexpr,
    store_state_boundaries: cutlass.Constexpr,
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

        if cutlass.const_expr(store_state_boundaries):
            for key_group in cutlass.range_constexpr(key_dim // 32):
                key_idx = lane + key_group * 32
                state_boundaries[batch, 0, head, key_idx, value_idx] = state[key_group]

        for chunk in range(0, n_chunks, 1):
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
    s_y_partial_all = allocator.allocate_tensor(
        cutlass.Float32, s_partial_layout, byte_alignment=1024
    )
    s_q_partial_all = allocator.allocate_tensor(
        cutlass.Float32, s_partial_layout, byte_alignment=1024
    )
    s_residual = allocator.allocate_tensor(operand_type, s_residual_layout, byte_alignment=1024)
    s_q_total = allocator.allocate_tensor(cutlass.Float32, s_q_total_layout, byte_alignment=1024)
    s_a = s_a_all[warp, None, None]
    s_q = s_q_all[warp, None, None]
    s_state = s_state_all[warp, None, None]
    s_y_partial = s_y_partial_all[warp, None, None]
    s_q_partial = s_q_partial_all[warp, None, None]
    s_state_as_b = cute.make_tensor(
        s_state.iterator,
        cute.make_layout((value_tile, 16), stride=(1, value_tile)),
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
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_b_state = cute.make_tiled_copy_B(copy_atom_b_state, tiled_mma)
    tiled_copy_b_residual = cute.make_tiled_copy_B(copy_atom_b_residual, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_b_state = tiled_copy_b_state.get_slice(lane)
    thr_copy_b_residual = tiled_copy_b_residual.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_ss_q_a = thr_copy_a.partition_S(s_q)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_ss_state = thr_copy_b_state.partition_S(s_state_as_b)
    t_sr_b_state = thr_copy_b_state.retile(t_cr_b)
    t_ss_residual = thr_copy_b_residual.partition_S(s_residual)
    t_sr_b_residual = thr_copy_b_residual.retile(t_cr_b)

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
        cute.autovec_copy(t_cr_state, thr_mma.partition_C(g_boundary))
    state_identity = cute.make_identity_tensor((16, value_tile))
    t_cp_state = thr_mma.partition_C(state_identity)

    for chunk in cutlass.range(n_chunks, unroll=1):
        start = chunk * _CHUNK_SIZE

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

        # Stage Y and Q together.  Their separate buffers let the same A
        # fragment consume both without a shared-memory overwrite barrier.
        for linear in cutlass.range(lane, _CHUNK_SIZE * 16, 32):
            row = linear // 16
            key_local = linear - row * 16
            s_a[row, key_local] = y[batch, start + row, head, key_start + key_local].to(
                operand_type
            )
            s_q[row, key_local] = q_gamma[batch, start + row, head, key_start + key_local].to(
                operand_type
            )
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
                y_total += s_y_partial_all[source_warp, row, value_local]
                q_total += s_q_partial_all[source_warp, row, value_local]
            residual_value = u[batch, start + row, head, value_start + value_local] - y_total
            s_residual[value_local, row] = residual_value.to(operand_type)
            s_q_total[row, value_local] = q_total
        cute.arch.sync_threads()

        # AQK @ residual is only one MMA atom.  Warp zero adds it to the
        # reduced Q partial and writes the chunk output while other warps can
        # proceed toward their independent state updates.
        if warp == 0:
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
        for state_element in cutlass.range_constexpr(cute.size(t_cr_state.shape)):
            key_local = t_cp_state[state_element][0]
            t_cr_state[state_element] = (
                t_cr_state[state_element] * decay_end[batch, chunk, head, key_start + key_local]
            )
        cute.copy(
            tiled_copy_b_residual,
            t_ss_residual[None, None, 0],
            t_sr_b_residual[None, None, 0],
        )
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
            cute.autovec_copy(t_cr_state, thr_mma.partition_C(g_boundary))

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
    stream: cuda.CUstream,
):
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
        k_tail,
        decay_end,
        aqk,
        time,
        n_chunks,
        heads,
        scale,
        _CHUNK_SIZE,
        _KEY_DIM,
        _VALUE_DIM,
    ).launch(
        grid=(batch * n_chunks * heads, 1, 1),
        block=(_PREPARE_THREADS, 1, 1),
        stream=stream,
    )
    if cutlass.const_expr(n_chunks < _K_SPLIT_MIN_CHUNKS or time % _CHUNK_SIZE != 0):
        _inter_chunk_kernel(
            y,
            u,
            q_gamma,
            k_tail,
            decay_end,
            aqk,
            initial_state,
            output,
            final_state,
            state_boundaries,
            time,
            n_chunks,
            heads,
            has_initial_state,
            store_state_boundaries,
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
        value_tile = _K_SPLIT_VALUE_TILE
        if cutlass.const_expr(n_chunks >= _K_SPLIT_LONG_MIN_CHUNKS):
            value_tile = _K_SPLIT_LONG_VALUE_TILE
        s_a_layout = cute.make_layout(
            (_K_SPLIT_WARPS, _CHUNK_SIZE, 16),
            stride=(_CHUNK_SIZE * 16, 16, 1),
        )
        s_state_layout = cute.make_layout(
            (_K_SPLIT_WARPS, 16, value_tile),
            stride=(16 * value_tile, value_tile, 1),
        )
        s_partial_layout = s_state_layout
        s_residual_layout = cute.make_layout((value_tile, _CHUNK_SIZE), stride=(_CHUNK_SIZE, 1))
        s_q_total_layout = cute.make_layout((_CHUNK_SIZE, value_tile), stride=(value_tile, 1))
        tiled_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(output.element_type, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, _K_SPLIT_VALUE_TILE, 16),
        )
        _inter_chunk_k_split_mma_kernel(
            y,
            u,
            q_gamma,
            k_tail,
            decay_end,
            aqk,
            initial_state,
            output,
            final_state,
            state_boundaries,
            time,
            n_chunks,
            heads,
            has_initial_state,
            store_state_boundaries,
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


@dataclass(frozen=True)
class ChunkForwardAux:
    """WY tensors and exact state boundaries consumed by checkpointed backward."""

    y: object
    u: object
    q_gamma: object
    k_tail: object
    decay_end: object
    aqk: object
    state_boundaries: object


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
    store_state_boundaries: bool,
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
    aux_dtype = input_dtype if compact_aux else f32

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
        _fake_tensor(aux_dtype, v_shape),  # u
        _fake_tensor(aux_dtype, q_shape),  # q_gamma
        _fake_tensor(aux_dtype, q_shape),  # k_tail
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, aqk_shape),
        _fake_tensor(input_dtype, v_shape),  # output
        _fake_tensor(f32, state_shape),  # final state
        _fake_tensor(
            f32, boundary_shape if store_state_boundaries else state_shape
        ),  # optional state boundaries
        batch,
        time,
        n_chunks,
        heads,
        0.0,  # runtime scale placeholder
        has_initial_state,
        store_state_boundaries,
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
    compact_aux = not return_aux and n_chunks >= _K_SPLIT_MIN_CHUNKS and time % _CHUNK_SIZE == 0
    aux_dtype = q.dtype if compact_aux else torch.float32
    output = torch.empty_like(v)
    final_state = torch.empty(expected_state_shape, device=q.device, dtype=torch.float32)
    # The tensor-core path consumes low-precision operands directly.  Keep the
    # public profiling auxiliaries FP32 when requested, while avoiding twice
    # the traffic in the normal long-sequence path.
    y = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    u = torch.empty(v.shape, device=q.device, dtype=aux_dtype)
    q_gamma = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    k_tail = torch.empty(q.shape, device=q.device, dtype=aux_dtype)
    decay_end = torch.empty((batch, n_chunks, heads, key_dim), device=q.device, dtype=torch.float32)
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
            dtype=torch.float32,
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
            return_aux,
        )
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        compiled(
            *tensors,
            state_arg,
            y,
            u,
            q_gamma,
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
        ChunkForwardAux(y, u, q_gamma, k_tail, decay_end, aqk, state_boundaries),
    )


__all__ = ["ChunkForwardAux", "chunk_forward"]

"""SM120 CuTe DSL backward for the Gated Delta Rule-2 recurrence.

The kernels are deliberately specialized for ``K == V == 128``.  The
final-state-only path handles up to 128 tokens with one CTA per eight-column
value tile; K-shaped gradients are emitted as FP32 tile partials and reduced
in a second kernel.  The latency-sensitive single-token case retains a fused
eight-warp CTA with no global partials.

Those short paths reconstruct previous states from the final state with the
Sherman--Morrison inverse

``y = S - k z^T; c = (r^T y) / (1 - r^T k); Sprev = (y + k c^T) / a``.

Applying that inverse across hundreds of rounded FP32 updates is unstable for
production-normalized keys.  When an initial state is available, the
checkpoint path is selected from 128 tokens: it scans forward to recompute
FP32 boundaries every 16 tokens, resets the reverse state to each saved chunk
end, and applies the inverse only inside that chunk.  This is a sequential
checkpoint/recompute algorithm, not a parallel WY backward.

All recurrence arithmetic and reductions are float32.  Sequence gradients
are cast back to the input dtype and ``d_initial_state`` remains float32.
"""

from __future__ import annotations

import math
import operator
import os
import threading

# CuTe DSL reads this while constructing its compilation target.  Do not
# silently replace an explicit caller choice, but make this SM120 module work
# without requiring shell setup.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_DIM = 128
_WARPS = 8
_THREADS = _WARPS * 32
_V_PER_WARP = _DIM // _WARPS
_TILED_V = 8
_VALUE_TILES = _DIM // _TILED_V
_TILED_KEYS_PER_LANE = 16
_TILED_LANE_VALUES = 4
_TILED_THREADS = 32
_TILED_KEY_FIELDS = 5
_TILED_SMEM_BYTES = (_DIM * _TILED_V + _TILED_KEY_FIELDS * _DIM) * 4
# Compatibility name retained for callers that imported the former hard cap.
# It is now the largest length handled by the final-state-only inverse path;
# longer inputs dispatch to checkpoint/recompute.
MAX_BACKWARD_TOKENS = 128

_CHECKPOINT_CHUNK = 16
_LONG_WARPS = 4
_LONG_WARP_V = 4
_LONG_V = _LONG_WARPS * _LONG_WARP_V
_LONG_VALUE_TILES = _DIM // _LONG_V
_LONG_KEYS_PER_LANE = 16
_LONG_THREADS = _LONG_WARPS * 32
_LONG_SMEM_BYTES = (_DIM * _LONG_V + 4 * _DIM + 4 * _LONG_WARPS * _DIM) * 4

# FP32 state [128, 128] plus four [8, 128] cross-warp partial buffers.
_SMEM_BYTES = (_DIM * _DIM + 4 * _WARPS * _DIM) * 4

_COMPILED: dict[tuple[object, ...], object] = {}
_TILED_COMPILED: dict[tuple[object, ...], object] = {}
_CHECKPOINT_COMPILED: dict[tuple[object, ...], object] = {}
_COMPILE_LOCK = threading.Lock()
_SINGLE_CUDA_DEVICE: bool | None = None


@cute.jit
def _half_warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    """All-reduce within each contiguous 16-lane half warp."""

    return cute.arch.warp_reduction(value, operator.add, threads_in_group=16)


@cute.jit
def _pair_sum(value: cutlass.Float32) -> cutlass.Float32:
    """All-reduce the two lanes that own one value column."""

    return value + cute.arch.shuffle_sync_bfly(value, offset=16)


@cute.jit
def _four_lane_sum(value: cutlass.Float32) -> cutlass.Float32:
    """All-reduce one four-column value tile."""

    return cute.arch.warp_reduction(value, operator.add, threads_in_group=4)


@cute.jit
def _oct_sum(value: cutlass.Float32) -> cutlass.Float32:
    """All-reduce the eight lanes that own one value column."""

    value = value + cute.arch.shuffle_sync_bfly(value, offset=4)
    value = value + cute.arch.shuffle_sync_bfly(value, offset=8)
    return value + cute.arch.shuffle_sync_bfly(value, offset=16)


@cute.kernel
def _checkpoint_recompute_backward_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    initial_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    checkpoints: cute.Tensor,
    partial: cute.Tensor,
    dv: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    scale: cutlass.Float32,
):
    """Stable long backward using recomputed FP32 recurrence checkpoints."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    value_tile = block % _LONG_VALUE_TILES
    batch_head = block // _LONG_VALUE_TILES
    batch = batch_head // q.shape[2]
    head = batch_head - batch * q.shape[2]

    warp = tidx >> 5
    lane = tidx & 31
    value_local = lane & 3
    key_group = lane >> 2
    tile_value_local = warp * _LONG_WARP_V + value_local
    value_idx = value_tile * _LONG_V + tile_value_local

    smem = cutlass.utils.SmemAllocator()
    state = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_DIM, _LONG_V), stride=(_LONG_V, 1)),
        byte_alignment=16,
    )
    # k, r=beta*k, exp(g), and reciprocal exp(g) for the current token.
    key_data = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )
    tile_partial = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (4, _LONG_WARPS, _DIM),
            stride=(_LONG_WARPS * _DIM, _DIM, 1),
        ),
        byte_alignment=16,
    )
    dstate = cute.make_rmem_tensor(cute.make_layout((_LONG_KEYS_PER_LANE,)), cutlass.Float32)

    for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
        key_idx = key_group + 8 * key_iter
        state[key_idx, tile_value_local] = initial_state[batch, head, key_idx, value_idx].to(
            cutlass.Float32
        )
        dstate[key_iter] = d_final_state[batch, head, key_idx, value_idx].to(cutlass.Float32)

    time = q.shape[1]
    n_chunks = checkpoints.shape[1] - 1

    # Materialize one FP32 recurrence state at every short-chunk boundary.
    for chunk in cutlass.range(n_chunks, unroll=1):
        for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            checkpoints[batch, chunk, head, key_idx, value_idx] = state[key_idx, tile_value_local]

        for token_local in cutlass.range_constexpr(_CHECKPOINT_CHUNK):
            token = chunk * _CHECKPOINT_CHUNK + token_local
            if token < time:
                for cache_iter in cutlass.range_constexpr(_DIM // _LONG_THREADS):
                    cache_key = tidx + _LONG_THREADS * cache_iter
                    key_value = k[batch, token, head, cache_key].to(cutlass.Float32)
                    beta_value = beta[batch, token, head, cache_key].to(cutlass.Float32)
                    decay = cute.math.exp(
                        g[batch, token, head, cache_key].to(cutlass.Float32),
                        fastmath=False,
                    )
                    key_data[0, cache_key] = key_value
                    key_data[1, cache_key] = beta_value * key_value
                    key_data[2, cache_key] = decay
                cute.arch.sync_threads()

                erased_part = cutlass.Float32(0.0)
                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    x_value = key_data[2, key_idx] * state[key_idx, tile_value_local]
                    erased_part += key_data[1, key_idx] * x_value
                update_value = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
                    batch, token, head, value_idx
                ].to(cutlass.Float32) - _oct_sum(erased_part)

                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    x_value = key_data[2, key_idx] * state[key_idx, tile_value_local]
                    state[key_idx, tile_value_local] = x_value + key_data[0, key_idx] * update_value
                # Every warp must finish consuming this token's shared key
                # factors before any warp overwrites them for the next token.
                cute.arch.sync_threads()

    # Include the final boundary so every reverse chunk can begin at its saved
    # end state without replaying that chunk's forward recurrence.
    for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
        key_idx = key_group + 8 * key_iter
        checkpoints[batch, n_chunks, head, key_idx, value_idx] = state[key_idx, tile_value_local]

    # Reverse chunks sequentially because dS crosses chunk boundaries.  Each
    # chunk resets S to its saved end and inverts at most 16 updates.
    for reverse_chunk in cutlass.range(n_chunks, unroll=1):
        chunk = n_chunks - 1 - reverse_chunk
        chunk_start = chunk * _CHECKPOINT_CHUNK
        length = cutlass.min(_CHECKPOINT_CHUNK, time - chunk_start)

        for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            state[key_idx, tile_value_local] = checkpoints[
                batch, chunk + 1, head, key_idx, value_idx
            ]

        for reverse_local in cutlass.range_constexpr(_CHECKPOINT_CHUNK):
            if reverse_local < length:
                token_local = length - 1 - reverse_local
                token = chunk_start + token_local
                do_value = do[batch, token, head, value_idx].to(cutlass.Float32)

                for cache_iter in cutlass.range_constexpr(_DIM // _LONG_THREADS):
                    cache_key = tidx + _LONG_THREADS * cache_iter
                    key_value = k[batch, token, head, cache_key].to(cutlass.Float32)
                    beta_value = beta[batch, token, head, cache_key].to(cutlass.Float32)
                    decay = cute.math.exp(
                        g[batch, token, head, cache_key].to(cutlass.Float32),
                        fastmath=False,
                    )
                    key_data[0, cache_key] = key_value
                    key_data[1, cache_key] = beta_value * key_value
                    key_data[2, cache_key] = decay
                    key_data[3, cache_key] = cutlass.Float32(1.0) / decay
                cute.arch.sync_threads()

                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                    current_state = state[key_idx, tile_value_local]
                    dstate[key_iter] += scale * q_value * do_value
                    dq_part = _four_lane_sum(scale * current_state * do_value)
                    if value_local == 0:
                        tile_partial[0, warp, key_idx] = dq_part

                denominator_part = cutlass.Float32(0.0)
                y_dot_part = cutlass.Float32(0.0)
                z_value = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
                    batch, token, head, value_idx
                ].to(cutlass.Float32)
                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    key_value = key_data[0, key_idx]
                    r_value = key_data[1, key_idx]
                    y_value = state[key_idx, tile_value_local] - key_value * z_value
                    denominator_part += r_value * key_value
                    y_dot_part += r_value * y_value

                denominator = cutlass.Float32(1.0) - _oct_sum(denominator_part)
                erased_value = _oct_sum(y_dot_part) / denominator
                update_value = z_value - erased_value

                du_part = cutlass.Float32(0.0)
                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    du_part += key_data[0, key_idx] * dstate[key_iter]
                du_value = _oct_sum(du_part)

                if key_group == 0:
                    v_value = v[batch, token, head, value_idx].to(cutlass.Float32)
                    w_value = w[batch, token, head, value_idx].to(cutlass.Float32)
                    dv[batch, token, head, value_idx] = (du_value * w_value).to(dv.element_type)
                    dw[batch, token, head, value_idx] = (du_value * v_value).to(dw.element_type)

                for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
                    key_idx = key_group + 8 * key_iter
                    key_value = key_data[0, key_idx]
                    r_value = key_data[1, key_idx]
                    decay = key_data[2, key_idx]
                    inverse_decay = key_data[3, key_idx]
                    current_state = state[key_idx, tile_value_local]
                    y_value = current_state - key_value * z_value
                    x_value = y_value + key_value * erased_value
                    previous_state = x_value * inverse_decay
                    state[key_idx, tile_value_local] = previous_state
                    ds_value = dstate[key_iter]

                    dk_direct_part = _four_lane_sum(ds_value * update_value)
                    dr_part = _four_lane_sum(-x_value * du_value)
                    dx_value = ds_value - r_value * du_value
                    da_part = _four_lane_sum(dx_value * previous_state)

                    if value_local == 0:
                        tile_partial[1, warp, key_idx] = dk_direct_part
                        tile_partial[2, warp, key_idx] = dr_part
                        tile_partial[3, warp, key_idx] = da_part
                    dstate[key_iter] = decay * dx_value

                # Merge the per-warp V4 contributions into this CTA's V8/V16
                # partial.  A second barrier prevents the next token from
                # overwriting shared slots while the leading threads read.
                cute.arch.sync_threads()
                for reduce_iter in cutlass.range_constexpr(_DIM // _LONG_THREADS):
                    key_idx = tidx + _LONG_THREADS * reduce_iter
                    dq_total = cutlass.Float32(0.0)
                    dk_direct_total = cutlass.Float32(0.0)
                    dr_total = cutlass.Float32(0.0)
                    da_total = cutlass.Float32(0.0)
                    for source_warp in cutlass.range_constexpr(_LONG_WARPS):
                        dq_total += tile_partial[0, source_warp, key_idx]
                        dk_direct_total += tile_partial[1, source_warp, key_idx]
                        dr_total += tile_partial[2, source_warp, key_idx]
                        da_total += tile_partial[3, source_warp, key_idx]
                    partial[0, batch, token, head, value_tile, key_idx] = dq_total
                    partial[1, batch, token, head, value_tile, key_idx] = dk_direct_total
                    partial[2, batch, token, head, value_tile, key_idx] = dr_total
                    partial[3, batch, token, head, value_tile, key_idx] = da_total
                cute.arch.sync_threads()

    for key_iter in cutlass.range(_LONG_KEYS_PER_LANE, unroll_full=True):
        key_idx = key_group + 8 * key_iter
        d_initial_state[batch, head, key_idx, value_idx] = dstate[key_iter]


@cute.kernel
def _reduce_long_partials_kernel(
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    partial: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block % k.shape[2]
    batch_token = block // k.shape[2]
    token = batch_token % k.shape[1]
    batch = batch_token // k.shape[1]
    key_idx = tidx

    dq_total = cutlass.Float32(0.0)
    dk_direct_total = cutlass.Float32(0.0)
    dr_total = cutlass.Float32(0.0)
    da_total = cutlass.Float32(0.0)
    for value_tile in cutlass.range_constexpr(_LONG_VALUE_TILES):
        dq_total += partial[0, batch, token, head, value_tile, key_idx]
        dk_direct_total += partial[1, batch, token, head, value_tile, key_idx]
        dr_total += partial[2, batch, token, head, value_tile, key_idx]
        da_total += partial[3, batch, token, head, value_tile, key_idx]

    key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
    beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
    decay = cute.math.exp(g[batch, token, head, key_idx].to(cutlass.Float32), fastmath=False)
    dq[batch, token, head, key_idx] = dq_total.to(dq.element_type)
    dk[batch, token, head, key_idx] = (dk_direct_total + beta_value * dr_total).to(dk.element_type)
    dbeta[batch, token, head, key_idx] = (key_value * dr_total).to(dbeta.element_type)
    dg[batch, token, head, key_idx] = (decay * da_total).to(dg.element_type)


@cute.kernel
def _chunk_backward_value_tiled_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    final_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    dv: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    partial: cute.Tensor,
    scale: cutlass.Float32,
):
    """Reverse one eight-column value tile per CTA.

    Splitting the value dimension is important for the relatively small-head
    chunk workload: B=1, H=16 otherwise exposes only 16 CTAs to an SM120 GPU.
    Each lane retains the former V4 ownership for two columns.  This preserves
    the old reduction trees while K-side loads and gate evaluation are shared
    across both V4 subtiles.  Final-form K-shaped gradients are kept as FP32
    tile partials and summed by a second kernel.  V-shaped gradients and dS
    need no reduction.
    """

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()

    value_tile = block % _VALUE_TILES
    batch_head = block // _VALUE_TILES
    batch = batch_head // q.shape[2]
    head = batch_head - batch * q.shape[2]

    value_lane = tidx & (_TILED_LANE_VALUES - 1)
    key_group = tidx >> 2
    value_idx_0 = value_tile * _TILED_V + value_lane
    value_idx_1 = value_idx_0 + _TILED_LANE_VALUES

    # Only the tile-local state is needed.  Each lane owns disjoint state
    # cells, so state accesses need no inter-thread synchronization.
    smem = cutlass.utils.SmemAllocator()
    state = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_DIM, _TILED_V), stride=(_TILED_V, 1)),
        byte_alignment=16,
    )
    # k, r=beta*k, exp(g), reciprocal exp(g), and beta are shared by all eight
    # columns.  Keeping beta lets this kernel emit final-form dk partials.
    key_data = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_TILED_KEY_FIELDS, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )
    dstate_0 = cute.make_rmem_tensor((_TILED_KEYS_PER_LANE,), cutlass.Float32)
    dstate_1 = cute.make_rmem_tensor((_TILED_KEYS_PER_LANE,), cutlass.Float32)
    for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
        key_idx = key_group + 8 * key_iter
        state[key_idx, value_lane] = final_state[batch, head, key_idx, value_idx_0].to(
            cutlass.Float32
        )
        state[key_idx, value_lane + _TILED_LANE_VALUES] = final_state[
            batch, head, key_idx, value_idx_1
        ].to(cutlass.Float32)
        dstate_0[key_iter] = d_final_state[batch, head, key_idx, value_idx_0].to(cutlass.Float32)
        dstate_1[key_iter] = d_final_state[batch, head, key_idx, value_idx_1].to(cutlass.Float32)

    time = q.shape[1]
    for reverse_iter in cutlass.range(time, unroll=1):
        token = time - 1 - reverse_iter
        do_value_0 = do[batch, token, head, value_idx_0].to(cutlass.Float32)
        do_value_1 = do[batch, token, head, value_idx_1].to(cutlass.Float32)

        # Evaluate the gate once per key and CTA, instead of once per state
        # cell.  Four coalesced rounds cover all 128 keys.
        for cache_iter in cutlass.range_constexpr(4):
            cache_key = tidx + 32 * cache_iter
            key_value = k[batch, token, head, cache_key].to(cutlass.Float32)
            beta_value = beta[batch, token, head, cache_key].to(cutlass.Float32)
            decay = cute.math.exp(
                g[batch, token, head, cache_key].to(cutlass.Float32), fastmath=False
            )
            key_data[0, cache_key] = key_value
            key_data[1, cache_key] = beta_value * key_value
            key_data[2, cache_key] = decay
            key_data[3, cache_key] = cutlass.Float32(1.0) / decay
            key_data[4, cache_key] = beta_value
        cute.arch.sync_warp()

        # Output VJP and this V8 tile's FP32 dq partial.  Each half retains the
        # old V4 reduction tree before lane zero combines them in value order.
        for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
            state_value_0 = state[key_idx, value_lane]
            state_value_1 = state[key_idx, value_lane + _TILED_LANE_VALUES]
            dstate_0[key_iter] += scale * q_value * do_value_0
            dstate_1[key_iter] += scale * q_value * do_value_1

            dq_part_0 = _four_lane_sum(scale * state_value_0 * do_value_0)
            dq_part_1 = _four_lane_sum(scale * state_value_1 * do_value_1)
            if value_lane == 0:
                partial[batch, token, head, value_tile, key_idx, 0] = dq_part_0 + dq_part_1

        denominator_part = cutlass.Float32(0.0)
        y_dot_part_0 = cutlass.Float32(0.0)
        y_dot_part_1 = cutlass.Float32(0.0)
        z_value_0 = w[batch, token, head, value_idx_0].to(cutlass.Float32) * v[
            batch, token, head, value_idx_0
        ].to(cutlass.Float32)
        z_value_1 = w[batch, token, head, value_idx_1].to(cutlass.Float32) * v[
            batch, token, head, value_idx_1
        ].to(cutlass.Float32)
        for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            key_value = key_data[0, key_idx]
            r_value = key_data[1, key_idx]
            y_value_0 = state[key_idx, value_lane] - key_value * z_value_0
            y_value_1 = state[key_idx, value_lane + _TILED_LANE_VALUES] - key_value * z_value_1
            denominator_part += r_value * key_value
            y_dot_part_0 += r_value * y_value_0
            y_dot_part_1 += r_value * y_value_1

        denominator = cutlass.Float32(1.0) - _oct_sum(denominator_part)
        erased_value_0 = _oct_sum(y_dot_part_0) / denominator
        erased_value_1 = _oct_sum(y_dot_part_1) / denominator
        update_value_0 = z_value_0 - erased_value_0
        update_value_1 = z_value_1 - erased_value_1

        du_part_0 = cutlass.Float32(0.0)
        du_part_1 = cutlass.Float32(0.0)
        for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            key_value = key_data[0, key_idx]
            du_part_0 += key_value * dstate_0[key_iter]
            du_part_1 += key_value * dstate_1[key_iter]
        du_value_0 = _oct_sum(du_part_0)
        du_value_1 = _oct_sum(du_part_1)

        if key_group == 0:
            v_value_0 = v[batch, token, head, value_idx_0].to(cutlass.Float32)
            w_value_0 = w[batch, token, head, value_idx_0].to(cutlass.Float32)
            v_value_1 = v[batch, token, head, value_idx_1].to(cutlass.Float32)
            w_value_1 = w[batch, token, head, value_idx_1].to(cutlass.Float32)
            dv[batch, token, head, value_idx_0] = (du_value_0 * w_value_0).to(dv.element_type)
            dw[batch, token, head, value_idx_0] = (du_value_0 * v_value_0).to(dw.element_type)
            dv[batch, token, head, value_idx_1] = (du_value_1 * w_value_1).to(dv.element_type)
            dw[batch, token, head, value_idx_1] = (du_value_1 * v_value_1).to(dw.element_type)

        for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
            key_idx = key_group + 8 * key_iter
            key_value = key_data[0, key_idx]
            r_value = key_data[1, key_idx]
            decay = key_data[2, key_idx]
            inverse_decay = key_data[3, key_idx]
            beta_value = key_data[4, key_idx]
            current_state_0 = state[key_idx, value_lane]
            current_state_1 = state[key_idx, value_lane + _TILED_LANE_VALUES]
            y_value_0 = current_state_0 - key_value * z_value_0
            y_value_1 = current_state_1 - key_value * z_value_1
            x_value_0 = y_value_0 + key_value * erased_value_0
            x_value_1 = y_value_1 + key_value * erased_value_1
            previous_state_0 = x_value_0 * inverse_decay
            previous_state_1 = x_value_1 * inverse_decay
            state[key_idx, value_lane] = previous_state_0
            state[key_idx, value_lane + _TILED_LANE_VALUES] = previous_state_1
            ds_value_0 = dstate_0[key_iter]
            ds_value_1 = dstate_1[key_iter]

            dk_direct_part_0 = _four_lane_sum(ds_value_0 * update_value_0)
            dk_direct_part_1 = _four_lane_sum(ds_value_1 * update_value_1)
            dr_part_0 = _four_lane_sum(-x_value_0 * du_value_0)
            dr_part_1 = _four_lane_sum(-x_value_1 * du_value_1)
            dx_value_0 = ds_value_0 - r_value * du_value_0
            dx_value_1 = ds_value_1 - r_value * du_value_1
            da_part_0 = _four_lane_sum(dx_value_0 * previous_state_0)
            da_part_1 = _four_lane_sum(dx_value_1 * previous_state_1)

            if value_lane == 0:
                dk_direct_part = dk_direct_part_0 + dk_direct_part_1
                dr_part = dr_part_0 + dr_part_1
                da_part = da_part_0 + da_part_1
                partial[batch, token, head, value_tile, key_idx, 1] = (
                    dk_direct_part + beta_value * dr_part
                )
                partial[batch, token, head, value_tile, key_idx, 2] = decay * da_part
                partial[batch, token, head, value_tile, key_idx, 3] = key_value * dr_part

            dstate_0[key_iter] = decay * dx_value_0
            dstate_1[key_iter] = decay * dx_value_1

    for key_iter in cutlass.range(_TILED_KEYS_PER_LANE, unroll_full=True):
        key_idx = key_group + 8 * key_iter
        d_initial_state[batch, head, key_idx, value_idx_0] = dstate_0[key_iter]
        d_initial_state[batch, head, key_idx, value_idx_1] = dstate_1[key_iter]


@cute.kernel
def _reduce_value_tile_partials_kernel(
    partial: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()

    head = block % dq.shape[2]
    batch_token = block // dq.shape[2]
    token = batch_token % dq.shape[1]
    batch = batch_token // dq.shape[1]
    key_idx = tidx

    dq_total = cutlass.Float32(0.0)
    dk_total = cutlass.Float32(0.0)
    dg_total = cutlass.Float32(0.0)
    dbeta_total = cutlass.Float32(0.0)
    for value_tile in cutlass.range_constexpr(_VALUE_TILES):
        dq_total += partial[batch, token, head, value_tile, key_idx, 0]
        dk_total += partial[batch, token, head, value_tile, key_idx, 1]
        dg_total += partial[batch, token, head, value_tile, key_idx, 2]
        dbeta_total += partial[batch, token, head, value_tile, key_idx, 3]

    dq[batch, token, head, key_idx] = dq_total.to(dq.element_type)
    dk[batch, token, head, key_idx] = dk_total.to(dk.element_type)
    dg[batch, token, head, key_idx] = dg_total.to(dg.element_type)
    dbeta[batch, token, head, key_idx] = dbeta_total.to(dbeta.element_type)


@cute.kernel
def _chunk_backward_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    final_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    scale: cutlass.Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()

    batch = block // q.shape[2]
    head = block - batch * q.shape[2]
    warp = tidx >> 5
    lane = tidx & 31

    # A half warp spans the warp's 16 value columns.  The lower and upper
    # half-warps own respectively the even and odd key rows.
    value_local = lane & 15
    key_parity = lane >> 4
    value_idx = warp * _V_PER_WARP + value_local

    smem = cutlass.utils.SmemAllocator()
    state = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_DIM, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )
    partial = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, _WARPS, _DIM), stride=(_WARPS * _DIM, _DIM, 1)),
        byte_alignment=16,
    )

    # One thread owns 64 state cells.  Keeping dS in an rmem tensor avoids a
    # second 64-KiB shared-memory matrix (SM120 permits 99 KiB per CTA).
    dstate = cute.make_rmem_tensor(cute.make_layout((64,)), cutlass.Float32)
    for key_iter in cutlass.range(64, unroll_full=True):
        key_idx = key_parity + 2 * key_iter
        state[key_idx, value_idx] = final_state[batch, head, key_idx, value_idx].to(cutlass.Float32)
        dstate[key_iter] = d_final_state[batch, head, key_idx, value_idx].to(cutlass.Float32)
    cute.arch.sync_threads()

    time = q.shape[1]
    for reverse_iter in cutlass.range(time, unroll=1):
        token = time - 1 - reverse_iter
        do_value = do[batch, token, head, value_idx].to(cutlass.Float32)

        # Add the output VJP to dS and form dq while S_t is still resident.
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
            state_value = state[key_idx, value_idx]
            dstate[key_iter] = dstate[key_iter] + scale * q_value * do_value

            dq_part = _half_warp_sum(scale * state_value * do_value)
            if value_local == 0:
                partial[0, warp, key_idx] = dq_part

        # r^T k is shared by all value columns.  Computing it redundantly in
        # each warp avoids a CTA-wide synchronization in the reconstruction.
        denominator_part = cutlass.Float32(0.0)
        y_dot_part = cutlass.Float32(0.0)
        z_value = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
            batch, token, head, value_idx
        ].to(cutlass.Float32)
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
            beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
            r_value = beta_value * key_value
            y_value = state[key_idx, value_idx] - key_value * z_value
            denominator_part = denominator_part + r_value * key_value
            y_dot_part = y_dot_part + r_value * y_value

        denominator = cutlass.Float32(1.0) - _pair_sum(denominator_part)
        erased_value = _pair_sum(y_dot_part) / denominator
        update_value = z_value - erased_value

        # Reconstruct S_{t-1} in place.  exp(g) and the inverse both stay FP32.
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
            decay = cute.math.exp(
                g[batch, token, head, key_idx].to(cutlass.Float32), fastmath=False
            )
            y_value = state[key_idx, value_idx] - key_value * z_value
            x_value = y_value + key_value * erased_value
            state[key_idx, value_idx] = x_value / decay

        # du = k^T dS.  The two lanes for a value column cover all 128 keys.
        du_part = cutlass.Float32(0.0)
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
            du_part = du_part + key_value * dstate[key_iter]
        du_value = _pair_sum(du_part)

        # z = w*v.  Only one of the two column lanes stores each V gradient.
        if key_parity == 0:
            v_value = v[batch, token, head, value_idx].to(cutlass.Float32)
            w_value = w[batch, token, head, value_idx].to(cutlass.Float32)
            dv[batch, token, head, value_idx] = (du_value * w_value).to(dv.element_type)
            dw[batch, token, head, value_idx] = (du_value * v_value).to(dw.element_type)

        # S = x + k*u^T, u = z-r^T*x, x=a*Sprev.
        # Reduce all K-shaped gradient contributions first over the warp's
        # 16 value columns, then below over all eight warps through shared.
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
            beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
            decay = cute.math.exp(
                g[batch, token, head, key_idx].to(cutlass.Float32), fastmath=False
            )
            previous_state = state[key_idx, value_idx]
            x_value = decay * previous_state
            r_value = beta_value * key_value
            ds_value = dstate[key_iter]

            dk_direct_part = _half_warp_sum(ds_value * update_value)
            dr_part = _half_warp_sum(-x_value * du_value)
            dx_value = ds_value - r_value * du_value
            da_part = _half_warp_sum(dx_value * previous_state)

            if value_local == 0:
                partial[1, warp, key_idx] = dk_direct_part
                partial[2, warp, key_idx] = dr_part
                partial[3, warp, key_idx] = da_part

            dstate[key_iter] = decay * dx_value

        cute.arch.sync_threads()

        # The first 128 threads own one key each for the cross-warp reduction.
        if tidx < _DIM:
            key_idx = tidx
            dq_total = cutlass.Float32(0.0)
            dk_direct_total = cutlass.Float32(0.0)
            dr_total = cutlass.Float32(0.0)
            da_total = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(_WARPS):
                dq_total = dq_total + partial[0, source_warp, key_idx]
                dk_direct_total = dk_direct_total + partial[1, source_warp, key_idx]
                dr_total = dr_total + partial[2, source_warp, key_idx]
                da_total = da_total + partial[3, source_warp, key_idx]

            key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
            beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
            decay = cute.math.exp(
                g[batch, token, head, key_idx].to(cutlass.Float32), fastmath=False
            )
            dq[batch, token, head, key_idx] = dq_total.to(dq.element_type)
            dk[batch, token, head, key_idx] = (dk_direct_total + beta_value * dr_total).to(
                dk.element_type
            )
            dbeta[batch, token, head, key_idx] = (key_value * dr_total).to(dbeta.element_type)
            dg[batch, token, head, key_idx] = (decay * da_total).to(dg.element_type)

        # No warp may overwrite shared partials for the next token yet.
        cute.arch.sync_threads()

    for key_iter in cutlass.range(64, unroll_full=True):
        key_idx = key_parity + 2 * key_iter
        d_initial_state[batch, head, key_idx, value_idx] = dstate[key_iter]


@cute.jit
def _launch_chunk_backward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    final_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _chunk_backward_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        final_state,
        do,
        d_final_state,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dw,
        d_initial_state,
        scale,
    ).launch(
        grid=(q.shape[0] * q.shape[2], 1, 1),
        block=(_THREADS, 1, 1),
        smem=_SMEM_BYTES,
        stream=stream,
    )


@cute.jit
def _launch_chunk_backward_value_tiled(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    final_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    partial: cute.Tensor,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _chunk_backward_value_tiled_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        final_state,
        do,
        d_final_state,
        dv,
        dw,
        d_initial_state,
        partial,
        scale,
    ).launch(
        grid=(q.shape[0] * q.shape[2] * _VALUE_TILES, 1, 1),
        block=(_TILED_THREADS, 1, 1),
        smem=_TILED_SMEM_BYTES,
        stream=stream,
    )
    _reduce_value_tile_partials_kernel(
        partial,
        dq,
        dk,
        dg,
        dbeta,
    ).launch(
        grid=(q.shape[0] * q.shape[1] * q.shape[2], 1, 1),
        block=(_DIM, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_checkpoint_recompute_backward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    initial_state: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    checkpoints: cute.Tensor,
    partial: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _checkpoint_recompute_backward_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        do,
        d_final_state,
        checkpoints,
        partial,
        dv,
        dw,
        d_initial_state,
        scale,
    ).launch(
        grid=(q.shape[0] * q.shape[2] * _LONG_VALUE_TILES, 1, 1),
        block=(_LONG_THREADS, 1, 1),
        smem=_LONG_SMEM_BYTES,
        stream=stream,
    )
    _reduce_long_partials_kernel(
        k,
        g,
        beta,
        partial,
        dq,
        dk,
        dg,
        dbeta,
    ).launch(
        grid=(q.shape[0] * q.shape[1] * q.shape[2], 1, 1),
        block=(_DIM, 1, 1),
        stream=stream,
    )


def _validate_backward_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    final_state: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
) -> None:
    key_shape = q.shape
    value_shape = v.shape
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/g/beta and v/w/do must be rank-4 [B, T, H, D] tensors")
    if key_shape != k.shape or key_shape != g.shape or key_shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if value_shape != w.shape or value_shape != do.shape or value_shape[:3] != key_shape[:3]:
        raise ValueError("v, w, and do must match each other and q's [B, T, H] dimensions")
    if key_shape[-1] != _DIM or value_shape[-1] != _DIM:
        raise ValueError("the SM120 backward kernel requires K == V == 128")

    expected_state_shape = (key_shape[0], key_shape[2], _DIM, _DIM)
    if final_state.shape != expected_state_shape or d_final_state.shape != expected_state_shape:
        raise ValueError(
            "final_state and d_final_state must have shape [B, H, 128, 128] = "
            f"{expected_state_shape}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q and all sequence tensors must use torch.float16 or torch.bfloat16")
    low_precision_tensors = (q, k, v, beta, w, do)
    if any(tensor.dtype != q.dtype for tensor in low_precision_tensors):
        raise TypeError("q, k, v, beta, w, and do must have the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must have the input dtype or torch.float32")
    if final_state.dtype != torch.float32 or d_final_state.dtype != torch.float32:
        raise TypeError("final_state and d_final_state must use torch.float32")
    all_tensors = (*low_precision_tensors, g, final_state, d_final_state)
    if not q.is_cuda or any(not tensor.is_cuda for tensor in all_tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if any(tensor.device != q.device for tensor in all_tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if any(not tensor.is_contiguous() for tensor in all_tensors):
        raise ValueError("all tensors must be contiguous")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("this kernel is specialized for SM120 GPUs")


def _chunk_backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    final_state: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run backward after public validation and CUDA-device selection."""

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dw = torch.empty_like(w)
    d_initial_state = torch.empty_like(final_state)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)

    # PyTorch deliberately rejects DLPack export of requires-grad tensors.
    # Backward is not itself differentiable, so pass storage-sharing detached
    # views to CuTe/TVM-FFI when called from the public autograd wrapper.
    q_arg = q.detach() if q.requires_grad else q
    k_arg = k.detach() if k.requires_grad else k
    v_arg = v.detach() if v.requires_grad else v
    g_arg = g.detach() if g.requires_grad else g
    beta_arg = beta.detach() if beta.requires_grad else beta
    w_arg = w.detach() if w.requires_grad else w
    final_state_arg = final_state.detach() if final_state.requires_grad else final_state
    do_arg = do.detach() if do.requires_grad else do
    d_final_state_arg = d_final_state.detach() if d_final_state.requires_grad else d_final_state

    use_checkpoint_path = q.shape[1] > MAX_BACKWARD_TOKENS or (
        q.shape[1] == MAX_BACKWARD_TOKENS and initial_state is not None
    )
    if use_checkpoint_path:
        initial_state_storage = (
            torch.zeros_like(final_state) if initial_state is None else initial_state
        )
        initial_state_arg = (
            initial_state_storage.detach()
            if initial_state_storage.requires_grad
            else initial_state_storage
        )
        n_chunks = math.ceil(q.shape[1] / _CHECKPOINT_CHUNK)
        checkpoints = torch.empty(
            (q.shape[0], n_chunks + 1, q.shape[2], _DIM, _DIM),
            device=q.device,
            dtype=torch.float32,
        )
        partial = torch.empty(
            (4, q.shape[0], q.shape[1], q.shape[2], _LONG_VALUE_TILES, _DIM),
            device=q.device,
            dtype=torch.float32,
        )
        compile_arguments = (
            from_dlpack(q_arg),
            from_dlpack(k_arg),
            from_dlpack(v_arg),
            from_dlpack(g_arg),
            from_dlpack(beta_arg),
            from_dlpack(w_arg),
            from_dlpack(initial_state_arg),
            from_dlpack(do_arg),
            from_dlpack(d_final_state_arg),
            from_dlpack(checkpoints),
            from_dlpack(partial),
            from_dlpack(dq),
            from_dlpack(dk),
            from_dlpack(dv),
            from_dlpack(dg),
            from_dlpack(dbeta),
            from_dlpack(dw),
            from_dlpack(d_initial_state),
            cutlass.Float32(output_scale),
            stream,
        )
        runtime_arguments = (
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            beta_arg,
            w_arg,
            initial_state_arg,
            do_arg,
            d_final_state_arg,
            checkpoints,
            partial,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dw,
            d_initial_state,
            output_scale,
            stream,
        )
        cache_key = (q.device.index, q.dtype, g.dtype, tuple(q.shape), tuple(v.shape))
        compiled = _CHECKPOINT_COMPILED.get(cache_key)
        if compiled is None:
            with _COMPILE_LOCK:
                compiled = _CHECKPOINT_COMPILED.get(cache_key)
                if compiled is None:
                    compiled = cute.compile(
                        _launch_checkpoint_recompute_backward,
                        *compile_arguments,
                        options="--enable-tvm-ffi",
                    )
                    _CHECKPOINT_COMPILED[cache_key] = compiled
        compiled(*runtime_arguments)
        return dq, dk, dv, dg, dbeta, dw, d_initial_state

    # A second launch pays for itself once there is more than one token.  Keep
    # the original single-CTA/head path for latency-sensitive recurrent use.
    if q.shape[1] > 1:
        partial = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2], _VALUE_TILES, _DIM, 4),
            device=q.device,
            dtype=torch.float32,
        )
        compile_arguments = (
            from_dlpack(q_arg),
            from_dlpack(k_arg),
            from_dlpack(v_arg),
            from_dlpack(g_arg),
            from_dlpack(beta_arg),
            from_dlpack(w_arg),
            from_dlpack(final_state_arg),
            from_dlpack(do_arg),
            from_dlpack(d_final_state_arg),
            from_dlpack(dq),
            from_dlpack(dk),
            from_dlpack(dv),
            from_dlpack(dg),
            from_dlpack(dbeta),
            from_dlpack(dw),
            from_dlpack(d_initial_state),
            from_dlpack(partial),
            cutlass.Float32(output_scale),
            stream,
        )
        runtime_arguments = (
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            beta_arg,
            w_arg,
            final_state_arg,
            do_arg,
            d_final_state_arg,
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dw,
            d_initial_state,
            partial,
            output_scale,
            stream,
        )
        cache_key = (q.device.index, q.dtype, g.dtype, tuple(q.shape), tuple(v.shape))
        compiled = _TILED_COMPILED.get(cache_key)
        if compiled is None:
            with _COMPILE_LOCK:
                compiled = _TILED_COMPILED.get(cache_key)
                if compiled is None:
                    compiled = cute.compile(
                        _launch_chunk_backward_value_tiled,
                        *compile_arguments,
                        options="--enable-tvm-ffi",
                    )
                    _TILED_COMPILED[cache_key] = compiled
        compiled(*runtime_arguments)
        return dq, dk, dv, dg, dbeta, dw, d_initial_state

    compile_arguments = (
        from_dlpack(q_arg),
        from_dlpack(k_arg),
        from_dlpack(v_arg),
        from_dlpack(g_arg),
        from_dlpack(beta_arg),
        from_dlpack(w_arg),
        from_dlpack(final_state_arg),
        from_dlpack(do_arg),
        from_dlpack(d_final_state_arg),
        from_dlpack(dq),
        from_dlpack(dk),
        from_dlpack(dv),
        from_dlpack(dg),
        from_dlpack(dbeta),
        from_dlpack(dw),
        from_dlpack(d_initial_state),
        cutlass.Float32(output_scale),
        stream,
    )
    runtime_arguments = (
        q_arg,
        k_arg,
        v_arg,
        g_arg,
        beta_arg,
        w_arg,
        final_state_arg,
        do_arg,
        d_final_state_arg,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dw,
        d_initial_state,
        output_scale,
        stream,
    )
    cache_key = (q.device.index, q.dtype, g.dtype, tuple(q.shape), tuple(v.shape))
    compiled = _COMPILED.get(cache_key)
    if compiled is None:
        with _COMPILE_LOCK:
            compiled = _COMPILED.get(cache_key)
            if compiled is None:
                compiled = cute.compile(
                    _launch_chunk_backward,
                    *compile_arguments,
                    options="--enable-tvm-ffi",
                )
                _COMPILED[cache_key] = compiled
    compiled(*runtime_arguments)
    return dq, dk, dv, dg, dbeta, dw, d_initial_state


def chunk_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    final_state: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return ``dq, dk, dv, dg, dbeta, dw, d_initial_state``.

    ``q/k/v/beta/w/do`` are contiguous ``[B, T, H, 128]`` FP16 or BF16
    tensors; ``g`` may additionally be FP32. ``final_state`` and
    ``d_final_state`` are contiguous
    ``[B, H, 128, 128]`` FP32 tensors.  The kernel reconstructs all earlier
    states.  For sequences longer than 128 tokens, pass the exact FP32 forward
    ``initial_state``; ``None`` means the forward recurrence started from zero.
    Supplying it at exactly 128 tokens also selects the faster checkpoint path.
    This path recomputes FP32 boundaries from ``initial_state`` and resets to
    each saved chunk end, avoiding ill-conditioned inversion across the full
    rounded final state.
    """

    global _SINGLE_CUDA_DEVICE

    _validate_backward_inputs(q, k, v, g, beta, w, final_state, do, d_final_state)
    if initial_state is not None:
        if initial_state.shape != final_state.shape or initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be contiguous float32 [B, H, 128, 128]")
        if not initial_state.is_cuda or not initial_state.is_contiguous():
            raise ValueError("initial_state must be a contiguous CUDA tensor")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same CUDA device as q")
    effective_arch = os.environ.get("CUTE_DSL_ARCH")
    if effective_arch != "sm_120":
        raise RuntimeError(
            "the SM120 backward kernel requires CUTE_DSL_ARCH=sm_120, "
            f"but the effective value is {effective_arch!r}"
        )

    # Avoid the CUDA device exchange on a single-GPU workstation: even a
    # no-op exchange is material at T=1 latency.  On multi-GPU systems,
    # compilation, current-stream lookup, allocations, and launch all execute
    # inside the tensor's explicit device context.
    if _SINGLE_CUDA_DEVICE is None:
        _SINGLE_CUDA_DEVICE = torch.cuda.device_count() == 1
    if _SINGLE_CUDA_DEVICE:
        return _chunk_backward_impl(
            q,
            k,
            v,
            g,
            beta,
            w,
            final_state,
            do,
            d_final_state,
            scale,
            initial_state,
        )
    with torch.cuda.device(q.device):
        return _chunk_backward_impl(
            q,
            k,
            v,
            g,
            beta,
            w,
            final_state,
            do,
            d_final_state,
            scale,
            initial_state,
        )


__all__ = ["MAX_BACKWARD_TOKENS", "chunk_backward"]

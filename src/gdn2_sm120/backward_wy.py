"""Compact-WY chunk-local VJPs for Gated DeltaNet-2.

The dimension-generic PyTorch implementation is an executable proof of the
VJP, while the staged SM120 CuTe implementation backs the production long
sequence autograd path.
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
import torch

_CHUNK_SIZE = 16
_DIM = 128
_MMA_WARPS = _DIM // 8
_MMA_THREADS = 32 * _MMA_WARPS
_CHAIN_THREADS = _DIM
_STATE_DOT_KEY_WARPS = 8
_STATE_DOT_THREADS = 32 * _STATE_DOT_KEY_WARPS
_LINEAR_THREADS = 256

_TRIANGLE_NONE = 0
_TRIANGLE_LOWER = 1
_TRIANGLE_STRICT_LOWER = 2

_COMBINE_COPY = 0
_COMBINE_SUBTRACT = 1
_COMBINE_NEGATE = 2


def _use_fused_state_decay_dot(
    _batch: int,
    _n_chunks: int,
    _heads: int,
    compact_boundaries: bool,
) -> bool:
    """Fold the state/gradient dot into an existing MMA for compact checkpoints."""

    return compact_boundaries


@dataclass(frozen=True)
class CompactWYVJPProof:
    """Boundary gradients recomputed by each independent chunk VJP."""

    dstate_starts: torch.Tensor


def _reverse_cumsum(value: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(value, dims=(dim,)), dim=dim), dims=(dim,))


def compact_wy_chunk_vjp_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state_boundaries: torch.Tensor,
    dstate_boundaries: torch.Tensor,
    aux: object,
    do: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = 16,
    verify_boundaries: bool = True,
) -> tuple[tuple[torch.Tensor, ...], CompactWYVJPProof]:
    """Evaluate the exact compact-WY parameter VJP in PyTorch.

    Every chunk uses only ``S0``, ``dS1``, its forward compact-WY tensors, and
    its local output gradient.  The loop is therefore an executable proof that
    all parameter VJPs may be launched independently once the two boundary
    tensors have been materialized.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q and v must be rank-4 [B, T, H, D] tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape != do.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v, w, and do must agree with q's [B, T, H] dimensions")

    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    n_chunks = math.ceil(time / chunk_size)
    boundary_shape = (batch, n_chunks + 1, heads, key_dim, value_dim)
    if state_boundaries.shape != boundary_shape or dstate_boundaries.shape != boundary_shape:
        raise ValueError(f"state boundaries must have shape {boundary_shape}")
    if any(
        tensor.shape[:3] != (batch, time, heads)
        for tensor in (aux.y, aux.u, aux.q_gamma, aux.k_tail)
    ):
        raise ValueError("compact-WY sequence auxiliaries have invalid leading dimensions")
    if aux.y.shape[-1] != key_dim or aux.q_gamma.shape[-1] != key_dim:
        raise ValueError("y and q_gamma must use the key dimension")
    if aux.u.shape[-1] != value_dim or aux.k_tail.shape[-1] != key_dim:
        raise ValueError("u or k_tail has an invalid feature dimension")
    if aux.decay_end.shape != (batch, n_chunks, heads, key_dim):
        raise ValueError("decay_end has an invalid shape")
    if aux.aqk.shape != (batch, n_chunks, heads, chunk_size, chunk_size):
        raise ValueError("aqk has an invalid shape")

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    compute_gradients = [
        torch.empty_like(tensor, dtype=compute_dtype) for tensor in (q, k, v, g, beta, w)
    ]
    dstate_starts = torch.empty(
        (batch, n_chunks, heads, key_dim, value_dim),
        device=q.device,
        dtype=compute_dtype,
    )

    for chunk in range(n_chunks):
        start = chunk * chunk_size
        stop = min(start + chunk_size, time)
        length = stop - start

        q_c = q[:, start:stop].transpose(1, 2).to(compute_dtype)
        k_c = k[:, start:stop].transpose(1, 2).to(compute_dtype)
        v_c = v[:, start:stop].transpose(1, 2).to(compute_dtype)
        g_c = g[:, start:stop].transpose(1, 2).to(compute_dtype)
        beta_c = beta[:, start:stop].transpose(1, 2).to(compute_dtype)
        w_c = w[:, start:stop].transpose(1, 2).to(compute_dtype)
        do_c = do[:, start:stop].transpose(1, 2).to(compute_dtype)
        y_c = aux.y[:, start:stop].transpose(1, 2).to(compute_dtype)
        value_aux_c = aux.u[:, start:stop].transpose(1, 2).to(compute_dtype)
        q_gamma_c = aux.q_gamma[:, start:stop].transpose(1, 2).to(compute_dtype)
        k_tail_c = aux.k_tail[:, start:stop].transpose(1, 2).to(compute_dtype)
        decay_end_c = aux.decay_end[:, chunk].to(compute_dtype)
        aqk_c = aux.aqk[:, chunk, :, :length, :length].to(compute_dtype)
        state0 = state_boundaries[:, chunk].to(compute_dtype)
        dstate1 = dstate_boundaries[:, chunk + 1].to(compute_dtype)

        cumulative_g = g_c.cumsum(dim=-2)
        gamma = cumulative_g.exp()
        k_bar = k_c / gamma
        erase_bar = gamma * beta_c * k_c
        lower = torch.tril(erase_bar @ k_bar.transpose(-1, -2), diagonal=-1)

        has_forward_residual = bool(getattr(aux, "u_is_residual", False))
        residual = value_aux_c if has_forward_residual else value_aux_c - y_c @ state0
        d_q_gamma = do_c @ state0.transpose(-1, -2)
        d_aqk = torch.tril(do_c @ residual.transpose(-1, -2))
        d_k_tail = residual @ dstate1.transpose(-1, -2)
        d_residual = aqk_c.transpose(-1, -2) @ do_c + k_tail_c @ dstate1
        d_y = -(d_residual @ state0.transpose(-1, -2))

        dstate0 = (
            decay_end_c.unsqueeze(-1) * dstate1
            + q_gamma_c.transpose(-1, -2) @ do_c
            - y_c.transpose(-1, -2) @ d_residual
        )
        dstate_starts[:, chunk] = dstate0
        if verify_boundaries:
            torch.testing.assert_close(
                dstate0,
                dstate_boundaries[:, chunk],
                atol=6e-5,
                rtol=6e-5,
            )

        system_t = (lower + torch.eye(length, device=q.device, dtype=compute_dtype)).transpose(
            -1, -2
        )
        d_z = torch.linalg.solve_triangular(system_t, d_residual, upper=True)
        d_e0 = torch.linalg.solve_triangular(system_t, d_y, upper=True)
        if has_forward_residual:
            # dE0 = -dZ @ S0.T, hence
            # dZ @ U.T + dE0 @ Y.T = dZ @ (U - Y @ S0).T.
            d_lower = -torch.tril(d_z @ residual.transpose(-1, -2), diagonal=-1)
        else:
            d_lower = -torch.tril(
                d_z @ value_aux_c.transpose(-1, -2) + d_e0 @ y_c.transpose(-1, -2),
                diagonal=-1,
            )

        d_erase = d_e0 + d_lower @ k_bar
        d_k_bar = d_lower.transpose(-1, -2) @ erase_bar
        d_q_gamma = d_q_gamma + d_aqk @ k_bar
        d_k_bar = d_k_bar + d_aqk.transpose(-1, -2) @ q_gamma_c
        d_k_bar = d_k_bar + d_k_tail * decay_end_c.unsqueeze(-2)
        d_decay_end = (d_k_tail * k_bar).sum(dim=-2) + (dstate1 * state0).sum(dim=-1)

        dq_c = output_scale * gamma * d_q_gamma
        dk_c = d_k_bar / gamma
        dgamma = output_scale * d_q_gamma * q_c - d_k_bar * k_bar / gamma
        dk_c = dk_c + d_erase * gamma * beta_c
        dbeta_c = d_erase * gamma * k_c
        dgamma = dgamma + d_erase * beta_c * k_c
        d_log_gamma = dgamma * gamma
        d_log_gamma[..., -1, :] = d_log_gamma[..., -1, :] + d_decay_end * decay_end_c
        dg_c = _reverse_cumsum(d_log_gamma, dim=-2)
        dv_c = d_z * w_c
        dw_c = d_z * v_c

        for destination, source in zip(
            compute_gradients,
            (dq_c, dk_c, dv_c, dg_c, dbeta_c, dw_c),
            strict=True,
        ):
            destination[:, start:stop] = source.transpose(1, 2)

    dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, w.dtype)
    gradients = tuple(
        gradient.to(dtype) for gradient, dtype in zip(compute_gradients, dtypes, strict=True)
    ) + (dstate_boundaries[:, 0].contiguous(),)
    return gradients, CompactWYVJPProof(dstate_starts)


@cute.kernel
def _prepare_compact_operands_kernel(
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    gamma: cute.Tensor,
    erase_bar: cute.Tensor,
    k_bar: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
):
    """Build FP32 ``gamma`` and padded low-precision ``E``/``K_bar``."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE

    if tidx < _DIM:
        cumulative_g = cutlass.Float32(0.0)
        for row in cutlass.range_constexpr(_CHUNK_SIZE):
            token = start + row
            gamma_value = cutlass.Float32(0.0)
            erase_value = cutlass.Float32(0.0)
            k_bar_value = cutlass.Float32(0.0)
            if token < time:
                cumulative_g += g[batch, token, head, tidx].to(cutlass.Float32)
                gamma_value = cute.math.exp(cumulative_g, fastmath=False)
                k_value = k[batch, token, head, tidx].to(cutlass.Float32)
                beta_value = beta[batch, token, head, tidx].to(cutlass.Float32)
                erase_value = gamma_value * beta_value * k_value
                k_bar_value = k_value / gamma_value
            gamma[batch, token, head, tidx] = gamma_value
            erase_bar[batch, token, head, tidx] = erase_value.to(erase_bar.element_type)
            k_bar[batch, token, head, tidx] = k_bar_value.to(k_bar.element_type)


@cute.kernel
def _m16_n128_k128_kernel(
    left: cute.Tensor,
    right_boundaries: cute.Tensor,
    combine_source: cute.Tensor,
    output: cute.Tensor,
    left_second: cute.Tensor,
    combine_source_second: cute.Tensor,
    output_second: cute.Tensor,
    dstate_dot_boundaries: cute.Tensor,
    state_decay_dot: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    boundary_offset: cutlass.Constexpr,
    transpose_right: cutlass.Constexpr,
    combine_mode: cutlass.Constexpr,
    combine_mode_second: cutlass.Constexpr,
    has_second_product: cutlass.Constexpr,
    compute_state_decay_dot: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_b_layout: cute.Layout,
):
    """Compute one padded ``[16, 128]`` product per chunk with tensor cores."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE
    output_start = warp * 8
    operand_type = tiled_mma.op.ab_dtype

    allocator = cutlass.utils.SmemAllocator()
    s_a = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_a_second = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_b_all = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    output_pair = warp >> 1
    output_half = warp & 1
    s_b_pair_storage = s_b_all[output_pair, None, None, None]
    s_b_pair = cute.make_tensor(
        s_b_pair_storage.iterator,
        cute.make_layout((_CHUNK_SIZE, 16), stride=(16, 1)),
    )
    s_b = s_b_all[output_pair, output_half, None, None]
    s_b_as_operand = cute.make_tensor(
        s_b.iterator,
        cute.make_layout((8, _CHUNK_SIZE), stride=(1, 16)),
    )

    thr_mma = tiled_mma.get_slice(lane)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cs_a_second = thr_mma.partition_A(s_a_second)
    t_cs_b = thr_mma.partition_B(s_b_as_operand)
    t_cr_a = thr_mma.make_fragment_A(t_cs_a)
    t_cr_a_second = thr_mma.make_fragment_A(t_cs_a_second)
    t_cr_b = thr_mma.make_fragment_B(t_cs_b)
    g_output = cute.local_tile(
        output[batch, None, head, None],
        (_CHUNK_SIZE, 8),
        (chunk, warp),
    )
    t_cg_output = thr_mma.partition_C(g_output)
    t_cr_output = thr_mma.make_fragment_C(t_cg_output)
    t_cr_output.fill(0.0)
    g_output_second = cute.local_tile(
        output_second[batch, None, head, None],
        (_CHUNK_SIZE, 8),
        (chunk, warp),
    )
    t_cg_output_second = thr_mma.partition_C(g_output_second)
    t_cr_output_second = thr_mma.make_fragment_C(t_cg_output_second)
    if cutlass.const_expr(has_second_product):
        t_cr_output_second.fill(0.0)

    copy_atom_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    copy_atom_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_b = tiled_copy_b.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_ss_a_second = thr_copy_a.partition_S(s_a_second)
    t_ss_b = thr_copy_b.partition_S(s_b_as_operand)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_sr_a_second = thr_copy_a.retile(t_cr_a_second)
    t_sr_b = thr_copy_b.retile(t_cr_b)
    state_dot_partial = cutlass.Float32(0.0)

    if cutlass.const_expr(not transpose_right and right_boundaries.element_type == operand_type):
        copy_atom_g2s = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), operand_type, num_bits_per_copy=128
        )
        tiled_copy_g2s = cute.make_tiled_copy_tv(
            copy_atom_g2s,
            cute.make_layout((16, 2), stride=(2, 1)),
            cute.make_layout((1, 8), stride=(8, 1)),
        )
        thr_copy_g2s = tiled_copy_g2s.get_slice(lane)
        t_ds_b_pair = thr_copy_g2s.partition_D(s_b_pair)

    for key_tile in cutlass.range_constexpr(_DIM // _CHUNK_SIZE):
        if tidx < _CHUNK_SIZE * _CHUNK_SIZE:
            row = tidx // _CHUNK_SIZE
            key_local = tidx - row * _CHUNK_SIZE
            token = start + row
            value = cutlass.Float32(0.0)
            if token < time:
                value = left[
                    batch,
                    token,
                    head,
                    key_tile * _CHUNK_SIZE + key_local,
                ].to(cutlass.Float32)
            s_a[row, key_local] = value.to(operand_type)
            if cutlass.const_expr(has_second_product):
                second_value = cutlass.Float32(0.0)
                if token < time:
                    second_value = left_second[
                        batch,
                        token,
                        head,
                        key_tile * _CHUNK_SIZE + key_local,
                    ].to(cutlass.Float32)
                s_a_second[row, key_local] = second_value.to(operand_type)
        if cutlass.const_expr(
            not transpose_right and right_boundaries.element_type == operand_type
        ):
            if output_half == 0:
                g_b_pair = cute.local_tile(
                    right_boundaries[batch, chunk + boundary_offset, head, None, None],
                    (_CHUNK_SIZE, 16),
                    (key_tile, output_pair),
                )
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_b_pair), t_ds_b_pair)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        else:
            if cutlass.const_expr(
                transpose_right and right_boundaries.element_type == operand_type
            ):
                # One aligned 128-bit load per half-warp lane replaces four
                # strided scalar rounds.  Lane pairs own one output key, load
                # its two contiguous V8 segments, scatter the transposed MMA
                # tile into shared memory, and retain the same values for the
                # state-decay dot.
                if lane < 16:
                    output_local = lane >> 1
                    segment = lane & 1
                    output_idx = output_start + output_local
                    vector_tile = key_tile * 2 + segment
                    g_state_vector = cute.local_tile(
                        right_boundaries[
                            batch,
                            chunk + boundary_offset,
                            head,
                            output_idx,
                            None,
                        ],
                        (8,),
                        (vector_tile,),
                    )
                    r_state_vector = cute.make_rmem_tensor((8,), operand_type)
                    cute.autovec_copy(g_state_vector, r_state_vector)
                    if cutlass.const_expr(compute_state_decay_dot):
                        g_dstate_vector = cute.local_tile(
                            dstate_dot_boundaries[
                                batch,
                                chunk + 1,
                                head,
                                output_idx,
                                None,
                            ],
                            (8,),
                            (vector_tile,),
                        )
                        r_dstate_vector = cute.make_rmem_tensor(
                            (8,), dstate_dot_boundaries.element_type
                        )
                        cute.autovec_copy(g_dstate_vector, r_dstate_vector)
                    for vector_idx in cutlass.range_constexpr(8):
                        key_local = segment * 8 + vector_idx
                        state_value = r_state_vector[vector_idx]
                        s_b[key_local, output_local] = state_value
                        if cutlass.const_expr(compute_state_decay_dot):
                            state_dot_partial += state_value.to(cutlass.Float32) * r_dstate_vector[
                                vector_idx
                            ].to(cutlass.Float32)
            else:
                for linear in cutlass.range(lane, _CHUNK_SIZE * 8, 32):
                    key_local = linear // 8
                    output_local = linear - key_local * 8
                    key_idx = key_tile * _CHUNK_SIZE + key_local
                    output_idx = output_start + output_local
                    if cutlass.const_expr(transpose_right):
                        value = right_boundaries[
                            batch,
                            chunk + boundary_offset,
                            head,
                            output_idx,
                            key_idx,
                        ]
                    else:
                        value = right_boundaries[
                            batch,
                            chunk + boundary_offset,
                            head,
                            key_idx,
                            output_idx,
                        ]
                    s_b[key_local, output_local] = value.to(operand_type)
            cute.arch.sync_threads()
        cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
        cute.copy(tiled_copy_b, t_ss_b[None, None, 0], t_sr_b[None, None, 0])
        cute.gemm(
            tiled_mma,
            t_cr_output,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_output,
        )
        if cutlass.const_expr(has_second_product):
            cute.copy(
                tiled_copy_a,
                t_ss_a_second[None, None, 0],
                t_sr_a_second[None, None, 0],
            )
            cute.gemm(
                tiled_mma,
                t_cr_output_second,
                t_cr_a_second[None, None, 0],
                t_cr_b[None, None, 0],
                t_cr_output_second,
            )
        cute.arch.sync_threads()

    if cutlass.const_expr(compute_state_decay_dot):
        state_dot_partial += cute.arch.shuffle_sync_bfly(state_dot_partial, 1)
        if lane < 16 and lane & 1 == 0:
            state_decay_dot[batch, chunk, head, output_start + (lane >> 1)] = state_dot_partial

    if cutlass.const_expr(combine_mode == _COMBINE_SUBTRACT):
        g_source = cute.local_tile(
            combine_source[batch, None, head, None],
            (_CHUNK_SIZE, 8),
            (chunk, warp),
        )
        t_cg_source = thr_mma.partition_C(g_source)
        t_cr_source = cute.make_fragment_like(t_cr_output, combine_source.element_type)
        cute.autovec_copy(t_cg_source, t_cr_source)
        t_cr_output[None] = t_cr_source.load().to(cutlass.Float32) - t_cr_output.load()
    elif cutlass.const_expr(combine_mode == _COMBINE_NEGATE):
        t_cr_output[None] = -t_cr_output.load()

    if cutlass.const_expr(output.element_type == cutlass.Float32):
        cute.autovec_copy(t_cr_output, t_cg_output)
    else:
        t_cr_output_cast = cute.make_fragment_like(t_cr_output, output.element_type)
        t_cr_output_cast[None] = t_cr_output.load().to(output.element_type)
        cute.autovec_copy(t_cr_output_cast, t_cg_output)

    if cutlass.const_expr(has_second_product):
        if cutlass.const_expr(combine_mode_second == _COMBINE_SUBTRACT):
            g_source_second = cute.local_tile(
                combine_source_second[batch, None, head, None],
                (_CHUNK_SIZE, 8),
                (chunk, warp),
            )
            t_cg_source_second = thr_mma.partition_C(g_source_second)
            t_cr_source_second = cute.make_fragment_like(
                t_cr_output_second, combine_source_second.element_type
            )
            cute.autovec_copy(t_cg_source_second, t_cr_source_second)
            t_cr_output_second[None] = (
                t_cr_source_second.load().to(cutlass.Float32) - t_cr_output_second.load()
            )
        elif cutlass.const_expr(combine_mode_second == _COMBINE_NEGATE):
            t_cr_output_second[None] = -t_cr_output_second.load()

        if cutlass.const_expr(output_second.element_type == cutlass.Float32):
            cute.autovec_copy(t_cr_output_second, t_cg_output_second)
        else:
            t_cr_output_second_cast = cute.make_fragment_like(
                t_cr_output_second, output_second.element_type
            )
            t_cr_output_second_cast[None] = t_cr_output_second.load().to(output_second.element_type)
            cute.autovec_copy(t_cr_output_second_cast, t_cg_output_second)


@cute.kernel
def _m16_n16_k128_kernel(
    left: cute.Tensor,
    right: cute.Tensor,
    left_second: cute.Tensor,
    right_second: cute.Tensor,
    output: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    add_output: cutlass.Constexpr,
    negate: cutlass.Constexpr,
    triangle: cutlass.Constexpr,
    has_second_product: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_b_layout: cute.Layout,
    s_partial_layout: cute.Layout,
):
    """Compute a 16x16 sequence Gram product with an eight-way K split."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    n_tile = warp & 1
    key_tile = warp >> 1
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE
    operand_type = tiled_mma.op.ab_dtype

    allocator = cutlass.utils.SmemAllocator()
    s_a_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_b_all = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    s_partial_all = allocator.allocate_tensor(
        cutlass.Float32, s_partial_layout, byte_alignment=1024
    )
    # The two N8 warps for one K16 tile share their identical A operand.
    s_a = s_a_all[key_tile, None, None]
    s_b = s_b_all[warp, None, None]
    s_partial = s_partial_all[warp, None, None]
    s_b_as_operand = cute.make_tensor(
        s_b.iterator,
        cute.make_layout((8, _CHUNK_SIZE), stride=(1, 8)),
    )

    thr_mma = tiled_mma.get_slice(lane)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cs_b = thr_mma.partition_B(s_b_as_operand)
    t_cs_partial = thr_mma.partition_C(s_partial)
    t_cr_a = thr_mma.make_fragment_A(t_cs_a)
    t_cr_b = thr_mma.make_fragment_B(t_cs_b)
    t_cr_partial = thr_mma.make_fragment_C(t_cs_partial)
    copy_atom_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    copy_atom_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_b = tiled_copy_b.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_ss_b = thr_copy_b.partition_S(s_b_as_operand)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_sr_b = thr_copy_b.retile(t_cr_b)

    if n_tile == 0:
        for linear in cutlass.range(lane, _CHUNK_SIZE * _CHUNK_SIZE, 32):
            row = linear // _CHUNK_SIZE
            key_local = linear - row * _CHUNK_SIZE
            token = start + row
            value = cutlass.Float32(0.0)
            if token < time:
                value = left[
                    batch,
                    token,
                    head,
                    key_tile * _CHUNK_SIZE + key_local,
                ].to(cutlass.Float32)
            s_a[row, key_local] = value.to(operand_type)
    for linear in cutlass.range(lane, _CHUNK_SIZE * 8, 32):
        key_local = linear // 8
        column_local = linear - key_local * 8
        token = start + n_tile * 8 + column_local
        value = cutlass.Float32(0.0)
        if token < time:
            value = right[
                batch,
                token,
                head,
                key_tile * _CHUNK_SIZE + key_local,
            ].to(cutlass.Float32)
        s_b[key_local, column_local] = value.to(operand_type)
    cute.arch.sync_threads()
    cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
    cute.copy(tiled_copy_b, t_ss_b[None, None, 0], t_sr_b[None, None, 0])
    t_cr_partial.fill(0.0)
    cute.gemm(
        tiled_mma,
        t_cr_partial,
        t_cr_a[None, None, 0],
        t_cr_b[None, None, 0],
        t_cr_partial,
    )
    if cutlass.const_expr(has_second_product):
        # Both N warps must finish their first ld.matrix before the owner warp
        # replaces the shared pair operand.
        cute.arch.sync_threads()
        if n_tile == 0:
            for linear in cutlass.range(lane, _CHUNK_SIZE * _CHUNK_SIZE, 32):
                row = linear // _CHUNK_SIZE
                key_local = linear - row * _CHUNK_SIZE
                token = start + row
                value = cutlass.Float32(0.0)
                if token < time:
                    value = left_second[
                        batch,
                        token,
                        head,
                        key_tile * _CHUNK_SIZE + key_local,
                    ].to(cutlass.Float32)
                s_a[row, key_local] = value.to(operand_type)
        for linear in cutlass.range(lane, _CHUNK_SIZE * 8, 32):
            key_local = linear // 8
            column_local = linear - key_local * 8
            token = start + n_tile * 8 + column_local
            value = cutlass.Float32(0.0)
            if token < time:
                value = right_second[
                    batch,
                    token,
                    head,
                    key_tile * _CHUNK_SIZE + key_local,
                ].to(cutlass.Float32)
            s_b[key_local, column_local] = value.to(operand_type)
        cute.arch.sync_threads()
        cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
        cute.copy(tiled_copy_b, t_ss_b[None, None, 0], t_sr_b[None, None, 0])
        cute.gemm(
            tiled_mma,
            t_cr_partial,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_partial,
        )
    cute.autovec_copy(t_cr_partial, t_cs_partial)
    cute.arch.sync_threads()

    if tidx < _CHUNK_SIZE * _CHUNK_SIZE:
        row = tidx // _CHUNK_SIZE
        column = tidx - row * _CHUNK_SIZE
        column_tile = column // 8
        column_local = column - column_tile * 8
        value = cutlass.Float32(0.0)
        for source_tile in cutlass.range_constexpr(_DIM // _CHUNK_SIZE):
            source_warp = source_tile * 2 + column_tile
            value += s_partial_all[source_warp, row, column_local]
        if cutlass.const_expr(add_output):
            value += output[batch, chunk, head, row, column]
        if cutlass.const_expr(negate):
            value = -value
        keep = True
        if cutlass.const_expr(triangle == _TRIANGLE_LOWER):
            keep = row >= column
        elif cutlass.const_expr(triangle == _TRIANGLE_STRICT_LOWER):
            keep = row > column
        if keep and start + row < time and start + column < time:
            output[batch, chunk, head, row, column] = value
        else:
            output[batch, chunk, head, row, column] = 0.0


@cute.kernel
def _m16_n128_k16_kernel(
    square: cute.Tensor,
    sequence: cute.Tensor,
    square_second: cute.Tensor,
    sequence_second: cute.Tensor,
    output: cute.Tensor,
    output_independent: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    transpose_square: cutlass.Constexpr,
    add_output: cutlass.Constexpr,
    has_second_product: cutlass.Constexpr,
    add_output_independent: cutlass.Constexpr,
    has_independent_output: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_b_layout: cute.Layout,
):
    """Multiply one 16x16 matrix by a padded 16x128 sequence."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE
    output_start = warp * 8
    operand_type = tiled_mma.op.ab_dtype

    allocator = cutlass.utils.SmemAllocator()
    s_a = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_a_independent = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_b_all = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    s_b = s_b_all[warp, None, None]
    s_b_as_operand = cute.make_tensor(
        s_b.iterator,
        cute.make_layout((8, _CHUNK_SIZE), stride=(1, 8)),
    )

    thr_mma = tiled_mma.get_slice(lane)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cs_a_independent = thr_mma.partition_A(s_a_independent)
    t_cs_b = thr_mma.partition_B(s_b_as_operand)
    t_cr_a = thr_mma.make_fragment_A(t_cs_a)
    t_cr_a_independent = thr_mma.make_fragment_A(t_cs_a_independent)
    t_cr_b = thr_mma.make_fragment_B(t_cs_b)
    g_output = cute.local_tile(
        output[batch, None, head, None],
        (_CHUNK_SIZE, 8),
        (chunk, warp),
    )
    t_cg_output = thr_mma.partition_C(g_output)
    t_cr_output = thr_mma.make_fragment_C(t_cg_output)
    if cutlass.const_expr(add_output):
        if cutlass.const_expr(output.element_type == cutlass.Float32):
            cute.autovec_copy(t_cg_output, t_cr_output)
        else:
            t_cr_existing = cute.make_fragment_like(t_cr_output, output.element_type)
            cute.autovec_copy(t_cg_output, t_cr_existing)
            t_cr_output[None] = t_cr_existing.load().to(cutlass.Float32)
    else:
        t_cr_output.fill(0.0)
    g_output_independent = cute.local_tile(
        output_independent[batch, None, head, None],
        (_CHUNK_SIZE, 8),
        (chunk, warp),
    )
    t_cg_output_independent = thr_mma.partition_C(g_output_independent)
    t_cr_output_independent = thr_mma.make_fragment_C(t_cg_output_independent)
    if cutlass.const_expr(has_independent_output):
        if cutlass.const_expr(add_output_independent):
            if cutlass.const_expr(output_independent.element_type == cutlass.Float32):
                cute.autovec_copy(t_cg_output_independent, t_cr_output_independent)
            else:
                t_cr_existing_independent = cute.make_fragment_like(
                    t_cr_output_independent,
                    output_independent.element_type,
                )
                cute.autovec_copy(t_cg_output_independent, t_cr_existing_independent)
                t_cr_output_independent[None] = t_cr_existing_independent.load().to(cutlass.Float32)
        else:
            t_cr_output_independent.fill(0.0)

    copy_atom_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    copy_atom_b = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom_b, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_b = tiled_copy_b.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_ss_a_independent = thr_copy_a.partition_S(s_a_independent)
    t_ss_b = thr_copy_b.partition_S(s_b_as_operand)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_sr_a_independent = thr_copy_a.retile(t_cr_a_independent)
    t_sr_b = thr_copy_b.retile(t_cr_b)

    if tidx < _CHUNK_SIZE * _CHUNK_SIZE:
        row = tidx // _CHUNK_SIZE
        inner = tidx - row * _CHUNK_SIZE
        if cutlass.const_expr(transpose_square):
            value = square[batch, chunk, head, inner, row]
        else:
            value = square[batch, chunk, head, row, inner]
        s_a[row, inner] = value.to(operand_type)
        if cutlass.const_expr(has_independent_output):
            if cutlass.const_expr(transpose_square):
                independent_value = square_second[batch, chunk, head, inner, row]
            else:
                independent_value = square_second[batch, chunk, head, row, inner]
            s_a_independent[row, inner] = independent_value.to(operand_type)
    for linear in cutlass.range(lane, _CHUNK_SIZE * 8, 32):
        inner = linear // 8
        output_local = linear - inner * 8
        token = start + inner
        value = cutlass.Float32(0.0)
        if token < time:
            value = sequence[batch, token, head, output_start + output_local].to(cutlass.Float32)
        s_b[inner, output_local] = value.to(operand_type)
    cute.arch.sync_threads()
    cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
    cute.copy(tiled_copy_b, t_ss_b[None, None, 0], t_sr_b[None, None, 0])
    cute.gemm(
        tiled_mma,
        t_cr_output,
        t_cr_a[None, None, 0],
        t_cr_b[None, None, 0],
        t_cr_output,
    )
    if cutlass.const_expr(has_independent_output):
        cute.copy(
            tiled_copy_a,
            t_ss_a_independent[None, None, 0],
            t_sr_a_independent[None, None, 0],
        )
        cute.gemm(
            tiled_mma,
            t_cr_output_independent,
            t_cr_a_independent[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_output_independent,
        )
    if cutlass.const_expr(has_second_product):
        # The unfused schedule stores the first product in the low-precision
        # sequence workspace before the second launch reloads it.  Preserve
        # that rounding point explicitly while avoiding the launch boundary.
        if cutlass.const_expr(output.element_type != cutlass.Float32):
            t_cr_rounded = cute.make_fragment_like(t_cr_output, output.element_type)
            t_cr_rounded[None] = t_cr_output.load().to(output.element_type)
            t_cr_output[None] = t_cr_rounded.load().to(cutlass.Float32)
        # ``s_a`` is shared by all output-column warps.  Do not let an early
        # warp overwrite the first square until every warp has loaded it.
        cute.arch.sync_threads()
        if tidx < _CHUNK_SIZE * _CHUNK_SIZE:
            row = tidx // _CHUNK_SIZE
            inner = tidx - row * _CHUNK_SIZE
            if cutlass.const_expr(transpose_square):
                value = square_second[batch, chunk, head, inner, row]
            else:
                value = square_second[batch, chunk, head, row, inner]
            s_a[row, inner] = value.to(operand_type)
        for linear in cutlass.range(lane, _CHUNK_SIZE * 8, 32):
            inner = linear // 8
            output_local = linear - inner * 8
            token = start + inner
            value = cutlass.Float32(0.0)
            if token < time:
                value = sequence_second[
                    batch,
                    token,
                    head,
                    output_start + output_local,
                ].to(cutlass.Float32)
            s_b[inner, output_local] = value.to(operand_type)
        cute.arch.sync_threads()
        cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
        cute.copy(tiled_copy_b, t_ss_b[None, None, 0], t_sr_b[None, None, 0])
        cute.gemm(
            tiled_mma,
            t_cr_output,
            t_cr_a[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_output,
        )
    if cutlass.const_expr(output.element_type == cutlass.Float32):
        cute.autovec_copy(t_cr_output, t_cg_output)
    else:
        t_cr_output_cast = cute.make_fragment_like(t_cr_output, output.element_type)
        t_cr_output_cast[None] = t_cr_output.load().to(output.element_type)
        cute.autovec_copy(t_cr_output_cast, t_cg_output)
    if cutlass.const_expr(has_independent_output):
        if cutlass.const_expr(output_independent.element_type == cutlass.Float32):
            cute.autovec_copy(t_cr_output_independent, t_cg_output_independent)
        else:
            t_cr_output_independent_cast = cute.make_fragment_like(
                t_cr_output_independent,
                output_independent.element_type,
            )
            t_cr_output_independent_cast[None] = t_cr_output_independent.load().to(
                output_independent.element_type
            )
            cute.autovec_copy(t_cr_output_independent_cast, t_cg_output_independent)


@cute.kernel
def _combine_sequence_kernel(
    source: cute.Tensor,
    product: cute.Tensor,
    output: cute.Tensor,
    time: cutlass.Constexpr,
    padded_time: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    mode: cutlass.Constexpr,
):
    """Copy or combine padded FP32 sequence workspaces."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * _LINEAR_THREADS + tidx
    total = output.shape[0] * padded_time * heads * _DIM
    if linear < total:
        feature = linear % _DIM
        sequence_linear = linear // _DIM
        head = sequence_linear % heads
        batch_token = sequence_linear // heads
        token = batch_token % padded_time
        batch = batch_token // padded_time
        source_value = cutlass.Float32(0.0)
        if token < time:
            source_value = source[batch, token, head, feature].to(cutlass.Float32)
        if cutlass.const_expr(mode == _COMBINE_SUBTRACT):
            value = source_value - product[batch, token, head, feature].to(cutlass.Float32)
        elif cutlass.const_expr(mode == _COMBINE_NEGATE):
            value = -source_value
        else:
            value = source_value
        output[batch, token, head, feature] = value.to(output.element_type)


@cute.kernel
def _transpose_triangular_solve_kernel(
    lower: cute.Tensor,
    d_residual: cute.Tensor,
    d_y: cute.Tensor,
    d_z: cute.Tensor,
    d_e0: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
):
    """Solve ``(I + T)^T [dZ,dE0] = [dR,dY]`` feature-wise."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE
    length = cutlass.min(_CHUNK_SIZE, time - start)

    if tidx < _DIM:
        z_values = cute.make_rmem_tensor((_CHUNK_SIZE,), cutlass.Float32)
        e_values = cute.make_rmem_tensor((_CHUNK_SIZE,), cutlass.Float32)
        for reverse_row in cutlass.range_constexpr(_CHUNK_SIZE):
            row = _CHUNK_SIZE - 1 - reverse_row
            z_value = cutlass.Float32(0.0)
            e_value = cutlass.Float32(0.0)
            if row < length:
                z_value = d_residual[batch, start + row, head, tidx].to(cutlass.Float32)
                e_value = d_y[batch, start + row, head, tidx].to(cutlass.Float32)
                for column in cutlass.range_constexpr(_CHUNK_SIZE):
                    if column > row and column < length:
                        coefficient = lower[batch, chunk, head, column, row]
                        z_value -= coefficient * z_values[column]
                        e_value -= coefficient * e_values[column]
            z_values[row] = z_value
            e_values[row] = e_value
            d_z[batch, start + row, head, tidx] = z_value.to(d_z.element_type)
            d_e0[batch, start + row, head, tidx] = e_value.to(d_e0.element_type)


@cute.kernel
def _state_decay_dot_kernel(
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    state_decay_dot: cute.Tensor,
    d_initial_state: cute.Tensor,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_precomputed_d_initial_state: cutlass.Constexpr,
):
    """Compute coalesced ``sum_v(dS1 * S0)`` rows for the gate chain."""

    tidx, _, _ = cute.arch.thread_idx()
    chunk_head, key_tile, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    head = chunk_head % heads
    batch_chunk = chunk_head // heads
    chunk = batch_chunk % n_chunks
    batch = batch_chunk // n_chunks
    key_idx = key_tile * _STATE_DOT_KEY_WARPS + warp

    partial = cutlass.Float32(0.0)
    for value_tile in cutlass.range_constexpr(_DIM // 32):
        value_idx = value_tile * 32 + lane
        partial += dstate_boundaries[batch, chunk + 1, head, key_idx, value_idx].to(
            cutlass.Float32
        ) * state_boundaries[batch, chunk, head, key_idx, value_idx].to(cutlass.Float32)
        if cutlass.const_expr(not has_precomputed_d_initial_state) and chunk == 0:
            d_initial_state[batch, head, key_idx, value_idx] = dstate_boundaries[
                batch, 0, head, key_idx, value_idx
            ].to(cutlass.Float32)
    row_dot = cute.arch.warp_reduction_sum(partial)
    if lane == 0:
        state_decay_dot[batch, chunk, head, key_idx] = row_dot


@cute.kernel
def _compact_chain_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_decay_dot: cute.Tensor,
    gamma: cute.Tensor,
    decay_end: cute.Tensor,
    d_q_gamma: cute.Tensor,
    d_k_tail: cute.Tensor,
    d_z: cute.Tensor,
    d_erase: cute.Tensor,
    d_k_bar: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
):
    """Apply Q/K/E/Z/gamma chains and write the six parameter gradients."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block % heads
    chunk_head = block // heads
    chunk = chunk_head % n_chunks
    batch = chunk_head // n_chunks
    start = chunk * _CHUNK_SIZE
    length = cutlass.min(_CHUNK_SIZE, time - start)

    key_idx = tidx
    if tidx < _DIM:
        decay_end_gradient = state_decay_dot[batch, chunk, head, key_idx]
        decay_end_value = decay_end[batch, chunk, head, key_idx]
        k_bar_values = cute.make_rmem_tensor((_CHUNK_SIZE,), cutlass.Float32)
        inv_gamma_values = cute.make_rmem_tensor((_CHUNK_SIZE,), cutlass.Float32)
        for row in cutlass.range_constexpr(_CHUNK_SIZE):
            k_bar_value = cutlass.Float32(0.0)
            inv_gamma_value = cutlass.Float32(0.0)
            if row < length:
                token = start + row
                inv_gamma_value = cutlass.Float32(1.0) / gamma[
                    batch, token, head, key_idx
                ]
                k_bar_value = (
                    k[batch, token, head, key_idx].to(cutlass.Float32) * inv_gamma_value
                )
                decay_end_gradient += d_k_tail[batch, token, head, key_idx] * k_bar_value
            k_bar_values[row] = k_bar_value
            inv_gamma_values[row] = inv_gamma_value

        running_gate_gradient = cutlass.Float32(0.0)
        for reverse_row in cutlass.range_constexpr(_CHUNK_SIZE):
            row = _CHUNK_SIZE - 1 - reverse_row
            if row < length:
                token = start + row
                gamma_value = gamma[batch, token, head, key_idx]
                # ``k_bar`` is persisted in the MMA operand dtype.  Reuse the
                # FP32 value materialized for the decay edge above, together
                # with the reciprocal shared by all reverse-chain consumers.
                k_bar_value = k_bar_values[row]
                inv_gamma_value = inv_gamma_values[row]
                d_q_gamma_value = d_q_gamma[batch, token, head, key_idx]
                d_k_bar_value = (
                    d_k_bar[batch, token, head, key_idx]
                    + d_k_tail[batch, token, head, key_idx] * decay_end_value
                )
                d_erase_value = d_erase[batch, token, head, key_idx]
                q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                k_value = k[batch, token, head, key_idx].to(cutlass.Float32)
                beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)

                dq[batch, token, head, key_idx] = (scale * gamma_value * d_q_gamma_value).to(
                    dq.element_type
                )
                dk[batch, token, head, key_idx] = (
                    d_k_bar_value * inv_gamma_value
                    + d_erase_value * gamma_value * beta_value
                ).to(dk.element_type)
                dbeta[batch, token, head, key_idx] = (d_erase_value * gamma_value * k_value).to(
                    dbeta.element_type
                )
                dv[batch, token, head, key_idx] = (
                    d_z[batch, token, head, key_idx]
                    * w[batch, token, head, key_idx].to(cutlass.Float32)
                ).to(dv.element_type)
                dw[batch, token, head, key_idx] = (
                    d_z[batch, token, head, key_idx]
                    * v[batch, token, head, key_idx].to(cutlass.Float32)
                ).to(dw.element_type)

                d_gamma_value = (
                    scale * d_q_gamma_value * q_value
                    - d_k_bar_value * k_bar_value * inv_gamma_value
                    + d_erase_value * beta_value * k_value
                )
                if row == length - 1:
                    # ``decay_end`` is the exact FP32 value saved by forward.
                    # Propagate its independent edge directly to log-gamma
                    # instead of multiplying by the recomputed last gamma.
                    running_gate_gradient += decay_end_gradient * decay_end_value
                running_gate_gradient += d_gamma_value * gamma_value
                dg[batch, token, head, key_idx] = running_gate_gradient.to(dg.element_type)


@cute.jit
def _launch_compact_wy_chunk_vjp(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    y: cute.Tensor,
    u: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    do: cute.Tensor,
    gamma: cute.Tensor,
    erase_bar: cute.Tensor,
    k_bar: cute.Tensor,
    lower: cute.Tensor,
    residual_product: cute.Tensor,
    residual: cute.Tensor,
    d_q_gamma: cute.Tensor,
    d_aqk: cute.Tensor,
    d_k_tail: cute.Tensor,
    d_residual: cute.Tensor,
    d_y: cute.Tensor,
    d_z: cute.Tensor,
    d_e0: cute.Tensor,
    d_lower: cute.Tensor,
    d_erase: cute.Tensor,
    d_k_bar: cute.Tensor,
    state_decay_dot: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    has_precomputed_d_residual: cutlass.Constexpr,
    has_precomputed_d_initial_state: cutlass.Constexpr,
    has_forward_residual: cutlass.Constexpr,
    time: cutlass.Constexpr,
    padded_time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    fuse_state_decay_dot: cutlass.Constexpr,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    operand_type = q.element_type
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(operand_type, cutlass.Float32, (16, 8, 16)),
        (1, 1, 1),
        permutation_mnk=(16, 8, 16),
    )
    s_a_layout = cute.make_layout((_CHUNK_SIZE, _CHUNK_SIZE), stride=(_CHUNK_SIZE, 1))
    s_b_layout = cute.make_layout(
        (_MMA_WARPS, _CHUNK_SIZE, 8),
        stride=(_CHUNK_SIZE * 8, 8, 1),
    )
    s_large_b_layout = cute.make_layout(
        (_MMA_WARPS // 2, 2, _CHUNK_SIZE, 8),
        stride=(_CHUNK_SIZE * 16, 8, 16, 1),
    )
    s_square_a_layout = cute.make_layout(
        (_MMA_WARPS // 2, _CHUNK_SIZE, _CHUNK_SIZE),
        stride=(_CHUNK_SIZE * _CHUNK_SIZE, _CHUNK_SIZE, 1),
    )
    s_partial_layout = cute.make_layout(
        (_MMA_WARPS, _CHUNK_SIZE, 8),
        stride=(_CHUNK_SIZE * 8, 8, 1),
    )
    chunk_grid = (q.shape[0] * n_chunks * heads, 1, 1)
    linear_grid = (
        (q.shape[0] * padded_time * heads * _DIM + _LINEAR_THREADS - 1) // _LINEAR_THREADS,
        1,
        1,
    )

    _prepare_compact_operands_kernel(
        k, g, beta, gamma, erase_bar, k_bar, time, n_chunks, heads
    ).launch(grid=chunk_grid, block=(_CHAIN_THREADS, 1, 1), stream=stream)
    _m16_n16_k128_kernel(
        erase_bar,
        k_bar,
        erase_bar,
        k_bar,
        lower,
        time,
        n_chunks,
        heads,
        False,
        False,
        _TRIANGLE_STRICT_LOWER,
        False,
        tiled_mma,
        s_square_a_layout,
        s_b_layout,
        s_partial_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)

    if cutlass.const_expr(has_forward_residual):
        # Forward already persisted R in the former U allocation.
        pass
    elif cutlass.const_expr(time % _CHUNK_SIZE == 0):
        _m16_n128_k128_kernel(
            y,
            state_boundaries,
            u,
            residual,
            y,
            residual,
            residual,
            dstate_boundaries,
            state_decay_dot,
            time,
            n_chunks,
            heads,
            0,
            False,
            _COMBINE_SUBTRACT,
            _COMBINE_SUBTRACT,
            False,
            False,
            tiled_mma,
            s_a_layout,
            s_large_b_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    else:
        _m16_n128_k128_kernel(
            y,
            state_boundaries,
            residual_product,
            residual_product,
            y,
            residual_product,
            residual_product,
            dstate_boundaries,
            state_decay_dot,
            time,
            n_chunks,
            heads,
            0,
            False,
            _COMBINE_COPY,
            _COMBINE_COPY,
            False,
            False,
            tiled_mma,
            s_a_layout,
            s_large_b_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
        _combine_sequence_kernel(
            u, residual_product, residual, time, padded_time, heads, _COMBINE_SUBTRACT
        ).launch(grid=linear_grid, block=(_LINEAR_THREADS, 1, 1), stream=stream)
    _m16_n16_k128_kernel(
        do,
        residual,
        do,
        residual,
        d_aqk,
        time,
        n_chunks,
        heads,
        False,
        False,
        _TRIANGLE_LOWER,
        False,
        tiled_mma,
        s_square_a_layout,
        s_b_layout,
        s_partial_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    _m16_n128_k128_kernel(
        residual,
        dstate_boundaries,
        d_k_tail,
        d_k_tail,
        residual,
        d_k_tail,
        d_k_tail,
        dstate_boundaries,
        state_decay_dot,
        time,
        n_chunks,
        heads,
        1,
        True,
        _COMBINE_COPY,
        _COMBINE_COPY,
        False,
        False,
        tiled_mma,
        s_a_layout,
        s_large_b_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    if cutlass.const_expr(not has_precomputed_d_residual):
        _m16_n128_k128_kernel(
            k_tail,
            dstate_boundaries,
            d_residual,
            d_residual,
            k_tail,
            d_residual,
            d_residual,
            dstate_boundaries,
            state_decay_dot,
            time,
            n_chunks,
            heads,
            1,
            False,
            _COMBINE_COPY,
            _COMBINE_COPY,
            False,
            False,
            tiled_mma,
            s_a_layout,
            s_large_b_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
        _m16_n128_k16_kernel(
            aqk,
            do,
            aqk,
            do,
            d_residual,
            d_residual,
            time,
            n_chunks,
            heads,
            True,
            True,
            False,
            False,
            False,
            tiled_mma,
            s_a_layout,
            s_b_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    _m16_n128_k128_kernel(
        do,
        state_boundaries,
        d_q_gamma,
        d_q_gamma,
        d_residual,
        d_y,
        d_y,
        dstate_boundaries,
        state_decay_dot,
        time,
        n_chunks,
        heads,
        0,
        True,
        _COMBINE_COPY,
        _COMBINE_NEGATE,
        True,
        fuse_state_decay_dot,
        tiled_mma,
        s_a_layout,
        s_large_b_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    _transpose_triangular_solve_kernel(
        lower, d_residual, d_y, d_z, d_e0, time, n_chunks, heads
    ).launch(grid=chunk_grid, block=(_CHAIN_THREADS, 1, 1), stream=stream)
    if cutlass.const_expr(has_forward_residual):
        # dE0 = -dZ @ S0.T, so the historical
        # dZ @ U.T + dE0 @ Y.T pair collapses to dZ @ R.T.
        _m16_n16_k128_kernel(
            d_z,
            residual,
            d_z,
            residual,
            d_lower,
            time,
            n_chunks,
            heads,
            False,
            True,
            _TRIANGLE_STRICT_LOWER,
            False,
            tiled_mma,
            s_square_a_layout,
            s_b_layout,
            s_partial_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    else:
        _m16_n16_k128_kernel(
            d_z,
            u,
            d_e0,
            y,
            d_lower,
            time,
            n_chunks,
            heads,
            False,
            True,
            _TRIANGLE_STRICT_LOWER,
            True,
            tiled_mma,
            s_square_a_layout,
            s_b_layout,
            s_partial_layout,
        ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    _m16_n128_k16_kernel(
        d_lower,
        k_bar,
        d_aqk,
        k_bar,
        d_erase,
        d_q_gamma,
        time,
        n_chunks,
        heads,
        False,
        True,
        False,
        True,
        True,
        tiled_mma,
        s_a_layout,
        s_b_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    _m16_n128_k16_kernel(
        d_lower,
        erase_bar,
        d_aqk,
        q_gamma,
        d_k_bar,
        d_k_bar,
        time,
        n_chunks,
        heads,
        True,
        False,
        True,
        False,
        False,
        tiled_mma,
        s_a_layout,
        s_b_layout,
    ).launch(grid=chunk_grid, block=(_MMA_THREADS, 1, 1), stream=stream)
    if cutlass.const_expr(not fuse_state_decay_dot):
        _state_decay_dot_kernel(
            state_boundaries,
            dstate_boundaries,
            state_decay_dot,
            d_initial_state,
            n_chunks,
            heads,
            has_precomputed_d_initial_state,
        ).launch(
            grid=(chunk_grid[0], _DIM // _STATE_DOT_KEY_WARPS, 1),
            block=(_STATE_DOT_THREADS, 1, 1),
            stream=stream,
        )
    _compact_chain_kernel(
        q,
        k,
        v,
        beta,
        w,
        state_decay_dot,
        gamma,
        decay_end,
        d_q_gamma,
        d_k_tail,
        d_z,
        d_erase,
        d_k_bar,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dw,
        time,
        n_chunks,
        heads,
        scale,
    ).launch(grid=chunk_grid, block=(_CHAIN_THREADS, 1, 1), stream=stream)


def _fake_tensor(dtype, shape):
    from cutlass.cute.runtime import make_fake_compact_tensor

    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=16,
    )


@lru_cache(maxsize=16)
def _compile_compact_wy_chunk_vjp(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    gate_dtype,
    aux_dtype,
    u_dtype,
    boundary_dtype,
    precomputed_residual_dtype,
    has_precomputed_d_residual: bool,
    has_precomputed_d_initial_state: bool,
    has_forward_residual: bool,
    fuse_state_decay_dot: bool,
):
    """Compile the staged implementation once per static tensor layout."""

    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    padded_time = n_chunks * _CHUNK_SIZE
    input_sequence_shape = (batch, time, heads, _DIM)
    workspace_sequence_shape = (batch, padded_time, heads, _DIM)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    decay_shape = (batch, n_chunks, heads, _DIM)
    square_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    f32 = cutlass.Float32

    input_sequence = _fake_tensor(input_dtype, input_sequence_shape)
    gate_sequence = _fake_tensor(gate_dtype, input_sequence_shape)
    aux_sequence = _fake_tensor(aux_dtype, input_sequence_shape)
    workspace_sequence = _fake_tensor(f32, workspace_sequence_shape)
    scratch_sequence = _fake_tensor(input_dtype, workspace_sequence_shape)
    square = _fake_tensor(f32, square_shape)
    return cute.compile(
        _launch_compact_wy_chunk_vjp,
        input_sequence,  # q
        input_sequence,  # k
        input_sequence,  # v
        gate_sequence,
        input_sequence,  # beta
        input_sequence,  # w
        _fake_tensor(boundary_dtype, boundary_shape),
        _fake_tensor(boundary_dtype, boundary_shape),
        aux_sequence,  # y
        _fake_tensor(u_dtype, input_sequence_shape),  # u
        aux_sequence,  # q_gamma
        aux_sequence,  # k_tail
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, square_shape),  # aqk
        input_sequence,  # do
        workspace_sequence,  # gamma
        scratch_sequence,  # erase_bar
        scratch_sequence,  # k_bar
        square,  # lower
        scratch_sequence,  # residual product
        (
            _fake_tensor(u_dtype, input_sequence_shape)
            if has_forward_residual
            else scratch_sequence
        ),  # residual or saved forward R
        scratch_sequence,  # d_q_gamma
        square,  # d_aqk
        scratch_sequence,  # d_k_tail
        _fake_tensor(
            precomputed_residual_dtype if has_precomputed_d_residual else input_dtype,
            input_sequence_shape if has_precomputed_d_residual else workspace_sequence_shape,
        ),  # d_residual
        scratch_sequence,  # d_y
        scratch_sequence,  # d_z
        scratch_sequence,  # d_e0
        square,  # d_lower
        scratch_sequence,  # d_erase
        scratch_sequence,  # d_k_bar
        _fake_tensor(f32, decay_shape),  # state decay dot
        input_sequence,  # dq
        input_sequence,  # dk
        input_sequence,  # dv
        gate_sequence,  # dg
        input_sequence,  # dbeta
        input_sequence,  # dw
        _fake_tensor(f32, state_shape),
        has_precomputed_d_residual,
        has_precomputed_d_initial_state,
        has_forward_residual,
        time,
        padded_time,
        n_chunks,
        heads,
        fuse_state_decay_dot,
        0.0,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def compact_wy_chunk_vjp_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state_boundaries: torch.Tensor,
    dstate_boundaries: torch.Tensor,
    aux: object,
    do: torch.Tensor,
    *,
    scale: float | None = None,
    precomputed_d_residual: torch.Tensor | None = None,
    precomputed_d_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the staged SM120 compact-WY VJP implementation.

    The standard precomputed-gradient-residual schedule uses 12 ordered
    launches and consumes forward/reverse state boundaries plus the saved
    compact-WY auxiliaries. A saved forward residual removes the ``Y @ S0``
    launch. Full-chunk compact BF16 also folds the state-decay dot into the
    shared-S0 product, leaving 10 launches; T=64, FP16, and partial-tail paths
    retain the separate dot and use 11. BF16 full chunks use BF16 checkpoints
    when the boundary scan also supplies an exact FP32
    ``precomputed_d_initial_state``. Passing its precomputed
    ``d_residual`` removes two duplicate products; this tensor is BF16 for
    compact boundaries and FP32 otherwise. Dual products, paired K16 products,
    and producer epilogues remove additional launch boundaries and operand
    reads. All large BT=16 products use warp ``m16n8k16`` MMA.

    Training scans at T=64 and T>=128 replace their saved FP32 U tensor in
    place with the forward residual. That specialization skips the backward
    ``Y @ S0`` product and evaluates ``dLower`` as ``-tril(dZ @ R.T)``,
    algebraically matching the historical paired U/Y products. The final
    parameter chain shares one reciprocal gamma across its K and decay
    expressions.

    Tensor-core operands are narrowed to the input FP16/BF16 dtype in shared
    memory. Long training checkpoints Y/Q-gamma/K-tail/A-qk in that dtype,
    while the U/R value auxiliary and decay remain FP32. The VJP is therefore
    a controlled low-precision approximation rather than a bit-exact FP32
    evaluation. Normalized-key oracle tests cover FP16/BF16,
    a partial tail, and the compact long schedule on non-default streams; the
    tests enforce per-gradient relative L2 below 1% and maximum error below
    5e-3.

    Every input must be contiguous and 16-byte aligned, matching the
    ``assumed_align=16`` TVM-FFI compile contract.
    """

    sequence_tensors = (q, k, v, g, beta, w, do)
    aux_tensors = (aux.y, aux.u, aux.q_gamma, aux.k_tail, aux.decay_end, aux.aqk)
    if any(not isinstance(tensor, torch.Tensor) for tensor in (*sequence_tensors, *aux_tensors)):
        raise TypeError("all sequence inputs and compact-WY auxiliaries must be tensors")
    if any(tensor.ndim != 4 for tensor in sequence_tensors):
        raise ValueError("all sequence inputs must be rank-4 tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape != do.shape or v.shape != q.shape:
        raise NotImplementedError("the SM120 compact-WY implementation requires K=V=128")
    batch, time, heads, key_dim = q.shape
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")
    if time <= 0:
        raise ValueError("time must be positive")
    if key_dim != _DIM:
        raise NotImplementedError("the SM120 compact-WY implementation requires K=V=128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must use float16 or bfloat16")
    if any(tensor.dtype != q.dtype for tensor in (k, v, beta, w, do)):
        raise TypeError("q, k, v, beta, w, and do must use the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must use the input dtype or float32")

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    padded_time = n_chunks * _CHUNK_SIZE
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    for name, tensor in (
        ("state_boundaries", state_boundaries),
        ("dstate_boundaries", dstate_boundaries),
    ):
        if tensor.shape != boundary_shape:
            raise ValueError(f"{name} must have shape {boundary_shape}")
    if state_boundaries.dtype != dstate_boundaries.dtype:
        raise TypeError("state and reverse-state boundaries must use one dtype")
    compact_boundaries = state_boundaries.dtype == q.dtype
    if state_boundaries.dtype != torch.float32 and not (
        compact_boundaries and q.dtype == torch.bfloat16 and time >= 128 and time % _CHUNK_SIZE == 0
    ):
        raise TypeError("boundaries must use FP32 or the BF16 full-chunk compact format")
    sequence_shape = (batch, time, heads, _DIM)
    if any(tensor.shape != sequence_shape for tensor in aux_tensors[:4]):
        raise ValueError("y, u, q_gamma, and k_tail must match the sequence shape")
    if aux.decay_end.shape != (batch, n_chunks, heads, _DIM):
        raise ValueError("decay_end has an invalid shape")
    square_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    if aux.aqk.shape != square_shape:
        raise ValueError("aqk has an invalid shape")
    if any(tensor.dtype != aux.y.dtype for tensor in (aux.q_gamma, aux.k_tail, aux.aqk)):
        raise TypeError("y, q_gamma, k_tail, and aqk must use one dtype")
    if aux.y.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("compact-WY auxiliaries must use float16, bfloat16, or float32")
    if aux.u.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("u must use float16, bfloat16, or float32")
    has_forward_residual = bool(getattr(aux, "u_is_residual", False))
    if aux.decay_end.dtype != torch.float32:
        raise TypeError("decay_end must use float32")
    if precomputed_d_residual is not None:
        if not isinstance(precomputed_d_residual, torch.Tensor):
            raise TypeError("precomputed_d_residual must be a tensor")
        if precomputed_d_residual.shape != sequence_shape:
            raise ValueError("precomputed_d_residual must match the sequence shape")
        if precomputed_d_residual.dtype != torch.float32 and not (
            compact_boundaries and precomputed_d_residual.dtype == q.dtype
        ):
            raise TypeError("precomputed_d_residual must use FP32, or BF16 with compact boundaries")
    if precomputed_d_initial_state is not None:
        if not isinstance(precomputed_d_initial_state, torch.Tensor):
            raise TypeError("precomputed_d_initial_state must be a tensor")
        if precomputed_d_initial_state.shape != (batch, heads, _DIM, _DIM):
            raise ValueError("precomputed_d_initial_state has an invalid shape")
        if precomputed_d_initial_state.dtype != torch.float32:
            raise TypeError("precomputed_d_initial_state must use float32")
    if compact_boundaries and precomputed_d_initial_state is None:
        raise ValueError("compact boundaries require precomputed_d_initial_state")

    tensors = (*sequence_tensors, state_boundaries, dstate_boundaries, *aux_tensors)
    if precomputed_d_residual is not None:
        tensors = (*tensors, precomputed_d_residual)
    if precomputed_d_initial_state is not None:
        tensors = (*tensors, precomputed_d_initial_state)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.data_ptr() % 16 != 0 for tensor in tensors):
        raise ValueError("all inputs must be 16-byte aligned")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("compact_wy_chunk_vjp_cute requires an SM120 CUDA device")
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("CUTE_DSL_ARCH must be sm_120 for compact_wy_chunk_vjp_cute")
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")
    fuse_state_decay_dot = _use_fused_state_decay_dot(
        batch,
        n_chunks,
        heads,
        compact_boundaries,
    )

    sequence_workspace_shape = (batch, padded_time, heads, _DIM)
    scratch_count = 5 if precomputed_d_residual is not None else 6
    gamma = torch.empty(sequence_workspace_shape, device=q.device, dtype=torch.float32)
    erase_bar = torch.empty(sequence_workspace_shape, device=q.device, dtype=q.dtype)
    k_bar = torch.empty(sequence_workspace_shape, device=q.device, dtype=q.dtype)
    scratch_workspaces = [
        torch.empty(sequence_workspace_shape, device=q.device, dtype=q.dtype)
        for _ in range(scratch_count)
    ]
    square_workspaces = [
        torch.empty(square_shape, device=q.device, dtype=torch.float32) for _ in range(3)
    ]

    state_shape = (batch, heads, _DIM, _DIM)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dw = torch.empty_like(w)
    d_initial_state = (
        torch.empty(state_shape, device=q.device, dtype=torch.float32)
        if precomputed_d_initial_state is None
        else precomputed_d_initial_state.detach()
    )

    input_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
    gate_dtype = cutlass.Float32 if g.dtype == torch.float32 else input_dtype
    aux_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[aux.y.dtype]
    u_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[aux.u.dtype]
    boundary_dtype = input_dtype if compact_boundaries else cutlass.Float32
    precomputed_residual_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[precomputed_d_residual.dtype if precomputed_d_residual is not None else q.dtype]
    device_index = q.device.index
    if device_index is None:
        raise RuntimeError("q must have a concrete CUDA device index")

    # TVM-FFI rejects autograd tensors.  These are storage-only views; this
    # implementation explicitly produces all VJPs and never asks PyTorch to
    # trace through a kernel launch.
    q_arg, k_arg, v_arg, g_arg, beta_arg, w_arg, do_arg = (
        tensor.detach() for tensor in sequence_tensors
    )
    state_arg = state_boundaries.detach()
    dstate_arg = dstate_boundaries.detach()
    y_arg, u_arg, q_gamma_arg, k_tail_arg, decay_arg, aqk_arg = (
        tensor.detach() for tensor in aux_tensors
    )
    residual_product = scratch_workspaces[0]
    residual_workspace = scratch_workspaces[1]
    residual = aux.u.detach() if has_forward_residual else residual_workspace
    d_q_gamma = scratch_workspaces[2]
    # Alias only across disjoint launch intervals; no kernel observes the
    # same storage through two tensor arguments.  Gamma remains FP32; the
    # tensor-core sequence operands and five precomputed-path scratch tensors
    # use the input dtype, keeping T512/H16 at 22 MiB.
    d_k_tail = residual_product
    d_z = residual_workspace
    if precomputed_d_residual is None:
        d_residual_workspace = scratch_workspaces[3]
        d_y = scratch_workspaces[4]
        d_e0 = scratch_workspaces[5]
    else:
        d_residual_workspace = None
        d_y = scratch_workspaces[3]
        d_e0 = scratch_workspaces[4]
    d_erase = d_e0
    d_k_bar = d_y
    d_residual = (
        d_residual_workspace
        if d_residual_workspace is not None
        else precomputed_d_residual.detach()
    )
    lower, d_aqk, d_lower = square_workspaces
    state_decay_shape = (batch, n_chunks, heads, _DIM)
    if fuse_state_decay_dot:
        # The dual S0 product produces these dots before ``erase_bar`` reaches
        # its final consumer, so the long fused schedule needs a small,
        # independent output (1 MiB at T2048/H16).
        state_decay_dot = torch.empty(
            state_decay_shape,
            device=q.device,
            dtype=torch.float32,
        )
    else:
        state_decay_dot = (
            erase_bar.view(torch.float32)
            .view(-1)[: math.prod(state_decay_shape)]
            .view(state_decay_shape)
        )

    with torch.cuda.device(q.device):
        compiled = _compile_compact_wy_chunk_vjp(
            device_index,
            batch,
            time,
            heads,
            input_dtype,
            gate_dtype,
            aux_dtype,
            u_dtype,
            boundary_dtype,
            precomputed_residual_dtype,
            precomputed_d_residual is not None,
            precomputed_d_initial_state is not None,
            has_forward_residual,
            fuse_state_decay_dot,
        )
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        compiled(
            q_arg,
            k_arg,
            v_arg,
            g_arg,
            beta_arg,
            w_arg,
            state_arg,
            dstate_arg,
            y_arg,
            u_arg,
            q_gamma_arg,
            k_tail_arg,
            decay_arg,
            aqk_arg,
            do_arg,
            gamma,
            erase_bar,
            k_bar,
            lower,
            residual_product,
            residual,
            d_q_gamma,
            d_aqk,
            d_k_tail,
            d_residual,
            d_y,
            d_z,
            d_e0,
            d_lower,
            d_erase,
            d_k_bar,
            state_decay_dot,
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
    return dq, dk, dv, dg, dbeta, dw, d_initial_state


__all__ = [
    "CompactWYVJPProof",
    "compact_wy_chunk_vjp_cute",
    "compact_wy_chunk_vjp_reference",
]

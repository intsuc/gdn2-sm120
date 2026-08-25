"""Chunk-parallel long-sequence GDN2 backward for SM120.

The production backward can expose sequence-length parallelism by splitting
the work into three stages:

1. save the state at every BT=16 forward boundary;
2. scan only the boundary state gradients in reverse;
3. launch every chunk independently from its local state and ``dS_end`` boundaries.

The dimension-generic PyTorch functions provide an executable proof of the
decomposition.  The CuTe kernels implement both the boundary scan and the
independent parameter VJPs for the production ``K == V == 128`` shape, using
the compact-WY tensors and state checkpoints already produced by
``chunk_forward(return_aux=True)``.  The reference decomposition keeps FP32
checkpoints.  Production BF16 full chunks at T>=128 compact both forward and
reverse checkpoints to BF16 while preserving the exact initial-state gradient
in a separate FP32 tensor; other paths retain FP32 checkpoints.

For one chunk, the forward map is

``S1 = (D - K_tail^T Y) S0 + K_tail^T U``

and

``O = (Q_gamma - A_qk Y) S0 + A_qk U``.

Consequently its boundary VJP is

``dS0 = D dS1 + Q_gamma^T dO - Y^T (K_tail dS1 + A_qk^T dO)``.

Only this small boundary scan remains ordered across chunks.  Once all
boundary gradients are materialized, the expensive parameter VJPs can be
computed by independent chunk CTAs.
"""

from __future__ import annotations

import math
import operator
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
_BOUNDARY_WARPS = 4
_BOUNDARY_THREADS = 32 * _BOUNDARY_WARPS
_BOUNDARY_VALUE_TILE = _BOUNDARY_WARPS
_BOUNDARY_MMA_WARPS = _DIM // 16
_BOUNDARY_MMA_THREADS = 32 * _BOUNDARY_MMA_WARPS
_BOUNDARY_MMA_VALUE_TILE = 8
_BOUNDARY_MMA_WIDE_VALUE_TILE = 16
# V8 exposes sixteen independent value CTAs per (batch, head), while V16
# halves duplicated Y/Q-gamma/K-tail reads.  V16 needs 24 batch-heads to put
# at least 192 CTAs on the 188-SM target.  Smaller midrange grids benefit from
# the duplicated-read saving only while the ordered scan remains short.
_BOUNDARY_MMA_MIDRANGE_MIN_BATCH_HEADS = 16
_BOUNDARY_MMA_MIDRANGE_MAX_TIME = 512
_BOUNDARY_MMA_WIDE_MIN_BATCH_HEADS = 24
_AQK_PRECOMPUTE_WARPS = 8
_AQK_PRECOMPUTE_THREADS = 32 * _AQK_PRECOMPUTE_WARPS
_AQK_PRECOMPUTE_VALUES = _AQK_PRECOMPUTE_WARPS * _BOUNDARY_MMA_VALUE_TILE
_LOCAL_THREADS = 32
_LOCAL_VALUE_TILE = 8
_LOCAL_VALUE_TILES = _DIM // _LOCAL_VALUE_TILE
_LOCAL_KEYS_PER_LANE = 32
_LOCAL_SMEM_BYTES = 4 * _DIM * 4
_FULL_WARPS = 8
_FULL_THREADS = 32 * _FULL_WARPS
_FULL_VALUES_PER_WARP = _DIM // _FULL_WARPS
_FULL_KEY_FIELDS = 5
_FULL_SMEM_BYTES = (_DIM * _DIM + 4 * _FULL_WARPS * _DIM + _FULL_KEY_FIELDS * _DIM) * 4
# One approximately 83-KiB CTA occupies an SM.  Use the full-value path only
# once chunk/head parallelism fills most of the 188-SM target; otherwise
# retain value tiling.
_FULL_MIN_CTAS = 384


def _select_boundary_mma_value_tile(batch: int, time: int, heads: int) -> int:
    """Choose V16 for filled grids and short underfilled midrange scans."""

    batch_heads = batch * heads
    if batch_heads >= _BOUNDARY_MMA_WIDE_MIN_BATCH_HEADS:
        return _BOUNDARY_MMA_WIDE_VALUE_TILE
    if (
        batch_heads >= _BOUNDARY_MMA_MIDRANGE_MIN_BATCH_HEADS
        and time <= _BOUNDARY_MMA_MIDRANGE_MAX_TIME
    ):
        return _BOUNDARY_MMA_WIDE_VALUE_TILE
    return _BOUNDARY_MMA_VALUE_TILE


@dataclass(frozen=True)
class WYBoundaryAux:
    """Compact-WY tensors required by the boundary-gradient scan.

    The field names and layouts intentionally match the corresponding fields
    of :class:`gdn2_sm120.chunk.ChunkForwardAux`.  ``u`` is not needed because
    the derivative with respect to a boundary state is independent of the
    chunk's affine write term.
    """

    y: torch.Tensor
    q_gamma: torch.Tensor
    k_tail: torch.Tensor
    decay_end: torch.Tensor
    aqk: torch.Tensor


@dataclass(frozen=True)
class ParallelBackwardProof:
    """Intermediate tensors from the executable backward decomposition."""

    state_boundaries: torch.Tensor
    dstate_boundaries: torch.Tensor


def _validate_chunk_size(chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")


def recurrent_forward_checkpoints_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the recurrence and save exact FP32 states at chunk boundaries.

    Returns ``(output, final_state, checkpoints)``.  Checkpoints use layout
    ``[B, ceil(T / BT) + 1, H, K, V]`` and include both the initial and final
    boundary.  This is the state layout required by a chunk-parallel CuTe VJP.
    """

    _validate_chunk_size(chunk_size)
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q and v must be rank-4 [B, T, H, D] tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must agree with q's [B, T, H] dimensions")

    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is None:
        state = torch.zeros(state_shape, device=q.device, dtype=torch.float32)
    else:
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must have shape {state_shape}")
        state = initial_state.float()

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    outputs: list[torch.Tensor] = []
    checkpoints = [state.clone()]
    for token in range(time):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        v_t = v[:, token].float()
        decay_t = g[:, token].float().exp()
        erase_t = beta[:, token].float() * k_t
        z_t = w[:, token].float() * v_t

        x_t = decay_t.unsqueeze(-1) * state
        update_t = z_t - torch.einsum("bhk,bhkv->bhv", erase_t, x_t)
        state = x_t + k_t.unsqueeze(-1) * update_t.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", output_scale * q_t, state))
        if (token + 1) % chunk_size == 0 or token + 1 == time:
            checkpoints.append(state.clone())

    output = (
        torch.stack(outputs, dim=1).to(q.dtype)
        if outputs
        else torch.empty((batch, 0, heads, value_dim), device=q.device, dtype=q.dtype)
    )
    return output, state, torch.stack(checkpoints, dim=1)


def build_wy_boundary_aux_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> WYBoundaryAux:
    """Build the subset of compact-WY auxiliaries used by the boundary VJP."""

    _validate_chunk_size(chunk_size)
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape or q.ndim != 4:
        raise ValueError("q, k, g, and beta must be same-shaped rank-4 tensors")
    batch, time, heads, key_dim = q.shape
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)

    y_chunks: list[torch.Tensor] = []
    q_gamma_chunks: list[torch.Tensor] = []
    k_tail_chunks: list[torch.Tensor] = []
    decay_chunks: list[torch.Tensor] = []
    aqk_chunks: list[torch.Tensor] = []
    for start in range(0, time, chunk_size):
        stop = min(start + chunk_size, time)
        length = stop - start
        q_c = q[:, start:stop].transpose(1, 2).float()
        k_c = k[:, start:stop].transpose(1, 2).float()
        g_c = g[:, start:stop].transpose(1, 2).float()
        beta_c = beta[:, start:stop].transpose(1, 2).float()

        cumulative_g = g_c.cumsum(dim=-2)
        gamma = cumulative_g.exp()
        k_bar = (-cumulative_g).exp() * k_c
        erase_bar = gamma * beta_c * k_c
        q_gamma = gamma * q_c * output_scale
        lower = torch.tril(erase_bar @ k_bar.transpose(-1, -2), diagonal=-1)
        system = lower + torch.eye(length, device=q.device, dtype=torch.float32)
        y = torch.linalg.solve_triangular(system, erase_bar, upper=False, unitriangular=True)
        aqk = torch.tril(q_gamma @ k_bar.transpose(-1, -2))
        decay_end = gamma[..., -1, :]
        k_tail = k_bar * decay_end.unsqueeze(-2)

        if length < chunk_size:
            token_pad = (0, 0, 0, chunk_size - length)
            square_pad = (0, chunk_size - length, 0, chunk_size - length)
            y = torch.nn.functional.pad(y, token_pad)
            q_gamma = torch.nn.functional.pad(q_gamma, token_pad)
            k_tail = torch.nn.functional.pad(k_tail, token_pad)
            aqk = torch.nn.functional.pad(aqk, square_pad)
        y_chunks.append(y)
        q_gamma_chunks.append(q_gamma)
        k_tail_chunks.append(k_tail)
        decay_chunks.append(decay_end)
        aqk_chunks.append(aqk)

    if not y_chunks:
        empty_sequence = torch.empty(
            (batch, 0, heads, key_dim), device=q.device, dtype=torch.float32
        )
        empty_decay = torch.empty((batch, 0, heads, key_dim), device=q.device, dtype=torch.float32)
        empty_aqk = torch.empty(
            (batch, 0, heads, chunk_size, chunk_size), device=q.device, dtype=torch.float32
        )
        return WYBoundaryAux(
            empty_sequence, empty_sequence.clone(), empty_sequence.clone(), empty_decay, empty_aqk
        )

    def flatten_token_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
        # [chunk][B,H,BT,K] -> [B,C*BT,H,K], then trim padding to T.
        return torch.cat([item.transpose(1, 2) for item in chunks], dim=1)[:, :time].contiguous()

    return WYBoundaryAux(
        y=flatten_token_chunks(y_chunks),
        q_gamma=flatten_token_chunks(q_gamma_chunks),
        k_tail=flatten_token_chunks(k_tail_chunks),
        decay_end=torch.stack(decay_chunks, dim=1),
        aqk=torch.stack(aqk_chunks, dim=1),
    )


def boundary_dstate_token_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    *,
    scale: float | None = None,
    chunk_size: int = _CHUNK_SIZE,
) -> torch.Tensor:
    """Reference boundary VJPs obtained from the token-level recurrence."""

    _validate_chunk_size(chunk_size)
    batch, time, heads, key_dim = q.shape
    value_dim = do.shape[-1]
    expected_state_shape = (batch, heads, key_dim, value_dim)
    if d_final_state.shape != expected_state_shape:
        raise ValueError(f"d_final_state must have shape {expected_state_shape}")
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    n_chunks = math.ceil(time / chunk_size)
    boundaries = torch.empty(
        (batch, n_chunks + 1, heads, key_dim, value_dim),
        device=q.device,
        dtype=torch.float32,
    )
    dstate = d_final_state.float()
    boundaries[:, n_chunks] = dstate
    for token in reversed(range(time)):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        decay_t = g[:, token].float().exp()
        erase_t = beta[:, token].float() * k_t
        do_t = do[:, token].float()
        total = dstate + output_scale * q_t.unsqueeze(-1) * do_t.unsqueeze(-2)
        du_t = torch.einsum("bhk,bhkv->bhv", k_t, total)
        dstate = decay_t.unsqueeze(-1) * (total - erase_t.unsqueeze(-1) * du_t.unsqueeze(-2))
        if token % chunk_size == 0:
            boundaries[:, token // chunk_size] = dstate
    return boundaries


def boundary_dstate_wy_reference(
    aux: WYBoundaryAux | object,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    *,
    chunk_size: int = _CHUNK_SIZE,
    return_d_residual: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the boundary VJP and optionally return its token residuals."""

    _validate_chunk_size(chunk_size)
    y = aux.y
    q_gamma = aux.q_gamma
    k_tail = aux.k_tail
    decay_end = aux.decay_end
    aqk = aux.aqk
    batch, time, heads, key_dim = y.shape
    value_dim = do.shape[-1]
    n_chunks = math.ceil(time / chunk_size)
    expected_decay = (batch, n_chunks, heads, key_dim)
    expected_aqk = (batch, n_chunks, heads, chunk_size, chunk_size)
    if decay_end.shape != expected_decay or aqk.shape != expected_aqk:
        raise ValueError("compact-WY auxiliary layouts do not match time/chunk_size")

    boundaries = torch.empty(
        (batch, n_chunks + 1, heads, key_dim, value_dim),
        device=do.device,
        dtype=torch.float32,
    )
    d_residual = None
    if return_d_residual:
        d_residual = torch.empty(
            (batch, time, heads, value_dim),
            device=do.device,
            dtype=torch.float32,
        )
    dstate = d_final_state.float()
    boundaries[:, n_chunks] = dstate
    for chunk in reversed(range(n_chunks)):
        start = chunk * chunk_size
        stop = min(start + chunk_size, time)
        y_c = y[:, start:stop].transpose(1, 2).float()
        q_gamma_c = q_gamma[:, start:stop].transpose(1, 2).float()
        k_tail_c = k_tail[:, start:stop].transpose(1, 2).float()
        do_c = do[:, start:stop].transpose(1, 2).float()
        aqk_c = aqk[:, chunk, :, : stop - start, : stop - start].float()

        residual_gradient = k_tail_c @ dstate + aqk_c.transpose(-1, -2) @ do_c
        if d_residual is not None:
            d_residual[:, start:stop] = residual_gradient.transpose(1, 2)
        dstate = (
            decay_end[:, chunk].float().unsqueeze(-1) * dstate
            + q_gamma_c.transpose(-1, -2) @ do_c
            - y_c.transpose(-1, -2) @ residual_gradient
        )
        boundaries[:, chunk] = dstate
    if d_residual is None:
        return boundaries
    return boundaries, d_residual


def parallel_chunk_backward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    chunk_size: int = _CHUNK_SIZE,
    state_boundaries: torch.Tensor | None = None,
    dstate_boundaries: torch.Tensor | None = None,
    verify_boundaries: bool = True,
) -> tuple[tuple[torch.Tensor, ...], ParallelBackwardProof]:
    """Executable proof that parameter VJPs are independent across chunks.

    Every loop iteration uses only its local sequence slice, exact forward
    start state, upstream output gradient, and exact end-boundary gradient.
    Therefore those iterations may become independent CuTe CTAs even though
    this clarity-first Python implementation executes them in a normal loop.
    """

    _validate_chunk_size(chunk_size)
    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    if state_boundaries is None:
        _, _, state_boundaries = recurrent_forward_checkpoints_reference(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            scale=output_scale,
            chunk_size=chunk_size,
        )
    if dstate_boundaries is None:
        aux = build_wy_boundary_aux_reference(
            q, k, g, beta, scale=output_scale, chunk_size=chunk_size
        )
        dstate_boundaries = boundary_dstate_wy_reference(
            aux, do, d_final_state, chunk_size=chunk_size
        )

    fp32_gradients = [
        torch.empty_like(tensor, dtype=torch.float32) for tensor in (q, k, v, g, beta, w)
    ]
    n_chunks = math.ceil(time / chunk_size)
    d_initial_state = torch.empty(
        (batch, heads, key_dim, value_dim), device=q.device, dtype=torch.float32
    )

    for chunk in range(n_chunks):
        start = chunk * chunk_size
        stop = min(start + chunk_size, time)
        state = state_boundaries[:, chunk].float()
        states = [state]
        x_values: list[torch.Tensor] = []
        update_values: list[torch.Tensor] = []
        for token in range(start, stop):
            k_t = k[:, token].float()
            decay_t = g[:, token].float().exp()
            erase_t = beta[:, token].float() * k_t
            z_t = w[:, token].float() * v[:, token].float()
            x_t = decay_t.unsqueeze(-1) * state
            update_t = z_t - torch.einsum("bhk,bhkv->bhv", erase_t, x_t)
            state = x_t + k_t.unsqueeze(-1) * update_t.unsqueeze(-2)
            x_values.append(x_t)
            update_values.append(update_t)
            states.append(state)

        dstate = dstate_boundaries[:, chunk + 1].float()
        for token in reversed(range(start, stop)):
            local = token - start
            q_t = q[:, token].float()
            k_t = k[:, token].float()
            v_t = v[:, token].float()
            decay_t = g[:, token].float().exp()
            beta_t = beta[:, token].float()
            erase_t = beta_t * k_t
            w_t = w[:, token].float()
            do_t = do[:, token].float()
            state_t = states[local + 1]
            previous_state = states[local]
            x_t = x_values[local]
            update_t = update_values[local]

            fp32_gradients[0][:, token] = output_scale * torch.einsum(
                "bhkv,bhv->bhk", state_t, do_t
            )
            total = dstate + output_scale * q_t.unsqueeze(-1) * do_t.unsqueeze(-2)
            du_t = torch.einsum("bhk,bhkv->bhv", k_t, total)
            dk_direct = torch.einsum("bhkv,bhv->bhk", total, update_t)
            dr_t = -torch.einsum("bhkv,bhv->bhk", x_t, du_t)
            dx_t = total - erase_t.unsqueeze(-1) * du_t.unsqueeze(-2)

            fp32_gradients[1][:, token] = dk_direct + beta_t * dr_t
            fp32_gradients[2][:, token] = du_t * w_t
            fp32_gradients[3][:, token] = decay_t * torch.einsum(
                "bhkv,bhkv->bhk", dx_t, previous_state
            )
            fp32_gradients[4][:, token] = k_t * dr_t
            fp32_gradients[5][:, token] = du_t * v_t
            dstate = decay_t.unsqueeze(-1) * dx_t

        if verify_boundaries:
            torch.testing.assert_close(
                dstate,
                dstate_boundaries[:, chunk],
                atol=5e-5,
                rtol=5e-5,
            )
        if chunk == 0:
            d_initial_state.copy_(dstate)

    input_dtypes = (q.dtype, k.dtype, v.dtype, g.dtype, beta.dtype, w.dtype)
    gradients = tuple(
        gradient.to(dtype) for gradient, dtype in zip(fp32_gradients, input_dtypes, strict=True)
    ) + (d_initial_state,)
    return gradients, ParallelBackwardProof(state_boundaries, dstate_boundaries)


@cute.jit
def _warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    return value + cute.arch.shuffle_sync_bfly(value, 1)


@cute.jit
def _value_tile_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce the eight adjacent lanes that own one key and value tile."""

    return cute.arch.warp_reduction(value, operator.add, threads_in_group=_LOCAL_VALUE_TILE)


@cute.jit
def _key_owner_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce the four strided lanes that own one value column."""

    value += cute.arch.shuffle_sync_bfly(value, 8)
    return value + cute.arch.shuffle_sync_bfly(value, 16)


@cute.jit
def _half_warp_sum(value: cutlass.Float32) -> cutlass.Float32:
    return cute.arch.warp_reduction(value, operator.add, threads_in_group=16)


@cute.jit
def _pair_sum(value: cutlass.Float32) -> cutlass.Float32:
    return value + cute.arch.shuffle_sync_bfly(value, offset=16)


@cute.kernel
def _precompute_aqk_do_kernel(
    aqk: cute.Tensor,
    do: cute.Tensor,
    aqk_do: cute.Tensor,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_b_layout: cute.Layout,
):
    """Compute every ``A_qk.T @ dO`` chunk concurrently.

    The boundary scan is ordered over chunks, so performing this state-
    independent product inside that scan serializes otherwise independent
    tensor-core work.  One eight-warp CTA reuses a single A_qk tile across 64
    value columns and writes an FP32 token/value tile for the subsequent scan.
    """

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_ctas = _DIM // _AQK_PRECOMPUTE_VALUES
    value_cta = block % value_ctas
    chunk_head = block // value_ctas
    head = chunk_head % heads
    batch_chunk = chunk_head // heads
    chunk = batch_chunk % n_chunks
    batch = batch_chunk // n_chunks
    value_tile = value_cta * _AQK_PRECOMPUTE_WARPS + warp
    start = chunk * _CHUNK_SIZE
    operand_type = do.element_type

    allocator = cutlass.utils.SmemAllocator()
    s_a = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_do_all = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    s_do = s_do_all[warp, None, None]

    thr_mma = tiled_mma.get_slice(lane)
    t_cs_a = thr_mma.partition_A(s_a)
    t_cr_a = thr_mma.make_fragment_A(t_cs_a)
    t_cs_do = thr_mma.partition_B(s_do)
    t_cr_do = thr_mma.make_fragment_B(t_cs_do)
    copy_atom = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_b = cute.make_tiled_copy_B(copy_atom, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_b = tiled_copy_b.get_slice(lane)
    t_ss_a = thr_copy_a.partition_S(s_a)
    t_sr_a = thr_copy_a.retile(t_cr_a)
    t_ss_do = thr_copy_b.partition_S(s_do)
    t_sr_do = thr_copy_b.retile(t_cr_do)

    # Store A_qk transposed in the MMA A tile.  All eight warps reuse it.
    if tidx < _CHUNK_SIZE * _CHUNK_SIZE:
        row = tidx // _CHUNK_SIZE
        output_row = tidx - row * _CHUNK_SIZE
        s_a[row, output_row] = aqk[batch, chunk, head, output_row, row].to(operand_type)
    for linear in cutlass.range(lane, _BOUNDARY_MMA_VALUE_TILE * _CHUNK_SIZE, 32):
        value_local = linear // _CHUNK_SIZE
        output_row = linear - value_local * _CHUNK_SIZE
        value_idx = value_tile * _BOUNDARY_MMA_VALUE_TILE + value_local
        s_do[value_local, output_row] = do[batch, start + output_row, head, value_idx].to(
            operand_type
        )
    cute.arch.sync_threads()

    cute.copy(tiled_copy_a, t_ss_a[None, None, 0], t_sr_a[None, None, 0])
    cute.copy(tiled_copy_b, t_ss_do[None, None, 0], t_sr_do[None, None, 0])
    g_output = cute.local_tile(
        aqk_do[batch, None, head, None],
        (_CHUNK_SIZE, _BOUNDARY_MMA_VALUE_TILE),
        (chunk, value_tile),
    )
    t_cg_output = thr_mma.partition_C(g_output)
    t_cr_output = thr_mma.make_fragment_C(t_cg_output)
    t_cr_output.fill(0.0)
    cute.gemm(
        tiled_mma,
        t_cr_output,
        t_cr_a[None, None, 0],
        t_cr_do[None, None, 0],
        t_cr_output,
    )
    cute.autovec_copy(t_cr_output, t_cg_output)


@cute.kernel
def _wy_boundary_dstate_kernel(
    y: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    boundaries: cute.Tensor,
    d_residual: cute.Tensor,
    d_scan_start_state: cute.Tensor,
    time: cutlass.Constexpr,
    last_chunk: cutlass.Constexpr,
    scan_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_d_final_state: cutlass.Constexpr,
    store_d_residual: cutlass.Constexpr,
    store_scan_start_state: cutlass.Constexpr,
):
    """Scan compact-WY state VJPs, four value columns per CTA."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_tiles = _DIM // _BOUNDARY_VALUE_TILE
    value_tile = block % value_tiles
    batch_head = block // value_tiles
    head = batch_head % heads
    batch = batch_head // heads
    value_idx = value_tile * _BOUNDARY_VALUE_TILE + warp

    dstate = cute.make_rmem_tensor((_DIM // 32,), cutlass.Float32)
    for key_group in cutlass.range_constexpr(_DIM // 32):
        key_idx = lane + 32 * key_group
        value = cutlass.Float32(0.0)
        if cutlass.const_expr(has_d_final_state):
            value = d_final_state[batch, head, key_idx, value_idx].to(cutlass.Float32)
        dstate[key_group] = value
        boundaries[batch, last_chunk, head, key_idx, value_idx] = value

    for reverse_chunk in range(scan_chunks):
        chunk = last_chunk - 1 - reverse_chunk
        start = chunk * _CHUNK_SIZE
        length = cutlass.min(_CHUNK_SIZE, time - start)
        residual_gradient = cute.make_rmem_tensor((_CHUNK_SIZE,), cutlass.Float32)

        for row in cutlass.range_constexpr(_CHUNK_SIZE):
            k_dot = cutlass.Float32(0.0)
            if row < length:
                token = start + row
                for key_group in cutlass.range_constexpr(_DIM // 32):
                    key_idx = lane + 32 * key_group
                    k_dot += (
                        k_tail[batch, token, head, key_idx].to(cutlass.Float32) * dstate[key_group]
                    )
            k_dot = _warp_sum(k_dot)

            output_term = cutlass.Float32(0.0)
            if row < length:
                for output_row in cutlass.range_constexpr(_CHUNK_SIZE):
                    if output_row < length:
                        output_token = start + output_row
                        output_term += aqk[batch, chunk, head, output_row, row].to(
                            cutlass.Float32
                        ) * do[batch, output_token, head, value_idx].to(cutlass.Float32)
            residual_gradient[row] = k_dot + output_term
            if cutlass.const_expr(store_d_residual):  # noqa: SIM102
                if lane == 0 and row < length:
                    d_residual[batch, start + row, head, value_idx] = residual_gradient[row]

        for key_group in cutlass.range_constexpr(_DIM // 32):
            key_idx = lane + 32 * key_group
            next_dstate = (
                decay_end[batch, chunk, head, key_idx].to(cutlass.Float32) * dstate[key_group]
            )
            for row in cutlass.range_constexpr(_CHUNK_SIZE):
                if row < length:
                    token = start + row
                    output_gradient = do[batch, token, head, value_idx].to(cutlass.Float32)
                    next_dstate += (
                        q_gamma[batch, token, head, key_idx].to(cutlass.Float32) * output_gradient
                    )
                    next_dstate -= (
                        y[batch, token, head, key_idx].to(cutlass.Float32) * residual_gradient[row]
                    )
            dstate[key_group] = next_dstate
            boundaries[batch, chunk, head, key_idx, value_idx] = next_dstate

    if cutlass.const_expr(store_scan_start_state):
        for key_group in cutlass.range_constexpr(_DIM // 32):
            key_idx = lane + 32 * key_group
            d_scan_start_state[batch, head, key_idx, value_idx] = dstate[key_group]


@cute.kernel
def _wy_boundary_dstate_mma_kernel(
    y: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk_do: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    boundaries: cute.Tensor,
    d_residual: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_d_final_state: cutlass.Constexpr,
    store_d_residual: cutlass.Constexpr,
    store_d_initial_state: cutlass.Constexpr,
    value_tile_size: cutlass.Constexpr,
    tiled_mma: cute.TiledMma,
    s_a_layout: cute.Layout,
    s_state_layout: cute.Layout,
    s_partial_layout: cute.Layout,
    s_b_layout: cute.Layout,
):
    """K-split tensor-core compact-WY boundary scan for full chunks."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_tiles = _DIM // value_tile_size
    value_tile = block % value_tiles
    batch_head = block // value_tiles
    head = batch_head % heads
    batch = batch_head // heads
    key_start = warp * 16
    operand_type = do.element_type

    allocator = cutlass.utils.SmemAllocator()
    s_k_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_q_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_y_all = allocator.allocate_tensor(operand_type, s_a_layout, byte_alignment=1024)
    s_state_all = allocator.allocate_tensor(operand_type, s_state_layout, byte_alignment=1024)
    s_partial_all = allocator.allocate_tensor(
        cutlass.Float32, s_partial_layout, byte_alignment=1024
    )
    s_do = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    s_residual = allocator.allocate_tensor(operand_type, s_b_layout, byte_alignment=1024)
    s_k = s_k_all[warp, None, None]
    s_q = s_q_all[warp, None, None]
    s_y = s_y_all[warp, None, None]
    s_state = s_state_all[warp, None, None]
    s_partial = s_partial_all[warp, None, None]
    s_state_as_b = cute.make_tensor(
        s_state.iterator,
        cute.make_layout((value_tile_size, 16), stride=(1, value_tile_size)),
    )
    s_q_as_a = cute.make_tensor(
        s_q.iterator,
        cute.make_layout((16, _CHUNK_SIZE), stride=(1, 16)),
    )
    s_y_as_a = cute.make_tensor(
        s_y.iterator,
        cute.make_layout((16, _CHUNK_SIZE), stride=(1, 16)),
    )

    thr_mma = tiled_mma.get_slice(lane)
    t_cs_k = thr_mma.partition_A(s_k)
    t_cs_q = thr_mma.partition_A(s_q_as_a)
    t_cs_y = thr_mma.partition_A(s_y_as_a)
    t_cs_state = thr_mma.partition_C(s_state)
    t_cs_partial = thr_mma.partition_C(s_partial)
    t_cr_k = thr_mma.make_fragment_A(t_cs_k)
    t_cr_q = thr_mma.make_fragment_A(t_cs_q)
    t_cr_y = thr_mma.make_fragment_A(t_cs_y)
    t_cr_b = thr_mma.make_fragment_B(thr_mma.partition_B(s_state_as_b))
    t_cr_partial = thr_mma.make_fragment_C(t_cs_partial)

    copy_atom_a = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), operand_type)
    copy_atom_a_transpose = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type
    )
    copy_atom_b_state = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(True, 4), operand_type
    )
    copy_atom_b_dense = copy_atom_a
    tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
    tiled_copy_a_transpose = cute.make_tiled_copy_A(copy_atom_a_transpose, tiled_mma)
    tiled_copy_b_state = cute.make_tiled_copy_B(copy_atom_b_state, tiled_mma)
    tiled_copy_b_dense = cute.make_tiled_copy_B(copy_atom_b_dense, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane)
    thr_copy_a_transpose = tiled_copy_a_transpose.get_slice(lane)
    thr_copy_b_state = tiled_copy_b_state.get_slice(lane)
    thr_copy_b_dense = tiled_copy_b_dense.get_slice(lane)
    t_ss_k = thr_copy_a.partition_S(s_k)
    t_ss_q = thr_copy_a_transpose.partition_S(s_q_as_a)
    t_ss_y = thr_copy_a_transpose.partition_S(s_y_as_a)
    t_sr_k = thr_copy_a.retile(t_cr_k)
    t_sr_q = thr_copy_a_transpose.retile(t_cr_q)
    t_sr_y = thr_copy_a_transpose.retile(t_cr_y)
    t_ss_state = thr_copy_b_state.partition_S(s_state_as_b)
    t_sr_b_state = thr_copy_b_state.retile(t_cr_b)
    t_ss_do = thr_copy_b_dense.partition_S(s_do)
    t_ss_residual = thr_copy_b_dense.partition_S(s_residual)
    t_sr_b_dense = thr_copy_b_dense.retile(t_cr_b)

    copy_atom_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(), operand_type, num_bits_per_copy=128
    )
    tiled_copy_g2s = cute.make_tiled_copy_tv(
        copy_atom_g2s,
        cute.make_layout((16, 2), stride=(2, 1)),
        cute.make_layout((1, 8), stride=(8, 1)),
    )
    thr_copy_g2s = tiled_copy_g2s.get_slice(lane)
    t_ds_k = thr_copy_g2s.partition_D(s_k)
    t_ds_q = thr_copy_g2s.partition_D(s_q)
    t_ds_y = thr_copy_g2s.partition_D(s_y)

    g_final = cute.local_tile(
        d_final_state[batch, head, None, None],
        (16, value_tile_size),
        (warp, value_tile),
    )
    t_cg_final = thr_mma.partition_C(g_final)
    t_cr_state = thr_mma.make_fragment_C(t_cg_final)
    if cutlass.const_expr(has_d_final_state):
        cute.autovec_copy(t_cg_final, t_cr_state)
    else:
        t_cr_state.fill(0.0)
    g_boundary = cute.local_tile(
        boundaries[batch, n_chunks, head, None, None],
        (16, value_tile_size),
        (warp, value_tile),
    )
    t_cg_boundary = thr_mma.partition_C(g_boundary)
    if cutlass.const_expr(boundaries.element_type == cutlass.Float32):
        cute.autovec_copy(t_cr_state, t_cg_boundary)
    else:
        t_cr_boundary = cute.make_fragment_like(t_cr_state, boundaries.element_type)
        t_cr_boundary[None] = t_cr_state.load().to(boundaries.element_type)
        cute.autovec_copy(t_cr_boundary, t_cg_boundary)
    state_identity = cute.make_identity_tensor((16, value_tile_size))
    t_cp_state = thr_mma.partition_C(state_identity)

    if cutlass.const_expr(y.element_type == operand_type):
        first_chunk = n_chunks - 1
        g_k_first = cute.local_tile(
            k_tail[batch, None, head, None], (_CHUNK_SIZE, 16), (first_chunk, warp)
        )
        g_q_first = cute.local_tile(
            q_gamma[batch, None, head, None], (_CHUNK_SIZE, 16), (first_chunk, warp)
        )
        g_y_first = cute.local_tile(
            y[batch, None, head, None], (_CHUNK_SIZE, 16), (first_chunk, warp)
        )
        cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_k_first), t_ds_k)
        cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_q_first), t_ds_q)
        cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_y_first), t_ds_y)
        cute.arch.cp_async_commit_group()

    for reverse_chunk in cutlass.range(n_chunks, unroll=1):
        chunk = n_chunks - 1 - reverse_chunk
        start = chunk * _CHUNK_SIZE

        if cutlass.const_expr(y.element_type == operand_type):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_warp()

        # Convert the persistent FP32 state to the low-precision MMA operand.
        t_cr_state_operand = cute.make_fragment_like(t_cr_state, operand_type)
        t_cr_state_operand[None] = t_cr_state.load().to(operand_type)
        cute.autovec_copy(t_cr_state_operand, t_cs_state)
        cute.arch.sync_warp()
        cute.copy(
            tiled_copy_b_state,
            t_ss_state[None, None, 0],
            t_sr_b_state[None, None, 0],
        )

        if cutlass.const_expr(y.element_type == operand_type):
            # All three current tiles reach registers before their shared
            # buffers are reused for the next reverse chunk.
            cute.copy(tiled_copy_a, t_ss_k[None, None, 0], t_sr_k[None, None, 0])
            cute.copy(
                tiled_copy_a_transpose,
                t_ss_q[None, None, 0],
                t_sr_q[None, None, 0],
            )
            cute.copy(
                tiled_copy_a_transpose,
                t_ss_y[None, None, 0],
                t_sr_y[None, None, 0],
            )

            if reverse_chunk + 1 < n_chunks:
                next_chunk = chunk - 1
                g_k_next = cute.local_tile(
                    k_tail[batch, None, head, None], (_CHUNK_SIZE, 16), (next_chunk, warp)
                )
                g_q_next = cute.local_tile(
                    q_gamma[batch, None, head, None],
                    (_CHUNK_SIZE, 16),
                    (next_chunk, warp),
                )
                g_y_next = cute.local_tile(
                    y[batch, None, head, None], (_CHUNK_SIZE, 16), (next_chunk, warp)
                )
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_k_next), t_ds_k)
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_q_next), t_ds_q)
                cute.copy(tiled_copy_g2s, thr_copy_g2s.partition_S(g_y_next), t_ds_y)
                cute.arch.cp_async_commit_group()
        else:
            # FP32 public auxiliaries require an explicit type conversion.
            for linear in cutlass.range(lane, _CHUNK_SIZE * 16, 32):
                row = linear // 16
                key_local = linear - row * 16
                s_k[row, key_local] = k_tail[batch, start + row, head, key_start + key_local].to(
                    operand_type
                )
                s_q[row, key_local] = q_gamma[batch, start + row, head, key_start + key_local].to(
                    operand_type
                )
                s_y[row, key_local] = y[batch, start + row, head, key_start + key_local].to(
                    operand_type
                )
            cute.arch.sync_warp()
            cute.copy(tiled_copy_a, t_ss_k[None, None, 0], t_sr_k[None, None, 0])
            cute.copy(
                tiled_copy_a_transpose,
                t_ss_q[None, None, 0],
                t_sr_q[None, None, 0],
            )
            cute.copy(
                tiled_copy_a_transpose,
                t_ss_y[None, None, 0],
                t_sr_y[None, None, 0],
            )

        # Every C-fragment lane reuses one of the same 16 decay values.
        # Fetch one coalesced copy per half warp before the K-tail MMA, then
        # broadcast from lanes 0..15 while that MMA result is still in flight.
        lane_decay = decay_end[
            batch,
            chunk,
            head,
            key_start + (lane & 15),
        ].to(cutlass.Float32)

        # K_tail @ dS: every warp contributes one K=16 slice, then the CTA
        # deterministically reduces the eight FP32 fragments.
        t_cr_partial.fill(0.0)
        cute.gemm(
            tiled_mma,
            t_cr_partial,
            t_cr_k[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_partial,
        )
        for state_element in cutlass.range_constexpr(cute.size(t_cr_state.shape)):
            key_local = t_cp_state[state_element][0]
            t_cr_state[state_element] *= cute.arch.shuffle_sync(lane_decay, key_local)
        cute.autovec_copy(t_cr_partial, t_cs_partial)
        cute.arch.sync_threads()

        if tidx < _CHUNK_SIZE * value_tile_size:
            row = tidx // value_tile_size
            value_local = tidx - row * value_tile_size
            value_idx = value_tile * value_tile_size + value_local
            total = cutlass.Float32(0.0)
            for source_warp in cutlass.range_constexpr(_BOUNDARY_MMA_WARPS):
                total += s_partial_all[source_warp, row, value_local]
            total += aqk_do[batch, start + row, head, value_idx]
            # Negating the B operand lets the already-staged +Y tile perform
            # the subtraction without another shared-memory rewrite.
            s_residual[value_local, row] = (-total).to(operand_type)
            s_do[value_local, row] = do[batch, start + row, head, value_idx].to(operand_type)
            if cutlass.const_expr(store_d_residual):
                d_residual[batch, start + row, head, value_idx] = total.to(d_residual.element_type)
        cute.arch.sync_threads()

        # Q_gamma^T*dO - Y^T*residual completes the update after the decay
        # multiply above.  Each warp owns exactly 16 K rows.
        cute.copy(
            tiled_copy_b_dense,
            t_ss_do[None, None, 0],
            t_sr_b_dense[None, None, 0],
        )
        cute.gemm(
            tiled_mma,
            t_cr_state,
            t_cr_q[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_state,
        )

        cute.copy(
            tiled_copy_b_dense,
            t_ss_residual[None, None, 0],
            t_sr_b_dense[None, None, 0],
        )
        cute.gemm(
            tiled_mma,
            t_cr_state,
            t_cr_y[None, None, 0],
            t_cr_b[None, None, 0],
            t_cr_state,
        )
        g_boundary = cute.local_tile(
            boundaries[batch, chunk, head, None, None],
            (16, value_tile_size),
            (warp, value_tile),
        )
        t_cg_boundary = thr_mma.partition_C(g_boundary)
        if cutlass.const_expr(boundaries.element_type == cutlass.Float32):
            cute.autovec_copy(t_cr_state, t_cg_boundary)
        else:
            t_cr_boundary = cute.make_fragment_like(t_cr_state, boundaries.element_type)
            t_cr_boundary[None] = t_cr_state.load().to(boundaries.element_type)
            cute.autovec_copy(t_cr_boundary, t_cg_boundary)

    if cutlass.const_expr(store_d_initial_state):
        g_initial = cute.local_tile(
            d_initial_state[batch, head, None, None],
            (16, value_tile_size),
            (warp, value_tile),
        )
        cute.autovec_copy(t_cr_state, thr_mma.partition_C(g_initial))


@cute.kernel
def _parallel_chunk_vjp_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    do: cute.Tensor,
    partial: cute.Tensor,
    dv: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
):
    """Compute one chunk/value-tile VJP independently in each CTA."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    value_tile = block % _LOCAL_VALUE_TILES
    chunk_head = block // _LOCAL_VALUE_TILES
    head = chunk_head % heads
    batch_chunk = chunk_head // heads
    chunk = batch_chunk % n_chunks
    batch = batch_chunk // n_chunks
    start = chunk * _CHUNK_SIZE
    length = cutlass.min(_CHUNK_SIZE, time - start)

    value_local = tidx & 7
    key_group = tidx >> 3
    value_idx = value_tile * _LOCAL_VALUE_TILE + value_local

    state = cute.make_rmem_tensor((_LOCAL_KEYS_PER_LANE,), cutlass.Float32)
    dstate = cute.make_rmem_tensor((_LOCAL_KEYS_PER_LANE,), cutlass.Float32)
    for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
        key_idx = key_group + 4 * key_iter
        state[key_iter] = state_boundaries[batch, chunk + 1, head, key_idx, value_idx].to(
            cutlass.Float32
        )
        dstate[key_iter] = dstate_boundaries[batch, chunk + 1, head, key_idx, value_idx].to(
            cutlass.Float32
        )

    allocator = cutlass.utils.SmemAllocator()
    key_data = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )

    for reverse_local in cutlass.range_constexpr(_CHUNK_SIZE):
        if reverse_local < length:
            token = start + length - 1 - reverse_local
            do_value = do[batch, token, head, value_idx].to(cutlass.Float32)

            # All 32 lanes cooperatively cache k, r=beta*k, exp(g), and its
            # reciprocal.  The subsequent strided K reductions remain in one
            # warp, so no CTA-wide synchronization is needed.
            for cache_iter in cutlass.range_constexpr(4):
                cache_key = tidx + 32 * cache_iter
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
            cute.arch.sync_warp()

            # Output VJP is applied while the exact/reconstructed S_t remains
            # live.  K-shaped gradients are emitted as value-tile partials.
            for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
                key_idx = key_group + 4 * key_iter
                state_value = state[key_iter]
                q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                dstate[key_iter] += scale * q_value * do_value
                dq_part = _value_tile_sum(scale * state_value * do_value)
                if value_local == 0:
                    partial[0, batch, token, head, value_tile, key_idx] = dq_part

            # Invert only this short chunk.  Exact state boundaries reset the
            # reconstruction every <=16 tokens, avoiding long-product drift.
            z_value = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
                batch, token, head, value_idx
            ].to(cutlass.Float32)
            denominator_part = cutlass.Float32(0.0)
            y_dot_part = cutlass.Float32(0.0)
            for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
                key_idx = key_group + 4 * key_iter
                key_value = key_data[0, key_idx]
                erase_value = key_data[1, key_idx]
                y_value = state[key_iter] - key_value * z_value
                denominator_part += erase_value * key_value
                y_dot_part += erase_value * y_value
            denominator = cutlass.Float32(1.0) - _key_owner_sum(denominator_part)
            erased_value = _key_owner_sum(y_dot_part) / denominator
            update_value = z_value - erased_value

            du_part = cutlass.Float32(0.0)
            for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
                key_idx = key_group + 4 * key_iter
                du_part += key_data[0, key_idx] * dstate[key_iter]
            du_value = _key_owner_sum(du_part)

            if key_group == 0:
                v_value = v[batch, token, head, value_idx].to(cutlass.Float32)
                w_value = w[batch, token, head, value_idx].to(cutlass.Float32)
                dv[batch, token, head, value_idx] = (du_value * w_value).to(dv.element_type)
                dw[batch, token, head, value_idx] = (du_value * v_value).to(dw.element_type)

            for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
                key_idx = key_group + 4 * key_iter
                key_value = key_data[0, key_idx]
                erase_value = key_data[1, key_idx]
                decay = key_data[2, key_idx]
                current_state = state[key_iter]
                y_value = current_state - key_value * z_value
                x_value = y_value + key_value * erased_value
                previous_state = x_value * key_data[3, key_idx]
                state[key_iter] = previous_state
                ds_value = dstate[key_iter]

                dk_direct_part = _value_tile_sum(ds_value * update_value)
                dr_part = _value_tile_sum(-x_value * du_value)
                dx_value = ds_value - erase_value * du_value
                da_part = _value_tile_sum(dx_value * previous_state)
                if value_local == 0:
                    partial[1, batch, token, head, value_tile, key_idx] = dk_direct_part
                    partial[2, batch, token, head, value_tile, key_idx] = dr_part
                    partial[3, batch, token, head, value_tile, key_idx] = da_part
                dstate[key_iter] = decay * dx_value

    if chunk == 0:
        for key_iter in cutlass.range_constexpr(_LOCAL_KEYS_PER_LANE):
            key_idx = key_group + 4 * key_iter
            d_initial_state[batch, head, key_idx, value_idx] = dstate[key_iter]


@cute.kernel
def _parallel_chunk_vjp_full_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
):
    """Process all 128 values in one independent chunk/head CTA.

    This long-sequence schedule trades about 83 KiB of shared memory per CTA for
    eliminating the global FP32 partial tensor and its reduction kernel.
    """

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    head = block % heads
    batch_chunk = block // heads
    chunk = batch_chunk % n_chunks
    batch = batch_chunk // n_chunks
    start = chunk * _CHUNK_SIZE
    length = cutlass.min(_CHUNK_SIZE, time - start)
    warp = tidx >> 5
    lane = tidx & 31
    value_local = lane & 15
    key_parity = lane >> 4
    value_idx = warp * _FULL_VALUES_PER_WARP + value_local

    allocator = cutlass.utils.SmemAllocator()
    state = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_DIM, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )
    partial = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((4, _FULL_WARPS, _DIM), stride=(_FULL_WARPS * _DIM, _DIM, 1)),
        byte_alignment=16,
    )
    key_data = allocator.allocate_tensor(
        cutlass.Float32,
        cute.make_layout((_FULL_KEY_FIELDS, _DIM), stride=(_DIM, 1)),
        byte_alignment=16,
    )
    dstate = cute.make_rmem_tensor((64,), cutlass.Float32)
    for key_iter in cutlass.range(64, unroll_full=True):
        key_idx = key_parity + 2 * key_iter
        state[key_idx, value_idx] = state_boundaries[batch, chunk + 1, head, key_idx, value_idx].to(
            cutlass.Float32
        )
        dstate[key_iter] = dstate_boundaries[batch, chunk + 1, head, key_idx, value_idx].to(
            cutlass.Float32
        )
    cute.arch.sync_threads()

    for reverse_local in cutlass.range_constexpr(_CHUNK_SIZE):
        if reverse_local < length:
            token = start + length - 1 - reverse_local
            do_value = do[batch, token, head, value_idx].to(cutlass.Float32)

            if tidx < _DIM:
                key_value = k[batch, token, head, tidx].to(cutlass.Float32)
                beta_value = beta[batch, token, head, tidx].to(cutlass.Float32)
                decay = cute.math.exp(
                    g[batch, token, head, tidx].to(cutlass.Float32), fastmath=False
                )
                key_data[0, tidx] = key_value
                key_data[1, tidx] = beta_value * key_value
                key_data[2, tidx] = decay
                key_data[3, tidx] = cutlass.Float32(1.0) / decay
                key_data[4, tidx] = beta_value
            cute.arch.sync_threads()

            for key_iter in cutlass.range(64, unroll_full=True):
                key_idx = key_parity + 2 * key_iter
                q_value = q[batch, token, head, key_idx].to(cutlass.Float32)
                state_value = state[key_idx, value_idx]
                dstate[key_iter] += scale * q_value * do_value
                dq_part = _half_warp_sum(scale * state_value * do_value)
                if value_local == 0:
                    partial[0, warp, key_idx] = dq_part

            z_value = w[batch, token, head, value_idx].to(cutlass.Float32) * v[
                batch, token, head, value_idx
            ].to(cutlass.Float32)
            denominator_part = cutlass.Float32(0.0)
            y_dot_part = cutlass.Float32(0.0)
            for key_iter in cutlass.range(64, unroll_full=True):
                key_idx = key_parity + 2 * key_iter
                key_value = key_data[0, key_idx]
                erase_value = key_data[1, key_idx]
                y_value = state[key_idx, value_idx] - key_value * z_value
                denominator_part += erase_value * key_value
                y_dot_part += erase_value * y_value
            denominator = cutlass.Float32(1.0) - _pair_sum(denominator_part)
            erased_value = _pair_sum(y_dot_part) / denominator
            update_value = z_value - erased_value

            for key_iter in cutlass.range(64, unroll_full=True):
                key_idx = key_parity + 2 * key_iter
                key_value = key_data[0, key_idx]
                y_value = state[key_idx, value_idx] - key_value * z_value
                state[key_idx, value_idx] = (y_value + key_value * erased_value) * key_data[
                    3, key_idx
                ]

            du_part = cutlass.Float32(0.0)
            for key_iter in cutlass.range(64, unroll_full=True):
                key_idx = key_parity + 2 * key_iter
                du_part += key_data[0, key_idx] * dstate[key_iter]
            du_value = _pair_sum(du_part)
            if key_parity == 0:
                v_value = v[batch, token, head, value_idx].to(cutlass.Float32)
                w_value = w[batch, token, head, value_idx].to(cutlass.Float32)
                dv[batch, token, head, value_idx] = (du_value * w_value).to(dv.element_type)
                dw[batch, token, head, value_idx] = (du_value * v_value).to(dw.element_type)

            for key_iter in cutlass.range(64, unroll_full=True):
                key_idx = key_parity + 2 * key_iter
                key_value = key_data[0, key_idx]
                beta_value = key_data[4, key_idx]
                decay = key_data[2, key_idx]
                previous_state = state[key_idx, value_idx]
                x_value = decay * previous_state
                erase_value = beta_value * key_value
                ds_value = dstate[key_iter]
                dk_direct_part = _half_warp_sum(ds_value * update_value)
                dr_part = _half_warp_sum(-x_value * du_value)
                dx_value = ds_value - erase_value * du_value
                da_part = _half_warp_sum(dx_value * previous_state)
                if value_local == 0:
                    partial[1, warp, key_idx] = dk_direct_part
                    partial[2, warp, key_idx] = dr_part
                    partial[3, warp, key_idx] = da_part
                dstate[key_iter] = decay * dx_value

            cute.arch.sync_threads()
            if tidx < _DIM:
                key_idx = tidx
                dq_total = cutlass.Float32(0.0)
                dk_direct_total = cutlass.Float32(0.0)
                dr_total = cutlass.Float32(0.0)
                da_total = cutlass.Float32(0.0)
                for source_warp in cutlass.range_constexpr(_FULL_WARPS):
                    dq_total += partial[0, source_warp, key_idx]
                    dk_direct_total += partial[1, source_warp, key_idx]
                    dr_total += partial[2, source_warp, key_idx]
                    da_total += partial[3, source_warp, key_idx]
                key_value = key_data[0, key_idx]
                beta_value = key_data[4, key_idx]
                decay = key_data[2, key_idx]
                dq[batch, token, head, key_idx] = dq_total.to(dq.element_type)
                dk[batch, token, head, key_idx] = (dk_direct_total + beta_value * dr_total).to(
                    dk.element_type
                )
                dg[batch, token, head, key_idx] = (decay * da_total).to(dg.element_type)
                dbeta[batch, token, head, key_idx] = (key_value * dr_total).to(dbeta.element_type)
            cute.arch.sync_threads()

    if chunk == 0:
        for key_iter in cutlass.range(64, unroll_full=True):
            key_idx = key_parity + 2 * key_iter
            d_initial_state[batch, head, key_idx, value_idx] = dstate[key_iter]


@cute.kernel
def _reduce_parallel_chunk_partials_kernel(
    k: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    partial: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
):
    """Reduce K-gradient contributions over the value tiles."""

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
    for value_tile in cutlass.range_constexpr(_LOCAL_VALUE_TILES):
        dq_total += partial[0, batch, token, head, value_tile, key_idx]
        dk_direct_total += partial[1, batch, token, head, value_tile, key_idx]
        dr_total += partial[2, batch, token, head, value_tile, key_idx]
        da_total += partial[3, batch, token, head, value_tile, key_idx]

    key_value = k[batch, token, head, key_idx].to(cutlass.Float32)
    beta_value = beta[batch, token, head, key_idx].to(cutlass.Float32)
    decay = cute.math.exp(g[batch, token, head, key_idx].to(cutlass.Float32), fastmath=False)
    dq[batch, token, head, key_idx] = dq_total.to(dq.element_type)
    dk[batch, token, head, key_idx] = (dk_direct_total + beta_value * dr_total).to(dk.element_type)
    dg[batch, token, head, key_idx] = (decay * da_total).to(dg.element_type)
    dbeta[batch, token, head, key_idx] = (key_value * dr_total).to(dbeta.element_type)


@cute.jit
def _launch_wy_boundary_dstate(
    y: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    boundaries: cute.Tensor,
    d_residual: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_d_final_state: cutlass.Constexpr,
    store_d_residual: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _wy_boundary_dstate_kernel(
        y,
        q_gamma,
        k_tail,
        decay_end,
        aqk,
        do,
        d_final_state,
        boundaries,
        d_residual,
        d_final_state,
        time,
        n_chunks,
        n_chunks,
        heads,
        has_d_final_state,
        store_d_residual,
        False,
    ).launch(
        grid=(y.shape[0] * heads * (_DIM // _BOUNDARY_VALUE_TILE), 1, 1),
        block=(_BOUNDARY_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_wy_boundary_dstate_mma(
    y: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    boundaries: cute.Tensor,
    d_residual: cute.Tensor,
    aqk_do: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_d_final_state: cutlass.Constexpr,
    store_d_residual: cutlass.Constexpr,
    store_d_initial_state: cutlass.Constexpr,
    value_tile_size: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    full_chunks = time // _CHUNK_SIZE
    mma_final_state = d_final_state
    mma_has_final_state = has_d_final_state
    mma_chunks = n_chunks
    if cutlass.const_expr(time % _CHUNK_SIZE != 0):
        # Reverse dependencies require the partial tail to run first.  Its
        # resulting dS at the full-prefix boundary is staged in the otherwise
        # optional FP32 d-initial buffer and consumed by the MMA prefix scan.
        _wy_boundary_dstate_kernel(
            y,
            q_gamma,
            k_tail,
            decay_end,
            aqk,
            do,
            d_final_state,
            boundaries,
            d_residual,
            d_initial_state,
            time,
            n_chunks,
            1,
            heads,
            has_d_final_state,
            store_d_residual,
            True,
        ).launch(
            grid=(y.shape[0] * heads * (_DIM // _BOUNDARY_VALUE_TILE), 1, 1),
            block=(_BOUNDARY_THREADS, 1, 1),
            stream=stream,
        )
        mma_final_state = d_initial_state
        mma_has_final_state = True
        mma_chunks = full_chunks

    precompute_a_layout = cute.make_layout((_CHUNK_SIZE, _CHUNK_SIZE), stride=(_CHUNK_SIZE, 1))
    precompute_b_layout = cute.make_layout(
        (_AQK_PRECOMPUTE_WARPS, _BOUNDARY_MMA_VALUE_TILE, _CHUNK_SIZE),
        stride=(_BOUNDARY_MMA_VALUE_TILE * _CHUNK_SIZE, _CHUNK_SIZE, 1),
    )
    s_a_layout = cute.make_layout(
        (_BOUNDARY_MMA_WARPS, _CHUNK_SIZE, 16),
        stride=(_CHUNK_SIZE * 16, 16, 1),
    )
    s_state_layout = cute.make_layout(
        (_BOUNDARY_MMA_WARPS, 16, value_tile_size),
        stride=(16 * value_tile_size, value_tile_size, 1),
    )
    s_b_layout = cute.make_layout((value_tile_size, _CHUNK_SIZE), stride=(_CHUNK_SIZE, 1))
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaF16BF16Op(do.element_type, cutlass.Float32, (16, 8, 16)),
        (1, 1, 1),
        permutation_mnk=(16, _BOUNDARY_MMA_VALUE_TILE, 16),
    )
    _precompute_aqk_do_kernel(
        aqk,
        do,
        aqk_do,
        mma_chunks,
        heads,
        tiled_mma,
        precompute_a_layout,
        precompute_b_layout,
    ).launch(
        grid=(
            y.shape[0] * mma_chunks * heads * (_DIM // _AQK_PRECOMPUTE_VALUES),
            1,
            1,
        ),
        block=(_AQK_PRECOMPUTE_THREADS, 1, 1),
        stream=stream,
    )
    _wy_boundary_dstate_mma_kernel(
        y,
        q_gamma,
        k_tail,
        decay_end,
        aqk_do,
        do,
        mma_final_state,
        boundaries,
        d_residual,
        d_initial_state,
        time,
        mma_chunks,
        heads,
        mma_has_final_state,
        store_d_residual,
        store_d_initial_state,
        value_tile_size,
        tiled_mma,
        s_a_layout,
        s_state_layout,
        s_state_layout,
        s_b_layout,
    ).launch(
        grid=(y.shape[0] * heads * (_DIM // value_tile_size), 1, 1),
        block=(_BOUNDARY_MMA_THREADS, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_wy_boundary_dstate_mma_compact(
    y: cute.Tensor,
    q_gamma: cute.Tensor,
    k_tail: cute.Tensor,
    decay_end: cute.Tensor,
    aqk: cute.Tensor,
    do: cute.Tensor,
    d_final_state: cute.Tensor,
    boundaries: cute.Tensor,
    d_residual: cute.Tensor,
    aqk_do: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_d_final_state: cutlass.Constexpr,
    store_d_residual: cutlass.Constexpr,
    value_tile_size: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    """Distinct trace entry for low-precision boundary checkpoint storage."""

    _launch_wy_boundary_dstate_mma(
        y,
        q_gamma,
        k_tail,
        decay_end,
        aqk,
        do,
        d_final_state,
        boundaries,
        d_residual,
        aqk_do,
        d_initial_state,
        time,
        n_chunks,
        heads,
        has_d_final_state,
        store_d_residual,
        True,
        value_tile_size,
        stream,
    )


@cute.jit
def _launch_parallel_chunk_vjp(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    do: cute.Tensor,
    partial: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _parallel_chunk_vjp_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        state_boundaries,
        dstate_boundaries,
        do,
        partial,
        dv,
        dw,
        d_initial_state,
        time,
        n_chunks,
        heads,
        scale,
    ).launch(
        grid=(q.shape[0] * n_chunks * heads * _LOCAL_VALUE_TILES, 1, 1),
        block=(_LOCAL_THREADS, 1, 1),
        smem=_LOCAL_SMEM_BYTES,
        stream=stream,
    )
    _reduce_parallel_chunk_partials_kernel(
        k,
        g,
        beta,
        partial,
        dq,
        dk,
        dg,
        dbeta,
    ).launch(
        grid=(q.shape[0] * time * heads, 1, 1),
        block=(_DIM, 1, 1),
        stream=stream,
    )


@cute.jit
def _launch_parallel_chunk_vjp_full(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    state_boundaries: cute.Tensor,
    dstate_boundaries: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dg: cute.Tensor,
    dbeta: cute.Tensor,
    dw: cute.Tensor,
    d_initial_state: cute.Tensor,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _parallel_chunk_vjp_full_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        state_boundaries,
        dstate_boundaries,
        do,
        dq,
        dk,
        dv,
        dg,
        dbeta,
        dw,
        d_initial_state,
        time,
        n_chunks,
        heads,
        scale,
    ).launch(
        grid=(q.shape[0] * n_chunks * heads, 1, 1),
        block=(_FULL_THREADS, 1, 1),
        smem=_FULL_SMEM_BYTES,
        stream=stream,
    )


def _fake_tensor(dtype, shape):
    from cutlass.cute.runtime import make_fake_compact_tensor

    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=16,
    )


@lru_cache(maxsize=32)
def _compile_wy_boundary_dstate(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    aux_dtype,
    has_d_final_state: bool,
    store_d_residual: bool,
):
    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    decay_shape = (batch, n_chunks, heads, _DIM)
    aqk_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    residual_shape = (batch, time, heads, _DIM)
    f32 = cutlass.Float32
    return cute.compile(
        _launch_wy_boundary_dstate,
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, aqk_shape),
        _fake_tensor(input_dtype, sequence_shape),
        _fake_tensor(f32, state_shape),
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(f32, residual_shape if store_d_residual else boundary_shape),
        time,
        n_chunks,
        heads,
        has_d_final_state,
        store_d_residual,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _compile_wy_boundary_dstate_mma(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    aux_dtype,
    has_d_final_state: bool,
    store_d_residual: bool,
    value_tile_size: int,
):
    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    decay_shape = (batch, n_chunks, heads, _DIM)
    aqk_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    residual_shape = (batch, time, heads, _DIM)
    f32 = cutlass.Float32
    return cute.compile(
        _launch_wy_boundary_dstate_mma,
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, aqk_shape),
        _fake_tensor(input_dtype, sequence_shape),
        _fake_tensor(f32, state_shape),
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(f32, residual_shape if store_d_residual else boundary_shape),
        _fake_tensor(f32, residual_shape),
        _fake_tensor(f32, state_shape),
        time,
        n_chunks,
        heads,
        has_d_final_state,
        store_d_residual,
        False,
        value_tile_size,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _compile_wy_boundary_dstate_mma_compact(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    aux_dtype,
    has_d_final_state: bool,
    store_d_residual: bool,
    value_tile_size: int,
):
    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    decay_shape = (batch, n_chunks, heads, _DIM)
    aqk_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    residual_shape = (batch, time, heads, _DIM)
    f32 = cutlass.Float32
    return cute.compile(
        _launch_wy_boundary_dstate_mma_compact,
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(aux_dtype, sequence_shape),
        _fake_tensor(f32, decay_shape),
        _fake_tensor(aux_dtype, aqk_shape),
        _fake_tensor(input_dtype, sequence_shape),
        _fake_tensor(f32, state_shape),
        _fake_tensor(input_dtype, boundary_shape),
        _fake_tensor(
            input_dtype if store_d_residual else f32,
            residual_shape if store_d_residual else state_shape,
        ),
        _fake_tensor(f32, residual_shape),
        _fake_tensor(f32, state_shape),
        time,
        n_chunks,
        heads,
        has_d_final_state,
        store_d_residual,
        value_tile_size,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _compile_parallel_chunk_vjp(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    gate_dtype,
):
    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    partial_shape = (4, batch, time, heads, _LOCAL_VALUE_TILES, _DIM)
    f32 = cutlass.Float32
    return cute.compile(
        _launch_parallel_chunk_vjp,
        _fake_tensor(input_dtype, sequence_shape),  # q
        _fake_tensor(input_dtype, sequence_shape),  # k
        _fake_tensor(input_dtype, sequence_shape),  # v
        _fake_tensor(gate_dtype, sequence_shape),  # g
        _fake_tensor(input_dtype, sequence_shape),  # beta
        _fake_tensor(input_dtype, sequence_shape),  # w
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(input_dtype, sequence_shape),  # do
        _fake_tensor(f32, partial_shape),
        _fake_tensor(input_dtype, sequence_shape),  # dq
        _fake_tensor(input_dtype, sequence_shape),  # dk
        _fake_tensor(input_dtype, sequence_shape),  # dv
        _fake_tensor(gate_dtype, sequence_shape),  # dg
        _fake_tensor(input_dtype, sequence_shape),  # dbeta
        _fake_tensor(input_dtype, sequence_shape),  # dw
        _fake_tensor(f32, state_shape),
        time,
        n_chunks,
        heads,
        0.0,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=32)
def _compile_parallel_chunk_vjp_full(
    device_index: int,
    batch: int,
    time: int,
    heads: int,
    input_dtype,
    gate_dtype,
):
    from cutlass.cute.runtime import make_fake_stream

    del device_index
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    f32 = cutlass.Float32
    return cute.compile(
        _launch_parallel_chunk_vjp_full,
        _fake_tensor(input_dtype, sequence_shape),  # q
        _fake_tensor(input_dtype, sequence_shape),  # k
        _fake_tensor(input_dtype, sequence_shape),  # v
        _fake_tensor(gate_dtype, sequence_shape),  # g
        _fake_tensor(input_dtype, sequence_shape),  # beta
        _fake_tensor(input_dtype, sequence_shape),  # w
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(f32, boundary_shape),
        _fake_tensor(input_dtype, sequence_shape),  # do
        _fake_tensor(input_dtype, sequence_shape),  # dq
        _fake_tensor(input_dtype, sequence_shape),  # dk
        _fake_tensor(input_dtype, sequence_shape),  # dv
        _fake_tensor(gate_dtype, sequence_shape),  # dg
        _fake_tensor(input_dtype, sequence_shape),  # dbeta
        _fake_tensor(input_dtype, sequence_shape),  # dw
        _fake_tensor(f32, state_shape),
        time,
        n_chunks,
        heads,
        0.0,
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def wy_boundary_dstate(
    aux: WYBoundaryAux | object,
    do: torch.Tensor,
    d_final_state: torch.Tensor | None,
    *,
    return_d_residual: bool = False,
    compact_boundaries: bool = False,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """Run the SM120 compact-WY boundary-gradient scan.

    ``aux`` may be either :class:`WYBoundaryAux` or ``ChunkForwardAux``.
    ``d_final_state=None`` specializes the scan for a zero terminal VJP
    without allocating, clearing, or reading a state-sized zero tensor.
    By default the boundary tensor is FP32 with shape
    ``[B, C + 1, H, 128, 128]``.  ``compact_boundaries=True`` is available for
    the full-chunk MMA path: it stores boundaries in ``do.dtype`` and appends
    the exact FP32 initial-state gradient to the return value.  With both
    options enabled the result is ``(boundaries, d_residual, d_initial)``;
    ``d_residual`` contains the already-computed
    ``K_tail @ dS1 + A_qk.T @ dO`` values as ``[B, T, H, 128]``.
    It remains FP32 for the standard path and uses BF16 together with compact
    BF16 boundaries, matching its downstream tensor-core operand contract.
    """

    tensors = (aux.y, aux.q_gamma, aux.k_tail, aux.decay_end, aux.aqk, do)
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all compact-WY auxiliaries and gradients must be tensors")
    if d_final_state is not None and not isinstance(d_final_state, torch.Tensor):
        raise TypeError("d_final_state must be a tensor or None")
    batch, time, heads, key_dim = aux.y.shape
    n_chunks = math.ceil(time / _CHUNK_SIZE)
    sequence_shape = (batch, time, heads, _DIM)
    state_shape = (batch, heads, _DIM, _DIM)
    if batch <= 0 or time <= 0 or heads <= 0:
        raise ValueError("batch, time, and heads must be positive")
    if key_dim != _DIM or do.shape != sequence_shape:
        raise NotImplementedError("the SM120 boundary scan requires K=V=128")
    if aux.q_gamma.shape != sequence_shape or aux.k_tail.shape != sequence_shape:
        raise ValueError("y, q_gamma, and k_tail must have identical layouts")
    if aux.decay_end.shape != (batch, n_chunks, heads, _DIM):
        raise ValueError("decay_end has an invalid layout")
    if aux.aqk.shape != (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE):
        raise ValueError("aqk has an invalid layout")
    if d_final_state is not None and (
        d_final_state.shape != state_shape or d_final_state.dtype != torch.float32
    ):
        raise ValueError("d_final_state must be float32 [B, H, 128, 128]")
    if do.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("do must use float16 or bfloat16")
    aux_tensors = (aux.y, aux.q_gamma, aux.k_tail, aux.aqk)
    if any(tensor.dtype != aux.y.dtype for tensor in aux_tensors):
        raise TypeError("y, q_gamma, k_tail, and aqk must use the same dtype")
    if aux.y.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("compact-WY auxiliaries must use float16, bfloat16, or float32")
    if aux.decay_end.dtype != torch.float32:
        raise TypeError("decay_end must use float32")
    checked_tensors = tensors + ((d_final_state,) if d_final_state is not None else ())
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in checked_tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if any(tensor.device != do.device for tensor in checked_tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.data_ptr() % 16 != 0 for tensor in checked_tensors):
        raise ValueError("all inputs must be 16-byte aligned")
    if torch.cuda.get_device_capability(do.device) != (12, 0):
        raise RuntimeError("wy_boundary_dstate requires an SM120 CUDA device")
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("CUTE_DSL_ARCH must be sm_120 for wy_boundary_dstate")
    if not isinstance(return_d_residual, bool):
        raise TypeError("return_d_residual must be bool")
    if not isinstance(compact_boundaries, bool):
        raise TypeError("compact_boundaries must be bool")
    if compact_boundaries and time < 128:
        raise ValueError("compact boundaries require T >= 128")

    use_mma = time == 64 or time >= 128
    if compact_boundaries and (not use_mma or time % _CHUNK_SIZE != 0):
        raise ValueError("compact boundaries require the full-chunk MMA path")
    if compact_boundaries and do.dtype != torch.bfloat16:
        raise ValueError("compact boundaries are supported only for BF16")

    boundaries = torch.empty(
        (batch, n_chunks + 1, heads, _DIM, _DIM),
        device=do.device,
        dtype=do.dtype if compact_boundaries else torch.float32,
    )
    d_residual = None
    if return_d_residual:
        d_residual = torch.empty(
            sequence_shape,
            device=do.device,
            dtype=do.dtype if compact_boundaries else torch.float32,
        )
    input_dtype = cutlass.BFloat16 if do.dtype == torch.bfloat16 else cutlass.Float16
    aux_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }[aux.y.dtype]
    d_initial_state = None
    if compact_boundaries:
        d_initial_state = torch.empty(state_shape, device=do.device, dtype=torch.float32)
    split_scan_state = None
    if use_mma and time % _CHUNK_SIZE != 0:
        split_scan_state = torch.empty(state_shape, device=do.device, dtype=torch.float32)
    # CuTe launch descriptors remain tensor-only.  In the zero-terminal-VJP
    # specialization, reuse already-allocated scratch as a shape-compatible
    # dummy; the constexpr branch guarantees that the kernel never reads it.
    if d_final_state is None:
        if d_initial_state is not None:
            d_final_state_arg = d_initial_state
        elif split_scan_state is not None:
            d_final_state_arg = split_scan_state
        else:
            d_final_state_arg = boundaries.flatten(0, 1)[:batch]
    else:
        d_final_state_arg = d_final_state
    residual_arg = d_residual if d_residual is not None else boundaries
    if compact_boundaries and d_residual is None:
        residual_arg = d_final_state_arg
    launch_tensors = (*tensors, d_final_state_arg)
    device_index = do.device.index
    if device_index is None:
        raise RuntimeError("do must have a concrete CUDA device index")
    with torch.cuda.device(do.device):
        if use_mma:
            value_tile_size = _select_boundary_mma_value_tile(batch, time, heads)
            compile_boundary = (
                _compile_wy_boundary_dstate_mma_compact
                if compact_boundaries
                else _compile_wy_boundary_dstate_mma
            )
            compiled = compile_boundary(
                device_index,
                batch,
                time,
                heads,
                input_dtype,
                aux_dtype,
                d_final_state is not None,
                return_d_residual,
                value_tile_size,
            )
        else:
            compiled = _compile_wy_boundary_dstate(
                device_index,
                batch,
                time,
                heads,
                input_dtype,
                aux_dtype,
                d_final_state is not None,
                return_d_residual,
            )
        stream = cuda.CUstream(torch.cuda.current_stream(do.device).cuda_stream)
        if use_mma:
            # The FP32 precompute scratch may alias the returned residual only
            # when that residual is also FP32.  The compact path rounds dR to
            # BF16 for its downstream MMA/solve consumers, while the ordered
            # boundary recurrence still consumes the unrounded FP32 product.
            aqk_do = d_residual if not compact_boundaries else None
            if aqk_do is None:
                aqk_do = torch.empty(sequence_shape, device=do.device, dtype=torch.float32)
            compiled(
                *launch_tensors,
                boundaries,
                residual_arg,
                aqk_do,
                (
                    d_initial_state
                    if d_initial_state is not None
                    else split_scan_state
                    if split_scan_state is not None
                    else d_final_state_arg
                ),
                stream,
            )
        else:
            compiled(
                *launch_tensors,
                boundaries,
                residual_arg,
                stream,
            )
    if d_initial_state is not None:
        if d_residual is None:
            return boundaries, d_initial_state
        return boundaries, d_residual, d_initial_state
    if d_residual is None:
        return boundaries
    return boundaries, d_residual


def parallel_chunk_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state_boundaries: torch.Tensor,
    dstate_boundaries: torch.Tensor,
    do: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run the SM120 chunk-parallel parameter VJP from exact boundaries."""

    sequence_tensors = (q, k, v, g, beta, w, do)
    if any(not isinstance(tensor, torch.Tensor) for tensor in sequence_tensors):
        raise TypeError("q, k, v, g, beta, w, and do must be tensors")
    if any(tensor.ndim != 4 for tensor in sequence_tensors):
        raise ValueError("all sequence inputs must be rank-4 tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape != do.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v, w, and do must agree with q's [B, T, H] dimensions")
    batch, time, heads, key_dim = q.shape
    if batch <= 0 or time <= 0 or heads <= 0:
        raise ValueError("batch, time, and heads must be positive")
    if key_dim != _DIM or v.shape[-1] != _DIM:
        raise NotImplementedError("the SM120 chunk VJP requires K=V=128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q must use float16 or bfloat16")
    if any(tensor.dtype != q.dtype for tensor in (k, v, beta, w, do)):
        raise TypeError("q, k, v, beta, w, and do must use the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must use the input dtype or float32")

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    boundary_shape = (batch, n_chunks + 1, heads, _DIM, _DIM)
    for name, tensor in (
        ("state_boundaries", state_boundaries),
        ("dstate_boundaries", dstate_boundaries),
    ):
        if tensor.shape != boundary_shape or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be contiguous float32 with shape {boundary_shape}")
    all_tensors = (*sequence_tensors, state_boundaries, dstate_boundaries)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in all_tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if any(tensor.device != q.device for tensor in all_tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    if any(tensor.data_ptr() % 16 != 0 for tensor in all_tensors):
        raise ValueError("all inputs must be 16-byte aligned")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("parallel_chunk_vjp requires an SM120 CUDA device")
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("CUTE_DSL_ARCH must be sm_120 for parallel_chunk_vjp")
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dg = torch.empty_like(g)
    dbeta = torch.empty_like(beta)
    dw = torch.empty_like(w)
    d_initial_state = torch.empty((batch, heads, _DIM, _DIM), device=q.device, dtype=torch.float32)
    cutlass_input_dtype = cutlass.BFloat16 if q.dtype == torch.bfloat16 else cutlass.Float16
    cutlass_gate_dtype = cutlass.Float32 if g.dtype == torch.float32 else cutlass_input_dtype
    device_index = q.device.index
    if device_index is None:
        raise RuntimeError("q must have a concrete CUDA device index")

    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        if batch * n_chunks * heads >= _FULL_MIN_CTAS:
            compiled = _compile_parallel_chunk_vjp_full(
                device_index,
                batch,
                time,
                heads,
                cutlass_input_dtype,
                cutlass_gate_dtype,
            )
            compiled(
                *sequence_tensors[:6],
                state_boundaries,
                dstate_boundaries,
                do,
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
        else:
            partial = torch.empty(
                (4, batch, time, heads, _LOCAL_VALUE_TILES, _DIM),
                device=q.device,
                dtype=torch.float32,
            )
            compiled = _compile_parallel_chunk_vjp(
                device_index,
                batch,
                time,
                heads,
                cutlass_input_dtype,
                cutlass_gate_dtype,
            )
            compiled(
                *sequence_tensors[:6],
                state_boundaries,
                dstate_boundaries,
                do,
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
    return dq, dk, dv, dg, dbeta, dw, d_initial_state


def parallel_chunk_backward_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state_boundaries: torch.Tensor,
    aux: WYBoundaryAux | object,
    do: torch.Tensor,
    d_final_state: torch.Tensor,
    *,
    scale: float | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Run boundary scan plus chunk-parallel VJP; return gradients and dS checkpoints."""

    dstate_boundaries = wy_boundary_dstate(aux, do, d_final_state)
    gradients = parallel_chunk_vjp(
        q,
        k,
        v,
        g,
        beta,
        w,
        state_boundaries,
        dstate_boundaries,
        do,
        scale=scale,
    )
    return gradients, dstate_boundaries


__all__ = [
    "ParallelBackwardProof",
    "WYBoundaryAux",
    "boundary_dstate_token_reference",
    "boundary_dstate_wy_reference",
    "build_wy_boundary_aux_reference",
    "parallel_chunk_backward_cute",
    "parallel_chunk_backward_reference",
    "parallel_chunk_vjp",
    "recurrent_forward_checkpoints_reference",
    "wy_boundary_dstate",
]

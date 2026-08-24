"""BT=16 chunkwise GDN2 forward kernels written in CuTe DSL.

The implementation deliberately separates work along the same boundary as the
WY algorithm:

1. every ``(batch, chunk, head)`` independently builds the asymmetric erase
   factors, solves the unit-lower-triangular system, and materializes the
   chunk-local auxiliaries;
2. every ``(batch, head, value tile)`` walks chunk states in order while value
   columns remain distributed across warps and the ``128 x V`` state stays in
   registers.

This SM120 schedule uses warp reductions today; the dense
``BT x K`` products are isolated in the first kernel so they can be replaced by
SM120 ``mma.sync`` without changing the public API or state schedule.
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
                q_gamma_out[batch, token, head, key_idx] = s_q_gamma[token_local, key_idx]
                k_tail_out[batch, token, head, key_idx] = gamma_end * s_k_bar[token_local, key_idx]

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
            aqk_out[batch, chunk, head, row, col] = causal

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
                    y_out[batch, token, head, dim] = s_solution[row_store, dim]
                else:
                    value_idx = dim - key_dim
                    u_out[batch, token, head, value_idx] = s_solution[row_store, dim]


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
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    has_initial_state: cutlass.Constexpr,
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

        for key_group in cutlass.range_constexpr(key_dim // 32):
            key_idx = lane + key_group * 32
            final_state[batch, head, key_idx, value_idx] = state[key_group]


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
    batch: cutlass.Int32,
    time: cutlass.Constexpr,
    n_chunks: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    has_initial_state: cutlass.Constexpr,
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
        time,
        n_chunks,
        heads,
        has_initial_state,
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
    """WY tensors exposed for profiling and a future checkpoint/WY backward."""

    y: object
    u: object
    q_gamma: object
    k_tail: object
    decay_end: object
    aqk: object


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
):
    """Compile once per static tensor layout; runtime calls bypass DSL tracing."""

    from cutlass.cute.runtime import make_fake_stream

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    q_shape = (batch, time, heads, _KEY_DIM)
    v_shape = (batch, time, heads, _VALUE_DIM)
    state_shape = (batch, heads, _KEY_DIM, _VALUE_DIM)
    decay_shape = (batch, n_chunks, heads, _KEY_DIM)
    aqk_shape = (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE)
    f32 = cutlass.Float32

    return cute.compile(
        _launch_chunk_forward,
        _fake_tensor(input_dtype, q_shape),  # q
        _fake_tensor(input_dtype, q_shape),  # k
        _fake_tensor(input_dtype, v_shape),  # v
        _fake_tensor(gate_dtype, q_shape),  # g
        _fake_tensor(input_dtype, q_shape),  # beta
        _fake_tensor(input_dtype, v_shape),  # w
        _fake_tensor(f32, state_shape),  # initial state
        _fake_tensor(f32, q_shape),  # y
        _fake_tensor(f32, v_shape),  # u
        _fake_tensor(f32, q_shape),  # q_gamma
        _fake_tensor(f32, q_shape),  # k_tail
        _fake_tensor(f32, decay_shape),
        _fake_tensor(f32, aqk_shape),
        _fake_tensor(input_dtype, v_shape),  # output
        _fake_tensor(f32, state_shape),  # final state
        batch,
        time,
        n_chunks,
        heads,
        0.0,  # runtime scale placeholder
        has_initial_state,
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
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("chunk_forward requires an SM120 CUDA device")
    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("CUTE_DSL_ARCH must be sm_120 for chunk_forward")
    expected_state_shape = (batch, heads, key_dim, value_dim)
    if initial_state is not None:
        if initial_state.shape != expected_state_shape or initial_state.dtype != torch.float32:
            raise ValueError("initial_state must be contiguous float32 [B, H, 128, 128]")
        if not initial_state.is_cuda or not initial_state.is_contiguous():
            raise ValueError("initial_state must be a contiguous CUDA tensor")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same CUDA device as q")

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    output = torch.empty_like(v)
    final_state = torch.empty(expected_state_shape, device=q.device, dtype=torch.float32)
    # FP32 auxiliaries keep this baseline useful as a strict numerical oracle.
    y = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    u = torch.empty(v.shape, device=q.device, dtype=torch.float32)
    q_gamma = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    k_tail = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    decay_end = torch.empty((batch, n_chunks, heads, key_dim), device=q.device, dtype=torch.float32)
    aqk = torch.empty(
        (batch, n_chunks, heads, _CHUNK_SIZE, _CHUNK_SIZE),
        device=q.device,
        dtype=torch.float32,
    )
    state_arg = initial_state if initial_state is not None else final_state

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
            batch,
            output_scale,
            stream,
        )
    if not return_aux:
        return output, final_state
    return output, final_state, ChunkForwardAux(y, u, q_gamma, k_tail, decay_end, aqk)


__all__ = ["ChunkForwardAux", "chunk_forward"]

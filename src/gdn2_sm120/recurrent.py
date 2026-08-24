"""SM120 CuTe DSL kernel for token-by-token Gated DeltaNet-2 inference.

The kernel is deliberately specialized for ``K == V == 128``.  A CTA owns one
``(batch, head, value tile)`` and consists of four warps.  A CTA owns 32 value
columns and warp ``w`` owns eight of them; lane ``l`` owns key rows
``l, l + 32, l + 64, l + 96``.  Consequently every lane keeps 32 float32
state elements live in register memory for the complete runtime token loop.

The two reductions in the recurrence (erase and output) are performed with
warp shuffles.  No inter-warp synchronization or shared memory is needed.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Sequence

# CuTe DSL consumes the target while importing its compilation support.
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

_DIM = 128
_WARPS = 4
_THREADS = _WARPS * 32
_ROWS_PER_LANE = _DIM // 32
_VALUE_TILE = 32
_VALUE_TILES = _DIM // _VALUE_TILE
_COLS_PER_WARP = _VALUE_TILE // _WARPS

# Calling a decorated JIT function retraces enough Python to dominate a small
# recurrent launch.  Keep explicit compiled executors instead.  T is dynamic in
# their tensor layouts, so a cache entry is reusable across sequence lengths.
_COMPILED_LAUNCHERS: dict[
    tuple[torch.device, torch.dtype, torch.dtype, int, int, bool], object
] = {}
_COMPILE_LOCK = threading.Lock()


@cute.kernel
def _token_forward_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    initial_state: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    time: cutlass.Int32,
    heads: cutlass.Int32,
    scale: cutlass.Float32,
    has_initial_state: cutlass.Constexpr,
):
    """Run one sequence/value tile for one ``(batch, head)`` per CTA."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_tile = block % _VALUE_TILES
    head_block = block // _VALUE_TILES
    head = head_block % heads
    batch = head_block // heads

    # (row-within-lane, value-column-within-warp).  This fixed-size rmem
    # allocation is scalarized by the compiler and remains live across T.
    state = cute.make_rmem_tensor((_ROWS_PER_LANE, _COLS_PER_WARP), cutlass.Float32)

    for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
        key_row = lane + row_slot * 32
        for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
            value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + col_slot
            if cutlass.const_expr(has_initial_state):
                state[row_slot, col_slot] = initial_state[batch, head, key_row, value_col]
            else:
                state[row_slot, col_slot] = cutlass.Float32(0.0)

    # Four row-dependent values are shared by all eight columns handled by a
    # warp.  Keeping them in rmem avoids reloading Q/K/G/beta for every column.
    q_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    k_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    decay_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    erase_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    decayed_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)

    token = cutlass.Int32(0)
    while token < time:
        for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
            key_row = lane + row_slot * 32
            q_value = q[batch, token, head, key_row].to(cutlass.Float32)
            k_value = k[batch, token, head, key_row].to(cutlass.Float32)
            beta_value = beta[batch, token, head, key_row].to(cutlass.Float32)
            g_value = g[batch, token, head, key_row].to(cutlass.Float32)
            q_rows[row_slot] = q_value * scale
            k_rows[row_slot] = k_value
            decay_rows[row_slot] = cute.exp(g_value)
            erase_rows[row_slot] = beta_value * k_value

        # Lanes 0..15 and 16..31 load the same set of 16 values.  This avoids
        # a divergent bounds check while preserving contiguous memory access;
        # each value is then broadcast from its low-half lane.
        lane_col = lane & (_COLS_PER_WARP - 1)
        lane_value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + lane_col
        v_lane = v[batch, token, head, lane_value_col].to(cutlass.Float32)
        w_lane = w[batch, token, head, lane_value_col].to(cutlass.Float32)

        for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
            erase_partial = cutlass.Float32(0.0)
            for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
                decayed = state[row_slot, col_slot] * decay_rows[row_slot]
                decayed_rows[row_slot] = decayed
                erase_partial += erase_rows[row_slot] * decayed

            erased = cute.arch.warp_reduction_sum(erase_partial)
            value_value = cute.arch.shuffle_sync(v_lane, col_slot)
            write_value = cute.arch.shuffle_sync(w_lane, col_slot)
            update_value = write_value * value_value - erased

            output_partial = cutlass.Float32(0.0)
            for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
                updated = decayed_rows[row_slot] + k_rows[row_slot] * update_value
                state[row_slot, col_slot] = updated
                output_partial += q_rows[row_slot] * updated

            output_value = cute.arch.warp_reduction_sum(output_partial)
            if lane == 0:
                value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + col_slot
                output[batch, token, head, value_col] = output_value.to(output.element_type)

        token += 1

    for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
        key_row = lane + row_slot * 32
        for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
            value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + col_slot
            final_state[batch, head, key_row, value_col] = state[row_slot, col_slot]


@cute.jit
def _launch_token_forward(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    g: cute.Tensor,
    beta: cute.Tensor,
    w: cute.Tensor,
    initial_state: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    batch: cutlass.Int32,
    time: cutlass.Int32,
    heads: cutlass.Int32,
    scale: cutlass.Float32,
    has_initial_state: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _token_forward_kernel(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        output,
        final_state,
        time,
        heads,
        scale,
        has_initial_state,
    ).launch(
        grid=(batch * heads * _VALUE_TILES, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


def _validate_token_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int]:
    tensors: Sequence[torch.Tensor] = (q, k, v, g, beta, w)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("q, k, v, g, beta, and w must be torch.Tensor objects")
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("all inputs must be rank-4 [B, T, H, D] tensors")
    if not (q.shape == k.shape == g.shape == beta.shape):
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must match each other and q's [B, T, H] dimensions")

    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if key_dim != _DIM or value_dim != _DIM:
        raise ValueError("the SM120 recurrent kernel requires K == V == 128")
    if batch <= 0 or heads <= 0:
        raise ValueError("batch and heads must be positive")

    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("inputs must use torch.float16 or torch.bfloat16")
    if any(tensor.dtype != q.dtype for tensor in (k, v, beta, w)):
        raise TypeError("q, k, v, beta, and w must have the same dtype")
    if g.dtype not in (q.dtype, torch.float32):
        raise TypeError("g must have the input dtype or torch.float32")
    if not q.is_cuda:
        raise ValueError("all inputs must be CUDA tensors")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same CUDA device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all inputs must be contiguous")
    if torch.cuda.get_device_capability(q.device) != (12, 0):
        raise RuntimeError("this kernel is specialized for SM120 GPUs")
    effective_arch = os.environ.get("CUTE_DSL_ARCH")
    if effective_arch != "sm_120":
        raise RuntimeError(
            f"the recurrent kernel requires CUTE_DSL_ARCH=sm_120, got {effective_arch!r}"
        )

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor or None")
        expected = (batch, heads, _DIM, _DIM)
        if initial_state.shape != expected:
            raise ValueError(f"initial_state must have shape {expected}")
        if initial_state.dtype != torch.float32:
            raise TypeError("initial_state must use torch.float32")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same CUDA device as q")
        if not initial_state.is_contiguous():
            raise ValueError("initial_state must be contiguous")

    return batch, time, heads


def _sequence_tensor_for_compile(tensor: torch.Tensor) -> cute.Tensor:
    """Create a compile-time tensor descriptor with runtime sequence length."""

    return from_dlpack(
        tensor, use_32bit_stride=True, enable_tvm_ffi=True
    ).mark_compact_shape_dynamic(
        mode=1,
        stride_order=tensor.dim_order(),
    )


def _get_compiled_launcher(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state_input: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    batch: int,
    time: int,
    heads: int,
    output_scale: float,
    has_initial_state: bool,
    stream: cuda.CUstream,
):
    device = torch.device(q.device.type, q.device.index)
    key = (device, q.dtype, g.dtype, batch, heads, has_initial_state)
    compiled = _COMPILED_LAUNCHERS.get(key)
    if compiled is not None:
        return compiled

    # The compiler and its in-memory cache are not guaranteed to be safe when
    # two Python threads specialize the same function simultaneously.
    with _COMPILE_LOCK:
        compiled = _COMPILED_LAUNCHERS.get(key)
        if compiled is None:
            compiled = cute.compile(
                _launch_token_forward,
                _sequence_tensor_for_compile(q),
                _sequence_tensor_for_compile(k),
                _sequence_tensor_for_compile(v),
                _sequence_tensor_for_compile(g),
                _sequence_tensor_for_compile(beta),
                _sequence_tensor_for_compile(w),
                from_dlpack(state_input, use_32bit_stride=True, enable_tvm_ffi=True),
                _sequence_tensor_for_compile(output),
                from_dlpack(final_state, use_32bit_stride=True, enable_tvm_ffi=True),
                cutlass.Int32(batch),
                cutlass.Int32(time),
                cutlass.Int32(heads),
                cutlass.Float32(output_scale),
                has_initial_state,
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILED_LAUNCHERS[key] = compiled
    return compiled


def token_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the GDN2 recurrence with the SM120 token-forward kernel.

    Inputs have contiguous ``[B, T, H, 128]`` layout.  Q/K/V/beta/w must all
    be fp16 or bf16; production-style float32 log-decay ``g`` is also accepted.
    ``initial_state`` is optional float32 ``[B, H, 128, 128]`` state.  Gate
    activations are intentionally outside this primitive, matching
    :func:`gdn2_sm120.reference.recurrent_forward_reference`.
    """

    batch, time, heads = _validate_token_inputs(q, k, v, g, beta, w, initial_state)
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")

    output = torch.empty((batch, time, heads, _DIM), device=q.device, dtype=q.dtype)
    final_state = torch.empty((batch, heads, _DIM, _DIM), device=q.device, dtype=torch.float32)
    if time == 0:
        if initial_state is None:
            final_state.zero_()
        else:
            final_state.copy_(initial_state)
        return output, final_state

    # final_state is a safe dummy input when the recurrence starts from zero;
    # the compile-time flag removes the uninitialized read from the kernel.
    state_input = final_state if initial_state is None else initial_state

    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
        compiled = _get_compiled_launcher(
            q,
            k,
            v,
            g,
            beta,
            w,
            state_input,
            output,
            final_state,
            batch,
            time,
            heads,
            output_scale,
            initial_state is not None,
            stream,
        )
        # Constexpr arguments are removed from a compiled executor's runtime
        # signature.  Passing torch tensors directly uses its cached adapters
        # and avoids retracing the sizeable unrolled kernel body.
        compiled(
            q,
            k,
            v,
            g,
            beta,
            w,
            state_input,
            output,
            final_state,
            batch,
            time,
            heads,
            output_scale,
            stream,
        )

    return output, final_state


__all__ = ["token_forward"]

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
import struct
import threading
from collections.abc import Sequence
from functools import cache, lru_cache

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
_COL_GROUP = 8

# Calling a decorated JIT function retraces enough Python to dominate a small
# recurrent launch.  Keep explicit compiled executors instead.  T is dynamic in
# their tensor layouts, so a cache entry is reusable across sequence lengths.
_COMPILED_LAUNCHERS: dict[
    tuple[torch.device, torch.dtype, torch.dtype, int, int, bool, bool, bool], object
] = {}
_COMPILED_T1_ZERO_LAUNCHERS: dict[tuple[torch.device, torch.dtype, int, int, bool], object] = {}
_COMPILE_LOCK = threading.Lock()


@cache
def _device_capability(device: torch.device) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device)


@lru_cache(maxsize=1)
def _cuda_device_count() -> int:
    return torch.cuda.device_count()


@lru_cache(maxsize=64)
def _driver_stream(stream_handle: int) -> cuda.CUstream:
    return cuda.CUstream(stream_handle)


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
    heads: cutlass.Constexpr,
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
        if cutlass.const_expr(has_initial_state):
            g_state_row = cute.local_tile(
                initial_state[batch, head, key_row, None],
                (_COLS_PER_WARP,),
                (value_tile * _WARPS + warp,),
            )
            state_row = cute.make_rmem_tensor((_COLS_PER_WARP,), cutlass.Float32)
            cute.autovec_copy(g_state_row, state_row)
            for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
                state[row_slot, col_slot] = state_row[col_slot]
        else:
            for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
                state[row_slot, col_slot] = cutlass.Float32(0.0)

    # Four row-dependent values are shared by all eight columns handled by a
    # warp.  Keeping them in rmem avoids reloading Q/K/G/beta for every column.
    q_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    k_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    decay_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)
    erase_rows = cute.make_rmem_tensor(_ROWS_PER_LANE, cutlass.Float32)

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

        # Only the eight source lanes load this warp's value columns; shuffle
        # broadcasts distribute them to the other lanes below.
        v_lane = cutlass.Float32(0.0)
        w_lane = cutlass.Float32(0.0)
        if lane < _COLS_PER_WARP:
            lane_value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + lane
            v_lane = v[batch, token, head, lane_value_col].to(cutlass.Float32)
            w_lane = w[batch, token, head, lane_value_col].to(cutlass.Float32)

        # Decay all columns first, then advance the eight independent columns
        # together. Interleaving their reductions hides the long sequence of
        # warp-shuffle dependencies in a column-at-a-time schedule.
        for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
            for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
                state[row_slot, col_slot] *= decay_rows[row_slot]

        for col_group in cutlass.range_constexpr(_COLS_PER_WARP // _COL_GROUP):
            erase_partials = cute.make_rmem_tensor((_COL_GROUP,), cutlass.Float32)
            erase_partials.fill(0.0)
            for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
                for group_slot in cutlass.range_constexpr(_COL_GROUP):
                    col_slot = col_group * _COL_GROUP + group_slot
                    erase_partials[group_slot] += erase_rows[row_slot] * state[row_slot, col_slot]

            update_values = cute.make_rmem_tensor((_COL_GROUP,), cutlass.Float32)
            for group_slot in cutlass.range_constexpr(_COL_GROUP):
                col_slot = col_group * _COL_GROUP + group_slot
                erased = cute.arch.warp_reduction_sum(erase_partials[group_slot])
                value_value = cute.arch.shuffle_sync(v_lane, col_slot)
                write_value = cute.arch.shuffle_sync(w_lane, col_slot)
                update_values[group_slot] = write_value * value_value - erased

            output_partials = cute.make_rmem_tensor((_COL_GROUP,), cutlass.Float32)
            output_partials.fill(0.0)
            for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
                for group_slot in cutlass.range_constexpr(_COL_GROUP):
                    col_slot = col_group * _COL_GROUP + group_slot
                    updated = (
                        state[row_slot, col_slot] + k_rows[row_slot] * update_values[group_slot]
                    )
                    state[row_slot, col_slot] = updated
                    output_partials[group_slot] += q_rows[row_slot] * updated

            for group_slot in cutlass.range_constexpr(_COL_GROUP):
                output_value = cute.arch.warp_reduction_sum(output_partials[group_slot])
                if lane == 0:
                    col_slot = col_group * _COL_GROUP + group_slot
                    value_col = value_tile * _VALUE_TILE + warp * _COLS_PER_WARP + col_slot
                    output[batch, token, head, value_col] = output_value.to(output.element_type)

        token += 1

    for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
        key_row = lane + row_slot * 32
        g_state_row = cute.local_tile(
            final_state[batch, head, key_row, None],
            (_COLS_PER_WARP,),
            (value_tile * _WARPS + warp,),
        )
        state_row = cute.make_rmem_tensor((_COLS_PER_WARP,), cutlass.Float32)
        for col_slot in cutlass.range_constexpr(_COLS_PER_WARP):
            state_row[col_slot] = state[row_slot, col_slot]
        cute.autovec_copy(state_row, g_state_row)


@cute.kernel
def _token_forward_t1_zero_kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    w: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
):
    """Evaluate the closed form for one token starting from zero state."""

    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tidx & 31
    warp = tidx >> 5
    value_tile = block % _VALUE_TILES
    head_block = block // _VALUE_TILES
    head = head_block % heads
    batch = head_block // heads

    k_rows = cute.make_rmem_tensor((_ROWS_PER_LANE,), cutlass.Float32)
    qk_partial = cutlass.Float32(0.0)
    for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
        key_row = lane + row_slot * 32
        q_value = q[batch, 0, head, key_row].to(cutlass.Float32)
        k_value = k[batch, 0, head, key_row].to(cutlass.Float32)
        k_rows[row_slot] = k_value
        qk_partial += q_value * scale * k_value
    qk = cute.arch.warp_reduction_sum(qk_partial)

    update_lane = cutlass.Float32(0.0)
    if lane < _COL_GROUP:
        value_col = value_tile * _VALUE_TILE + warp * _COL_GROUP + lane
        value_value = v[batch, 0, head, value_col].to(cutlass.Float32)
        write_value = w[batch, 0, head, value_col].to(cutlass.Float32)
        update_lane = write_value * value_value

    updates = cute.make_rmem_tensor((_COL_GROUP,), cutlass.Float32)
    for group_col in cutlass.range_constexpr(_COL_GROUP):
        update = cute.arch.shuffle_sync(update_lane, group_col)
        updates[group_col] = update
        if lane == 0:
            value_col = value_tile * _VALUE_TILE + warp * _COL_GROUP + group_col
            output[batch, 0, head, value_col] = (qk * update).to(output.element_type)

    for row_slot in cutlass.range_constexpr(_ROWS_PER_LANE):
        key_row = lane + row_slot * 32
        g_state_row = cute.local_tile(
            final_state[batch, head, key_row, None],
            (_COL_GROUP,),
            (value_tile * _WARPS + warp,),
        )
        state_row = cute.make_rmem_tensor((_COL_GROUP,), cutlass.Float32)
        for group_col in cutlass.range_constexpr(_COL_GROUP):
            state_row[group_col] = k_rows[row_slot] * updates[group_col]
        cute.autovec_copy(state_row, g_state_row)


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
    batch: cutlass.Constexpr,
    time: cutlass.Int32,
    heads: cutlass.Constexpr,
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


@cute.jit
def _launch_token_forward_t1_zero(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    w: cute.Tensor,
    output: cute.Tensor,
    final_state: cute.Tensor,
    batch: cutlass.Constexpr,
    heads: cutlass.Constexpr,
    scale: cutlass.Float32,
    stream: cuda.CUstream,
):
    _token_forward_t1_zero_kernel(
        q,
        k,
        v,
        w,
        output,
        final_state,
        heads,
        scale,
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
    if _device_capability(q.device) != (12, 0):
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


def _canonical_strides(shape: torch.Size | tuple[int, ...]) -> tuple[int, ...]:
    running = 1
    reversed_strides = []
    for size in reversed(shape):
        reversed_strides.append(running)
        running *= max(size, 1)
    return tuple(reversed(reversed_strides))


def _canonical_view(tensor: torch.Tensor, strides: tuple[int, ...]) -> torch.Tensor:
    if tensor.stride() == strides:
        return tensor
    return tensor.as_strided(tensor.shape, strides)


def _validate_output_buffer(
    name: str,
    tensor: torch.Tensor,
    shape: torch.Size | tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same CUDA device as q")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    """Return the half-open byte range occupied by a contiguous tensor."""

    start = tensor.data_ptr()
    return start, start + tensor.nbytes


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < left[1] and right[0] < right[1] and left[0] < right[1] and right[0] < left[1]


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
    device = q.device
    state_aligned = state_input.data_ptr() % 16 == 0
    final_state_aligned = final_state.data_ptr() % 16 == 0
    key = (
        device,
        q.dtype,
        g.dtype,
        batch,
        heads,
        has_initial_state,
        state_aligned,
        final_state_aligned,
    )
    compiled = _COMPILED_LAUNCHERS.get(key)
    if compiled is not None:
        return compiled

    # The compiler and its in-memory cache are not guaranteed to be safe when
    # two Python threads specialize the same function simultaneously.
    with _COMPILE_LOCK:
        compiled = _COMPILED_LAUNCHERS.get(key)
        if compiled is None:
            # PyTorch permits arbitrary strides only on size-zero/one axes of
            # an otherwise contiguous tensor. Canonicalize compile descriptors;
            # runtime tensors have the same logical addresses and can retain
            # their original views without per-launch wrapper work.
            sequence_strides = _canonical_strides(q.shape)
            q_compile, k_compile, v_compile, g_compile, beta_compile, w_compile = (
                _canonical_view(tensor, sequence_strides) for tensor in (q, k, v, g, beta, w)
            )
            state_strides = _canonical_strides(state_input.shape)
            state_input_compile = _canonical_view(state_input, state_strides)
            output_compile = _canonical_view(output, sequence_strides)
            final_state_compile = _canonical_view(final_state, state_strides)
            compiled = cute.compile(
                _launch_token_forward,
                _sequence_tensor_for_compile(q_compile),
                _sequence_tensor_for_compile(k_compile),
                _sequence_tensor_for_compile(v_compile),
                _sequence_tensor_for_compile(g_compile),
                _sequence_tensor_for_compile(beta_compile),
                _sequence_tensor_for_compile(w_compile),
                from_dlpack(
                    state_input_compile,
                    assumed_align=16 if state_aligned else None,
                    use_32bit_stride=True,
                    enable_tvm_ffi=True,
                ),
                _sequence_tensor_for_compile(output_compile),
                from_dlpack(
                    final_state_compile,
                    assumed_align=16 if final_state_aligned else None,
                    use_32bit_stride=True,
                    enable_tvm_ffi=True,
                ),
                batch,
                cutlass.Int32(time),
                heads,
                cutlass.Float32(output_scale),
                has_initial_state,
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILED_LAUNCHERS[key] = compiled
    return compiled


def _get_compiled_t1_zero_launcher(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    batch: int,
    heads: int,
    output_scale: float,
    stream: cuda.CUstream,
):
    final_state_aligned = final_state.data_ptr() % 16 == 0
    key = (q.device, q.dtype, batch, heads, final_state_aligned)
    compiled = _COMPILED_T1_ZERO_LAUNCHERS.get(key)
    if compiled is not None:
        return compiled

    with _COMPILE_LOCK:
        compiled = _COMPILED_T1_ZERO_LAUNCHERS.get(key)
        if compiled is None:
            sequence_strides = _canonical_strides(q.shape)
            q_compile, k_compile, v_compile, w_compile = (
                _canonical_view(tensor, sequence_strides) for tensor in (q, k, v, w)
            )
            output_compile = _canonical_view(output, sequence_strides)
            state_strides = _canonical_strides(final_state.shape)
            final_state_compile = _canonical_view(final_state, state_strides)
            compiled = cute.compile(
                _launch_token_forward_t1_zero,
                from_dlpack(q_compile, use_32bit_stride=True, enable_tvm_ffi=True),
                from_dlpack(k_compile, use_32bit_stride=True, enable_tvm_ffi=True),
                from_dlpack(v_compile, use_32bit_stride=True, enable_tvm_ffi=True),
                from_dlpack(w_compile, use_32bit_stride=True, enable_tvm_ffi=True),
                from_dlpack(output_compile, use_32bit_stride=True, enable_tvm_ffi=True),
                from_dlpack(
                    final_state_compile,
                    assumed_align=16 if final_state_aligned else None,
                    use_32bit_stride=True,
                    enable_tvm_ffi=True,
                ),
                batch,
                heads,
                cutlass.Float32(output_scale),
                stream,
                options="--enable-tvm-ffi",
            )
            _COMPILED_T1_ZERO_LAUNCHERS[key] = compiled
    return compiled


def _invoke_token_forward(
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
) -> None:
    stream = _driver_stream(torch.cuda.current_stream().cuda_stream)
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
        has_initial_state,
        stream,
    )
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
        time,
        output_scale,
        stream,
    )


def _invoke_token_forward_t1_zero(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    output: torch.Tensor,
    final_state: torch.Tensor,
    batch: int,
    heads: int,
    output_scale: float,
) -> None:
    stream = _driver_stream(torch.cuda.current_stream().cuda_stream)
    compiled = _get_compiled_t1_zero_launcher(
        q,
        k,
        v,
        w,
        output,
        final_state,
        batch,
        heads,
        output_scale,
        stream,
    )
    compiled(q, k, v, w, output, final_state, output_scale, stream)


def token_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    scale: float | None = None,
    *,
    out: torch.Tensor | None = None,
    final_state_out: torch.Tensor | None = None,
    inplace_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the GDN2 recurrence with the SM120 token-forward kernel.

    Inputs have contiguous ``[B, T, H, 128]`` layout.  Q/K/V/beta/w must all
    be fp16 or bf16; production-style float32 log-decay ``g`` is also accepted.
    ``initial_state`` is optional float32 ``[B, H, 128, 128]`` state.  Gate
    activations are intentionally outside this primitive, matching
    :func:`gdn2_sm120.reference.recurrent_forward_reference`.

    ``out`` and ``final_state_out`` may provide reusable output buffers.
    ``inplace_final_state=True`` explicitly updates and returns
    ``initial_state``; the default remains allocation-based and does not
    mutate inputs.
    """

    batch, time, heads = _validate_token_inputs(q, k, v, g, beta, w, initial_state)
    requested_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(requested_scale):
        raise ValueError("scale must be finite")
    try:
        output_scale = struct.unpack("=f", struct.pack("=f", requested_scale))[0]
    except OverflowError as error:
        raise ValueError("scale must be representable as float32") from error
    if not math.isfinite(output_scale):
        raise ValueError("scale must be representable as finite float32")

    if out is None:
        output = torch.empty_like(q)
    else:
        _validate_output_buffer("out", out, q.shape, q.dtype, q.device)
        output = out

    state_shape = (batch, heads, _DIM, _DIM)
    if inplace_final_state:
        if initial_state is None:
            raise ValueError("inplace_final_state=True requires initial_state")
        if final_state_out is not None:
            raise ValueError("final_state_out cannot be used with inplace_final_state=True")
        final_state = initial_state
    elif final_state_out is not None:
        _validate_output_buffer(
            "final_state_out", final_state_out, state_shape, torch.float32, q.device
        )
        final_state = final_state_out
    else:
        final_state = (
            torch.empty_like(initial_state)
            if initial_state is not None
            else torch.empty(state_shape, device=q.device, dtype=torch.float32)
        )

    inputs = (q, k, v, g, beta, w)
    if out is not None or final_state_out is not None or inplace_final_state:
        sequence_spans = tuple(_byte_span(tensor) for tensor in inputs)
        initial_span = _byte_span(initial_state) if initial_state is not None else None
        output_span = _byte_span(output) if out is not None else None
        final_span = _byte_span(final_state)
    else:
        sequence_spans = ()
        initial_span = output_span = final_span = None

    if out is not None and output.numel() != 0:
        assert output_span is not None
        if any(_spans_overlap(output_span, span) for span in sequence_spans) or (
            initial_span is not None and _spans_overlap(output_span, initial_span)
        ):
            raise ValueError("out must not overlap an input tensor")
        assert final_span is not None
        if _spans_overlap(output_span, final_span):
            raise ValueError("out and final_state must not overlap")
    if final_state_out is not None:
        assert final_span is not None
        if any(_spans_overlap(final_span, span) for span in sequence_spans) or (
            initial_span is not None and _spans_overlap(final_span, initial_span)
        ):
            raise ValueError("final_state_out must not overlap an input tensor")
    if inplace_final_state:
        assert final_span is not None
        if any(_spans_overlap(final_span, span) for span in sequence_spans):
            raise ValueError("initial_state must not overlap sequence inputs for in-place update")
    if time == 0:
        if initial_state is None:
            final_state.zero_()
        elif final_state is not initial_state:
            final_state.copy_(initial_state)
        return output, final_state

    # final_state is a safe dummy input when the recurrence starts from zero;
    # the compile-time flag removes the uninitialized read from the kernel.
    state_input = final_state if initial_state is None else initial_state
    use_t1_zero = time == 1 and initial_state is None

    if _cuda_device_count() == 1:
        if use_t1_zero:
            _invoke_token_forward_t1_zero(
                q,
                k,
                v,
                w,
                output,
                final_state,
                batch,
                heads,
                output_scale,
            )
        else:
            _invoke_token_forward(
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
            )
    else:
        with torch.cuda.device(q.device):
            if use_t1_zero:
                _invoke_token_forward_t1_zero(
                    q,
                    k,
                    v,
                    w,
                    output,
                    final_state,
                    batch,
                    heads,
                    output_scale,
                )
            else:
                _invoke_token_forward(
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
                )

    return output, final_state


__all__ = ["token_forward"]

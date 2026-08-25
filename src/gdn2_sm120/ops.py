"""PyTorch-facing Gated DeltaNet-2 operations backed by the SM120 kernels."""

from __future__ import annotations

import math

import torch

from .backward import chunk_backward
from .backward_parallel import WYBoundaryAux, parallel_chunk_vjp, wy_boundary_dstate
from .backward_wy import compact_wy_chunk_vjp_cute
from .chunk import ChunkForwardAux, chunk_forward
from .recurrent import token_forward

_DIM = 128
_CHUNK_SIZE = 16
_PARALLEL_BACKWARD_MIN_TOKENS = 64
_COMPACT_WY_BACKWARD_MIN_TOKENS = 128
_T64_COMPACT_WY_MIN_CHUNK_HEADS = 64
# CuTe's generated compact tensor layouts use a 32-bit byte offset.  Keep
# every saved forward/reverse state-boundary tensor at or below that address
# space so a large leading batch cannot wrap an otherwise valid access.
_CUTE_TENSOR_BYTE_ADDRESS_LIMIT = 1 << 32


def _use_compact_wy_backward(batch: int, time: int, heads: int) -> bool:
    """Select the staged VJP only when the T=64 grid can fill the GPU."""

    chunk_heads = batch * (time // _CHUNK_SIZE) * heads
    return time >= _COMPACT_WY_BACKWARD_MIN_TOKENS or (
        time == 64 and chunk_heads >= _T64_COMPACT_WY_MIN_CHUNK_HEADS
    )


def _training_batch_ranges(
    batch: int,
    time: int,
    heads: int,
    dtype: torch.dtype,
) -> tuple[tuple[int, int], ...]:
    """Return balanced batch ranges whose state boundaries fit 32-bit offsets."""

    if batch <= 0 or time < _PARALLEL_BACKWARD_MIN_TOKENS or heads <= 0:
        return ((0, batch),)

    n_chunks = math.ceil(time / _CHUNK_SIZE)
    compact_boundaries = (
        dtype == torch.bfloat16
        and time >= _COMPACT_WY_BACKWARD_MIN_TOKENS
        and time % _CHUNK_SIZE == 0
    )
    boundary_element_size = 2 if compact_boundaries else 4
    boundary_bytes_per_batch = (n_chunks + 1) * heads * _DIM * _DIM * boundary_element_size
    max_batch = max(1, _CUTE_TENSOR_BYTE_ADDRESS_LIMIT // boundary_bytes_per_batch)
    if batch <= max_batch:
        return ((0, batch),)

    # Balance the slices rather than emitting max_batch plus a small tail.
    # B4/T32768/H16 BF16 therefore becomes B2+B2 instead of B3+B1, letting
    # both calls reuse one compiled specialization and similar grid sizes.
    n_ranges = math.ceil(batch / max_batch)
    base_size, larger_ranges = divmod(batch, n_ranges)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(n_ranges):
        size = base_size + (index < larger_ranges)
        stop = start + size
        ranges.append((start, stop))
        start = stop
    return tuple(ranges)


class _ChunkGDN2(torch.autograd.Function):
    """Connect the independent CuTe forward and backward executors."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        w: torch.Tensor,
        initial_state: torch.Tensor | None,
        scale: float,
        prepare_backward: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        use_parallel_backward = q.shape[1] >= _PARALLEL_BACKWARD_MIN_TOKENS and prepare_backward
        forward_result = chunk_forward(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            scale=scale,
            return_aux=use_parallel_backward,
        )
        if use_parallel_backward:
            output, final_state, aux = forward_result
            ctx.save_for_backward(
                q,
                k,
                v,
                g,
                beta,
                w,
                aux.y,
                aux.u,
                aux.q_gamma,
                aux.k_tail,
                aux.decay_end,
                aux.aqk,
                aux.state_boundaries,
            )
        else:
            output, final_state = forward_result
            ctx.save_for_backward(q, k, v, g, beta, w, final_state)
        ctx.set_materialize_grads(False)
        ctx.has_initial_state = initial_state is not None
        ctx.use_parallel_backward = use_parallel_backward
        ctx.scale = scale
        return output, final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        d_output: torch.Tensor | None,
        d_final_state: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None,
        None,
    ]:
        q, k, v, g, beta, w, *saved = ctx.saved_tensors
        if d_output is None:
            d_output = torch.zeros_like(v)
        elif not d_output.is_contiguous():
            d_output = d_output.contiguous()
        if ctx.use_parallel_backward:
            y, u, q_gamma, k_tail, decay_end, aqk, state_boundaries = saved
            if d_final_state is None:
                d_final_state = torch.zeros(
                    (q.shape[0], q.shape[2], _DIM, _DIM),
                    device=q.device,
                    dtype=torch.float32,
                )
            elif not d_final_state.is_contiguous():
                d_final_state = d_final_state.contiguous()
            boundary_aux = WYBoundaryAux(y, q_gamma, k_tail, decay_end, aqk)
            use_compact_wy = _use_compact_wy_backward(q.shape[0], q.shape[1], q.shape[2])
            if use_compact_wy:
                compact_boundaries = state_boundaries.dtype == q.dtype
                boundary_result = wy_boundary_dstate(
                    boundary_aux,
                    d_output,
                    d_final_state,
                    return_d_residual=True,
                    compact_boundaries=compact_boundaries,
                )
                if compact_boundaries:
                    dstate_boundaries, d_residual, d_initial_state = boundary_result
                else:
                    dstate_boundaries, d_residual = boundary_result
                    d_initial_state = None
                forward_aux = ChunkForwardAux(
                    y,
                    u,
                    q_gamma,
                    k_tail,
                    decay_end,
                    aqk,
                    state_boundaries,
                )
                gradients = compact_wy_chunk_vjp_cute(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    w,
                    state_boundaries,
                    dstate_boundaries,
                    forward_aux,
                    d_output,
                    scale=ctx.scale,
                    precomputed_d_residual=d_residual,
                    precomputed_d_initial_state=d_initial_state,
                )
            else:
                dstate_boundaries = wy_boundary_dstate(
                    boundary_aux,
                    d_output,
                    d_final_state,
                )
                gradients = parallel_chunk_vjp(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    w,
                    state_boundaries,
                    dstate_boundaries,
                    d_output,
                    scale=ctx.scale,
                )
        else:
            (final_state,) = saved
            if d_final_state is None:
                d_final_state = torch.zeros_like(final_state)
            elif not d_final_state.is_contiguous():
                d_final_state = d_final_state.contiguous()
            gradients = chunk_backward(
                q,
                k,
                v,
                g,
                beta,
                w,
                final_state,
                d_output,
                d_final_state,
                ctx.scale,
            )
        *sequence_gradients, d_initial_state = gradients
        return (
            *sequence_gradients,
            d_initial_state if ctx.has_initial_state else None,
            None,
            None,
        )


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run chunkwise GDN2 with the native SM120 backward under autograd.

    Sequence tensors use contiguous ``[B, T, H, 128]`` layout. Q/K/V/beta/w
    are FP16 or BF16, ``g`` may additionally be FP32, and state tensors are
    FP32 ``[B, H, 128, 128]``. Gate activations and Q/K normalization remain
    caller-side operations.
    """

    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(_DIM)
    if not math.isfinite(output_scale):
        raise ValueError("scale must be finite")
    differentiable_inputs = (q, k, v, g, beta, w, initial_state)
    prepare_backward = torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad
        for tensor in differentiable_inputs
    )
    sequence_inputs = (q, k, v, g, beta, w)
    can_slice_batch = (
        prepare_backward
        and all(
            isinstance(tensor, torch.Tensor) and tensor.ndim == 4 and tensor.shape[0] == q.shape[0]
            for tensor in sequence_inputs
        )
        and (
            initial_state is None
            or (
                isinstance(initial_state, torch.Tensor)
                and initial_state.ndim == 4
                and initial_state.shape[0] == q.shape[0]
            )
        )
    )
    batch_ranges = (
        _training_batch_ranges(q.shape[0], q.shape[1], q.shape[2], q.dtype)
        if can_slice_batch
        else None
    )
    if batch_ranges is None or len(batch_ranges) == 1:
        output, final_state = _ChunkGDN2.apply(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            output_scale,
            prepare_backward,
        )
    else:
        outputs: list[torch.Tensor] = []
        final_states: list[torch.Tensor] = []
        for start, stop in batch_ranges:
            sliced_inputs = tuple(tensor[start:stop] for tensor in sequence_inputs)
            sliced_initial_state = initial_state[start:stop] if initial_state is not None else None
            sliced_output, sliced_final_state = _ChunkGDN2.apply(
                *sliced_inputs,
                sliced_initial_state,
                output_scale,
                prepare_backward,
            )
            outputs.append(sliced_output)
            final_states.append(sliced_final_state)
        output = torch.cat(outputs, dim=0)
        final_state = torch.cat(final_states, dim=0)
    return output, final_state if output_final_state else None


def recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    out: torch.Tensor | None = None,
    final_state_out: torch.Tensor | None = None,
    inplace_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward-only token/recurrent SM120 kernel."""

    return token_forward(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        scale=scale,
        out=out,
        final_state_out=final_state_out,
        inplace_final_state=inplace_final_state,
    )


__all__ = ["chunk_gdn2", "recurrent_gdn2"]

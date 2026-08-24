"""PyTorch-facing Gated DeltaNet-2 operations backed by the SM120 kernels."""

from __future__ import annotations

import math

import torch

from .backward import MAX_BACKWARD_TOKENS, chunk_backward
from .chunk import chunk_forward
from .recurrent import token_forward

_DIM = 128


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, final_state = chunk_forward(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            scale=scale,
        )
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(q, k, v, g, beta, w, final_state)
        ctx.has_initial_state = initial_state is not None
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
    ]:
        q, k, v, g, beta, w, final_state = ctx.saved_tensors
        if d_output is None:
            d_output = torch.zeros_like(v)
        elif not d_output.is_contiguous():
            d_output = d_output.contiguous()
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
    needs_backward = torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad
        for tensor in differentiable_inputs
    )
    if (
        needs_backward
        and isinstance(q, torch.Tensor)
        and q.ndim == 4
        and q.shape[1] > MAX_BACKWARD_TOKENS
    ):
        raise NotImplementedError(
            f"the native backward supports at most {MAX_BACKWARD_TOKENS} tokens; "
            "checkpoint/WY long-sequence backward is not implemented"
        )
    output, final_state = _ChunkGDN2.apply(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        output_scale,
    )
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the forward-only token/recurrent SM120 kernel."""

    return token_forward(q, k, v, g, beta, w, initial_state, scale=scale)


__all__ = ["chunk_gdn2", "recurrent_gdn2"]

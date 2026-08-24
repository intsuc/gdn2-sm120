"""Readable PyTorch definitions used as the numerical oracle.

The public kernels use the log-decay convention from the official GDN2
implementation: ``g <= 0`` and the actual channel-wise decay is ``exp(g)``.
Inputs use the contiguous ``[batch, time, heads, dim]`` layout. States use
``[batch, heads, key_dim, value_dim]`` and are accumulated in float32.
"""

from __future__ import annotations

import math

import torch


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("q/k/g/beta and v/w must be rank-4 [B, T, H, D] tensors")
    if q.shape != k.shape or q.shape != g.shape or q.shape != beta.shape:
        raise ValueError("q, k, g, and beta must have identical shapes")
    if v.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("v and w must match each other and q's [B, T, H] dimensions")
    if not all(x.device == q.device for x in (k, v, g, beta, w)):
        raise ValueError("all inputs must be on the same device")
    batch, time, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if initial_state is not None and initial_state.shape != (
        batch,
        heads,
        key_dim,
        value_dim,
    ):
        raise ValueError(
            f"initial_state must have shape [B, H, K, V] = {(batch, heads, key_dim, value_dim)}"
        )
    return batch, time, heads, key_dim, value_dim


def recurrent_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    return_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the Gated Delta Rule-2 recurrence in float32.

    ``beta`` is the channel-wise erase gate. ``w`` is the independent
    channel-wise write gate. Gate activations are intentionally outside this
    primitive so callers can compare equivalent preprocessed inputs across
    implementations.
    """

    batch, time, heads, key_dim, value_dim = _validate_inputs(q, k, v, g, beta, w, initial_state)
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    state = (
        torch.zeros(
            (batch, heads, key_dim, value_dim),
            device=q.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.float()
    )

    outputs: list[torch.Tensor] = []
    for token in range(time):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        v_t = v[:, token].float()
        decay_t = g[:, token].float().exp()
        beta_t = beta[:, token].float()
        w_t = w[:, token].float()

        decayed = decay_t.unsqueeze(-1) * state
        erase_key = beta_t * k_t
        erased = torch.einsum("bhk,bhkv->bhv", erase_key, decayed)
        update_value = w_t * v_t - erased
        state = decayed + k_t.unsqueeze(-1) * update_value.unsqueeze(-2)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t * output_scale, state))

    if time:
        output = torch.stack(outputs, dim=1).to(q.dtype)
    else:
        output = torch.empty((batch, 0, heads, value_dim), device=q.device, dtype=q.dtype)
    return output, state if return_final_state else None


def chunkwise_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    chunk_size: int = 64,
    return_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Vectorized WY reference for the chunkwise training algorithm.

    This follows the paper's asymmetric erase-factor construction. Matrix
    products are evaluated in float32, while the returned output matches
    ``q.dtype``. The implementation is intentionally composed from PyTorch
    primitives so autograd provides an independent oracle for every backward
    gradient.
    """

    batch, time, heads, key_dim, value_dim = _validate_inputs(q, k, v, g, beta, w, initial_state)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_scale = float(scale) if scale is not None else 1.0 / math.sqrt(key_dim)
    state = (
        torch.zeros(
            (batch, heads, key_dim, value_dim),
            device=q.device,
            dtype=torch.float32,
        )
        if initial_state is None
        else initial_state.float()
    )
    chunks: list[torch.Tensor] = []

    for start in range(0, time, chunk_size):
        stop = min(start + chunk_size, time)
        # Work in [B, H, C, D] so the last two modes are matrix modes.
        q_c = q[:, start:stop].transpose(1, 2).float()
        k_c = k[:, start:stop].transpose(1, 2).float()
        v_c = v[:, start:stop].transpose(1, 2).float()
        g_c = g[:, start:stop].transpose(1, 2).float()
        beta_c = beta[:, start:stop].transpose(1, 2).float()
        w_c = w[:, start:stop].transpose(1, 2).float()

        cumulative_g = g_c.cumsum(dim=-2)
        gamma = cumulative_g.exp()
        reciprocal_gamma = (-cumulative_g).exp()
        k_bar = reciprocal_gamma * k_c
        erase_bar = gamma * beta_c * k_c
        z = w_c * v_c

        length = stop - start
        lower = torch.tril(torch.matmul(erase_bar, k_bar.transpose(-1, -2)), diagonal=-1)
        system = lower + torch.eye(length, device=q.device, dtype=torch.float32)
        y = torch.linalg.solve_triangular(system, erase_bar, upper=False, unitriangular=True)
        u = torch.linalg.solve_triangular(system, z, upper=False, unitriangular=True)
        residual = u - torch.matmul(y, state)

        q_gamma = q_c * gamma * output_scale
        causal_qk = torch.tril(torch.matmul(q_gamma, k_bar.transpose(-1, -2)))
        out_c = torch.matmul(q_gamma, state) + torch.matmul(causal_qk, residual)
        chunks.append(out_c.transpose(1, 2))

        tail_decay = gamma[..., -1, :]
        tail_keys = k_bar * tail_decay.unsqueeze(-2)
        state = tail_decay.unsqueeze(-1) * state + torch.matmul(
            tail_keys.transpose(-1, -2), residual
        )

    output = (
        torch.cat(chunks, dim=1).to(q.dtype)
        if chunks
        else torch.empty((batch, 0, heads, value_dim), device=q.device, dtype=q.dtype)
    )
    return output, state if return_final_state else None

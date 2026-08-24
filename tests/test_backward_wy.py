from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from gdn2_sm120.backward_parallel import WYBoundaryAux, wy_boundary_dstate
from gdn2_sm120.backward_wy import (
    compact_wy_chunk_vjp_cute,
    compact_wy_chunk_vjp_reference,
)
from gdn2_sm120.chunk import chunk_forward


def _compact_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
):
    state = initial_state
    outputs = []
    states = [state]
    y_all = []
    u_all = []
    q_gamma_all = []
    k_tail_all = []
    decay_all = []
    aqk_all = []
    time = q.shape[1]
    for start in range(0, time, chunk_size):
        stop = min(start + chunk_size, time)
        length = stop - start
        q_c = q[:, start:stop].transpose(1, 2)
        k_c = k[:, start:stop].transpose(1, 2)
        v_c = v[:, start:stop].transpose(1, 2)
        g_c = g[:, start:stop].transpose(1, 2)
        beta_c = beta[:, start:stop].transpose(1, 2)
        w_c = w[:, start:stop].transpose(1, 2)
        gamma = g_c.cumsum(dim=-2).exp()
        k_bar = k_c / gamma
        erase = gamma * beta_c * k_c
        q_gamma = scale * gamma * q_c
        lower = torch.tril(erase @ k_bar.transpose(-1, -2), diagonal=-1)
        system = lower + torch.eye(length, dtype=q.dtype, device=q.device)
        y = torch.linalg.solve_triangular(system, erase, upper=False, unitriangular=True)
        z = w_c * v_c
        u = torch.linalg.solve_triangular(system, z, upper=False, unitriangular=True)
        aqk = torch.tril(q_gamma @ k_bar.transpose(-1, -2))
        residual = u - y @ state
        outputs.append((q_gamma @ state + aqk @ residual).transpose(1, 2))
        decay_end = gamma[..., -1, :]
        k_tail = k_bar * decay_end.unsqueeze(-2)
        state = decay_end.unsqueeze(-1) * state + k_tail.transpose(-1, -2) @ residual
        states.append(state)

        y_all.append(y.transpose(1, 2))
        u_all.append(u.transpose(1, 2))
        q_gamma_all.append(q_gamma.transpose(1, 2))
        k_tail_all.append(k_tail.transpose(1, 2))
        decay_all.append(decay_end)
        aqk_all.append(
            torch.nn.functional.pad(aqk, (0, chunk_size - length, 0, chunk_size - length))
        )

    aux = SimpleNamespace(
        y=torch.cat(y_all, dim=1),
        u=torch.cat(u_all, dim=1),
        q_gamma=torch.cat(q_gamma_all, dim=1),
        k_tail=torch.cat(k_tail_all, dim=1),
        decay_end=torch.stack(decay_all, dim=1),
        aqk=torch.stack(aqk_all, dim=1),
    )
    return torch.cat(outputs, dim=1), state, torch.stack(states, dim=1), aux


def _boundary_vjp(
    aux,
    do,
    d_final_state,
    *,
    chunk_size: int,
    return_d_residual: bool = False,
):
    batch, time, heads, key_dim = aux.y.shape
    value_dim = do.shape[-1]
    n_chunks = (time + chunk_size - 1) // chunk_size
    result = torch.empty(
        (batch, n_chunks + 1, heads, key_dim, value_dim),
        dtype=d_final_state.dtype,
        device=d_final_state.device,
    )
    dstate = d_final_state
    result[:, -1] = dstate
    residual_sequence = torch.empty(
        (batch, time, heads, value_dim),
        dtype=d_final_state.dtype,
        device=d_final_state.device,
    )
    compute_dtype = d_final_state.dtype
    for chunk in reversed(range(n_chunks)):
        start = chunk * chunk_size
        stop = min(start + chunk_size, time)
        length = stop - start
        y = aux.y[:, start:stop].transpose(1, 2).to(compute_dtype)
        q_gamma = aux.q_gamma[:, start:stop].transpose(1, 2).to(compute_dtype)
        k_tail = aux.k_tail[:, start:stop].transpose(1, 2).to(compute_dtype)
        aqk = aux.aqk[:, chunk, :, :length, :length].to(compute_dtype)
        do_c = do[:, start:stop].transpose(1, 2).to(compute_dtype)
        d_residual = k_tail @ dstate + aqk.transpose(-1, -2) @ do_c
        residual_sequence[:, start:stop] = d_residual.transpose(1, 2)
        dstate = (
            aux.decay_end[:, chunk].unsqueeze(-1) * dstate
            + q_gamma.transpose(-1, -2) @ do_c
            - y.transpose(-1, -2) @ d_residual
        )
        result[:, chunk] = dstate
    if return_d_residual:
        return result, residual_sequence
    return result


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.parametrize("shape", [(0, 16, 1, 128), (1, 16, 0, 128)])
def test_compact_wy_rejects_empty_batch_or_heads(shape: tuple[int, ...]) -> None:
    batch, time, heads, dim = shape
    sequence = torch.empty(shape, dtype=torch.float16)
    n_chunks = 1
    boundaries = torch.empty((batch, n_chunks + 1, heads, dim, dim), dtype=torch.float32)
    aux = SimpleNamespace(
        y=torch.empty(shape, dtype=torch.float32),
        u=torch.empty(shape, dtype=torch.float32),
        q_gamma=torch.empty(shape, dtype=torch.float32),
        k_tail=torch.empty(shape, dtype=torch.float32),
        decay_end=torch.empty((batch, n_chunks, heads, dim), dtype=torch.float32),
        aqk=torch.empty((batch, n_chunks, heads, 16, 16), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="batch and heads must be positive"):
        compact_wy_chunk_vjp_cute(
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            boundaries,
            boundaries,
            aux,
            sequence,
        )


def _cuda_compact_case(time: int, dtype: torch.dtype, *, heads: int = 1):
    generator = torch.Generator(device="cuda").manual_seed(4_170 + time)
    batch, dim = 1, 128
    shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
    base = (
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        (torch.randn(shape, generator=generator, device="cuda") * 0.15).to(dtype),
        -torch.rand(shape, generator=generator, device="cuda") * 0.02,
        (torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")) * 0.7).to(dtype),
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
    )
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.02
    )
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.12).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.05
    )
    _, _, aux = chunk_forward(
        *base,
        initial_state,
        scale=0.125,
        return_aux=True,
    )
    # Most direct kernel tests exercise the generic FP32-boundary contract.
    # Production compact BF16 checkpoints are covered through the public
    # autograd path and a dedicated low-boundary test.
    if aux.state_boundaries.dtype != torch.float32:
        aux = replace(aux, state_boundaries=aux.state_boundaries.float())
    dstate_boundaries, d_residual = _boundary_vjp(
        aux,
        do,
        d_final_state,
        chunk_size=16,
        return_d_residual=True,
    )
    return base, do, aux, dstate_boundaries, d_residual


def _assert_long_accuracy_contract(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> None:
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        difference = gradient.float() - reference.float()
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference.float()
        ).clamp_min(1e-12)
        assert relative_l2.item() < 0.01
        assert difference.abs().max().item() < 5e-3
    torch.testing.assert_close(actual[-1], expected[-1], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("time", [4, 7], ids=["full-chunk", "partial-tail"])
def test_compact_wy_chunk_vjp_proof_matches_fp64_autograd(time: int) -> None:
    generator = torch.Generator().manual_seed(821_000 + time)
    batch, heads, key_dim, value_dim = 1, 2, 7, 5
    shape_k = (batch, time, heads, key_dim)
    shape_v = (batch, time, heads, value_dim)
    state_shape = (batch, heads, key_dim, value_dim)
    dtype = torch.float64
    scale = 0.23
    chunk_size = 4
    base = (
        torch.nn.functional.normalize(
            torch.randn(shape_k, generator=generator, dtype=dtype), dim=-1
        ),
        torch.nn.functional.normalize(
            torch.randn(shape_k, generator=generator, dtype=dtype), dim=-1
        ),
        torch.randn(shape_v, generator=generator, dtype=dtype) * 0.2,
        -torch.rand(shape_k, generator=generator, dtype=dtype) * 0.03,
        torch.sigmoid(torch.randn(shape_k, generator=generator, dtype=dtype)) * 0.7,
        torch.sigmoid(torch.randn(shape_v, generator=generator, dtype=dtype)),
    )
    initial_state = torch.randn(state_shape, generator=generator, dtype=dtype) * 0.03
    do = torch.randn(shape_v, generator=generator, dtype=dtype) * 0.2
    d_final_state = torch.randn(state_shape, generator=generator, dtype=dtype) * 0.1

    differentiable = tuple(tensor.detach().requires_grad_() for tensor in base)
    differentiable_state = initial_state.detach().requires_grad_()
    output, final_state, boundaries, aux = _compact_forward(
        *differentiable,
        differentiable_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    expected = torch.autograd.grad(
        (output, final_state),
        (*differentiable, differentiable_state),
        (do, d_final_state),
    )
    detached_aux = SimpleNamespace(**{name: getattr(aux, name).detach() for name in vars(aux)})
    dstate_boundaries = _boundary_vjp(
        detached_aux,
        do,
        d_final_state,
        chunk_size=chunk_size,
    )
    actual, proof = compact_wy_chunk_vjp_reference(
        *base,
        boundaries.detach(),
        dstate_boundaries,
        detached_aux,
        do,
        scale=scale,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(
        proof.dstate_starts, dstate_boundaries[:, :-1], atol=2e-13, rtol=2e-13
    )
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, atol=2e-12, rtol=2e-12)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("time", [16, 19], ids=["full-chunk", "partial-tail"])
def test_compact_wy_cute_matches_reference(
    time: int,
    dtype: torch.dtype,
) -> None:
    base, do, aux, dstate_boundaries, _ = _cuda_compact_case(time, dtype)
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
    )
    # Exercise the TVM-FFI detach contract used by an autograd caller.
    differentiable = tuple(tensor.detach().requires_grad_() for tensor in base)
    actual = compact_wy_chunk_vjp_cute(
        *differentiable,
        aux.state_boundaries.detach().requires_grad_(),
        dstate_boundaries.detach().requires_grad_(),
        aux,
        do.detach().requires_grad_(),
        scale=0.125,
    )
    torch.cuda.synchronize()

    atol = 5e-4 if dtype == torch.float16 else 3e-3
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=atol, rtol=4e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=0.0, rtol=0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_compact_wy_chain_uses_saved_decay_end() -> None:
    base, do, aux, dstate_boundaries, _ = _cuda_compact_case(16, torch.float16)
    modified_aux = replace(aux, decay_end=aux.decay_end * 1.25)
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        modified_aux.state_boundaries,
        dstate_boundaries,
        modified_aux,
        do,
        scale=0.125,
        verify_boundaries=False,
    )
    actual = compact_wy_chunk_vjp_cute(
        *base,
        modified_aux.state_boundaries,
        dstate_boundaries,
        modified_aux,
        do,
        scale=0.125,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual[3], expected[3], atol=5e-4, rtol=4e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_compact_wy_rejects_misaligned_contiguous_input() -> None:
    base, do, aux, dstate_boundaries, _ = _cuda_compact_case(16, torch.float16)
    storage = torch.empty(base[0].numel() + 1, device="cuda", dtype=torch.float16)
    misaligned_q = storage[1:].view_as(base[0])
    misaligned_q.copy_(base[0])
    assert misaligned_q.is_contiguous() and misaligned_q.data_ptr() % 16 != 0

    with pytest.raises(ValueError, match="16-byte aligned"):
        compact_wy_chunk_vjp_cute(
            misaligned_q,
            *base[1:],
            aux.state_boundaries,
            dstate_boundaries,
            aux,
            do,
            scale=0.125,
        )


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("time", [128, 256, 512])
def test_compact_wy_long_normalized_keys_remain_stable(time: int) -> None:
    base, do, aux, dstate_boundaries, d_residual = _cuda_compact_case(time, torch.bfloat16)
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
    )
    actual = compact_wy_chunk_vjp_cute(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
        precomputed_d_residual=d_residual,
    )
    torch.cuda.synchronize()

    _assert_long_accuracy_contract(actual, expected)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_compact_wy_h16_t128_matches_reference() -> None:
    base, do, aux, dstate_boundaries, d_residual = _cuda_compact_case(
        128,
        torch.bfloat16,
        heads=16,
    )
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
    )
    actual = compact_wy_chunk_vjp_cute(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
        precomputed_d_residual=d_residual,
    )
    torch.cuda.synchronize()

    _assert_long_accuracy_contract(actual, expected)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_compact_wy_t2048_low_boundaries_match_reference_on_current_stream() -> None:
    generator = torch.Generator(device="cuda").manual_seed(4_170 + 2048)
    shape = (1, 2048, 1, 128)
    state_shape = (1, 1, 128, 128)
    dtype = torch.bfloat16
    base = (
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        (torch.randn(shape, generator=generator, device="cuda") * 0.15).to(dtype),
        -torch.rand(shape, generator=generator, device="cuda") * 0.02,
        (torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")) * 0.7).to(dtype),
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
    )
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.02
    )
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.12).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.05
    )
    _, _, aux = chunk_forward(
        *base,
        initial_state,
        scale=0.125,
        return_aux=True,
    )
    assert aux.state_boundaries.dtype == torch.bfloat16
    boundary_aux = WYBoundaryAux(
        aux.y,
        aux.q_gamma,
        aux.k_tail,
        aux.decay_end,
        aux.aqk,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dstate_boundaries, d_residual, exact_d_initial = wy_boundary_dstate(
            boundary_aux,
            do,
            d_final_state,
            return_d_residual=True,
            compact_boundaries=True,
        )
        actual = compact_wy_chunk_vjp_cute(
            *base,
            aux.state_boundaries,
            dstate_boundaries,
            aux,
            do,
            scale=0.125,
            precomputed_d_residual=d_residual,
            precomputed_d_initial_state=exact_d_initial,
        )
    stream.synchronize()

    assert dstate_boundaries.dtype == torch.bfloat16
    assert exact_d_initial.dtype == torch.float32
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
        verify_boundaries=False,
    )
    _assert_long_accuracy_contract(
        (*actual[:-1], exact_d_initial),
        (*expected[:-1], exact_d_initial),
    )
    torch.testing.assert_close(actual[-1], exact_d_initial, atol=0.0, rtol=0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_compact_wy_partial_tail_uses_current_stream() -> None:
    base, do, aux, dstate_boundaries, d_residual = _cuda_compact_case(19, torch.bfloat16)
    expected, _ = compact_wy_chunk_vjp_reference(
        *base,
        aux.state_boundaries,
        dstate_boundaries,
        aux,
        do,
        scale=0.125,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = compact_wy_chunk_vjp_cute(
            *base,
            aux.state_boundaries,
            dstate_boundaries,
            aux,
            do,
            scale=0.125,
            precomputed_d_residual=d_residual,
        )
    # Synchronizing only the caller-selected stream covers every staged launch
    # in the precomputed-residual specialization, including output.
    stream.synchronize()

    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=3e-3, rtol=4e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=0.0, rtol=0.0)

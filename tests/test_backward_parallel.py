from __future__ import annotations

import pytest
import torch

from gdn2_sm120.backward_parallel import (
    boundary_dstate_token_reference,
    boundary_dstate_wy_reference,
    build_wy_boundary_aux_reference,
    parallel_chunk_backward_cute,
    parallel_chunk_backward_reference,
    parallel_chunk_vjp,
    recurrent_forward_checkpoints_reference,
    wy_boundary_dstate,
)
from gdn2_sm120.chunk import chunk_forward
from gdn2_sm120.reference import recurrent_forward_reference


def _inputs(
    *,
    batch: int = 1,
    time: int = 11,
    heads: int = 2,
    key_dim: int = 7,
    value_dim: int = 5,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(20260825)
    shape_k = (batch, time, heads, key_dim)
    shape_v = (batch, time, heads, value_dim)
    state_shape = (batch, heads, key_dim, value_dim)
    q = torch.nn.functional.normalize(torch.randn(shape_k, generator=generator), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(shape_k, generator=generator), dim=-1)
    v = torch.randn(shape_v, generator=generator) * 0.2
    g = -torch.rand(shape_k, generator=generator) * 0.04
    beta = torch.sigmoid(torch.randn(shape_k, generator=generator)) * 0.7
    w = torch.sigmoid(torch.randn(shape_v, generator=generator))
    initial_state = torch.randn(state_shape, generator=generator) * 0.03
    do = torch.randn(shape_v, generator=generator) * 0.2
    d_final_state = torch.randn(state_shape, generator=generator) * 0.1
    return q, k, v, g, beta, w, initial_state, do, d_final_state


def test_forward_checkpoints_include_partial_final_chunk() -> None:
    q, k, v, g, beta, w, initial_state, *_ = _inputs(time=11)
    output, final_state, checkpoints = recurrent_forward_checkpoints_reference(
        q, k, v, g, beta, w, initial_state, scale=0.21, chunk_size=4
    )
    expected_output, expected_final_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state, scale=0.21
    )

    assert checkpoints.shape == (1, 4, 2, 7, 5)
    torch.testing.assert_close(checkpoints[:, 0], initial_state)
    torch.testing.assert_close(checkpoints[:, -1], final_state)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_final_state)


@pytest.mark.parametrize("time", [1, 8, 11], ids=["partial", "exact", "multiple"])
def test_wy_boundary_vjp_matches_token_scan(time: int) -> None:
    q, k, _, g, beta, _, _, do, d_final_state = _inputs(time=time)
    scale = 0.21
    chunk_size = 4
    aux = build_wy_boundary_aux_reference(q, k, g, beta, scale=scale, chunk_size=chunk_size)
    expected = boundary_dstate_token_reference(
        q,
        k,
        g,
        beta,
        do,
        d_final_state,
        scale=scale,
        chunk_size=chunk_size,
    )
    actual = boundary_dstate_wy_reference(aux, do, d_final_state, chunk_size=chunk_size)

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_independent_chunk_vjps_match_full_autograd() -> None:
    q, k, v, g, beta, w, initial_state, do, d_final_state = _inputs(time=11)
    base = (q, k, v, g, beta, w)
    differentiable = tuple(tensor.detach().requires_grad_() for tensor in base)
    differentiable_initial_state = initial_state.detach().requires_grad_()
    output, final_state = recurrent_forward_reference(
        *differentiable,
        differentiable_initial_state,
        scale=0.21,
    )
    assert final_state is not None
    expected = torch.autograd.grad(
        (output, final_state),
        (*differentiable, differentiable_initial_state),
        (do, d_final_state),
    )

    actual, proof = parallel_chunk_backward_reference(
        *base,
        do,
        d_final_state,
        initial_state,
        scale=0.21,
        chunk_size=4,
    )

    assert proof.state_boundaries.shape == (1, 4, 2, 7, 5)
    assert proof.dstate_boundaries.shape == (1, 4, 2, 7, 5)
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, atol=6e-5, rtol=6e-5)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("time", [16, 19], ids=["full-chunk", "partial-tail"])
def test_cute_wy_boundary_scan_matches_reference(time: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260825 + time)
    shape = (1, time, 1, 128)
    state_shape = (1, 1, 128, 128)
    dtype = torch.bfloat16
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    g = -torch.rand(shape, generator=generator, device="cuda") * 0.03
    beta = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype)
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )
    aux = build_wy_boundary_aux_reference(q, k, g, beta, scale=0.125, chunk_size=16)
    expected, expected_d_residual = boundary_dstate_wy_reference(
        aux,
        do,
        d_final_state,
        chunk_size=16,
        return_d_residual=True,
    )

    default_result = wy_boundary_dstate(aux, do, d_final_state)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual, d_residual = wy_boundary_dstate(
            aux,
            do,
            d_final_state,
            return_d_residual=True,
        )
    stream.synchronize()

    assert isinstance(default_result, torch.Tensor)
    torch.testing.assert_close(default_result, expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
    assert d_residual.shape == do.shape
    assert d_residual.dtype == torch.float32
    torch.testing.assert_close(d_residual, expected_d_residual, atol=3e-5, rtol=3e-4)

    if time == 16:
        storage = torch.empty(do.numel() + 1, device="cuda", dtype=do.dtype)
        misaligned_do = storage[1:].view_as(do)
        assert misaligned_do.data_ptr() % 16 != 0
        with pytest.raises(ValueError, match="16-byte aligned"):
            wy_boundary_dstate(aux, misaligned_do, d_final_state)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_cute_parallel_chunk_vjp_matches_proven_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(912_019)
    batch, time, heads, dim = 1, 19, 1, 128
    shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
    dtype = torch.bfloat16
    scale = 0.125
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    v = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    g = -torch.rand(shape, generator=generator, device="cuda") * 0.03
    beta = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype)
    w = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype)
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
    )
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )
    base = (q, k, v, g, beta, w)
    _, _, state_boundaries = recurrent_forward_checkpoints_reference(
        *base,
        initial_state,
        scale=scale,
        chunk_size=16,
    )
    aux = build_wy_boundary_aux_reference(q, k, g, beta, scale=scale, chunk_size=16)
    expected, proof = parallel_chunk_backward_reference(
        *base,
        do,
        d_final_state,
        initial_state,
        scale=scale,
        chunk_size=16,
        state_boundaries=state_boundaries,
    )

    actual, dstate_boundaries = parallel_chunk_backward_cute(
        *base,
        state_boundaries,
        aux,
        do,
        d_final_state,
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(dstate_boundaries, proof.dstate_boundaries, atol=2e-5, rtol=2e-4)
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=2e-5, rtol=2e-4)

    storage = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)
    misaligned_q = storage[1:].view_as(q)
    assert misaligned_q.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte aligned"):
        parallel_chunk_vjp(
            misaligned_q,
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


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_cute_mma_boundary_scan_matches_reference(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(812_800)
    shape = (1, 128, 1, 128)
    state_shape = (1, 1, 128, 128)
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    g = -torch.rand(shape, generator=generator, device="cuda") * 0.03
    beta = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype)
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )
    aux = build_wy_boundary_aux_reference(q, k, g, beta, scale=0.125, chunk_size=16)
    expected, expected_d_residual = boundary_dstate_wy_reference(
        aux,
        do,
        d_final_state,
        chunk_size=16,
        return_d_residual=True,
    )

    default_result = wy_boundary_dstate(aux, do, d_final_state)
    actual, d_residual = wy_boundary_dstate(
        aux,
        do,
        d_final_state,
        return_d_residual=True,
    )
    torch.cuda.synchronize()

    atol = 4e-4 if dtype == torch.bfloat16 else 5e-5
    assert isinstance(default_result, torch.Tensor)
    torch.testing.assert_close(default_result, expected, atol=atol, rtol=2e-3)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-3)
    residual_atol = 1e-3 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(d_residual, expected_d_residual, atol=residual_atol, rtol=3e-3)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_partial_tail_requires_grad_and_uses_current_stream() -> None:
    """T=129 covers scalar boundary fallback and independent final chunk."""

    generator = torch.Generator(device="cuda").manual_seed(129_129)
    batch, time, heads, dim = 1, 129, 1, 128
    shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
    dtype = torch.bfloat16
    scale = 0.125
    base = (
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        torch.nn.functional.normalize(
            torch.randn(shape, generator=generator, device="cuda"), dim=-1
        ).to(dtype),
        (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype),
        -torch.rand(shape, generator=generator, device="cuda") * 0.03,
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
    )
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
    )
    do = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )
    differentiable = tuple(tensor.detach().requires_grad_() for tensor in base)
    _, _, aux = chunk_forward(
        *differentiable,
        initial_state.requires_grad_(),
        scale=scale,
        return_aux=True,
    )
    torch.cuda.synchronize()
    dstate_reference = boundary_dstate_wy_reference(aux, do, d_final_state, chunk_size=16)
    expected, _ = parallel_chunk_backward_reference(
        *(tensor.detach() for tensor in base),
        do,
        d_final_state,
        initial_state.detach(),
        scale=scale,
        chunk_size=16,
        state_boundaries=aux.state_boundaries,
        dstate_boundaries=dstate_reference,
        verify_boundaries=False,
    )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual, dstate_boundaries = parallel_chunk_backward_cute(
            *differentiable,
            aux.state_boundaries,
            aux,
            do,
            d_final_state,
            scale=scale,
        )
    stream.synchronize()

    torch.testing.assert_close(dstate_boundaries, dstate_reference, atol=2e-5, rtol=2e-4)
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=3e-5, rtol=3e-4)

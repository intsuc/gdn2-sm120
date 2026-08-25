from __future__ import annotations

import pytest
import torch

from gdn2_sm120.backward_parallel import (
    _select_boundary_mma_value_tile,
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


def test_boundary_mma_value_tile_is_shape_dependent() -> None:
    assert _select_boundary_mma_value_tile(1, 512, 16) == 16
    assert _select_boundary_mma_value_tile(1, 768, 16) == 8
    assert _select_boundary_mma_value_tile(1, 2048, 16) == 8
    assert _select_boundary_mma_value_tile(1, 4096, 16) == 8
    assert _select_boundary_mma_value_tile(1, 32768, 23) == 8
    assert _select_boundary_mma_value_tile(1, 32768, 24) == 16
    assert _select_boundary_mma_value_tile(2, 32768, 16) == 16
    assert _select_boundary_mma_value_tile(4, 16384, 16) == 16


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
    expected_zero_terminal = boundary_dstate_wy_reference(
        aux,
        do,
        torch.zeros_like(d_final_state),
        chunk_size=16,
    )

    default_result = wy_boundary_dstate(aux, do, d_final_state)
    zero_terminal_result = wy_boundary_dstate(aux, do, None)
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
    torch.testing.assert_close(zero_terminal_result, expected_zero_terminal, atol=2e-5, rtol=2e-4)
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
@pytest.mark.parametrize(
    ("time", "compact_aux"),
    [(64, False), (128, False), (128, True)],
    ids=["t64-fp32-aux", "t128-fp32-aux", "t128-compact-aux"],
)
def test_cute_mma_boundary_scan_matches_reference(
    time: int,
    dtype: torch.dtype,
    compact_aux: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(812_800 + time)
    shape = (1, time, 1, 128)
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
    if compact_aux:
        v = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
        w = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype)
        initial_state = (
            torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
        )
        _, _, aux = chunk_forward(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            scale=0.125,
            return_aux=True,
        )
        assert aux.y.dtype == dtype
    else:
        aux = build_wy_boundary_aux_reference(q, k, g, beta, scale=0.125, chunk_size=16)
    expected, expected_d_residual = boundary_dstate_wy_reference(
        aux,
        do,
        d_final_state,
        chunk_size=16,
        return_d_residual=True,
    )
    expected_zero_terminal = boundary_dstate_wy_reference(
        aux,
        do,
        torch.zeros_like(d_final_state),
        chunk_size=16,
    )

    default_result = wy_boundary_dstate(aux, do, d_final_state)
    zero_terminal_result = wy_boundary_dstate(aux, do, None)
    if time == 64:
        with pytest.raises(ValueError, match="compact boundaries require T >= 128"):
            wy_boundary_dstate(
                aux,
                do,
                d_final_state,
                compact_boundaries=True,
            )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual, d_residual = wy_boundary_dstate(
            aux,
            do,
            d_final_state,
            return_d_residual=True,
        )
    stream.synchronize()

    atol = 4e-4 if dtype == torch.bfloat16 else 5e-5
    assert isinstance(default_result, torch.Tensor)
    torch.testing.assert_close(default_result, expected, atol=atol, rtol=2e-3)
    torch.testing.assert_close(
        zero_terminal_result,
        expected_zero_terminal,
        atol=atol,
        rtol=2e-3,
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=2e-3)
    residual_atol = 1e-3 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(d_residual, expected_d_residual, atol=residual_atol, rtol=3e-3)

    if dtype == torch.bfloat16 and compact_aux:
        with torch.cuda.stream(stream):
            compact, compact_residual, exact_d_initial = wy_boundary_dstate(
                aux,
                do,
                d_final_state,
                return_d_residual=True,
                compact_boundaries=True,
            )
        stream.synchronize()
        assert compact.dtype == dtype
        assert compact_residual.dtype == dtype
        assert exact_d_initial.dtype == torch.float32
        torch.testing.assert_close(exact_d_initial, actual[:, 0], atol=0, rtol=0)
        torch.testing.assert_close(compact.float(), actual, atol=2e-2, rtol=3e-2)
        torch.testing.assert_close(
            compact_residual.float(),
            expected_d_residual,
            atol=4e-3,
            rtol=3e-3,
        )
        compact_zero, compact_zero_residual, exact_zero_d_initial = wy_boundary_dstate(
            aux,
            do,
            None,
            return_d_residual=True,
            compact_boundaries=True,
        )
        compact_zero_only, exact_zero_only_d_initial = wy_boundary_dstate(
            aux,
            do,
            None,
            compact_boundaries=True,
        )
        torch.testing.assert_close(
            compact_zero.float(),
            zero_terminal_result,
            atol=2e-2,
            rtol=3e-2,
        )
        torch.testing.assert_close(
            exact_zero_d_initial,
            zero_terminal_result[:, 0],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(compact_zero_only, compact_zero, atol=0, rtol=0)
        torch.testing.assert_close(
            exact_zero_only_d_initial,
            exact_zero_d_initial,
            atol=0,
            rtol=0,
        )
        assert compact_zero_residual.dtype == dtype


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize(
    ("batch", "time", "dtype"),
    [
        (1, 128, torch.bfloat16),
        (1, 128, torch.float16),
        (1, 512, torch.bfloat16),
        (2, 64, torch.bfloat16),
        (2, 128, torch.bfloat16),
        (2, 512, torch.bfloat16),
        (4, 64, torch.bfloat16),
        (4, 128, torch.bfloat16),
        (4, 512, torch.bfloat16),
    ],
    ids=[
        "b1-t128-bf16",
        "b1-t128-fp16",
        "b1-t512",
        "b2-t64",
        "b2-t128",
        "b2-t512",
        "b4-t64",
        "b4-t128",
        "b4-t512",
    ],
)
def test_cute_wide_mma_boundary_scan_matches_reference(
    batch: int,
    time: int,
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(32_000 + batch * 1_000 + time)
    heads, dim = 16, 128
    shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
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
    _, _, aux = chunk_forward(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        scale=0.125,
        return_aux=True,
    )
    expected, expected_d_residual = boundary_dstate_wy_reference(
        aux,
        do,
        d_final_state,
        chunk_size=16,
        return_d_residual=True,
    )

    actual, d_residual = wy_boundary_dstate(
        aux,
        do,
        d_final_state,
        return_d_residual=True,
    )
    torch.cuda.synchronize()

    assert _select_boundary_mma_value_tile(batch, time, heads) == 16
    torch.testing.assert_close(actual, expected, atol=4e-4, rtol=2e-3)
    torch.testing.assert_close(d_residual, expected_d_residual, atol=1e-3, rtol=3e-3)
    if dtype == torch.bfloat16 and time >= 128:
        compact, compact_residual, exact_d_initial = wy_boundary_dstate(
            aux,
            do,
            d_final_state,
            return_d_residual=True,
            compact_boundaries=True,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(compact.float(), actual, atol=2e-2, rtol=3e-2)
        torch.testing.assert_close(
            compact_residual.float(), expected_d_residual, atol=4e-3, rtol=3e-3
        )
        torch.testing.assert_close(exact_d_initial, actual[:, 0], atol=0, rtol=0)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_partial_tail_requires_grad_and_uses_current_stream() -> None:
    """T=129 covers scalar-tail plus MMA-prefix reverse scan ordering."""

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
    zero_terminal_reference = boundary_dstate_wy_reference(
        aux,
        do,
        torch.zeros_like(d_final_state),
        chunk_size=16,
    )
    zero_terminal_actual = wy_boundary_dstate(aux, do, None)
    torch.testing.assert_close(
        zero_terminal_actual,
        zero_terminal_reference,
        atol=4e-4,
        rtol=2e-3,
    )
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

    torch.testing.assert_close(dstate_boundaries, dstate_reference, atol=4e-4, rtol=2e-3)
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=1.5e-2, rtol=2e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=1e-4, rtol=2e-3)

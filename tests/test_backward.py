from __future__ import annotations

import pytest
import torch

from gdn2_sm120.backward import chunk_backward
from gdn2_sm120.reference import chunkwise_forward_reference


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def test_backward_rejects_non_128_dimensions_before_launch() -> None:
    sequence = torch.zeros(1, 1, 1, 16, dtype=torch.float16)
    state = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
    with pytest.raises(ValueError, match="K == V == 128"):
        chunk_backward(
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            state,
            sequence,
            state,
            scale=0.25,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_backward_rejects_mismatched_cute_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (1, 1, 1, 128)
    sequence = torch.zeros(shape, device="cuda", dtype=torch.float16)
    state = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.float32)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100")

    with pytest.raises(RuntimeError, match="requires CUTE_DSL_ARCH=sm_120"):
        chunk_backward(
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            sequence,
            state,
            sequence,
            state,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize(
    ("dtype", "gate_dtype"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("time", [1, 3], ids=["fused-t1", "value-tiled"])
def test_chunk_backward_matches_reference_autograd(
    dtype: torch.dtype, gate_dtype: torch.dtype, time: int
) -> None:
    torch.manual_seed(20260824)
    batch, heads, dim = 1, 2, 128
    sequence_shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
    device = torch.device("cuda")
    scale = 0.19

    # Small keys keep 1-r^T k comfortably away from zero, which is the
    # invertibility condition of the exact reverse reconstruction.
    base = [
        torch.randn(sequence_shape, device=device, dtype=dtype) * 0.2,
        torch.randn(sequence_shape, device=device, dtype=dtype) * 0.015,
        torch.randn(sequence_shape, device=device, dtype=dtype) * 0.2,
        -torch.rand(sequence_shape, device=device, dtype=gate_dtype) * 0.03,
        torch.sigmoid(torch.randn(sequence_shape, device=device, dtype=dtype)),
        torch.sigmoid(torch.randn(sequence_shape, device=device, dtype=dtype)),
    ]
    initial_state = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.03
    do = torch.randn(sequence_shape, device=device, dtype=dtype) * 0.2
    d_final_state = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.1

    differentiable = [tensor.detach().requires_grad_() for tensor in base]
    differentiable_initial_state = initial_state.detach().requires_grad_()
    output, final_state = chunkwise_forward_reference(
        *differentiable,
        differentiable_initial_state,
        scale=scale,
        chunk_size=2,
    )
    assert final_state is not None
    loss = (output * do).sum() + (final_state * d_final_state).sum()
    expected = torch.autograd.grad(
        loss,
        (*differentiable, differentiable_initial_state),
    )

    actual = chunk_backward(
        *base,
        final_state.detach().contiguous(),
        do,
        d_final_state,
        scale,
    )
    torch.cuda.synchronize()

    assert [gradient.dtype for gradient in actual[:-1]] == [
        dtype,
        dtype,
        dtype,
        gate_dtype,
        dtype,
        dtype,
    ]
    assert actual[-1].dtype == torch.float32
    sequence_atol = 2e-3 if dtype == torch.float16 else 1.5e-2
    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=sequence_atol, rtol=0.02)
    torch.testing.assert_close(actual[-1], expected[-1], atol=2e-5, rtol=2e-4)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_normalized_key_t128_backward_matches_reference() -> None:
    """Production-normalized keys remain accurate at the longest safe length."""

    generator = torch.Generator(device="cuda").manual_seed(512128)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch, time, heads, dim = 1, 128, 1, 128
    sequence_shape = (batch, time, heads, dim)
    state_shape = (batch, heads, dim, dim)
    scale = 0.125

    q = torch.nn.functional.normalize(
        torch.randn(sequence_shape, generator=generator, device=device), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(sequence_shape, generator=generator, device=device), dim=-1
    ).to(dtype)
    base = [
        q,
        k,
        torch.randn(sequence_shape, generator=generator, device=device, dtype=dtype) * 0.2,
        -torch.rand(sequence_shape, generator=generator, device=device) * 0.03,
        torch.sigmoid(torch.randn(sequence_shape, generator=generator, device=device, dtype=dtype)),
        torch.sigmoid(torch.randn(sequence_shape, generator=generator, device=device, dtype=dtype)),
    ]
    initial_state = (
        torch.randn(state_shape, generator=generator, device=device, dtype=torch.float32) * 0.03
    )
    do = torch.randn(sequence_shape, generator=generator, device=device, dtype=dtype) * 0.2
    d_final_state = (
        torch.randn(state_shape, generator=generator, device=device, dtype=torch.float32) * 0.1
    )

    differentiable = [tensor.detach().requires_grad_() for tensor in base]
    differentiable_initial_state = initial_state.detach().requires_grad_()
    output, final_state = chunkwise_forward_reference(
        *differentiable,
        differentiable_initial_state,
        scale=scale,
        chunk_size=64,
    )
    assert final_state is not None
    expected = torch.autograd.grad(
        (output, final_state),
        (*differentiable, differentiable_initial_state),
        (do, d_final_state),
    )
    actual = chunk_backward(
        *base,
        final_state.detach().contiguous(),
        do,
        d_final_state,
        scale,
    )
    torch.cuda.synchronize()

    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=1.5e-2, rtol=0.02)
    torch.testing.assert_close(actual[-1], expected[-1], atol=2e-5, rtol=2e-4)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("time", [129, 256, 512])
def test_long_normalized_key_backward_requires_checkpoints(time: int) -> None:
    device = torch.device("cuda")
    dtype = torch.bfloat16
    shape = (1, time, 1, 128)
    state_shape = (1, 1, 128, 128)
    sequence = torch.zeros(shape, device=device, dtype=dtype)
    key = torch.zeros_like(sequence)
    key[..., 0] = 1.0
    gate = torch.full(shape, -0.015, device=device, dtype=torch.float32)
    beta = torch.full(shape, 0.5, device=device, dtype=dtype)
    state = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.03

    with pytest.raises(RuntimeError, match="at most 128 tokens"):
        chunk_backward(
            sequence,
            key,
            sequence,
            gate,
            beta,
            sequence,
            state,
            sequence,
            state,
            scale=0.125,
        )


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_value_tiled_backward_uses_current_stream() -> None:
    """The multi-token main and reduction launches must use the caller's stream."""

    device = torch.device("cuda")
    batch, time, heads, dim = 1, 2, 1, 128
    shape = (batch, time, heads, dim)
    sequence = torch.zeros(shape, device=device, dtype=torch.bfloat16)
    gate = torch.zeros(shape, device=device, dtype=torch.float32)
    do = torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.1
    state = torch.randn((batch, heads, dim, dim), device=device, dtype=torch.float32) * 0.03
    d_final_state = torch.randn_like(state) * 0.1
    scale = 0.125

    expected_dq = (scale * torch.einsum("bhkv,bthv->bthk", state, do.float())).to(torch.bfloat16)
    expected_dg = torch.einsum("bhkv,bhkv->bhk", d_final_state, state)
    expected_dg = expected_dg[:, None].expand(shape).contiguous()

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        actual = chunk_backward(
            sequence,
            sequence,
            sequence,
            gate,
            sequence,
            sequence,
            state,
            do,
            d_final_state,
            scale,
        )
    # Synchronize only the caller-selected stream: this catches either tiled
    # launch accidentally escaping to the default stream.
    stream.synchronize()

    torch.testing.assert_close(actual[0], expected_dq, atol=2e-3, rtol=2e-2)
    for gradient in (actual[1], actual[2], actual[4], actual[5]):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))
    torch.testing.assert_close(actual[3], expected_dg, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(actual[6], d_final_state, atol=0.0, rtol=0.0)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_zero_length_backward_copies_terminal_state_gradient() -> None:
    device = torch.device("cuda")
    sequence = torch.empty(1, 0, 1, 128, device=device, dtype=torch.float16)
    state = torch.randn(1, 1, 128, 128, device=device, dtype=torch.float32)
    gradients = chunk_backward(
        sequence,
        sequence,
        sequence,
        sequence,
        sequence,
        sequence,
        state,
        sequence,
        state,
        scale=0.25,
    )
    torch.cuda.synchronize()

    for gradient in gradients[:-1]:
        assert gradient.shape == sequence.shape
    torch.testing.assert_close(gradients[-1], state, atol=0.0, rtol=0.0)

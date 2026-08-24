from __future__ import annotations

import pytest
import torch

from gdn2_sm120.ops import chunk_gdn2
from gdn2_sm120.reference import chunkwise_forward_reference


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def test_chunk_gdn2_rejects_long_differentiable_sequence_before_cuda() -> None:
    sequence = torch.zeros(1, 129, 1, 128, dtype=torch.bfloat16, requires_grad=True)
    with pytest.raises(NotImplementedError, match="at most 128 tokens"):
        chunk_gdn2(sequence, sequence, sequence, sequence, sequence, sequence)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_chunk_gdn2_autograd_matches_reference() -> None:
    torch.manual_seed(2605)
    device = torch.device("cuda")
    shape = (1, 3, 1, 128)
    state_shape = (1, 1, 128, 128)
    dtype = torch.bfloat16
    scale = 0.125

    base = [
        torch.randn(shape, device=device, dtype=dtype) * 0.2,
        torch.randn(shape, device=device, dtype=dtype) * 0.015,
        torch.randn(shape, device=device, dtype=dtype) * 0.2,
        -torch.rand(shape, device=device, dtype=torch.float32) * 0.03,
        torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)),
        torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)),
    ]
    initial_state = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.03
    d_output = torch.randn(shape, device=device, dtype=dtype) * 0.2
    d_final_state = torch.randn(state_shape, device=device, dtype=torch.float32) * 0.1

    actual_inputs = [tensor.detach().requires_grad_() for tensor in base]
    actual_initial = initial_state.detach().requires_grad_()
    output, final_state = chunk_gdn2(
        *actual_inputs,
        actual_initial,
        scale=scale,
        output_final_state=True,
    )
    assert final_state is not None
    actual = torch.autograd.grad(
        (output, final_state),
        (*actual_inputs, actual_initial),
        (d_output, d_final_state),
    )

    expected_inputs = [tensor.detach().requires_grad_() for tensor in base]
    expected_initial = initial_state.detach().requires_grad_()
    expected_output, expected_final = chunkwise_forward_reference(
        *expected_inputs,
        expected_initial,
        scale=scale,
        chunk_size=16,
    )
    assert expected_final is not None
    expected = torch.autograd.grad(
        (expected_output, expected_final),
        (*expected_inputs, expected_initial),
        (d_output, d_final_state),
    )

    for gradient, reference in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(gradient, reference, atol=1.5e-2, rtol=0.02)
    torch.testing.assert_close(actual[-1], expected[-1], atol=2e-5, rtol=2e-4)


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_chunk_gdn2_can_hide_final_state() -> None:
    shape = (1, 1, 1, 128)
    tensors = [torch.zeros(shape, device="cuda", dtype=torch.float16) for _ in range(6)]
    tensors[3].fill_(-0.01)
    tensors = [tensor.requires_grad_() for tensor in tensors]
    output, final_state = chunk_gdn2(*tensors)
    output.float().sum().backward()

    assert output.shape == shape
    assert final_state is None
    assert all(tensor.grad is not None for tensor in tensors)

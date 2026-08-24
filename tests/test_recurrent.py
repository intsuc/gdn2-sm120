from __future__ import annotations

import pytest
import torch

from gdn2_sm120.recurrent import token_forward
from gdn2_sm120.reference import recurrent_forward_reference

pytestmark = pytest.mark.cuda


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("g_fp32", [False, True])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_token_forward_matches_reference(
    dtype: torch.dtype, g_fp32: bool, with_initial_state: bool
) -> None:
    torch.manual_seed(1234)
    device = torch.device("cuda")
    shape = (1, 5, 2, 128)
    q = (torch.randn(shape, device=device) * 0.1).to(dtype)
    k = (torch.randn(shape, device=device) * 0.1).to(dtype)
    v = (torch.randn(shape, device=device) * 0.2).to(dtype)
    g = -torch.rand(shape, device=device) * 0.08
    if not g_fp32:
        g = g.to(dtype)
    beta = torch.sigmoid(torch.randn(shape, device=device)).to(dtype)
    w = torch.sigmoid(torch.randn(shape, device=device)).to(dtype)
    initial_state = (
        torch.randn((1, 2, 128, 128), device=device, dtype=torch.float32) * 0.1
        if with_initial_state
        else None
    )

    expected_output, expected_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state, scale=0.125
    )
    output, final_state = token_forward(q, k, v, g, beta, w, initial_state, scale=0.125)

    assert expected_state is not None
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_empty_sequence_copies_initial_state() -> None:
    device = torch.device("cuda")
    shape = (1, 0, 1, 128)
    inputs = [torch.empty(shape, device=device, dtype=torch.bfloat16) for _ in range(6)]
    initial_state = torch.randn((1, 1, 128, 128), device=device, dtype=torch.float32)

    output, final_state = token_forward(*inputs, initial_state)

    assert output.shape == shape
    torch.testing.assert_close(final_state, initial_state, rtol=0, atol=0)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_uses_current_stream() -> None:
    torch.manual_seed(99)
    device = torch.device("cuda")
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        shape = (1, 3, 1, 128)
        q = (torch.randn(shape, device=device) * 0.1).bfloat16()
        k = (torch.randn(shape, device=device) * 0.1).bfloat16()
        v = (torch.randn(shape, device=device) * 0.1).bfloat16()
        g = -torch.rand(shape, device=device, dtype=torch.float32) * 0.05
        beta = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
        w = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
        initial_state = torch.randn((1, 1, 128, 128), device=device, dtype=torch.float32)
        expected_output, expected_state = recurrent_forward_reference(
            q, k, v, g, beta, w, initial_state
        )
        output, final_state = token_forward(q, k, v, g, beta, w, initial_state)
        finished = stream.record_event()

    torch.cuda.current_stream(device).wait_event(finished)
    assert expected_state is not None
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


def test_token_forward_rejects_unsupported_dimensions() -> None:
    x = torch.empty((1, 1, 1, 64), dtype=torch.float16)
    with pytest.raises(ValueError, match="K == V == 128"):
        token_forward(x, x, x, x, x, x)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_rejects_incompatible_cute_target(monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (1, 1, 1, 128)
    x = torch.zeros(shape, device="cuda", dtype=torch.float16)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100")
    with pytest.raises(RuntimeError, match="requires CUTE_DSL_ARCH=sm_120"):
        token_forward(x, x, x, x, x, x)

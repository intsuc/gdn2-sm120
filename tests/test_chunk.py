from __future__ import annotations

import pytest
import torch

from gdn2_sm120.chunk import chunk_forward
from gdn2_sm120.reference import chunkwise_forward_reference


def _inputs(
    time: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    batch: int = 1,
    heads: int = 2,
    normalized_qk: bool = False,
):
    torch.manual_seed(2026 + time)
    shape = (batch, time, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype) * 0.25
    k = torch.randn(shape, device="cuda", dtype=dtype) * 0.05
    if normalized_qk:
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype) * 0.25
    g = -torch.rand(shape, device="cuda", dtype=torch.float32) * 0.03
    beta = torch.sigmoid(torch.randn(shape, device="cuda", dtype=dtype))
    w = torch.sigmoid(torch.randn(shape, device="cuda", dtype=dtype))
    state = (
        torch.randn(
            batch,
            heads,
            128,
            128,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.01
    )
    return q, k, v, g, beta, w, state


@pytest.mark.cuda
@pytest.mark.parametrize("time", [1, 16, 19])
def test_chunk_forward_matches_wy_reference(time: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time)
    output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=2e-4, rtol=2e-3)


@pytest.mark.cuda
def test_chunk_forward_without_initial_state() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(7)
    output, final_state = chunk_forward(*args[:6])
    expected_output, expected_state = chunkwise_forward_reference(*args[:6], chunk_size=16)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=2e-4, rtol=2e-3)


@pytest.mark.cuda
def test_chunk_forward_rejects_invalid_static_inputs() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(1)

    with pytest.raises(ValueError, match="rank-4"):
        chunk_forward(args[0][0], *args[1:6])

    empty_shape = (0, 1, 1, 128)
    empty_inputs = [
        torch.empty(
            empty_shape,
            device="cuda",
            dtype=torch.float32 if index == 3 else torch.bfloat16,
        )
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="must be positive"):
        chunk_forward(*empty_inputs)

    for scale in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="scale must be finite"):
            chunk_forward(*args[:6], args[6], scale=scale)


@pytest.mark.cuda
def test_chunk_forward_rejects_incompatible_dsl_arch(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(1)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100")
    with pytest.raises(RuntimeError, match="CUTE_DSL_ARCH must be sm_120"):
        chunk_forward(*args[:6], args[6])


@pytest.mark.cuda
def test_chunk_forward_fp16_normalized_qk_batch_two() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(
        19,
        torch.float16,
        batch=2,
        heads=1,
        normalized_qk=True,
    )
    output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=2e-4, rtol=2e-3)


@pytest.mark.cuda
def test_chunk_forward_uses_current_stream() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(19)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    stream.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=2e-4, rtol=2e-3)

from __future__ import annotations

import pytest
import torch

from gdn2_sm120 import recurrent as recurrent_module
from gdn2_sm120.ops import recurrent_gdn2
from gdn2_sm120.recurrent import token_forward
from gdn2_sm120.reference import recurrent_forward_reference

pytestmark = pytest.mark.cuda


def _cuda_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _t1_zero_closed_form(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_t = q[:, 0].float()
    k_t = k[:, 0].float()
    update = w[:, 0].float() * v[:, 0].float()
    state = k_t.unsqueeze(-1) * update.unsqueeze(-2)
    qk = (q_t * scale * k_t).sum(dim=-1, keepdim=True)
    return (qk * update).unsqueeze(1).to(q.dtype), state


def test_t1_zero_closed_form_matches_recurrence_on_cpu() -> None:
    torch.manual_seed(611)
    shape = (2, 1, 3, 128)
    q, k, v, beta, w = [(torch.randn(shape) * 0.1).bfloat16() for _ in range(5)]
    g = -torch.rand(shape) * 0.07
    expected_output, expected_state = recurrent_forward_reference(q, k, v, g, beta, w, scale=0.0625)
    closed_output, closed_state = _t1_zero_closed_form(q, k, v, w, 0.0625)

    assert expected_state is not None
    torch.testing.assert_close(closed_output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(closed_state, expected_state, rtol=1e-5, atol=1e-5)


def test_empty_byte_span_does_not_overlap() -> None:
    assert not recurrent_module._spans_overlap((128, 128), (0, 256))
    assert not recurrent_module._spans_overlap((0, 256), (128, 128))


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_supports_offset_singleton_slices() -> None:
    torch.manual_seed(831)
    device = torch.device("cuda")

    def token_slice(dtype: torch.dtype) -> torch.Tensor:
        backing = torch.randn((1, 8, 1, 128), device=device, dtype=dtype)
        return backing[:, 4:5]

    q, k, v, beta, w = [token_slice(torch.bfloat16) for _ in range(5)]
    g = -token_slice(torch.float32).abs() * 0.04
    initial_backing = torch.randn((1, 3, 128, 128), device=device) * 0.1
    initial_state = initial_backing[:, 1:2]
    output_backing = torch.empty((1, 8, 1, 128), device=device, dtype=q.dtype)
    output_buffer = output_backing[:, 4:5]
    final_state_backing = torch.empty((1, 3, 128, 128), device=device)
    final_state_buffer = final_state_backing[:, 1:2]

    tensors = (q, k, v, g, beta, w, initial_state, output_buffer, final_state_buffer)
    assert all(tensor.is_contiguous() for tensor in tensors)
    assert any(
        tensor.stride() != recurrent_module._canonical_strides(tensor.shape) for tensor in tensors
    )

    expected_output, expected_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state, scale=0.0625
    )
    output, final_state = recurrent_gdn2(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        scale=0.0625,
        out=output_buffer,
        final_state_out=final_state_buffer,
    )

    assert expected_state is not None
    assert output is output_buffer
    assert final_state is final_state_buffer
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_supports_arbitrary_singleton_strides() -> None:
    torch.manual_seed(919)
    device = torch.device("cuda")
    shape = (2, 1, 3, 128)
    strides = (384, 999, 128, 1)

    def strided_random(dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.empty_strided(shape, strides, device=device, dtype=dtype)
        tensor.copy_(torch.randn(shape, device=device).to(dtype) * 0.1)
        return tensor

    q, k, v, beta, w = [strided_random(torch.bfloat16) for _ in range(5)]
    g = torch.empty_strided(shape, strides, device=device, dtype=torch.float32)
    g.copy_(-torch.rand(shape, device=device) * 0.05)
    output_buffer = torch.empty_strided(shape, strides, device=device, dtype=q.dtype)
    initial_state = torch.randn((2, 3, 128, 128), device=device) * 0.1

    assert q.is_contiguous()
    assert q.stride() == strides
    expected_output, expected_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state
    )
    output, final_state = token_forward(q, k, v, g, beta, w, initial_state, out=output_buffer)

    assert expected_state is not None
    assert output is output_buffer
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_token_forward_t1_zero_uses_closed_form_on_current_stream(dtype: torch.dtype) -> None:
    torch.manual_seed(271)
    device = torch.device("cuda")
    shape = (2, 1, 3, 128)
    q, k, v, beta, w = [(torch.randn(shape, device=device) * 0.1).to(dtype) for _ in range(5)]
    g_low = (-torch.rand(shape, device=device) * 0.05).to(dtype)
    expected_output, expected_state = _t1_zero_closed_form(q, k, v, w, 0.0625)

    output, final_state = token_forward(q, k, v, g_low, beta, w, scale=0.0625)
    launcher_keys = set(recurrent_module._COMPILED_T1_ZERO_LAUNCHERS)

    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    g_fp32 = -torch.rand(shape, device=device) * 0.11
    beta_other = torch.randn(shape, device=device).to(dtype)
    with torch.cuda.stream(stream):
        output_other, state_other = token_forward(q, k, v, g_fp32, beta_other, w, scale=0.0625)
        finished = stream.record_event()
    torch.cuda.current_stream(device).wait_event(finished)
    scaled_output, scaled_state = token_forward(q, k, v, g_low, beta, w, scale=0.0375)
    expected_scaled_output, _ = _t1_zero_closed_form(q, k, v, w, 0.0375)

    assert set(recurrent_module._COMPILED_T1_ZERO_LAUNCHERS) == launcher_keys
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_other, output, rtol=0, atol=0)
    torch.testing.assert_close(state_other, final_state, rtol=0, atol=0)
    torch.testing.assert_close(scaled_output, expected_scaled_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(scaled_state, final_state, rtol=0, atol=0)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_t1_zero_supports_unaligned_output_buffers() -> None:
    torch.manual_seed(613)
    device = torch.device("cuda")
    shape = (1, 1, 1, 128)
    q, k, v, beta, w = [(torch.randn(shape, device=device) * 0.1).bfloat16() for _ in range(5)]
    g = -torch.rand(shape, device=device) * 0.05
    expected_output, expected_state = _t1_zero_closed_form(q, k, v, w, 0.0625)
    output_backing = torch.empty(q.numel() + 1, device=device, dtype=q.dtype)
    output_buffer = output_backing[1:].view_as(q)
    state_backing = torch.empty(128 * 128 + 1, device=device)
    state_buffer = state_backing[1:].view(1, 1, 128, 128)

    output, final_state = token_forward(
        q,
        k,
        v,
        g,
        beta,
        w,
        scale=0.0625,
        out=output_buffer,
        final_state_out=state_buffer,
    )

    assert output is output_buffer
    assert final_state is state_buffer
    assert output.data_ptr() % 16 != 0
    assert final_state.data_ptr() % 16 != 0
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


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
    output_buffer = torch.empty_like(inputs[0])
    final_state_buffer = torch.empty_like(initial_state)

    output, final_state = recurrent_gdn2(
        *inputs,
        initial_state,
        out=output_buffer,
        final_state_out=final_state_buffer,
    )

    assert output is output_buffer
    assert final_state is final_state_buffer
    assert output.shape == shape
    torch.testing.assert_close(final_state, initial_state, rtol=0, atol=0)

    zero_state_buffer = torch.empty_like(initial_state)
    output_without_state, zero_state = recurrent_gdn2(
        *inputs,
        out=torch.empty_like(inputs[0]),
        final_state_out=zero_state_buffer,
    )
    assert output_without_state.shape == shape
    assert zero_state is zero_state_buffer
    torch.testing.assert_close(zero_state, torch.zeros_like(zero_state), rtol=0, atol=0)


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


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("time", [3, 64], ids=["recurrent", "chunk-fallback"])
def test_token_forward_supports_unaligned_contiguous_initial_state(time: int) -> None:
    torch.manual_seed(2026)
    device = torch.device("cuda")
    shape = (1, time, 1, 128)
    q = (torch.randn(shape, device=device) * 0.1).bfloat16()
    k = (torch.randn(shape, device=device) * 0.1).bfloat16()
    v = (torch.randn(shape, device=device) * 0.1).bfloat16()
    g = -torch.rand(shape, device=device, dtype=torch.float32) * 0.05
    beta = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
    w = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
    backing = torch.randn(128 * 128 + 1, device=device, dtype=torch.float32) * 0.1
    initial_state = backing[1:].view(1, 1, 128, 128)
    assert initial_state.is_contiguous()
    assert initial_state.data_ptr() % 16 != 0

    expected_output, expected_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state
    )
    output_backing = torch.empty(q.numel() + 1, device=device, dtype=q.dtype)
    output_buffer = output_backing[1:].view_as(q)
    assert output_buffer.is_contiguous()
    assert output_buffer.data_ptr() % 16 != 0
    output, final_state = recurrent_gdn2(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        out=output_buffer,
        inplace_final_state=True,
    )

    assert expected_state is not None
    assert output is output_buffer
    assert final_state is initial_state
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_supports_preallocated_outputs() -> None:
    torch.manual_seed(77)
    device = torch.device("cuda")
    shape = (1, 2, 1, 128)
    q, k, v, beta, w = [(torch.randn(shape, device=device) * 0.1).half() for _ in range(5)]
    g = (-torch.rand(shape, device=device) * 0.04).half()
    initial_state = torch.randn((1, 1, 128, 128), device=device) * 0.1
    expected_output, expected_state = recurrent_forward_reference(
        q, k, v, g, beta, w, initial_state
    )
    output_buffer = torch.empty_like(q)
    state_buffer = torch.empty_like(initial_state)

    output, final_state = recurrent_gdn2(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        out=output_buffer,
        final_state_out=state_buffer,
    )

    assert output is output_buffer
    assert final_state is state_buffer
    assert expected_state is not None
    torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)

    with pytest.raises(ValueError, match="out must not overlap"):
        token_forward(q, k, v, g, beta, w, initial_state, out=q)

    alias_backing = torch.empty(q.numel() + 1, device=device, dtype=q.dtype)
    alias_q = alias_backing[:-1].view_as(q)
    shifted_alias_out = torch.from_dlpack(alias_backing[1:]).view_as(q)
    assert alias_q.untyped_storage().data_ptr() != shifted_alias_out.untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="out must not overlap"):
        token_forward(alias_q, k, v, g, beta, w, initial_state, out=shifted_alias_out)

    with pytest.raises(ValueError, match="final_state_out must not overlap"):
        token_forward(
            q,
            k,
            v,
            g,
            beta,
            w,
            initial_state,
            final_state_out=initial_state,
        )


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize("inplace", [False, True], ids=["preallocated", "inplace"])
def test_recurrent_api_chunk_dispatch_preserves_output_buffers(inplace: bool) -> None:
    torch.manual_seed(6401 + inplace)
    device = torch.device("cuda")
    shape = (1, 64, 1, 128)
    q, k, v, beta, w = [(torch.randn(shape, device=device) * 0.1).bfloat16() for _ in range(5)]
    g = -torch.rand(shape, device=device, dtype=torch.float32) * 0.04
    initial_state = torch.randn((1, 1, 128, 128), device=device) * 0.1
    expected_output, expected_state = recurrent_forward_reference(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state.clone(),
        scale=0.125,
    )
    output_buffer = torch.empty_like(q)
    state_buffer = None if inplace else torch.empty_like(initial_state)

    output, final_state = recurrent_gdn2(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        scale=0.125,
        out=output_buffer,
        final_state_out=state_buffer,
        inplace_final_state=inplace,
    )

    assert output is output_buffer
    assert final_state is (initial_state if inplace else state_buffer)
    assert expected_state is not None
    torch.testing.assert_close(output, expected_output, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(final_state, expected_state, rtol=3e-2, atol=1.5e-3)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_reuses_dynamic_time_launcher_across_streams_and_maps_batch_heads() -> None:
    torch.manual_seed(314)
    device = torch.device("cuda")
    streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
    calls = []

    for time, stream, scale in zip((1, 4), streams, (0.0375, 0.0625), strict=True):
        shape = (2, time, 3, 128)
        q = (torch.randn(shape, device=device) * 0.1).bfloat16()
        k = (torch.randn(shape, device=device) * 0.1).bfloat16()
        v = (torch.randn(shape, device=device) * 0.2).bfloat16()
        g = -torch.rand(shape, device=device, dtype=torch.float32) * 0.06
        beta = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
        w = torch.sigmoid(torch.randn(shape, device=device)).bfloat16()
        initial_state = torch.randn((2, 3, 128, 128), device=device) * 0.1
        expected = recurrent_forward_reference(q, k, v, g, beta, w, initial_state, scale=scale)
        calls.append(((q, k, v, g, beta, w, initial_state), expected, stream, scale))

    inputs, expected, stream, scale = calls[0]
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        actual = token_forward(*inputs, scale=scale)
        finished = stream.record_event()
    torch.cuda.current_stream(device).wait_event(finished)
    launcher_keys_after_first = set(recurrent_module._COMPILED_LAUNCHERS)

    inputs_second, expected_second, stream_second, scale_second = calls[1]
    stream_second.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream_second):
        actual_second = token_forward(*inputs_second, scale=scale_second)
        finished_second = stream_second.record_event()
    torch.cuda.current_stream(device).wait_event(finished_second)

    assert set(recurrent_module._COMPILED_LAUNCHERS) == launcher_keys_after_first
    for (output, state), (expected_output, expected_state) in (
        (actual, expected),
        (actual_second, expected_second),
    ):
        assert expected_state is not None
        torch.testing.assert_close(output, expected_output, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(state, expected_state, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not _cuda_available(), reason="requires an SM120 CUDA GPU")
def test_token_forward_normalizes_scale_to_finite_float32() -> None:
    device = torch.device("cuda")
    shape = (1, 1, 1, 128)
    inputs = [torch.zeros(shape, device=device, dtype=torch.bfloat16) for _ in range(6)]

    negative_zero = token_forward(*inputs, scale=-0.0)
    positive_zero = token_forward(*inputs, scale=0.0)
    torch.testing.assert_close(negative_zero[0], positive_zero[0], rtol=0, atol=0)
    torch.testing.assert_close(negative_zero[1], positive_zero[1], rtol=0, atol=0)

    with pytest.raises(ValueError, match="representable as float32"):
        token_forward(*inputs, scale=1e100)


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

from __future__ import annotations

import pytest
import torch

import gdn2_sm120.ops as ops_module
from gdn2_sm120.ops import chunk_gdn2
from gdn2_sm120.reference import chunkwise_forward_reference


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.parametrize(
    ("batch", "time", "heads", "chunk_api", "expected"),
    [
        (1, 32, 16, False, True),
        (1, 48, 16, False, True),
        (4, 48, 16, False, True),
        (1, 64, 16, False, False),
        (1, 40, 16, True, True),
        (1, 48, 16, True, False),
        (4, 48, 16, True, False),
        (1, 49, 1, True, True),
        (1, 56, 16, True, True),
        (1, 63, 1, True, True),
        (4, 56, 16, True, True),
        (1, 64, 16, True, False),
    ],
)
def test_forward_kernel_crossover_dispatch(
    batch: int,
    time: int,
    heads: int,
    chunk_api: bool,
    expected: bool,
) -> None:
    assert ops_module._use_recurrent_kernel(batch, time, heads, chunk_api=chunk_api) is expected


@pytest.mark.cuda
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
def test_chunk_gdn2_empty_sequence_preserves_autograd_state_edge() -> None:
    shape = (1, 0, 1, 128)
    inputs = [
        torch.empty(
            shape,
            device="cuda",
            dtype=torch.float32 if index == 3 else torch.bfloat16,
            requires_grad=True,
        )
        for index in range(6)
    ]
    initial_state = torch.randn(
        (1, 1, 128, 128), device="cuda", dtype=torch.float32, requires_grad=True
    )

    output, final_state = chunk_gdn2(
        *inputs,
        initial_state,
        output_final_state=True,
    )

    assert output.shape == shape
    assert final_state is not None
    torch.testing.assert_close(final_state, initial_state, atol=0, rtol=0)
    gradients = torch.autograd.grad(
        output.float().sum() + final_state.sum(),
        (*inputs, initial_state),
    )
    assert all(gradient.shape == shape and gradient.numel() == 0 for gradient in gradients[:-1])
    torch.testing.assert_close(gradients[-1], torch.ones_like(initial_state), atol=0, rtol=0)


@pytest.mark.parametrize(
    ("batch", "time", "heads", "expected"),
    [
        (1, 63, 64, False),
        (1, 64, 15, False),
        (1, 64, 16, True),
        (2, 64, 8, True),
        (1, 80, 16, False),
        (1, 127, 16, False),
        (1, 128, 1, True),
        (1, 129, 1, True),
    ],
)
def test_compact_wy_backward_dispatch(
    batch: int,
    time: int,
    heads: int,
    expected: bool,
) -> None:
    assert ops_module._use_compact_wy_backward(batch, time, heads) is expected


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_chunk_gdn2_uses_one_launch_for_supported_batches(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
) -> None:
    observed_batches: list[int] = []

    def fake_apply(q, _k, v, _g, _beta, _w, _state, _scale, prepare_backward):
        assert prepare_backward
        observed_batches.append(q.shape[0])
        final_state = torch.zeros(
            (q.shape[0], q.shape[2], 128, 128),
            device=q.device,
            dtype=torch.float32,
        )
        return v.clone(), final_state

    monkeypatch.setattr(ops_module._ChunkGDN2, "apply", staticmethod(fake_apply))
    sequence = torch.zeros((batch, 128, 1, 128), dtype=torch.bfloat16, requires_grad=True)

    output, final_state = chunk_gdn2(*([sequence] * 6), output_final_state=True)

    assert output.shape == sequence.shape
    assert final_state is not None and final_state.shape == (batch, 1, 128, 128)
    assert observed_batches == [batch]


def test_chunk_gdn2_rejects_b4_t32768_training_before_kernel_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Meta storage exercises the exact production shape without allocating or
    # launching the unsafe B4/T32768 kernel.  The CUDA traits are overridden
    # only far enough for chunk_forward's address-range guard to run.
    shape = (4, 32768, 16, 128)
    sequence = torch.empty(shape, device="meta", dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _tensor: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=None: (12, 0))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_120")

    with pytest.raises(ValueError, match="4-GiB per-launch address limit"):
        chunk_gdn2(*([sequence] * 6))


def test_long_inference_does_not_materialize_backward_aux(monkeypatch) -> None:
    observed_return_aux: list[bool] = []

    def fake_chunk_forward(q, _k, v, _g, _beta, _w, _state, *, scale, return_aux):
        del scale
        observed_return_aux.append(return_aux)
        final_state = torch.zeros(
            (q.shape[0], q.shape[2], 128, 128), device=q.device, dtype=torch.float32
        )
        return torch.empty_like(v), final_state

    monkeypatch.setattr(ops_module, "chunk_forward", fake_chunk_forward)
    monkeypatch.setattr(
        ops_module._ChunkGDN2,
        "apply",
        staticmethod(lambda *_args: pytest.fail("inference must bypass autograd.Function")),
    )
    sequence = torch.zeros(1, 512, 1, 128, dtype=torch.bfloat16)

    output, final_state = chunk_gdn2(*([sequence] * 6), output_final_state=True)

    assert output.shape == sequence.shape
    assert final_state is not None
    assert observed_return_aux == [False]


def test_hidden_final_state_propagates_none_to_parallel_boundary_scan(monkeypatch) -> None:
    shape = (1, 128, 1, 128)
    state_shape = (1, 1, 128, 128)
    n_chunks = shape[1] // 16
    observed_terminal_vjps: list[torch.Tensor | None] = []

    def fake_chunk_forward(q, _k, v, _g, _beta, _w, _state, *, scale, return_aux):
        del scale
        assert return_aux
        aux = ops_module.ChunkForwardAux(
            torch.empty_like(q),
            torch.empty_like(v),
            torch.empty_like(q),
            torch.empty_like(q),
            torch.empty((1, n_chunks, 1, 128), dtype=torch.float32),
            torch.empty((1, n_chunks, 1, 16, 16), dtype=q.dtype),
            torch.empty((1, n_chunks + 1, 1, 128, 128), dtype=torch.float32),
        )
        return torch.empty_like(v), torch.empty(state_shape, dtype=torch.float32), aux

    def fake_boundary_scan(aux, _d_output, d_final_state, **_kwargs):
        del aux
        observed_terminal_vjps.append(d_final_state)
        return (
            torch.empty((1, n_chunks + 1, 1, 128, 128), dtype=torch.float32),
            torch.empty(shape, dtype=torch.float32),
        )

    def fake_compact_vjp(q, k, v, g, beta, w, *_args, **_kwargs):
        return (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(g),
            torch.empty_like(beta),
            torch.empty_like(w),
            torch.empty(state_shape, dtype=torch.float32),
        )

    monkeypatch.setattr(ops_module, "chunk_forward", fake_chunk_forward)
    monkeypatch.setattr(ops_module, "wy_boundary_dstate", fake_boundary_scan)
    monkeypatch.setattr(ops_module, "compact_wy_chunk_vjp_cute", fake_compact_vjp)
    inputs = [torch.empty(shape, dtype=torch.bfloat16, requires_grad=True) for _ in range(6)]

    output, final_state = chunk_gdn2(*inputs)
    torch.autograd.grad(output, inputs, torch.empty_like(output))

    assert final_state is None
    assert observed_terminal_vjps == [None]


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_disabled_grad_mode_does_not_materialize_backward_aux(monkeypatch, context) -> None:
    observed_return_aux: list[bool] = []

    def fake_chunk_forward(q, _k, v, _g, _beta, _w, _state, *, scale, return_aux):
        del scale
        observed_return_aux.append(return_aux)
        final_state = torch.zeros(
            (q.shape[0], q.shape[2], 128, 128), device=q.device, dtype=torch.float32
        )
        return torch.empty_like(v), final_state

    monkeypatch.setattr(ops_module, "chunk_forward", fake_chunk_forward)
    monkeypatch.setattr(
        ops_module._ChunkGDN2,
        "apply",
        staticmethod(
            lambda *_args: pytest.fail("disabled grad mode must bypass autograd.Function")
        ),
    )
    sequence = torch.zeros(1, 512, 1, 128, dtype=torch.bfloat16, requires_grad=True)

    with context():
        output, final_state = chunk_gdn2(*([sequence] * 6), output_final_state=True)

    assert not output.requires_grad
    assert final_state is not None
    assert observed_return_aux == [False]


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
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize(
    "time",
    [129, 143, 513],
    ids=["one-token-tail", "masked-tail", "algebra-prefix"],
)
def test_chunk_gdn2_parallel_backward_handles_partial_tail(time: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(2_605_000 + time)
    shape = (1, time, 1, 128)
    state_shape = (1, 1, 128, 128)
    dtype = torch.bfloat16
    scale = 0.125

    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    base = [
        q,
        k,
        (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype),
        -torch.rand(shape, generator=generator, device="cuda") * 0.03,
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
    ]
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
    )
    d_output = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )

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
        torch.testing.assert_close(gradient, reference, atol=2e-2, rtol=3e-2)
    torch.testing.assert_close(actual[-1], expected[-1], atol=5e-4, rtol=3e-3)


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not _sm120_available(), reason="requires an SM120 CUDA GPU")
@pytest.mark.parametrize(
    ("time", "heads", "dtype", "has_initial_state", "include_final_state_gradient"),
    [
        (64, 16, torch.bfloat16, True, True),
        (64, 16, torch.float16, True, True),
        (128, 1, torch.bfloat16, True, True),
        (512, 1, torch.bfloat16, True, False),
        (512, 1, torch.bfloat16, False, False),
        (128, 1, torch.bfloat16, False, True),
        (128, 1, torch.float16, True, True),
    ],
    ids=[
        "t64-h16-bf16",
        "t64-h16-fp16",
        "t128-all-seven",
        "t512-hidden-final",
        "t512-no-initial-hidden-final",
        "t128-no-initial",
        "fp16-fallback",
    ],
)
def test_chunk_gdn2_full_chunk_backward_matches_reference(
    time: int,
    heads: int,
    dtype: torch.dtype,
    has_initial_state: bool,
    include_final_state_gradient: bool,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        2_605_000 + time + 17 * has_initial_state + 31 * include_final_state_gradient
    )
    shape = (1, time, heads, 128)
    state_shape = (1, heads, 128, 128)
    scale = 0.125
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda"), dim=-1
    ).to(dtype)
    base = [
        q,
        k,
        (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype),
        -torch.rand(shape, generator=generator, device="cuda") * 0.03,
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
        torch.sigmoid(torch.randn(shape, generator=generator, device="cuda")).to(dtype),
    ]
    initial_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
        if has_initial_state
        else None
    )
    d_output = (torch.randn(shape, generator=generator, device="cuda") * 0.2).to(dtype)
    d_final_state = (
        torch.randn(state_shape, generator=generator, device="cuda", dtype=torch.float32) * 0.1
    )

    actual_inputs = [tensor.detach().requires_grad_() for tensor in base]
    actual_initial = initial_state.detach().requires_grad_() if initial_state is not None else None
    actual_output, actual_final = chunk_gdn2(
        *actual_inputs,
        actual_initial,
        scale=scale,
        output_final_state=include_final_state_gradient,
    )
    assert (actual_final is not None) == include_final_state_gradient
    actual_targets = (
        (*actual_inputs, actual_initial) if actual_initial is not None else tuple(actual_inputs)
    )
    if include_final_state_gradient:
        assert actual_final is not None
        actual = torch.autograd.grad(
            (actual_output, actual_final),
            actual_targets,
            (d_output, d_final_state),
        )
    else:
        # The custom Function specializes the boundary scan for a zero
        # terminal VJP without materializing a state-sized tensor.
        actual = torch.autograd.grad(actual_output, actual_targets, d_output)

    expected_inputs = [tensor.detach().requires_grad_() for tensor in base]
    expected_initial = (
        initial_state.detach().requires_grad_() if initial_state is not None else None
    )
    expected_output, expected_final = chunkwise_forward_reference(
        *expected_inputs,
        expected_initial,
        scale=scale,
        chunk_size=16,
    )
    assert expected_final is not None
    expected_targets = (
        (*expected_inputs, expected_initial)
        if expected_initial is not None
        else tuple(expected_inputs)
    )
    if include_final_state_gradient:
        expected = torch.autograd.grad(
            (expected_output, expected_final),
            expected_targets,
            (d_output, d_final_state),
        )
    else:
        expected = torch.autograd.grad(expected_output, expected_targets, d_output)

    for gradient, reference in zip(actual, expected, strict=True):
        difference = gradient.float() - reference.float()
        relative_l2 = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(
            reference.float()
        ).clamp_min(1e-12)
        assert relative_l2.item() < 0.01
        assert difference.abs().max().item() < 5e-3


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

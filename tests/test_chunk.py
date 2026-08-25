from __future__ import annotations

import pytest
import torch

import gdn2_sm120.chunk as chunk_module
from gdn2_sm120.chunk import (
    _MAX_CUTE_TENSOR_BYTES,
    _select_k_split_value_tile,
    _state_boundary_storage_bytes,
    chunk_forward,
)
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


@pytest.mark.parametrize(
    ("batch", "heads", "n_chunks", "expected"),
    [
        pytest.param(1, 64, 31, 8, id="short-sequence-keeps-v8"),
        pytest.param(1, 1, 32, 8, id="one-head-long-v8"),
        pytest.param(1, 4, 32, 8, id="four-head-long-v8"),
        pytest.param(1, 8, 32, 8, id="eight-head-long-v8"),
        pytest.param(1, 12, 32, 16, id="empirical-v16-crossover"),
        pytest.param(1, 16, 32, 16, id="sixteen-head-long-v16"),
        pytest.param(2, 32, 32, 16, id="multi-batch-long-v16"),
    ],
)
def test_select_k_split_value_tile(batch: int, heads: int, n_chunks: int, expected: int) -> None:
    assert _select_k_split_value_tile(batch, heads, n_chunks) == expected


def test_state_boundary_storage_identifies_four_gib_address_limit() -> None:
    assert _state_boundary_storage_bytes(3, 32768, 16, 2) <= _MAX_CUTE_TENSOR_BYTES
    assert _state_boundary_storage_bytes(4, 32768, 16, 2) > _MAX_CUTE_TENSOR_BYTES
    assert _state_boundary_storage_bytes(1, 32768, 16, 4) <= _MAX_CUTE_TENSOR_BYTES
    assert _state_boundary_storage_bytes(2, 32768, 16, 4) > _MAX_CUTE_TENSOR_BYTES


@pytest.mark.cuda
def test_chunk_forward_rejects_aux_boundary_above_address_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    time = 128
    args = _inputs(time, batch=2, heads=1)
    per_batch_bytes = _state_boundary_storage_bytes(1, time, 1, 2)
    monkeypatch.setattr(chunk_module, "_MAX_CUTE_TENSOR_BYTES", per_batch_bytes)

    with pytest.raises(ValueError, match="4-GiB per-launch address limit"):
        chunk_forward(*args[:6], args[6], return_aux=True)


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
    # The tensor-core path converts the persistent FP32 state to the input
    # dtype for each MMA, matching the official kernel's accumulation model.
    uses_mma_scan = time >= 16
    state_atol = 8e-4 if uses_mma_scan else 2e-4
    state_rtol = 3e-2 if uses_mma_scan else 2e-3
    torch.testing.assert_close(final_state, expected_state, atol=state_atol, rtol=state_rtol)


@pytest.mark.cuda
@pytest.mark.parametrize("time", [7, 16, 512])
def test_chunk_forward_without_initial_state(time: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time)
    output, final_state = chunk_forward(*args[:6])
    expected_output, expected_state = chunkwise_forward_reference(*args[:6], chunk_size=16)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    state_atol = 1.5e-3 if time >= 512 else 8e-4 if time % 16 == 0 else 2e-4
    state_rtol = 3e-2 if time % 16 == 0 else 2e-3
    torch.testing.assert_close(final_state, expected_state, atol=state_atol, rtol=state_rtol)


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

    storage = torch.empty(args[0].numel() + 1, device="cuda", dtype=args[0].dtype)
    misaligned_q = storage[1:].view_as(args[0])
    assert misaligned_q.is_contiguous() and misaligned_q.data_ptr() % 16 != 0
    with pytest.raises(ValueError, match="16-byte aligned"):
        chunk_forward(misaligned_q, *args[1:6], args[6])


@pytest.mark.cuda
def test_chunk_forward_rejects_incompatible_dsl_arch(monkeypatch) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(1)
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100")
    with pytest.raises(RuntimeError, match="CUTE_DSL_ARCH must be sm_120"):
        chunk_forward(*args[:6], args[6])


@pytest.mark.cuda
def test_chunk_forward_rejects_preallocated_output_overlap() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(16)

    with pytest.raises(ValueError, match="out must not overlap"):
        chunk_forward(*args[:6], args[6], out=args[0])


@pytest.mark.cuda
def test_chunk_forward_rejects_shifted_state_output_overlap() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(16)
    state_shape = args[6].shape
    state_elements = args[6].numel()
    backing = torch.empty(state_elements + 4, device="cuda", dtype=torch.float32)
    initial_state = backing[:state_elements].view(state_shape)
    shifted_output = backing[4:].view(state_shape)
    initial_state.copy_(args[6])

    with pytest.raises(ValueError, match="must not partially overlap"):
        chunk_forward(*args[:6], initial_state, final_state_out=shifted_output)


@pytest.mark.cuda
def test_chunk_forward_fp16_normalized_qk_batch_two() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(
        32,
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
    torch.testing.assert_close(final_state, expected_state, atol=8e-4, rtol=3e-2)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("time", "dtype", "return_aux", "expected_use_algebra"),
    [
        pytest.param(47, torch.bfloat16, False, False, id="inference-before-threshold"),
        pytest.param(48, torch.bfloat16, False, True, id="inference-at-threshold"),
        pytest.param(49, torch.bfloat16, False, True, id="inference-partial-tail"),
        pytest.param(512, torch.float16, False, False, id="fp16-keeps-legacy"),
        pytest.param(511, torch.bfloat16, True, False, id="training-before-threshold"),
        pytest.param(512, torch.bfloat16, True, True, id="training-at-threshold"),
    ],
)
def test_chunk_forward_selects_mode_specific_algebra_threshold(
    time: int,
    dtype: torch.dtype,
    return_aux: bool,
    expected_use_algebra: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    compile_flags = []

    def compile_stub(*args):
        compile_flags.append((args[-2], args[-1]))
        return lambda *runtime_args: None

    monkeypatch.setattr(chunk_module, "_compile_chunk_forward", compile_stub)
    args = _inputs(time, dtype, heads=1)
    chunk_forward(*args[:6], args[6], scale=0.125, return_aux=return_aux)

    assert compile_flags == [(expected_use_algebra, return_aux)]


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("time", "dtype"),
    [
        pytest.param(47, torch.bfloat16, id="before-algebra-threshold"),
        pytest.param(48, torch.bfloat16, id="at-algebra-threshold"),
        pytest.param(49, torch.bfloat16, id="algebra-partial-tail"),
        pytest.param(48, torch.float16, id="fp16-original-expression"),
    ],
)
def test_chunk_forward_compact_dispatch_boundaries(time: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time, dtype, heads=1, normalized_qk=True)
    output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    uses_mma_scan = time >= 16
    state_atol = 1.5e-3 if uses_mma_scan else 2e-4
    state_rtol = 3e-2 if uses_mma_scan else 2e-3
    torch.testing.assert_close(final_state, expected_state, atol=state_atol, rtol=state_rtol)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("batch", "heads"),
    [
        pytest.param(2, 3, id="v8-underfilled"),
        pytest.param(2, 6, id="v16-crossover-mapping"),
    ],
)
def test_chunk_forward_long_algebra_maps_batches_and_heads(batch: int, heads: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(512, batch=batch, heads=heads, normalized_qk=True)
    output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=1.5e-3, rtol=3e-2)


@pytest.mark.cuda
@pytest.mark.parametrize("time", [16, 19])
def test_chunk_forward_uses_current_stream(time: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    stream.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    uses_mma_scan = time >= 16
    state_atol = 8e-4 if uses_mma_scan else 2e-4
    state_rtol = 3e-2 if uses_mma_scan else 2e-3
    torch.testing.assert_close(final_state, expected_state, atol=state_atol, rtol=state_rtol)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("time", "dtype"),
    [(1024, torch.bfloat16), (512, torch.float16)],
)
def test_chunk_forward_long_k_split_uses_current_stream(time: int, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time, dtype, heads=1, normalized_qk=True)
    expected_output, expected_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        output, final_state = chunk_forward(*args[:6], args[6], scale=0.125)
    stream.synchronize()

    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(final_state, expected_state, atol=1.5e-3, rtol=3e-2)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("time", "dtype"),
    [
        (19, torch.bfloat16),
        (32, torch.bfloat16),
        (64, torch.bfloat16),
        (129, torch.bfloat16),
        (1024, torch.bfloat16),
        (128, torch.float16),
    ],
)
def test_chunk_forward_aux_uses_training_checkpoint_dtype_contract(
    time: int, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    args = _inputs(time, dtype=dtype, heads=1)
    output, final_state, aux = chunk_forward(*args[:6], args[6], scale=0.125, return_aux=True)
    torch.cuda.synchronize()

    assert aux.state_boundaries.shape == (1, (time + 15) // 16 + 1, 1, 128, 128)
    expected_boundary_dtype = (
        args[0].dtype
        if args[0].dtype == torch.bfloat16 and time >= 128 and time % 16 == 0
        else torch.float32
    )
    assert aux.state_boundaries.dtype == expected_boundary_dtype
    expected_wy_dtype = args[0].dtype if time >= 128 else torch.float32
    assert all(
        tensor.dtype == expected_wy_dtype
        for tensor in (
            aux.y,
            aux.q_gamma,
            aux.k_tail,
            aux.aqk,
        )
    )
    assert aux.u.dtype == torch.float32
    assert aux.u_is_residual is (time == 64 or time >= 128)
    assert aux.decay_end.dtype == torch.float32
    torch.testing.assert_close(
        aux.state_boundaries[:, 0], args[6].to(expected_boundary_dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        aux.state_boundaries[:, -1], final_state.to(expected_boundary_dtype), atol=0, rtol=0
    )

    expected_output, expected_final_state = chunkwise_forward_reference(
        *args[:6], args[6], scale=0.125, chunk_size=16
    )
    torch.testing.assert_close(output, expected_output, atol=3e-2, rtol=3e-2)
    uses_mma_scan = time >= 16
    final_atol = 1.5e-3 if time >= 512 else 1e-3 if uses_mma_scan else 2e-4
    final_rtol = 3e-2 if uses_mma_scan else 2e-3
    torch.testing.assert_close(
        final_state,
        expected_final_state,
        atol=final_atol,
        rtol=final_rtol,
    )

    expected_boundaries = [args[6]]
    running_state = args[6]
    for start in range(0, time, 16):
        stop = min(start + 16, time)
        _, running_state = chunkwise_forward_reference(
            *(tensor[:, start:stop] for tensor in args[:6]),
            running_state,
            scale=0.125,
            chunk_size=16,
        )
        expected_boundaries.append(running_state)
    expected = torch.stack(expected_boundaries, dim=1)
    boundary_atol = 1e-3 if uses_mma_scan else 2e-4
    boundary_rtol = 3e-2 if uses_mma_scan else 2e-3
    torch.testing.assert_close(
        aux.state_boundaries.float(),
        expected,
        atol=boundary_atol,
        rtol=boundary_rtol,
    )

    if time == 1024:
        compact_output, compact_final_state = chunk_forward(*args[:6], args[6], scale=0.125)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, compact_output, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(
            final_state,
            compact_final_state,
            atol=1.5e-3,
            rtol=3e-2,
        )


@pytest.mark.cuda
def test_chunk_forward_training_algebra_preserves_raw_q_gamma_and_aqk() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    time = 512
    args = _inputs(time, heads=1, normalized_qk=True)
    _, _, aux = chunk_forward(*args[:6], args[6], scale=0.125, return_aux=True)
    torch.cuda.synchronize()
    assert aux.u_is_residual

    batch, _, heads, key_dim = args[0].shape
    n_chunks = time // 16
    q_chunks = args[0].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    k_chunks = args[1].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    v_chunks = args[2].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    g_chunks = args[3].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    beta_chunks = args[4].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    w_chunks = args[5].view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4)
    gamma = g_chunks.float().cumsum(dim=-2).exp()
    raw_q_gamma = q_chunks.float() * gamma * 0.125
    k_bar = k_chunks.float() / gamma
    raw_aqk = torch.tril(torch.matmul(raw_q_gamma, k_bar.transpose(-1, -2)))
    raw_q_gamma = raw_q_gamma.permute(0, 1, 3, 2, 4).reshape_as(args[0])

    erase_bar = gamma * beta_chunks.float() * k_chunks.float()
    lower = torch.tril(torch.matmul(erase_bar, k_bar.transpose(-1, -2)), diagonal=-1)
    system = lower + torch.eye(16, device=args[0].device)
    raw_u = torch.linalg.solve_triangular(
        system,
        w_chunks.float() * v_chunks.float(),
        upper=False,
        unitriangular=True,
    )
    y_chunks = aux.y.view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4).float()
    expected_residual = raw_u - torch.matmul(y_chunks, aux.state_boundaries[:, :-1].float())
    saved_residual = aux.u.view(batch, n_chunks, 16, heads, key_dim).permute(0, 1, 3, 2, 4).float()

    torch.testing.assert_close(aux.q_gamma.float(), raw_q_gamma, atol=1e-3, rtol=2e-3)
    torch.testing.assert_close(aux.aqk.float(), raw_aqk, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(saved_residual, expected_residual, atol=2e-6, rtol=2e-5)


@pytest.mark.cuda
def test_chunk_forward_fp16_avoids_algebra_overflow() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    shape = (1, 48, 1, 128)
    q = torch.zeros(shape, device="cuda", dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    g = torch.zeros(shape, device="cuda", dtype=torch.float32)
    beta = torch.zeros_like(q)
    w = torch.zeros_like(q)
    initial_state = torch.zeros((1, 1, 128, 128), device="cuda", dtype=torch.float32)
    q[0, 0, 0, 0] = 8.0
    k[0, 0, 0, 0] = 2.0
    v[0, 0, 0, 0] = 60_000.0
    beta[0, 0, 0, 0] = 1.0
    w[0, 0, 0, 0] = 1.0
    initial_state[0, 0, 0, 0] = 40_000.0

    output, final_state = chunk_forward(q, k, v, g, beta, w, initial_state, scale=0.125)
    expected_output, expected_state = chunkwise_forward_reference(
        q,
        k,
        v,
        g,
        beta,
        w,
        initial_state,
        scale=0.125,
        chunk_size=16,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()
    torch.testing.assert_close(output, expected_output, atol=1.0, rtol=0)
    torch.testing.assert_close(final_state, expected_state, atol=1.0, rtol=0)

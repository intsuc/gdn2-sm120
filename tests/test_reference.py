from __future__ import annotations

import math

import pytest
import torch

from gdn2_sm120.reference import chunkwise_forward_reference, recurrent_forward_reference


def test_zero_key_leaves_decayed_state_and_zero_output() -> None:
    q = torch.zeros(2, 3, 2, 4)
    k = torch.zeros_like(q)
    v = torch.randn(2, 3, 2, 5)
    g = torch.full_like(q, -0.1)
    beta = torch.sigmoid(torch.randn_like(q))
    w = torch.sigmoid(torch.randn_like(v))
    state = torch.randn(2, 2, 4, 5)

    output, final_state = recurrent_forward_reference(q, k, v, g, beta, w, state)

    assert torch.count_nonzero(output) == 0
    expected = state.float() * math.exp(-0.3)
    torch.testing.assert_close(final_state, expected)


def test_reference_is_differentiable() -> None:
    tensors = [torch.randn(1, 3, 2, 4, dtype=torch.float64) for _ in range(4)]
    q, k, g_raw, beta_raw = tensors
    v = torch.randn(1, 3, 2, 5, dtype=torch.float64, requires_grad=True)
    w_raw = torch.randn_like(v, requires_grad=True)
    q.requires_grad_()
    k.requires_grad_()
    g = -torch.nn.functional.softplus(g_raw.requires_grad_())
    beta = torch.sigmoid(beta_raw.requires_grad_())
    w = torch.sigmoid(w_raw)

    output, final_state = recurrent_forward_reference(q, k, v, g, beta, w)
    assert final_state is not None
    loss = output.float().square().mean() + final_state.square().mean()
    loss.backward()

    for tensor in (q, k, v, g_raw, beta_raw, w_raw):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_shape_validation() -> None:
    x = torch.randn(1, 2, 3, 4)
    with pytest.raises(ValueError, match="v and w"):
        recurrent_forward_reference(x, x, x, x, x, x[..., :3])


@pytest.mark.parametrize("time", [1, 3, 4, 5, 9])
@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_chunkwise_matches_recurrent(time: int, chunk_size: int) -> None:
    torch.manual_seed(17)
    shape_k = (2, time, 2, 4)
    shape_v = (2, time, 2, 5)
    q = torch.randn(shape_k, dtype=torch.float64)
    k = torch.randn(shape_k, dtype=torch.float64) * 0.25
    v = torch.randn(shape_v, dtype=torch.float64)
    g = -torch.rand(shape_k, dtype=torch.float64) * 0.2
    beta = torch.sigmoid(torch.randn(shape_k, dtype=torch.float64))
    w = torch.sigmoid(torch.randn(shape_v, dtype=torch.float64))
    state = torch.randn(2, 2, 4, 5, dtype=torch.float64) * 0.1

    recurrent = recurrent_forward_reference(q, k, v, g, beta, w, state, scale=0.37)
    chunkwise = chunkwise_forward_reference(
        q,
        k,
        v,
        g,
        beta,
        w,
        state,
        scale=0.37,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(chunkwise[0], recurrent[0], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(chunkwise[1], recurrent[1], atol=2e-6, rtol=2e-6)


def test_chunkwise_gradients_match_recurrent() -> None:
    torch.manual_seed(23)
    key_shape = (1, 5, 2, 4)
    value_shape = (1, 5, 2, 3)
    base = [
        torch.randn(key_shape, dtype=torch.float64),
        torch.randn(key_shape, dtype=torch.float64) * 0.2,
        torch.randn(value_shape, dtype=torch.float64),
        -torch.rand(key_shape, dtype=torch.float64) * 0.1,
        torch.sigmoid(torch.randn(key_shape, dtype=torch.float64)),
        torch.sigmoid(torch.randn(value_shape, dtype=torch.float64)),
        torch.randn(1, 2, 4, 3, dtype=torch.float64) * 0.1,
    ]
    upstream_o = torch.randn(value_shape, dtype=torch.float64)
    upstream_state = torch.randn(1, 2, 4, 3, dtype=torch.float64)

    def gradients(fn):
        args = [tensor.detach().requires_grad_() for tensor in base]
        output, final_state = fn(*args, scale=0.5)
        assert final_state is not None
        loss = (output * upstream_o).sum() + (final_state * upstream_state).sum()
        return torch.autograd.grad(loss, args)

    recurrent_grads = gradients(recurrent_forward_reference)
    chunkwise_grads = gradients(
        lambda *args, scale: chunkwise_forward_reference(*args, scale=scale, chunk_size=4)
    )
    for actual, expected in zip(chunkwise_grads, recurrent_grads, strict=True):
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-6)

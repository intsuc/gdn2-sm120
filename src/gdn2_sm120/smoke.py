"""Minimal CuTe DSL compile/launch smoke test for the active GPU."""

from __future__ import annotations

import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120")

import cutlass.cute as cute


@cute.kernel
def _scale_kernel(x: cute.Tensor, y: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx < 32:
        y[tidx] = x[tidx] * 2.0


@cute.jit
def _launch_scale(x: cute.Tensor, y: cute.Tensor):
    _scale_kernel(x, y).launch(grid=(1, 1, 1), block=(32, 1, 1))


def main() -> None:
    import torch
    from cutlass.cute.runtime import from_dlpack

    if os.environ.get("CUTE_DSL_ARCH") != "sm_120":
        raise RuntimeError("smoke test requires CUTE_DSL_ARCH=sm_120")

    x = torch.arange(32, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    _launch_scale(from_dlpack(x), from_dlpack(y))
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x * 2)
    print("CuTe DSL compile/launch: PASS")


if __name__ == "__main__":
    main()

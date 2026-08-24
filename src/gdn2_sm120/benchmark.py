"""Reproducible SM120 latency comparison against the official Triton code."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import subprocess
import sys
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .ops import chunk_gdn2, recurrent_gdn2

OFFICIAL_COMMIT = "95709fc250357c2dd109361c353192f2aa5913f9"
FLA_COMMIT = "4b02d15d6a68700181b180235be62a9fb95d2a38"


@dataclass(frozen=True)
class Timing:
    implementation: str
    median_us: float
    minimum_us: float


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    batch: int
    time: int
    heads: int
    dtype: str
    warmup: int
    repeats: int
    cute: Timing
    triton: Timing | None
    speedup: float | None
    validation_max_abs: float | None
    official_commit: str | None
    fla_commit: str
    qk_l2_normalized: bool
    scale: float
    device: str
    torch_version: str
    cuda_runtime: str


def _measure(call: Callable[[], object], warmup: int, repeats: int) -> Timing:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeats):
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1_000.0)
    return Timing("", statistics.median(samples), min(samples))


def _make_inputs(batch: int, time: int, heads: int, dtype: torch.dtype):
    generator = torch.Generator(device="cuda").manual_seed(20260824)
    shape = (batch, time, heads, 128)
    # The production layer L2-normalizes Q/K. Do that once outside both timed
    # calls, then disable each implementation's fused normalization so the
    # benchmark measures the same kernel primitive.
    q = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(dtype)
    k = torch.nn.functional.normalize(
        torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32),
        dim=-1,
    ).to(dtype)
    v = torch.randn(shape, generator=generator, device="cuda", dtype=dtype) * 0.2
    g = -torch.rand(shape, generator=generator, device="cuda", dtype=torch.float32) * 0.03
    beta = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda", dtype=dtype))
    w = torch.sigmoid(torch.randn(shape, generator=generator, device="cuda", dtype=dtype))
    state = (
        torch.randn(
            (batch, heads, 128, 128),
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.03
    )
    return q, k, v, g, beta, w, state


def _load_official(repo: Path, allow_mismatch: bool):
    ops_dir = repo / "lit_gpt" / "gdn2_ops"
    if not (ops_dir / "chunk_gdn2.py").is_file():
        raise FileNotFoundError(f"not an NVlabs/GatedDeltaNet-2 checkout: {repo}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_changes = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT and not allow_mismatch:
        raise RuntimeError(
            f"official checkout is {commit}, expected {OFFICIAL_COMMIT}; "
            "checkout the pinned commit or pass --allow-reference-mismatch"
        )
    if tracked_changes and not allow_mismatch:
        raise RuntimeError(
            "official checkout has tracked modifications; restore it or pass "
            "--allow-reference-mismatch"
        )

    package_name = "_gdn2_official_ops"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ops_dir)]
    sys.modules[package_name] = package
    chunk_module = importlib.import_module(f"{package_name}.chunk_gdn2")
    token_module = importlib.import_module(f"{package_name}.fused_recurrent_gdn2")
    return chunk_module.chunk_gdn2, token_module.fused_recurrent_gdn2, commit


def _validate_equivalent(cute_result: object, triton_result: object) -> float:
    """Check untimed outputs and return their largest absolute difference."""

    cute_tensors = (cute_result,) if isinstance(cute_result, torch.Tensor) else tuple(cute_result)
    triton_tensors = (
        (triton_result,) if isinstance(triton_result, torch.Tensor) else tuple(triton_result)
    )
    if len(cute_tensors) != len(triton_tensors):
        raise AssertionError(
            f"result arity differs: CuTe={len(cute_tensors)}, Triton={len(triton_tensors)}"
        )

    maximum = 0.0
    for cute_tensor, triton_tensor in zip(cute_tensors, triton_tensors, strict=True):
        if cute_tensor.shape != triton_tensor.shape:
            raise AssertionError(
                f"result shape differs: CuTe={cute_tensor.shape}, Triton={triton_tensor.shape}"
            )
        difference = (cute_tensor.float() - triton_tensor.float()).abs()
        if difference.numel():
            maximum = max(maximum, difference.max().item())
        if cute_tensor.dtype == torch.float32:
            torch.testing.assert_close(cute_tensor, triton_tensor, atol=5e-3, rtol=3e-2)
        else:
            torch.testing.assert_close(cute_tensor, triton_tensor, atol=3e-2, rtol=5e-2)
    return maximum


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("benchmark requires an SM120 CUDA GPU")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    inputs = _make_inputs(args.batch, args.time, args.heads, dtype)
    q, k, v, g, beta, w, state = inputs
    scale = float(args.scale)
    if not math.isfinite(scale):
        raise ValueError("benchmark scale must be finite")

    official_chunk = official_token = official_commit = None
    if args.official_repo is not None:
        official_chunk, official_token, official_commit = _load_official(
            args.official_repo.resolve(), args.allow_reference_mismatch
        )

    triton_call: Callable[[], object] | None = None
    if args.mode == "chunk-forward":

        def cute_call():
            return chunk_gdn2(
                *inputs[:6],
                state,
                scale=scale,
                output_final_state=True,
            )

        if official_chunk is not None:

            def triton_call():
                return official_chunk(
                    *inputs[:6],
                    scale=scale,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=False,
                )

    elif args.mode == "token-forward":

        def cute_call():
            return recurrent_gdn2(*inputs[:6], state, scale=scale)

        if official_token is not None:

            def triton_call():
                return official_token(
                    *inputs[:6],
                    scale=scale,
                    initial_state=state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=False,
                )

    elif args.mode in ("chunk-backward", "chunk-training"):
        gradient_generator = torch.Generator(device="cuda").manual_seed(20260825)
        upstream_o = (
            torch.randn(
                v.shape,
                generator=gradient_generator,
                device=v.device,
                dtype=v.dtype,
            )
            * 0.2
        )
        upstream_state = (
            torch.randn(
                state.shape,
                generator=gradient_generator,
                device=state.device,
                dtype=state.dtype,
            )
            * 0.1
        )
        cute_differentiable = [tensor.detach().requires_grad_() for tensor in inputs[:6]]
        cute_state = state.detach().requires_grad_()
        cute_grad_inputs = (*cute_differentiable, cute_state)
        if args.mode == "chunk-backward":
            cute_output, cute_final = chunk_gdn2(
                *cute_differentiable,
                cute_state,
                scale=scale,
                output_final_state=True,
            )
            assert cute_final is not None

            def cute_call():
                return torch.autograd.grad(
                    (cute_output, cute_final),
                    cute_grad_inputs,
                    (upstream_o, upstream_state),
                    retain_graph=True,
                )

        else:

            def cute_call():
                cute_output, cute_final = chunk_gdn2(
                    *cute_differentiable,
                    cute_state,
                    scale=scale,
                    output_final_state=True,
                )
                assert cute_final is not None
                return torch.autograd.grad(
                    (cute_output, cute_final),
                    cute_grad_inputs,
                    (upstream_o, upstream_state),
                )

        if official_chunk is not None:
            differentiable = [tensor.detach().requires_grad_() for tensor in inputs[:6]]
            differentiable_state = state.detach().requires_grad_()
            grad_inputs = (*differentiable, differentiable_state)
            if args.mode == "chunk-backward":
                official_output, official_final = official_chunk(
                    *differentiable,
                    scale=scale,
                    initial_state=differentiable_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    use_gate_in_kernel=False,
                )

                def triton_call():
                    return torch.autograd.grad(
                        (official_output, official_final),
                        grad_inputs,
                        (upstream_o, upstream_state),
                        retain_graph=True,
                    )

            else:

                def triton_call():
                    official_output, official_final = official_chunk(
                        *differentiable,
                        scale=scale,
                        initial_state=differentiable_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,
                        use_gate_in_kernel=False,
                    )
                    return torch.autograd.grad(
                        (official_output, official_final),
                        grad_inputs,
                        (upstream_o, upstream_state),
                    )

    else:
        raise AssertionError(f"unhandled benchmark mode: {args.mode}")

    validation_max_abs = None
    if triton_call is not None:
        validation_max_abs = _validate_equivalent(cute_call(), triton_call())

    cute_timing = _measure(cute_call, args.warmup, args.repeats)
    cute_timing = Timing("cute-sm120", cute_timing.median_us, cute_timing.minimum_us)
    triton_timing = None
    speedup = None
    if triton_call is not None:
        measured = _measure(triton_call, args.warmup, args.repeats)
        triton_timing = Timing("official-triton", measured.median_us, measured.minimum_us)
        speedup = triton_timing.median_us / cute_timing.median_us

    return BenchmarkResult(
        mode=args.mode,
        batch=args.batch,
        time=args.time,
        heads=args.heads,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        cute=cute_timing,
        triton=triton_timing,
        speedup=speedup,
        validation_max_abs=validation_max_abs,
        official_commit=official_commit,
        fla_commit=FLA_COMMIT,
        qk_l2_normalized=True,
        scale=scale,
        device=torch.cuda.get_device_name(),
        torch_version=str(torch.__version__),
        cuda_runtime=str(torch.version.cuda),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("chunk-forward", "chunk-backward", "chunk-training", "token-forward"),
        default="chunk-forward",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--time", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.125,
        help="explicit output scale applied identically to both implementations",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--official-repo",
        type=Path,
        help="checkout of NVlabs/GatedDeltaNet-2 at the pinned commit",
    )
    parser.add_argument("--allow-reference-mismatch", action="store_true")
    parser.add_argument("--json", type=Path, help="also write the result as JSON")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if min(args.batch, args.time, args.heads, args.warmup, args.repeats) <= 0:
        raise SystemExit("batch, time, heads, warmup, and repeats must be positive")
    result = run_benchmark(args)
    print(
        f"{result.mode} B={result.batch} T={result.time} H={result.heads} "
        f"{result.dtype}: CuTe={result.cute.median_us:.1f} us"
    )
    if result.triton is not None:
        print(
            f"official Triton={result.triton.median_us:.1f} us, "
            f"speedup={result.speedup:.2f}x, "
            f"validation max|diff|={result.validation_max_abs:.3g}"
        )
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

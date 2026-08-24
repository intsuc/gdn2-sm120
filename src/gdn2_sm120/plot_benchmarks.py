"""Render validated benchmark JSON files as a static README-friendly figure."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODES = ("chunk-forward", "chunk-backward", "token-forward")
MODE_TITLES = {
    "chunk-forward": "Chunk forward",
    "chunk-backward": "Chunk backward (T ≤ 128)",
    "token-forward": "Token forward",
}


@dataclass(frozen=True)
class BenchmarkPoint:
    """The validated subset of one current benchmark result used by the plot."""

    mode: str
    batch: int
    time: int
    heads: int
    dtype: str
    warmup: int
    repeats: int
    cute_median_us: float
    cute_minimum_us: float
    triton_median_us: float
    triton_minimum_us: float
    stored_speedup: float
    validation_max_abs: float
    official_commit: str
    qk_l2_normalized: bool
    device: str
    torch_version: str
    cuda_runtime: str

    @property
    def speedup(self) -> float:
        """Official Triton median divided by the CuTe SM120 median."""

        return self.triton_median_us / self.cute_median_us

    @property
    def logical_key(self) -> tuple[str, int, int, int, str]:
        return self.mode, self.batch, self.time, self.heads, self.dtype


@dataclass(frozen=True)
class BenchmarkSuite:
    """A metadata-consistent, deterministically ordered collection of points."""

    points: tuple[BenchmarkPoint, ...]
    device: str
    dtype: str
    official_commit: str
    qk_l2_normalized: bool
    torch_version: str
    cuda_runtime: str


def _mapping(value: object, name: str, origin: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{origin}: {name} must be a JSON object")
    return value


def _positive_int(value: object, name: str, origin: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{origin}: {name} must be a positive integer")
    return value


def _finite_number(value: object, name: str, origin: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{origin}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{origin}: {name} must be {qualifier}")
    return result


def _nonempty_string(value: object, name: str, origin: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{origin}: {name} must be a non-empty string")
    return value


def _timing(record: dict[str, Any], name: str, origin: str) -> tuple[float, float]:
    timing = _mapping(record.get(name), name, origin)
    _nonempty_string(timing.get("implementation"), f"{name}.implementation", origin)
    median = _finite_number(timing.get("median_us"), f"{name}.median_us", origin, positive=True)
    minimum = _finite_number(timing.get("minimum_us"), f"{name}.minimum_us", origin, positive=True)
    if minimum > median:
        raise ValueError(f"{origin}: {name}.minimum_us cannot exceed its median_us")
    return median, minimum


def _parse_point(record: object, origin: str) -> BenchmarkPoint:
    item = _mapping(record, "benchmark result", origin)
    mode = _nonempty_string(item.get("mode"), "mode", origin)
    if mode not in MODES:
        raise ValueError(f"{origin}: unsupported mode {mode!r}")
    dtype = _nonempty_string(item.get("dtype"), "dtype", origin)
    cute_median, cute_minimum = _timing(item, "cute", origin)
    if item.get("triton") is None:
        raise ValueError(f"{origin}: triton timing is required for a comparison plot")
    triton_median, triton_minimum = _timing(item, "triton", origin)
    stored_speedup = _finite_number(item.get("speedup"), "speedup", origin, positive=True)
    computed_speedup = triton_median / cute_median
    if not math.isclose(stored_speedup, computed_speedup, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError(
            f"{origin}: speedup {stored_speedup:g} disagrees with medians ({computed_speedup:g})"
        )
    validation = _finite_number(
        item.get("validation_max_abs"), "validation_max_abs", origin, positive=False
    )
    if validation < 0.0:
        raise ValueError(f"{origin}: validation_max_abs cannot be negative")
    normalized = item.get("qk_l2_normalized")
    if not isinstance(normalized, bool):
        raise ValueError(f"{origin}: qk_l2_normalized must be boolean")

    return BenchmarkPoint(
        mode=mode,
        batch=_positive_int(item.get("batch"), "batch", origin),
        time=_positive_int(item.get("time"), "time", origin),
        heads=_positive_int(item.get("heads"), "heads", origin),
        dtype=dtype,
        warmup=_positive_int(item.get("warmup"), "warmup", origin),
        repeats=_positive_int(item.get("repeats"), "repeats", origin),
        cute_median_us=cute_median,
        cute_minimum_us=cute_minimum,
        triton_median_us=triton_median,
        triton_minimum_us=triton_minimum,
        stored_speedup=stored_speedup,
        validation_max_abs=validation,
        official_commit=_nonempty_string(item.get("official_commit"), "official_commit", origin),
        qk_l2_normalized=normalized,
        device=_nonempty_string(item.get("device"), "device", origin),
        torch_version=_nonempty_string(item.get("torch_version"), "torch_version", origin),
        cuda_runtime=_nonempty_string(item.get("cuda_runtime"), "cuda_runtime", origin),
    )


def _records(payload: object, origin: str) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        version = payload.get("schema_version")
        if version != 1:
            raise ValueError(f"{origin}: unsupported schema_version {version!r}")
        results = payload["results"]
        if not isinstance(results, list):
            raise ValueError(f"{origin}: results must be a JSON array")
        return results
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"{origin}: expected a benchmark object, array, or suite object")


def _input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"benchmark input does not exist: {path}")
    if not files:
        raise ValueError("no benchmark JSON files were found")
    return files


def _uniform(points: list[BenchmarkPoint], field: str) -> object:
    values = {getattr(point, field) for point in points}
    if len(values) != 1:
        rendered = ", ".join(sorted(map(str, values)))
        raise ValueError(f"benchmark inputs mix {field}: {rendered}")
    return next(iter(values))


def load_benchmarks(paths: list[Path]) -> BenchmarkSuite:
    """Load current benchmark JSON, rejecting duplicates and mixed environments."""

    points: list[BenchmarkPoint] = []
    origins: dict[tuple[str, int, int, int, str], str] = {}
    for path in _input_files(paths):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
        for index, record in enumerate(_records(payload, str(path))):
            origin = f"{path}:results[{index}]"
            point = _parse_point(record, origin)
            if point.logical_key in origins:
                raise ValueError(
                    f"{origin}: duplicate logical benchmark key; first seen in "
                    f"{origins[point.logical_key]}"
                )
            origins[point.logical_key] = origin
            points.append(point)
    if not points:
        raise ValueError("benchmark inputs contain no results")

    order = {mode: index for index, mode in enumerate(MODES)}
    points.sort(key=lambda point: (order[point.mode], point.time, point.batch, point.heads))
    return BenchmarkSuite(
        points=tuple(points),
        device=str(_uniform(points, "device")),
        dtype=str(_uniform(points, "dtype")),
        official_commit=str(_uniform(points, "official_commit")),
        qk_l2_normalized=bool(_uniform(points, "qk_l2_normalized")),
        torch_version=str(_uniform(points, "torch_version")),
        cuda_runtime=str(_uniform(points, "cuda_runtime")),
    )


def _speedup_label(speedup: float) -> tuple[str, str]:
    if speedup >= 1.01:
        return f"{speedup:.2f}×", "#166534"
    if speedup <= 0.99:
        return f"{speedup:.2f}×", "#B91C1C"
    return f"≈{speedup:.2f}×", "#854D0E"


def _plot_chunk_panel(axis: Any, points: list[BenchmarkPoint], title: str) -> None:
    cute_color = "#2563EB"
    triton_color = "#475569"
    xs = [math.log2(point.time) for point in points]
    cute = [point.cute_median_us for point in points]
    triton = [point.triton_median_us for point in points]
    maximum = max((*cute, *triton))

    axis.plot(xs, cute, color=cute_color, marker="o", linewidth=2.4, label="CuTe SM120")
    axis.plot(
        xs,
        triton,
        color=triton_color,
        marker="s",
        linewidth=2.2,
        label="Official Triton",
    )
    for x, point in zip(xs, points, strict=True):
        low = min(point.cute_median_us, point.triton_median_us)
        high = max(point.cute_median_us, point.triton_median_us)
        axis.vlines(x, low, high, color="#CBD5E1", linewidth=1.0, zorder=0)
        axis.annotate(
            f"{point.cute_median_us:.1f}",
            (x, point.cute_median_us),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=cute_color,
            fontsize=8.5,
        )
        axis.annotate(
            f"{point.triton_median_us:.1f}",
            (x, point.triton_median_us),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            color=triton_color,
            fontsize=8.5,
        )
        label, color = _speedup_label(point.speedup)
        axis.text(
            x,
            maximum * 1.16,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )

    for left, right in zip(points, points[1:], strict=False):
        if (left.speedup - 1.0) * (right.speedup - 1.0) < 0.0:
            left_x, right_x = math.log2(left.time), math.log2(right.time)
            axis.axvspan(left_x, right_x, color="#F59E0B", alpha=0.08, zorder=-1)

    axis.set_title(title, loc="left", fontweight="bold", fontsize=12)
    axis.set_xticks(xs, [str(point.time) for point in points])
    axis.set_xlabel("Sequence length T (log₂ spacing)")
    axis.set_ylabel("Median latency (µs / call)")
    axis.set_ylim(0.0, maximum * 1.27)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)


def _plot_token_panel(axis: Any, points: list[BenchmarkPoint]) -> None:
    cute_color = "#2563EB"
    triton_color = "#475569"
    positions = list(range(len(points)))
    width = 0.34
    cute = axis.bar(
        [position - width / 2 for position in positions],
        [point.cute_median_us for point in points],
        width,
        color=cute_color,
        label="CuTe SM120",
    )
    triton = axis.bar(
        [position + width / 2 for position in positions],
        [point.triton_median_us for point in points],
        width,
        color=triton_color,
        label="Official Triton",
    )
    maximum = max(
        *(point.cute_median_us for point in points),
        *(point.triton_median_us for point in points),
    )
    axis.bar_label(cute, fmt="%.1f", padding=3, color=cute_color, fontsize=8.5)
    axis.bar_label(triton, fmt="%.1f", padding=3, color=triton_color, fontsize=8.5)
    for position, point in zip(positions, points, strict=True):
        label, color = _speedup_label(point.speedup)
        axis.text(
            position,
            maximum * 1.16,
            label,
            ha="center",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
        )
    axis.set_title(MODE_TITLES["token-forward"], loc="left", fontweight="bold", fontsize=12)
    axis.set_xticks(
        positions,
        [f"B{point.batch} · T{point.time}\nH{point.heads}" for point in points],
    )
    axis.set_xlabel("Measured shape (different H; not a scaling line)")
    axis.set_ylabel("Median latency (µs / call)")
    axis.set_ylim(0.0, maximum * 1.27)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.set_axisbelow(True)


def render_benchmarks(
    suite: BenchmarkSuite,
    output: Path,
    *,
    title: str | None = None,
) -> None:
    """Render one PNG or SVG without requiring an interactive display."""

    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("output filename must end in .png or .svg")
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required; run this command through `uv run --group visualization`"
        ) from error
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": "#0F172A",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "gdn2-sm120-benchmarks",
        }
    )

    by_mode = {mode: [point for point in suite.points if point.mode == mode] for mode in MODES}
    missing = [mode for mode, points in by_mode.items() if not points]
    if missing:
        raise ValueError(f"plot requires all three modes; missing: {', '.join(missing)}")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 5.7),
        gridspec_kw={"width_ratios": (1.25, 1.05, 0.85), "wspace": 0.30},
    )
    _plot_chunk_panel(axes[0], by_mode["chunk-forward"], MODE_TITLES["chunk-forward"])
    _plot_chunk_panel(axes[1], by_mode["chunk-backward"], MODE_TITLES["chunk-backward"])
    _plot_token_panel(axes[2], by_mode["token-forward"])

    display_device = suite.device.removeprefix("NVIDIA ")
    figure.suptitle(
        title or f"Gated DeltaNet-2 latency on {display_device}",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    normalized = "Q/K L2-normalized" if suite.qk_l2_normalized else "Q/K not normalized"
    figure.text(
        0.04,
        0.925,
        f"{suite.dtype.upper()} · K=V=128 · {normalized} · lower is better · "
        "labels are official/CuTe speedup",
        ha="left",
        fontsize=10,
        color="#475569",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.985),
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    figure.text(
        0.04,
        0.02,
        f"Synchronized CUDA events · PyTorch {suite.torch_version} / CUDA {suite.cuda_runtime} · "
        f"official commit {suite.official_commit[:12]} · shaded spans mark a winner flip "
        "between measured points",
        ha="left",
        fontsize=8.5,
        color="#64748B",
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.84, bottom=0.18)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white", metadata={"Creator": "gdn2-sm120"})
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="current benchmark JSON files, a suite JSON, or directories containing JSON",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/assets/benchmark-results-sm120.png"),
        help="destination .png or .svg (default: %(default)s)",
    )
    parser.add_argument("--title", help="optional figure title override")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        suite = load_benchmarks(args.inputs)
        render_benchmarks(suite, args.output, title=args.title)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

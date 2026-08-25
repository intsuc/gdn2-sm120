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
    "chunk-backward": "Chunk backward",
    "token-forward": "Token forward",
}

_CHUNK_DATA_TOP_FRACTION = 0.69
_CHUNK_RAIL_BOTTOM = 0.735
_CHUNK_CUTE_ROW_Y = 0.775
_CHUNK_TRITON_ROW_Y = 0.855
_CHUNK_SPEEDUP_ROW_Y = 0.945
_BATCH_LINESTYLES = ("-", "--", ":", "-.")
_CHUNK_GROUP_DODGE_MAX = 0.20
_CHUNK_IMPLEMENTATION_DODGE_MAX = 0.045
_CHUNK_IMPLEMENTATION_LANE_FRACTION = 0.35
_CHUNK_TICK_GAP_FRACTION = 0.40


@dataclass(frozen=True)
class PlotPalette:
    """Semantic colors for one static plot theme."""

    background: str
    foreground: str
    axis_label: str
    tick: str
    muted: str
    border: str
    grid: str
    connector: str
    cute: str
    triton: str
    speedup_positive: str
    speedup_negative: str
    speedup_neutral: str
    crossover: str
    crossover_alpha: float


_LIGHT_PALETTE = PlotPalette(
    background="#FFFFFF",
    foreground="#0F172A",
    axis_label="#334155",
    tick="#475569",
    muted="#64748B",
    border="#CBD5E1",
    grid="#E2E8F0",
    connector="#CBD5E1",
    cute="#2563EB",
    triton="#475569",
    speedup_positive="#166534",
    speedup_negative="#B91C1C",
    speedup_neutral="#854D0E",
    crossover="#F59E0B",
    crossover_alpha=0.08,
)
_DARK_PALETTE = PlotPalette(
    background="#0D1117",
    foreground="#F0F6FC",
    axis_label="#C9D1D9",
    tick="#8B949E",
    muted="#8B949E",
    border="#484F58",
    grid="#30363D",
    connector="#484F58",
    cute="#58A6FF",
    triton="#C9D1D9",
    speedup_positive="#3FB950",
    speedup_negative="#FF7B72",
    speedup_neutral="#D29922",
    crossover="#D29922",
    crossover_alpha=0.14,
)
_PLOT_PALETTES = {"light": _LIGHT_PALETTE, "dark": _DARK_PALETTE}


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
    fla_commit: str
    qk_l2_normalized: bool
    scale: float
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
    fla_commit: str
    qk_l2_normalized: bool
    scale: float
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
        fla_commit=_nonempty_string(item.get("fla_commit"), "fla_commit", origin),
        qk_l2_normalized=normalized,
        scale=_finite_number(item.get("scale"), "scale", origin, positive=False),
        device=_nonempty_string(item.get("device"), "device", origin),
        torch_version=_nonempty_string(item.get("torch_version"), "torch_version", origin),
        cuda_runtime=_nonempty_string(item.get("cuda_runtime"), "cuda_runtime", origin),
    )


def _records(payload: object, origin: str) -> list[object]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "results" in payload:
        version = payload.get("schema_version")
        if version != 2:
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
        fla_commit=str(_uniform(points, "fla_commit")),
        qk_l2_normalized=bool(_uniform(points, "qk_l2_normalized")),
        scale=float(_uniform(points, "scale")),
        torch_version=str(_uniform(points, "torch_version")),
        cuda_runtime=str(_uniform(points, "cuda_runtime")),
    )


def _plot_palette(theme: str) -> PlotPalette:
    try:
        return _PLOT_PALETTES[theme]
    except KeyError as error:
        choices = ", ".join(sorted(_PLOT_PALETTES))
        raise ValueError(f"unsupported plot theme {theme!r}; choose one of: {choices}") from error


def _speedup_label(
    speedup: float,
    palette: PlotPalette = _LIGHT_PALETTE,
) -> tuple[str, str]:
    if speedup >= 1.01:
        return f"{speedup:.2f}×", palette.speedup_positive
    if speedup <= 0.99:
        return f"{speedup:.2f}×", palette.speedup_negative
    return f"≈{speedup:.2f}×", palette.speedup_neutral


def _plot_chunk_panel(
    axis: Any,
    points: list[BenchmarkPoint],
    title: str,
    *,
    log_latency: bool = False,
    palette: PlotPalette = _LIGHT_PALETTE,
) -> None:
    from matplotlib.patches import Rectangle

    shapes = {(point.batch, point.heads) for point in points}
    if len(shapes) > 1:
        _plot_multi_shape_chunk_panel(
            axis,
            points,
            title,
            log_latency=log_latency,
            palette=palette,
        )
        return

    cute_color = palette.cute
    triton_color = palette.triton
    xs = [math.log2(point.time) for point in points]
    cute = [point.cute_median_us for point in points]
    triton = [point.triton_median_us for point in points]
    maximum = max((*cute, *triton))
    minimum = min((*cute, *triton))

    axis.plot(
        xs,
        cute,
        color=cute_color,
        marker="o",
        linewidth=2.4,
        label="CuTe SM120",
        zorder=3,
    )
    axis.plot(
        xs,
        triton,
        color=triton_color,
        marker="s",
        linewidth=2.2,
        label="Official Triton",
        zorder=3,
    )
    for x, point in zip(xs, points, strict=True):
        low = min(point.cute_median_us, point.triton_median_us)
        high = max(point.cute_median_us, point.triton_median_us)
        axis.vlines(x, low, high, color=palette.connector, linewidth=1.0, zorder=1)

    for left, right in zip(points, points[1:], strict=False):
        if (left.speedup - 1.0) * (right.speedup - 1.0) < 0.0:
            left_x, right_x = math.log2(left.time), math.log2(right.time)
            axis.axvspan(
                left_x,
                right_x,
                color=palette.crossover,
                alpha=palette.crossover_alpha,
                zorder=-1,
            )

    axis.set_title(title, loc="left", fontweight="bold", fontsize=12)
    axis.set_xticks(
        xs,
        [str(point.time) for point in points],
        rotation=35 if len(points) > 6 else 0,
        ha="right" if len(points) > 6 else "center",
        rotation_mode="anchor",
    )
    axis.set_xlabel("Sequence length T (log₂ spacing)")

    # Keep every observation below a fixed three-row label rail. Computing the
    # limit in the scale's coordinate space preserves the same separation for
    # either chunk panel despite their different latency ranges.
    if log_latency:
        axis.set_yscale("log", base=2)
        lower_power = math.floor(math.log2(minimum)) - 0.45
        maximum_power = math.log2(maximum)
        upper_power = lower_power + (maximum_power - lower_power) / _CHUNK_DATA_TOP_FRACTION
        ticks = [
            2.0**power for power in range(math.ceil(lower_power), math.floor(maximum_power) + 1)
        ]
        axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        axis.set_ylabel("Median latency (µs / call, log₂ scale)")
        axis.set_ylim(2.0**lower_power, 2.0**upper_power)
        axis.grid(axis="y", which="major", color=palette.grid, linewidth=0.8)
    else:
        axis.set_ylabel("Median latency (µs / call)")
        axis.set_ylim(0.0, maximum / _CHUNK_DATA_TOP_FRACTION)
        axis.grid(axis="y", color=palette.grid, linewidth=0.8)
    axis.set_axisbelow(True)

    # Values use predictable rows keyed with the same marker shapes and colors
    # as the series. The opaque rail keeps labels separate from every data line.
    axis.add_patch(
        Rectangle(
            (0.0, _CHUNK_RAIL_BOTTOM),
            1.0,
            1.0 - _CHUNK_RAIL_BOTTOM,
            transform=axis.transAxes,
            facecolor=palette.background,
            edgecolor="none",
            zorder=5,
            gid="chunk-value-rail",
        )
    )
    for y in (_CHUNK_RAIL_BOTTOM, 0.815, 0.895):
        axis.plot(
            [0.0, 1.0],
            [y, y],
            transform=axis.transAxes,
            color=palette.grid,
            linewidth=0.75,
            zorder=6,
            clip_on=False,
        )

    for symbol, y, color, size in (
        ("×", _CHUNK_SPEEDUP_ROW_Y, palette.speedup_positive, 8.2),
        ("■", _CHUNK_TRITON_ROW_Y, triton_color, 7.2),
        ("●", _CHUNK_CUTE_ROW_Y, cute_color, 7.2),
    ):
        axis.text(
            -0.012,
            y,
            symbol,
            transform=axis.transAxes,
            ha="right",
            va="center",
            color=color,
            fontsize=size,
            fontweight="bold" if symbol == "×" else "normal",
            zorder=7,
            clip_on=False,
            gid="chunk-rail-key",
        )

    rail_transform = axis.get_xaxis_transform()
    speedup_fontsize = 6.2 if len(points) >= 11 else 7.8
    latency_fontsize = 6.2 if len(points) >= 11 else 7.7
    for x, point in zip(xs, points, strict=True):
        speedup_label, speedup_color = _speedup_label(point.speedup, palette)
        for text, y, color, size, weight in (
            (
                speedup_label,
                _CHUNK_SPEEDUP_ROW_Y,
                speedup_color,
                speedup_fontsize,
                "bold",
            ),
            (
                f"{point.triton_median_us:.1f}",
                _CHUNK_TRITON_ROW_Y,
                triton_color,
                latency_fontsize,
                "normal",
            ),
            (
                f"{point.cute_median_us:.1f}",
                _CHUNK_CUTE_ROW_Y,
                cute_color,
                latency_fontsize,
                "normal",
            ),
        ):
            axis.text(
                x,
                y,
                text,
                transform=rail_transform,
                ha="center",
                va="center",
                color=color,
                fontsize=size,
                fontweight=weight,
                zorder=7,
                gid="chunk-rail-label",
            )
        axis.plot(
            [x, x],
            [_CHUNK_RAIL_BOTTOM - 0.012, _CHUNK_RAIL_BOTTOM + 0.012],
            transform=rail_transform,
            color=palette.connector,
            linewidth=0.8,
            zorder=7,
            clip_on=False,
        )


def _chunk_dodge_geometry(
    shapes: list[tuple[int, int]],
    times: list[int],
) -> tuple[dict[tuple[int, int], float], float]:
    """Return ordered group centers and the nested implementation offset."""

    log_times = [math.log2(time) for time in times]
    gaps = [right - left for left, right in zip(log_times, log_times[1:], strict=False)]
    maximum_half_width = _CHUNK_GROUP_DODGE_MAX + _CHUNK_IMPLEMENTATION_DODGE_MAX
    scale = min(1.0, _CHUNK_TICK_GAP_FRACTION * min(gaps) / maximum_half_width) if gaps else 1.0
    group_spread = _CHUNK_GROUP_DODGE_MAX * scale
    implementation_dodge = _CHUNK_IMPLEMENTATION_DODGE_MAX * scale
    if len(shapes) == 1:
        return {shapes[0]: 0.0}, implementation_dodge

    center_step = 2.0 * group_spread / (len(shapes) - 1)
    implementation_dodge = min(
        implementation_dodge,
        _CHUNK_IMPLEMENTATION_LANE_FRACTION * center_step,
    )
    centers = {shape: -group_spread + index * center_step for index, shape in enumerate(shapes)}
    return centers, implementation_dodge


def _plot_multi_shape_chunk_panel(
    axis: Any,
    points: list[BenchmarkPoint],
    title: str,
    *,
    log_latency: bool,
    palette: PlotPalette,
) -> None:
    """Draw batch-dodged CuTe/Triton dumbbells around shared length ticks."""

    _plot_multi_shape_scaling_panel(
        axis,
        points,
        title,
        log_latency=log_latency,
        palette=palette,
        gid_prefix="chunk",
        connect_pairs=True,
    )


def _plot_multi_shape_scaling_panel(
    axis: Any,
    points: list[BenchmarkPoint],
    title: str,
    *,
    log_latency: bool,
    palette: PlotPalette,
    gid_prefix: str,
    connect_pairs: bool,
) -> None:
    """Draw implementation-colored, batch-styled scaling series and speedup rows."""

    from matplotlib.patches import Rectangle

    grouped: dict[tuple[int, int], list[BenchmarkPoint]] = {}
    for point in points:
        grouped.setdefault((point.batch, point.heads), []).append(point)
    groups = [
        (shape, sorted(shape_points, key=lambda point: point.time))
        for shape, shape_points in sorted(grouped.items())
    ]
    shapes = [shape for shape, _ in groups]
    batches = sorted({batch for batch, _ in shapes})
    linestyle_by_batch = {
        batch: _BATCH_LINESTYLES[index % len(_BATCH_LINESTYLES)]
        for index, batch in enumerate(batches)
    }
    times = sorted({point.time for point in points})
    if connect_pairs:
        center_by_shape, implementation_dodge = _chunk_dodge_geometry(shapes, times)
    else:
        center_by_shape = dict.fromkeys(shapes, 0.0)
        implementation_dodge = 0.0
    xs = [math.log2(time) for time in times]
    cute = [point.cute_median_us for point in points]
    triton = [point.triton_median_us for point in points]
    maximum = max((*cute, *triton))
    minimum = min((*cute, *triton))

    for index, (shape, shape_points) in enumerate(groups):
        batch, _ = shape
        center_xs = [math.log2(point.time) + center_by_shape[shape] for point in shape_points]
        cute_xs = [x - implementation_dodge for x in center_xs]
        triton_xs = [x + implementation_dodge for x in center_xs]
        linestyle = linestyle_by_batch[batch]
        axis.plot(
            cute_xs,
            [point.cute_median_us for point in shape_points],
            color=palette.cute,
            linestyle=linestyle,
            linewidth=1.45,
            alpha=0.38,
            label="CuTe SM120" if index == 0 else "_nolegend_",
            zorder=2,
            gid=f"{gid_prefix}-trend",
        )
        axis.plot(
            triton_xs,
            [point.triton_median_us for point in shape_points],
            color=palette.triton,
            linestyle=linestyle,
            linewidth=1.45,
            alpha=0.38,
            label="Official Triton" if index == 0 else "_nolegend_",
            zorder=2,
            gid=f"{gid_prefix}-trend",
        )
        if connect_pairs:
            for cute_x, triton_x, point in zip(cute_xs, triton_xs, shape_points, strict=True):
                axis.plot(
                    [cute_x, triton_x],
                    [point.cute_median_us, point.triton_median_us],
                    color=palette.tick,
                    linewidth=1.8,
                    alpha=0.78,
                    solid_capstyle="round",
                    zorder=3,
                    gid=f"{gid_prefix}-pair-connector",
                )
        axis.scatter(
            cute_xs,
            [point.cute_median_us for point in shape_points],
            s=54,
            marker="o",
            facecolor=palette.cute,
            edgecolor=palette.background,
            linewidth=1.45,
            zorder=4,
            gid=f"{gid_prefix}-observation-cute",
        )
        axis.scatter(
            triton_xs,
            [point.triton_median_us for point in shape_points],
            s=52,
            marker="s",
            facecolor=palette.triton,
            edgecolor=palette.background,
            linewidth=1.45,
            zorder=4,
            gid=f"{gid_prefix}-observation-triton",
        )

    axis.set_title(title, loc="left", fontweight="bold", fontsize=12)
    axis.set_xticks(
        xs,
        [str(time) for time in times],
        rotation=0,
        ha="center",
    )
    head_counts = {heads for (_, heads), _ in groups}
    shape_suffix = (
        f"fixed H{next(iter(head_counts))}" if len(head_counts) == 1 else "grouped by B/H"
    )
    axis.set_xlabel(f"Sequence length T (log₂ spacing) · {shape_suffix}")

    if log_latency:
        axis.set_yscale("log", base=2)
        lower_power = math.floor(math.log2(minimum)) - 0.45
        maximum_power = math.log2(maximum)
        upper_power = lower_power + (maximum_power - lower_power) / _CHUNK_DATA_TOP_FRACTION
        ticks = [
            2.0**power for power in range(math.ceil(lower_power), math.floor(maximum_power) + 1)
        ]
        axis.set_yticks(ticks, [f"{tick:g}" for tick in ticks])
        axis.set_ylabel("Median latency (µs / call, log₂ scale)")
        axis.set_ylim(2.0**lower_power, 2.0**upper_power)
        axis.grid(axis="y", which="major", color=palette.grid, linewidth=0.8)
    else:
        axis.set_ylabel("Median latency (µs / call)")
        axis.set_ylim(0.0, maximum / _CHUNK_DATA_TOP_FRACTION)
        axis.grid(axis="y", color=palette.grid, linewidth=0.8)
    axis.yaxis.set_label_coords(-0.04, _CHUNK_RAIL_BOTTOM / 2.0)
    axis.grid(axis="x", color=palette.grid, linewidth=0.6, alpha=0.55)
    axis.set_axisbelow(True)
    axis.set_gid(f"{gid_prefix}-multi-shape-panel")

    axis.add_patch(
        Rectangle(
            (0.0, _CHUNK_RAIL_BOTTOM),
            1.0,
            1.0 - _CHUNK_RAIL_BOTTOM,
            transform=axis.transAxes,
            facecolor=palette.background,
            edgecolor="none",
            zorder=5,
            gid=f"{gid_prefix}-value-rail",
        )
    )
    row_height = (1.0 - _CHUNK_RAIL_BOTTOM) / len(groups)
    row_ys = [_CHUNK_RAIL_BOTTOM + (index + 0.5) * row_height for index in range(len(groups))]
    for index in range(len(groups) + 1):
        y = _CHUNK_RAIL_BOTTOM + index * row_height
        axis.plot(
            [0.0, 1.0],
            [y, y],
            transform=axis.transAxes,
            color=palette.grid,
            linewidth=0.75,
            zorder=6,
            clip_on=False,
            gid=f"{gid_prefix}-rail-separator",
        )

    rail_transform = axis.get_xaxis_transform()
    label_fontsize = 7.5 if len(times) >= 11 else 7.8
    fixed_heads = len(head_counts) == 1
    for ((batch, heads_for_group), shape_points), y in zip(groups, row_ys, strict=True):
        key = f"B{batch}" if fixed_heads else f"B{batch}/H{heads_for_group}"
        axis.plot(
            [-0.008, 0.004],
            [y, y],
            transform=axis.transAxes,
            color=palette.foreground,
            linestyle=linestyle_by_batch[batch],
            linewidth=1.8,
            zorder=7,
            clip_on=False,
            gid=f"{gid_prefix}-rail-key-line",
        )
        axis.text(
            -0.012,
            y,
            key,
            transform=axis.transAxes,
            ha="right",
            va="center",
            color=palette.foreground,
            fontsize=7.5,
            fontweight="bold",
            zorder=7,
            clip_on=False,
            gid=f"{gid_prefix}-rail-key",
        )
        for point in shape_points:
            label, color = _speedup_label(point.speedup, palette)
            axis.text(
                math.log2(point.time),
                y,
                label,
                transform=rail_transform,
                ha="center",
                va="center",
                color=color,
                fontsize=label_fontsize,
                fontweight="bold",
                zorder=7,
                gid=f"{gid_prefix}-rail-label",
            )


def _plot_token_panel(
    axis: Any,
    points: list[BenchmarkPoint],
    *,
    palette: PlotPalette = _LIGHT_PALETTE,
) -> None:
    _plot_multi_shape_scaling_panel(
        axis,
        points,
        MODE_TITLES["token-forward"],
        log_latency=True,
        palette=palette,
        gid_prefix="token",
        connect_pairs=True,
    )


def _render_benchmark_figure(
    plt: Any,
    suite: BenchmarkSuite,
    output: Path,
    *,
    title: str | None,
    palette: PlotPalette,
) -> None:
    by_mode = {mode: [point for point in suite.points if point.mode == mode] for mode in MODES}
    missing = [mode for mode, points in by_mode.items() if not points]
    if missing:
        raise ValueError(f"plot requires all three modes; missing: {', '.join(missing)}")

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(16.5, 15.2),
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0), "hspace": 0.47},
    )
    _plot_chunk_panel(
        axes[0],
        by_mode["chunk-forward"],
        MODE_TITLES["chunk-forward"],
        log_latency=True,
        palette=palette,
    )
    _plot_chunk_panel(
        axes[1],
        by_mode["chunk-backward"],
        MODE_TITLES["chunk-backward"],
        log_latency=True,
        palette=palette,
    )
    _plot_token_panel(axes[2], by_mode["token-forward"], palette=palette)

    display_device = suite.device.removeprefix("NVIDIA ")
    figure.suptitle(
        title or f"Gated DeltaNet-2 latency on {display_device}",
        x=0.065,
        y=0.982,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    normalized = "Q/K L2-normalized" if suite.qk_l2_normalized else "Q/K not normalized"
    figure.text(
        0.065,
        0.951,
        f"{suite.dtype.upper()} · K=V=128 · scale={suite.scale:g} · {normalized} · "
        "lower is better · "
        "labels are official/CuTe speedup",
        ha="left",
        fontsize=10,
        color=palette.tick,
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=palette.cute,
            marker="o",
            markerfacecolor=palette.cute,
            markeredgecolor=palette.background,
            markeredgewidth=1.2,
            linestyle="none",
            label="CuTe SM120",
        ),
        Line2D(
            [0],
            [0],
            color=palette.triton,
            marker="s",
            markerfacecolor=palette.triton,
            markeredgecolor=palette.background,
            markeredgewidth=1.2,
            linestyle="none",
            label="Official Triton",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.955, 0.986),
        frameon=False,
        ncol=2,
        fontsize=10,
    )
    figure.text(
        0.065,
        0.025,
        f"Synchronized CUDA events · PyTorch {suite.torch_version} / CUDA {suite.cuda_runtime} · "
        f"official commit {suite.official_commit[:12]}",
        ha="left",
        fontsize=8.5,
        color=palette.muted,
    )
    figure.subplots_adjust(left=0.095, right=0.975, top=0.92, bottom=0.075)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(
            output,
            dpi=180,
            facecolor=palette.background,
            metadata={"Creator": "gdn2-sm120"},
        )
    finally:
        plt.close(figure)


def _validate_output(output: Path) -> None:
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("output filename must end in .png or .svg")


def render_benchmarks(
    suite: BenchmarkSuite,
    output: Path,
    *,
    title: str | None = None,
    theme: str = "light",
) -> None:
    """Render one themed PNG or SVG without requiring an interactive display."""

    _validate_output(output)
    palette = _plot_palette(theme)
    try:
        import matplotlib
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required; run this command through `uv run --group visualization`"
        ) from error
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    style = {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": palette.border,
        "axes.labelcolor": palette.axis_label,
        "xtick.color": palette.tick,
        "ytick.color": palette.tick,
        "text.color": palette.foreground,
        "figure.facecolor": palette.background,
        "axes.facecolor": palette.background,
        "svg.hashsalt": "gdn2-sm120-benchmarks",
    }
    with matplotlib.rc_context(style):
        _render_benchmark_figure(
            plt,
            suite,
            output,
            title=title,
            palette=palette,
        )


def _default_dark_output(light_output: Path) -> Path:
    return light_output.with_name(f"{light_output.stem}-dark{light_output.suffix}")


def render_benchmark_themes(
    suite: BenchmarkSuite,
    light_output: Path,
    *,
    dark_output: Path | None = None,
    title: str | None = None,
) -> tuple[Path, Path]:
    """Render light/fallback and dark variants, returning their output paths."""

    resolved_dark_output = dark_output or _default_dark_output(light_output)
    if resolved_dark_output == light_output:
        raise ValueError("light and dark output paths must be different")
    _validate_output(light_output)
    _validate_output(resolved_dark_output)
    render_benchmarks(suite, light_output, title=title, theme="light")
    render_benchmarks(suite, resolved_dark_output, title=title, theme="dark")
    return light_output, resolved_dark_output


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
        help=(
            "light/fallback destination .png or .svg; a -dark sibling is also written "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dark-output",
        type=Path,
        help="optional dark-theme destination (default: -dark sibling of --output)",
    )
    parser.add_argument("--title", help="optional figure title override")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        suite = load_benchmarks(args.inputs)
        light_output, dark_output = render_benchmark_themes(
            suite,
            args.output,
            dark_output=args.dark_output,
            title=args.title,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(f"wrote {light_output}")
    print(f"wrote {dark_output}")


if __name__ == "__main__":
    main()

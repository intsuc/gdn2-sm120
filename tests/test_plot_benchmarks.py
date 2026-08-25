from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import pytest

import gdn2_sm120.plot_benchmarks as plot_benchmarks
from gdn2_sm120.plot_benchmarks import (
    load_benchmarks,
    render_benchmark_themes,
    render_benchmarks,
)


def _record(
    mode: str,
    time: int,
    *,
    batch: int = 1,
    heads: int = 16,
    cute_us: float = 40.0,
    triton_us: float = 100.0,
) -> dict[str, object]:
    return {
        "mode": mode,
        "batch": batch,
        "time": time,
        "heads": heads,
        "dtype": "bf16",
        "warmup": 5,
        "repeats": 50,
        "cute": {
            "implementation": "cute-sm120",
            "median_us": cute_us,
            "minimum_us": cute_us - 1.0,
        },
        "triton": {
            "implementation": "official-triton",
            "median_us": triton_us,
            "minimum_us": triton_us - 1.0,
        },
        "speedup": triton_us / cute_us,
        "validation_max_abs": 0.001,
        "official_commit": "95709fc250357c2dd109361c353192f2aa5913f9",
        "fla_commit": "4b02d15d6a68700181b180235be62a9fb95d2a38",
        "qk_l2_normalized": True,
        "scale": 0.125,
        "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "torch_version": "2.13.0+cu130",
        "cuda_runtime": "13.0",
    }


def _write_suite(path: Path, results: list[dict[str, object]]) -> None:
    payload = {"schema_version": 2, "results": results}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _three_mode_suite(path: Path) -> None:
    _write_suite(
        path,
        [
            _record("token-forward", 1, cute_us=21.0, triton_us=25.2),
            _record("chunk-backward", 16, cute_us=112.0, triton_us=277.0),
            _record("chunk-backward", 16384, cute_us=5810.6, triton_us=6353.5),
            _record("chunk-forward", 64, cute_us=64.0, triton_us=185.0),
            _record("chunk-forward", 16, cute_us=43.5, triton_us=189.7),
        ],
    )


def _mixed_batch_chunk_records(mode: str = "chunk-forward") -> list[dict[str, object]]:
    return [
        _record(mode, 16, batch=1, cute_us=40.0, triton_us=80.0),
        _record(mode, 64, batch=1, cute_us=60.0, triton_us=90.0),
        _record(mode, 16, batch=2, cute_us=75.0, triton_us=90.0),
        _record(mode, 128, batch=2, cute_us=110.0, triton_us=99.0),
        _record(mode, 64, batch=4, cute_us=150.0, triton_us=165.0),
        _record(mode, 256, batch=4, cute_us=210.0, triton_us=252.0),
    ]


def _mixed_batch_token_records() -> list[dict[str, object]]:
    return [
        _record(
            "token-forward",
            time,
            batch=batch,
            cute_us=18.0 + 7.0 * batch + 0.35 * time,
            triton_us=22.0 + 8.5 * batch + 0.42 * time,
        )
        for batch in (1, 2, 4)
        for time in (1, 2, 4, 8, 16, 32, 64, 128)
    ]


def test_loads_and_sorts_current_suite(tmp_path: Path) -> None:
    source = tmp_path / "suite.json"
    _three_mode_suite(source)

    suite = load_benchmarks([source])

    assert [(point.mode, point.time) for point in suite.points] == [
        ("chunk-forward", 16),
        ("chunk-forward", 64),
        ("chunk-backward", 16),
        ("chunk-backward", 16384),
        ("token-forward", 1),
    ]
    assert suite.points[0].speedup == pytest.approx(189.7 / 43.5)
    assert suite.device.startswith("NVIDIA RTX PRO 6000")


def test_loads_same_chunk_times_for_distinct_batches(tmp_path: Path) -> None:
    source = tmp_path / "mixed-batch.json"
    records = _mixed_batch_chunk_records()
    records.append(_record("chunk-forward", 16, batch=4, cute_us=130.0, triton_us=143.0))
    _write_suite(source, records)

    points = load_benchmarks([source]).points

    assert {(point.batch, point.time, point.heads) for point in points} == {
        (1, 16, 16),
        (1, 64, 16),
        (2, 16, 16),
        (2, 128, 16),
        (4, 16, 16),
        (4, 64, 16),
        (4, 256, 16),
    }


def test_tracked_sweeps_cover_all_published_batches_and_lengths() -> None:
    source = Path(__file__).parents[1] / "docs/data/benchmark-results-sm120.json"
    points = load_benchmarks([source]).points
    chunk_times = (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    complete_chunks = {(batch, time, 16) for batch in (1, 2, 4) for time in chunk_times}
    token_times = (1, 2, 4, 8, 16, 32, 64, 128)
    expected_by_mode = {
        "chunk-forward": complete_chunks,
        # B4/T32768 backward exceeds the CuTe per-launch 4-GiB byte-address
        # range for one saved state-boundary tensor and is intentionally absent.
        "chunk-backward": complete_chunks - {(4, 32768, 16)},
        "token-forward": {(batch, time, 16) for batch in (1, 2, 4) for time in token_times},
    }

    for mode, expected in expected_by_mode.items():
        actual = {(point.batch, point.time, point.heads) for point in points if point.mode == mode}
        assert actual == expected


def test_rejects_duplicate_logical_shape(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.json"
    point = _record("chunk-forward", 16)
    _write_suite(source, [point, copy.deepcopy(point)])

    with pytest.raises(ValueError, match="duplicate logical benchmark key"):
        load_benchmarks([source])


def test_rejects_stale_or_malformed_measurements(tmp_path: Path) -> None:
    bad_speedup = _record("chunk-forward", 16)
    bad_speedup["speedup"] = 99.0
    source = tmp_path / "bad-speedup.json"
    _write_suite(source, [bad_speedup])
    with pytest.raises(ValueError, match="disagrees with medians"):
        load_benchmarks([source])

    missing_validation = _record("chunk-forward", 16)
    missing_validation.pop("validation_max_abs")
    _write_suite(source, [missing_validation])
    with pytest.raises(ValueError, match="validation_max_abs must be numeric"):
        load_benchmarks([source])

    no_baseline = _record("chunk-forward", 16)
    no_baseline["triton"] = None
    _write_suite(source, [no_baseline])
    with pytest.raises(ValueError, match="triton timing is required"):
        load_benchmarks([source])


def test_rejects_mixed_environment(tmp_path: Path) -> None:
    first = _record("chunk-forward", 16)
    second = _record("chunk-backward", 16)
    second["device"] = "another GPU"
    source = tmp_path / "mixed.json"
    _write_suite(source, [first, second])

    with pytest.raises(ValueError, match="mix device"):
        load_benchmarks([source])

    second = _record("chunk-backward", 16)
    second["scale"] = 0.25
    _write_suite(source, [first, second])
    with pytest.raises(ValueError, match="mix scale"):
        load_benchmarks([source])


def test_rejects_nonfinite_latency(tmp_path: Path) -> None:
    point = _record("chunk-forward", 16)
    assert isinstance(point["cute"], dict)
    point["cute"]["median_us"] = float("nan")
    source = tmp_path / "nonfinite.json"
    _write_suite(source, [point])

    with pytest.raises(ValueError, match="positive and finite"):
        load_benchmarks([source])


def test_renders_headless_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    source = tmp_path / "suite.json"
    destination = tmp_path / "plot.png"
    _three_mode_suite(source)

    chunk_scales: dict[str, str] = {}
    plot_chunk_panel = plot_benchmarks._plot_chunk_panel

    def capture_chunk_scale(*args: object, **kwargs: object) -> None:
        plot_chunk_panel(*args, **kwargs)
        axis, _, title = args
        chunk_scales[str(title)] = str(axis.get_yscale())

    monkeypatch.setattr(plot_benchmarks, "_plot_chunk_panel", capture_chunk_scale)

    render_benchmarks(load_benchmarks([source]), destination)

    image = destination.read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 10_000
    assert chunk_scales == {"Chunk forward": "log", "Chunk backward": "log"}


def test_main_panels_are_stacked_vertically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "suite.json"
    destination = tmp_path / "plot.png"
    _three_mode_suite(source)
    suite = load_benchmarks([source])
    real_subplots = plt.subplots
    captured: dict[str, object] = {}

    def capture_subplots(*args: object, **kwargs: object) -> tuple[object, object]:
        figure, axes = real_subplots(*args, **kwargs)
        captured.update(args=args, kwargs=kwargs, figure=figure, axes=axes)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)

    plot_benchmarks._render_benchmark_figure(
        plt,
        suite,
        destination,
        title=None,
        palette=plot_benchmarks._plot_palette("light"),
    )

    assert captured["args"] == (3, 1)
    axes = list(captured["axes"])
    assert len(axes) == 3
    assert [axis.get_title(loc="left") for axis in axes] == [
        "Chunk forward",
        "Chunk backward",
        "Token forward",
    ]
    positions = [axis.get_position() for axis in axes]
    assert [position.x0 for position in positions] == pytest.approx([positions[0].x0] * 3)
    assert [position.width for position in positions] == pytest.approx([positions[0].width] * 3)
    assert positions[0].y0 > positions[1].y1
    assert positions[1].y0 > positions[2].y1
    plt.close(captured["figure"])


def test_renders_light_and_dark_pngs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import image as matplotlib_image

    source = tmp_path / "suite.json"
    light_output = tmp_path / "plot.png"
    dark_output = tmp_path / "plot-dark.png"
    _three_mode_suite(source)
    suite = load_benchmarks([source])

    outputs = render_benchmark_themes(suite, light_output)

    assert outputs == (light_output, dark_output)
    for output in outputs:
        image = output.read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(image) > 10_000
    assert light_output.read_bytes() != dark_output.read_bytes()
    assert matplotlib_image.imread(light_output)[0, 0, :3] == pytest.approx((1.0, 1.0, 1.0))
    assert matplotlib_image.imread(dark_output)[0, 0, :3] == pytest.approx(
        (13 / 255, 17 / 255, 23 / 255)
    )

    with pytest.raises(ValueError, match="must be different"):
        render_benchmark_themes(suite, light_output, dark_output=light_output)
    partial_output = tmp_path / "partial.png"
    with pytest.raises(ValueError, match="must end in"):
        render_benchmark_themes(suite, partial_output, dark_output=tmp_path / "dark.txt")
    assert not partial_output.exists()
    with pytest.raises(ValueError, match="unsupported plot theme"):
        render_benchmarks(suite, tmp_path / "invalid.png", theme="sepia")


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_multi_batch_token_sweeps_use_chunk_style_series_and_speedup_rows(
    tmp_path: Path,
    theme: str,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "token-sweep.json"
    _write_suite(source, _mixed_batch_token_records())
    points = list(load_benchmarks([source]).points)
    palette = plot_benchmarks._plot_palette(theme)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_token_panel(axis, points, palette=palette)
        figure.canvas.draw()

        trends = [line for line in axis.lines if line.get_gid() == "token-trend"]
        assert axis.get_gid() == "token-multi-shape-panel"
        assert axis.get_yscale() == "log"
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · fixed H16"
        assert len(trends) == 6
        expected_trends = Counter(
            (
                color,
                plot_benchmarks._BATCH_LINESTYLES[index],
                tuple(
                    getattr(point, implementation)
                    for point in sorted(
                        (candidate for candidate in points if candidate.batch == batch),
                        key=lambda point: point.time,
                    )
                ),
            )
            for index, batch in enumerate((1, 2, 4))
            for color, implementation in (
                (palette.cute, "cute_median_us"),
                (palette.triton, "triton_median_us"),
            )
        )
        actual_trends = Counter(
            (line.get_color(), line.get_linestyle(), tuple(line.get_ydata())) for line in trends
        )
        assert actual_trends == expected_trends
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "1",
            "2",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",
        ]
        assert [text.get_text() for text in axis.texts if text.get_gid() == "token-rail-key"] == [
            "B1",
            "B2",
            "B4",
        ]
        labels = [text for text in axis.texts if text.get_gid() == "token-rail-label"]
        assert Counter(text.get_text() for text in labels) == Counter(
            plot_benchmarks._speedup_label(point.speedup, palette)[0] for point in points
        )
        assert len(labels) == 24
        connectors = [line for line in axis.lines if line.get_gid() == "token-pair-connector"]
        assert len(connectors) == 24
        midpoints_by_tick: dict[float, list[float]] = {float(x): [] for x in range(8)}
        for connector in connectors:
            connector_xs = [float(x) for x in connector.get_xdata()]
            assert connector_xs[0] < connector_xs[1]
            midpoint = sum(connector_xs) / 2.0
            tick = min(midpoints_by_tick, key=lambda true_x: abs(midpoint - true_x))
            assert abs(midpoint - tick) <= 0.21
            assert all(abs(x - tick) <= 0.26 for x in connector_xs)
            midpoints_by_tick[tick].append(midpoint)
        assert all(
            len(midpoints) == len({round(midpoint, 8) for midpoint in midpoints}) == 3
            for midpoints in midpoints_by_tick.values()
        )
        assert [patch.get_gid() for patch in axis.patches] == ["token-value-rail"]
        assert Counter(collection.get_gid() for collection in axis.collections) == Counter(
            {"token-observation-cute": 3, "token-observation-triton": 3}
        )
    finally:
        plt.close(figure)


def test_single_batch_token_sweep_uses_the_shared_scaling_panel(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "single-batch-token-sweep.json"
    _write_suite(
        source,
        [
            _record(
                "token-forward",
                time,
                batch=2,
                cute_us=25.0 + time,
                triton_us=30.0 + time,
            )
            for time in (1, 2, 4, 8)
        ],
    )
    points = list(load_benchmarks([source]).points)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_token_panel(axis, points)
        figure.canvas.draw()

        assert axis.get_gid() == "token-multi-shape-panel"
        assert axis.get_yscale() == "log"
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · fixed H16"
        assert len([line for line in axis.lines if line.get_gid() == "token-trend"]) == 2
        assert [text.get_text() for text in axis.texts if text.get_gid() == "token-rail-key"] == [
            "B2"
        ]
        assert len([text for text in axis.texts if text.get_gid() == "token-rail-label"]) == 4
        assert len(
            [line for line in axis.lines if line.get_gid() == "token-pair-connector"]
        ) == len(points)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("theme", ("light", "dark"))
def test_chunk_values_render_in_a_rail_above_the_data(tmp_path: Path, theme: str) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    source = tmp_path / "single-batch-chunks.json"
    _write_suite(
        source,
        [
            _record(mode, time, cute_us=40.0 + time, triton_us=70.0 + time * 1.25)
            for mode in ("chunk-forward", "chunk-backward")
            for time in (16, 64, 128)
        ],
    )
    suite = load_benchmarks([source])
    palette = plot_benchmarks._plot_palette(theme)

    for mode in ("chunk-forward", "chunk-backward"):
        points = [point for point in suite.points if point.mode == mode]
        figure, axis = plt.subplots()
        try:
            plot_benchmarks._plot_chunk_panel(
                axis,
                points,
                mode,
                log_latency=True,
                palette=palette,
            )
            figure.canvas.draw()

            rail = next(patch for patch in axis.patches if patch.get_gid() == "chunk-value-rail")
            assert to_hex(rail.get_facecolor()).upper() == palette.background
            rail_bottom = min(
                axis.transAxes.inverted().transform(rail.get_transform().transform(vertex))[1]
                for vertex in rail.get_path().vertices
            )

            series = [
                line for line in axis.lines if line.get_label() in {"CuTe SM120", "Official Triton"}
            ]
            assert {line.get_color() for line in series} == {palette.cute, palette.triton}
            data_top = max(
                axis.transAxes.inverted().transform(line.get_transform().transform((x, y)))[1]
                for line in series
                for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True)
            )
            assert data_top < rail_bottom

            labels = [text for text in axis.texts if text.get_gid() == "chunk-rail-label"]
            actual = Counter(text.get_text() for text in labels)
            expected = Counter(
                label
                for point in points
                for label in (
                    plot_benchmarks._speedup_label(point.speedup)[0],
                    f"{point.triton_median_us:.1f}",
                    f"{point.cute_median_us:.1f}",
                )
            )
            assert actual == expected
            assert all(
                axis.transAxes.inverted().transform(
                    text.get_transform().transform(text.get_position())
                )[1]
                > rail_bottom
                for text in labels
            )
        finally:
            plt.close(figure)


@pytest.mark.parametrize("theme", ("light", "dark"))
@pytest.mark.parametrize("mode", ("chunk-forward", "chunk-backward"))
def test_mixed_batch_chunk_sweeps_use_dodged_dumbbells_trends_and_speedup_rows(
    tmp_path: Path,
    theme: str,
    mode: str,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "mixed-batch-chunks.json"
    records = _mixed_batch_chunk_records(mode)
    records.append(_record(mode, 16, batch=4, cute_us=130.0, triton_us=143.0))
    _write_suite(source, records)
    points = list(load_benchmarks([source]).points)
    palette = plot_benchmarks._plot_palette(theme)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_chunk_panel(
            axis,
            points,
            plot_benchmarks.MODE_TITLES[mode],
            log_latency=True,
            palette=palette,
        )
        figure.canvas.draw()

        groups = {(point.batch, point.heads) for point in points}
        trends = [line for line in axis.lines if line.get_gid() == "chunk-trend"]
        connectors = [line for line in axis.lines if line.get_gid() == "chunk-pair-connector"]
        cute_collections = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "chunk-observation-cute"
        ]
        triton_collections = [
            collection
            for collection in axis.collections
            if collection.get_gid() == "chunk-observation-triton"
        ]

        assert axis.get_gid() == "chunk-multi-shape-panel"
        assert len(trends) == 2 * len(groups)
        assert Counter(line.get_color() for line in trends) == Counter(
            {palette.cute: len(groups), palette.triton: len(groups)}
        )
        expected_trend_values = Counter(
            tuple(
                getattr(point, implementation)
                for point in sorted(
                    (candidate for candidate in points if candidate.batch == batch),
                    key=lambda point: point.time,
                )
            )
            for batch in (1, 2, 4)
            for implementation in ("cute_median_us", "triton_median_us")
        )
        assert Counter(tuple(line.get_ydata()) for line in trends) == expected_trend_values

        assert len(connectors) == len(points)
        assert len(cute_collections) == len(groups)
        assert len(triton_collections) == len(groups)
        cute_offsets = [
            (float(x), float(y))
            for collection in cute_collections
            for x, y in collection.get_offsets()
        ]
        triton_offsets = [
            (float(x), float(y))
            for collection in triton_collections
            for x, y in collection.get_offsets()
        ]
        assert len(cute_offsets) == len(points)
        assert len(triton_offsets) == len(points)
        assert Counter(round(y, 8) for _, y in cute_offsets) == Counter(
            round(point.cute_median_us, 8) for point in points
        )
        assert Counter(round(y, 8) for _, y in triton_offsets) == Counter(
            round(point.triton_median_us, 8) for point in points
        )

        true_xs = [math.log2(point.time) for point in points]
        connector_midpoints_by_tick: dict[float, list[float]] = {}
        for connector in connectors:
            connector_xs = [float(x) for x in connector.get_xdata()]
            connector_ys = [float(y) for y in connector.get_ydata()]
            assert len(connector_xs) == len(connector_ys) == 2
            assert connector_xs[0] < connector_xs[1]
            midpoint = sum(connector_xs) / 2.0
            tick = min(true_xs, key=lambda true_x: abs(midpoint - true_x))
            assert abs(midpoint - tick) <= 0.21
            assert all(abs(x - tick) <= 0.26 for x in connector_xs)
            connector_midpoints_by_tick.setdefault(tick, []).append(midpoint)
            assert any(
                x == pytest.approx(connector_xs[0]) and y == pytest.approx(connector_ys[0])
                for x, y in cute_offsets
            )
            assert any(
                x == pytest.approx(connector_xs[1]) and y == pytest.approx(connector_ys[1])
                for x, y in triton_offsets
            )
        for midpoints in connector_midpoints_by_tick.values():
            assert len({round(midpoint, 8) for midpoint in midpoints}) == len(midpoints)

        assert list(axis.get_xticks()) == pytest.approx([4.0, 6.0, 7.0, 8.0])
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["16", "64", "128", "256"]
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · fixed H16"

        keys = [text for text in axis.texts if text.get_gid() == "chunk-rail-key"]
        assert [text.get_text() for text in keys] == ["B1", "B2", "B4"]
        renderer = figure.canvas.get_renderer()
        y_label_bounds = axis.yaxis.label.get_window_extent(renderer)
        assert all(not y_label_bounds.overlaps(key.get_window_extent(renderer)) for key in keys)

        labels = [text for text in axis.texts if text.get_gid() == "chunk-rail-label"]
        assert Counter(text.get_text() for text in labels) == Counter(
            plot_benchmarks._speedup_label(point.speedup, palette)[0] for point in points
        )
        assert len(labels) == len(points)
        assert Counter(round(text.get_position()[0], 8) for text in labels) == Counter(
            round(math.log2(point.time), 8) for point in points
        )
        assert all(text.get_text().endswith("×") for text in labels)
        assert len([patch for patch in axis.patches if patch.get_gid() == "chunk-value-rail"]) == 1
    finally:
        plt.close(figure)


def test_chunk_dodge_separates_groups_with_the_same_batch_and_different_heads(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "same-batch-different-heads.json"
    records = [
        _record(
            "chunk-forward",
            time,
            batch=1,
            heads=heads,
            cute_us=40.0 + time,
            triton_us=80.0 + time,
        )
        for heads in (16, 24)
        for time in (16, 64)
    ]
    _write_suite(source, records)
    points = list(load_benchmarks([source]).points)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_chunk_panel(
            axis,
            points,
            "Chunk forward",
            log_latency=True,
        )
        figure.canvas.draw()

        connectors = [line for line in axis.lines if line.get_gid() == "chunk-pair-connector"]
        assert len(connectors) == len(points)
        midpoints_by_time: dict[int, list[float]] = {16: [], 64: []}
        for connector in connectors:
            midpoint = sum(float(x) for x in connector.get_xdata()) / 2.0
            time = min(midpoints_by_time, key=lambda value: abs(midpoint - math.log2(value)))
            assert abs(midpoint - math.log2(time)) <= 0.21
            midpoints_by_time[time].append(midpoint)
        assert all(
            len(midpoints) == len({round(midpoint, 8) for midpoint in midpoints}) == 2
            for midpoints in midpoints_by_time.values()
        )

        assert [text.get_text() for text in axis.texts if text.get_gid() == "chunk-rail-key"] == [
            "B1/H16",
            "B1/H24",
        ]
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · grouped by B/H"
        assert (
            len(
                [
                    collection
                    for collection in axis.collections
                    if collection.get_gid()
                    in {"chunk-observation-cute", "chunk-observation-triton"}
                ]
            )
            == 4
        )
    finally:
        plt.close(figure)


def test_chunk_dodge_adapts_to_close_ticks_and_many_groups() -> None:
    shapes = [(batch, 16) for batch in range(1, 7)]
    times = [16, 17]

    centers, implementation_dodge = plot_benchmarks._chunk_dodge_geometry(shapes, times)

    ordered_centers = [centers[shape] for shape in shapes]
    center_steps = [
        right - left for left, right in zip(ordered_centers, ordered_centers[1:], strict=False)
    ]
    tick_gap = math.log2(times[1]) - math.log2(times[0])
    assert ordered_centers == sorted(ordered_centers)
    maximum_offset = max(abs(center) + implementation_dodge for center in ordered_centers)
    assert maximum_offset <= plot_benchmarks._CHUNK_TICK_GAP_FRACTION * tick_gap + 1e-12
    assert 2.0 * implementation_dodge < min(center_steps)

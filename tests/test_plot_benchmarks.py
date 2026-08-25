from __future__ import annotations

import copy
import json
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
            _record("token-forward", 1, heads=32, cute_us=21.0, triton_us=25.2),
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


def test_tracked_chunk_sweeps_cover_all_published_batches_and_lengths() -> None:
    source = Path(__file__).parents[1] / "docs/data/benchmark-results-sm120.json"
    points = load_benchmarks([source]).points
    times = (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    complete = {(batch, time, 16) for batch in (1, 2, 4) for time in times}
    expected_by_mode = {
        "chunk-forward": complete,
        # B4/T32768 backward exceeds the CuTe per-launch 4-GiB byte-address
        # range for one saved state-boundary tensor and is intentionally absent.
        "chunk-backward": complete - {(4, 32768, 16)},
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
def test_fixed_shape_token_sweep_uses_log2_t_lines(
    tmp_path: Path,
    theme: str,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "token-sweep.json"
    _write_suite(
        source,
        [
            _record(
                "token-forward",
                time,
                heads=32,
                cute_us=20.0 + time,
                triton_us=24.0 + time * 1.2,
            )
            for time in (1, 2, 4, 8, 16, 32, 64, 128)
        ],
    )
    points = list(load_benchmarks([source]).points)
    palette = plot_benchmarks._plot_palette(theme)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_token_panel(axis, points, palette=palette)
        figure.canvas.draw()

        assert axis.get_gid() == "token-scaling-panel"
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · B1/H32"
        series = [
            line for line in axis.lines if line.get_label() in {"CuTe SM120", "Official Triton"}
        ]
        assert len(series) == 2
        assert {line.get_color() for line in series} == {palette.cute, palette.triton}
        assert all(list(line.get_xdata()) == pytest.approx(range(8)) for line in series)
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
        labels = [text.get_text() for text in axis.texts if text.get_gid() == "token-speedup-label"]
        assert labels == [plot_benchmarks._speedup_label(point.speedup)[0] for point in points]
        assert not axis.patches
    finally:
        plt.close(figure)


def test_mixed_token_shapes_keep_grouped_bar_fallback(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "mixed-token-shapes.json"
    _write_suite(
        source,
        [
            _record("token-forward", 1, heads=32, cute_us=21.0, triton_us=25.0),
            _record("token-forward", 16, heads=16, cute_us=35.0, triton_us=36.0),
            _record("token-forward", 32, batch=2, heads=16, cute_us=50.0, triton_us=52.0),
        ],
    )
    points = list(load_benchmarks([source]).points)
    figure, axis = plt.subplots()
    try:
        plot_benchmarks._plot_token_panel(axis, points)
        figure.canvas.draw()

        assert axis.get_gid() != "token-scaling-panel"
        assert axis.get_xlabel() == "Measured shape (mixed B/H; not a scaling line)"
        assert len(axis.patches) == 2 * len(points)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == [
            "B1 · T1\nH32",
            "B1 · T16\nH16",
            "B2 · T32\nH16",
        ]
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
def test_mixed_batch_chunk_sweeps_use_independent_lines_union_ticks_and_speedup_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    theme: str,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    source = tmp_path / "mixed-batch-chunks.json"
    records = _mixed_batch_chunk_records()
    _write_suite(source, records)
    points = list(load_benchmarks([source]).points)
    palette = plot_benchmarks._plot_palette(theme)
    figure, axis = plt.subplots()
    connectors: list[object] = []
    crossovers: list[object] = []
    monkeypatch.setattr(axis, "vlines", lambda *args, **kwargs: connectors.append((args, kwargs)))
    monkeypatch.setattr(axis, "axvspan", lambda *args, **kwargs: crossovers.append((args, kwargs)))
    try:
        plot_benchmarks._plot_chunk_panel(
            axis,
            points,
            "Chunk forward",
            log_latency=True,
            palette=palette,
        )
        figure.canvas.draw()

        series = [line for line in axis.lines if line.get_gid() == "chunk-series"]
        assert len(series) == 6
        assert [line.get_color() for line in series] == [
            palette.cute,
            palette.triton,
        ] * 3
        assert [line.get_marker() for line in series] == ["o", "s"] * 3
        assert [line.get_linestyle() for line in series] == ["-", "-", "--", "--", ":", ":"]
        assert [list(line.get_xdata()) for line in series] == [
            [4.0, 6.0],
            [4.0, 6.0],
            [4.0, 7.0],
            [4.0, 7.0],
            [6.0, 8.0],
            [6.0, 8.0],
        ]
        assert [line.get_label() for line in series] == [
            "CuTe SM120",
            "Official Triton",
            "_nolegend_",
            "_nolegend_",
            "_nolegend_",
            "_nolegend_",
        ]
        assert connectors == []
        assert crossovers == []

        assert list(axis.get_xticks()) == pytest.approx([4.0, 6.0, 7.0, 8.0])
        assert [tick.get_text() for tick in axis.get_xticklabels()] == ["16", "64", "128", "256"]
        assert axis.get_xlabel() == "Sequence length T (log₂ spacing) · fixed H16"

        keys = [text.get_text() for text in axis.texts if text.get_gid() == "chunk-rail-key"]
        assert keys == ["B1", "B2", "B4"]
        labels = [text.get_text() for text in axis.texts if text.get_gid() == "chunk-rail-label"]
        assert Counter(labels) == Counter(
            plot_benchmarks._speedup_label(point.speedup, palette)[0] for point in points
        )
        assert len(labels) == len(points)
        assert all(label.endswith("×") for label in labels)
        assert len([patch for patch in axis.patches if patch.get_gid() == "chunk-value-rail"]) == 1
    finally:
        plt.close(figure)

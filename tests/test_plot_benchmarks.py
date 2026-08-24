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
    heads: int = 16,
    cute_us: float = 40.0,
    triton_us: float = 100.0,
) -> dict[str, object]:
    return {
        "mode": mode,
        "batch": 1,
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
def test_chunk_values_render_in_a_rail_above_the_data(theme: str) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    source = Path(__file__).parents[1] / "docs/data/benchmark-results-sm120.json"
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

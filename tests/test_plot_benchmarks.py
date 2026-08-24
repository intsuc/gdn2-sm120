from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gdn2_sm120.plot_benchmarks import load_benchmarks, render_benchmarks


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
        "qk_l2_normalized": True,
        "device": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "torch_version": "2.13.0+cu130",
        "cuda_runtime": "13.0",
    }


def _write_suite(path: Path, results: list[dict[str, object]]) -> None:
    payload = {"schema_version": 1, "results": results}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _three_mode_suite(path: Path) -> None:
    _write_suite(
        path,
        [
            _record("token-forward", 1, heads=32, cute_us=21.0, triton_us=25.2),
            _record("chunk-backward", 16, cute_us=112.0, triton_us=277.0),
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


def test_rejects_nonfinite_latency(tmp_path: Path) -> None:
    point = _record("chunk-forward", 16)
    assert isinstance(point["cute"], dict)
    point["cute"]["median_us"] = float("nan")
    source = tmp_path / "nonfinite.json"
    _write_suite(source, [point])

    with pytest.raises(ValueError, match="positive and finite"):
        load_benchmarks([source])


def test_renders_headless_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    source = tmp_path / "suite.json"
    destination = tmp_path / "plot.png"
    _three_mode_suite(source)

    render_benchmarks(load_benchmarks([source]), destination)

    image = destination.read_bytes()
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(image) > 10_000

"""Audit locks for the repository's real ingestion keyframe selector.

These tests deliberately assert literal outputs.  They are evidence for
INGEST-KF01, not a second implementation of the selection algorithm.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from mvr.ingestion.keyframes import make_keyframes, select_frame_indices
from mvr.ingestion.models import Shot


def _make_shot(*, frame_count: int, fps: float, method: str = "pyscenedetect") -> Shot:
    return Shot("AUDIT", method, 7, 0, frame_count - 1, fps)


@pytest.mark.parametrize(
    ("label", "fps", "frame_count", "expected_indices", "expected_timestamps"),
    [
        ("0.1s", 30.0, 3, [1], [1 / 30]),
        ("1s", 25.0, 25, [12], [0.48]),
        ("threshold-epsilon", 25.0, 299, [149], [5.96]),
        ("threshold", 25.0, 300, [149, 200], [5.96, 8.0]),
        ("threshold+epsilon", 25.0, 301, [150, 200], [6.0, 8.0]),
        ("2*interval", 30.0, 480, [239, 240], [239 / 30, 8.0]),
        ("2*interval+epsilon", 30.0, 481, [240], [8.0]),
        (
            "30s-at-29.97fps",
            29.97,
            899,
            [240, 449, 480, 720],
            [240 / 29.97, 449 / 29.97, 480 / 29.97, 720 / 29.97],
        ),
        (
            "60s-at-29.97fps",
            29.97,
            1798,
            [240, 480, 720, 898, 960, 1200, 1440, 1680],
            [
                240 / 29.97,
                480 / 29.97,
                720 / 29.97,
                898 / 29.97,
                960 / 29.97,
                1200 / 29.97,
                1440 / 29.97,
                1680 / 29.97,
            ],
        ),
    ],
)
def test_audit_boundary_outputs_from_real_make_keyframes(
    label: str,
    fps: float,
    frame_count: int,
    expected_indices: list[int],
    expected_timestamps: list[float],
) -> None:
    del label
    keyframes = make_keyframes([_make_shot(frame_count=frame_count, fps=fps)], 12.0, 8.0)

    assert [item.frame_idx for item in keyframes] == expected_indices
    assert [item.timestamp for item in keyframes] == pytest.approx(expected_timestamps)
    assert [item.ordinal for item in keyframes] == list(range(1, len(keyframes) + 1))
    assert [item.keyframe_id for item in keyframes] == [
        f"AUDIT_S0007_K{ordinal:02d}"
        for ordinal in range(1, len(keyframes) + 1)
    ]


@pytest.mark.parametrize(
    ("fps", "interval", "expected"),
    [
        (25.0, 0.1, [2, 4, 6, 8]),  # round(2.5) -> 2 (ties to even)
        (30.0, 0.05, [2, 4, 6, 8]),  # round(1.5) -> 2 (ties to even)
        (29.97, 0.1, [3, 4, 6]),  # round(2.997) -> 3; end frame 9 is excluded
    ],
)
def test_audit_between_frame_step_rounding_uses_python_round(
    fps: float,
    interval: float,
    expected: list[int],
) -> None:
    shot = _make_shot(frame_count=10, fps=fps)
    assert select_frame_indices(shot, 0.0, interval) == expected


def test_audit_periodic_endpoint_is_exclusive_and_midpoint_duplicate_is_removed() -> None:
    endpoint = _make_shot(frame_count=10, fps=30.0)
    assert select_frame_indices(endpoint, 0.0, 0.1) == [3, 4, 6]

    duplicate = _make_shot(frame_count=481, fps=30.0)
    assert select_frame_indices(duplicate, 12.0, 8.0) == [240]


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("pyscenedetect", [240, 299, 480]),
        ("transnetv2", [240, 299, 480]),
    ],
)
def test_audit_detector_label_does_not_change_selection(method: str, expected: list[int]) -> None:
    shot = _make_shot(frame_count=600, fps=30.0, method=method)
    assert select_frame_indices(shot, 12.0, 8.0) == expected


@pytest.mark.parametrize(
    ("threshold", "interval", "expected_by_duration"),
    [
        (
            12.0,
            8.0,
            {
                5: [74],
                10: [149],
                12: [179, 240],
                15: [224, 240],
                20: [240, 299, 480],
                30: [240, 449, 480, 720],
                45: [240, 480, 674, 720, 960, 1200],
                60: [240, 480, 720, 899, 960, 1200, 1440, 1680],
            },
        ),
        (
            15.0,
            5.0,
            {
                5: [74],
                10: [149],
                12: [179],
                15: [150, 224, 300],
                20: [150, 299, 300, 450],
                30: [150, 300, 449, 450, 600, 750],
                45: [150, 300, 450, 600, 674, 750, 900, 1050, 1200],
                60: [150, 300, 450, 600, 750, 899, 900, 1050, 1200, 1350, 1500, 1650],
            },
        ),
    ],
)
def test_audit_active_profile_examples_at_30fps(
    threshold: float,
    interval: float,
    expected_by_duration: dict[int, list[int]],
) -> None:
    observed = {
        seconds: select_frame_indices(
            _make_shot(frame_count=seconds * 30, fps=30.0),
            threshold,
            interval,
        )
        for seconds in expected_by_duration
    }
    assert observed == expected_by_duration


def test_audit_package_cli_ingest_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from mvr import cli

    calls: list[dict] = []
    monkeypatch.setattr(cli, "run_pipeline", lambda **values: calls.append(values))
    cli.main(["ingest", str(tmp_path / "input.mp4"), "--output", str(tmp_path / "out")])

    assert calls[0]["methods"] == ["pyscenedetect"]
    assert calls[0]["primary"] == "pyscenedetect"
    assert calls[0]["long_shot_seconds"] == 12.0
    assert calls[0]["sample_every_seconds"] == 8.0
    assert calls[0]["device"] == "auto"


def test_audit_run_ingestion_script_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script_path = Path(__file__).parents[2] / "scripts" / "run_ingestion.py"
    spec = importlib.util.spec_from_file_location("ingest_kf01_run_ingestion", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[dict] = []
    monkeypatch.setattr(module, "run_pipeline", lambda **values: calls.append(values))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "--input", str(tmp_path / "input.mp4"), "--output", str(tmp_path / "out")],
    )
    module.main()

    assert calls[0]["methods"] == ["pyscenedetect"]
    assert calls[0]["primary"] == "pyscenedetect"
    assert calls[0]["long_shot_seconds"] == 15.0
    assert calls[0]["sample_every_seconds"] == 5.0
    assert calls[0]["device"] == "cpu"

import json
from pathlib import Path
from types import SimpleNamespace


def test_probe_video_preserves_fractional_fps_and_count_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from mvr.ingestion import probe

    video_path = tmp_path / "sample.mp4"
    payload = {
        "streams": [
            {
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "nb_frames": "N/A",
                "nb_read_frames": None,
                "duration": None,
                "width": 1920,
                "height": 1080,
            }
        ],
        "format": {"duration": "2.002"},
    }

    monkeypatch.setattr(
        probe.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload)),
    )

    info = probe.probe_video(video_path)

    assert info.video_id == "sample"
    assert info.path == video_path.resolve()
    assert info.fps == 30000 / 1001
    assert info.frame_count == 60
    assert info.duration == 2.002
    assert info.width == 1920
    assert info.height == 1080

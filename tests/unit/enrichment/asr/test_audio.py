import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("timestamp", "duration", "expected"),
    [
        (5.0, 20.0, (2.0, 8.0)),
        (1.0, 20.0, (0.0, 4.0)),
        (19.0, 20.0, (16.0, 20.0)),
    ],
)
def test_calculate_audio_window_clamps_to_video(
    timestamp: float, duration: float, expected: tuple[float, float]
) -> None:
    from mvr.enrichment.asr.audio import calculate_audio_window

    assert calculate_audio_window(timestamp, duration) == expected


def test_cached_provider_decodes_each_video_once_and_slices_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvr.enrichment.asr import audio as audio_module

    videos = [tmp_path / "V1.mp4", tmp_path / "V2.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    decoded: list[str] = []
    samples = np.arange(20, dtype=np.float32)

    def fake_run(command, **kwargs):
        if "ffprobe" in command[0]:
            return SimpleNamespace(stdout=json.dumps({"streams": [{"index": 0}]}))
        decoded.append(Path(command[command.index("-i") + 1]).name)
        kwargs["stdout"].write(samples.astype("<f4").tobytes())
        return SimpleNamespace(stderr=b"")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    provider = audio_module.FfmpegCachedAudioWindowProvider(
        tmp_path / "pcm", cleanup=True
    )

    assert provider.has_audio_stream(videos[0]) is True
    assert provider.decode(videos[0], 0.2, 0.6, 10).tolist() == [2.0, 3.0, 4.0, 5.0]
    assert provider.decode(videos[0], 1.0, 1.3, 10).tolist() == [10.0, 11.0, 12.0]
    first_pcm = next((tmp_path / "pcm").glob("*.f32"))
    assert provider.decode(videos[1], 0.0, 0.2, 10).tolist() == [0.0, 1.0]

    assert decoded == ["V1.mp4", "V2.mp4"]
    assert not first_pcm.exists()
    provider.close()
    assert not list((tmp_path / "pcm").glob("*.f32"))

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


def calculate_audio_window(
    timestamp: float,
    video_duration: float,
    pre_seconds: float = 3.0,
    post_seconds: float = 3.0,
) -> tuple[float, float]:
    if timestamp < 0 or video_duration < 0:
        raise ValueError("timestamp and video_duration must be non-negative")
    if pre_seconds < 0 or post_seconds < 0:
        raise ValueError("pre_seconds and post_seconds must be non-negative")
    start = max(0.0, timestamp - pre_seconds)
    end = min(video_duration, timestamp + post_seconds)
    if end < start:
        raise ValueError(
            f"keyframe timestamp {timestamp} is beyond video duration {video_duration}"
        )
    return start, end


class FfmpegAudioWindowProvider:
    """Decode one independent mono float32 audio window through FFmpeg."""

    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> None:
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def has_audio_stream(self, video_path: Path) -> bool:
        command = [
            self.ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise AudioDecodeError(f"could not inspect audio stream: {exc}") from exc
        return bool(json.loads(result.stdout).get("streams"))

    def decode(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        sample_rate: int,
    ) -> np.ndarray:
        duration = end_seconds - start_seconds
        if duration <= 0:
            return np.empty(0, dtype=np.float32)
        command = [
            self.ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", b"")
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise AudioDecodeError(detail or str(exc)) from exc
        return np.frombuffer(result.stdout, dtype="<f4").copy()

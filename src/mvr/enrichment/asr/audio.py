from __future__ import annotations

import hashlib
import json
import os
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


class FfmpegCachedAudioWindowProvider(FfmpegAudioWindowProvider):
    """Decode each video's mono audio once, then slice keyframe windows locally.

    The ASR pipeline still asks for one independent ``[t-pre, t+post]`` window
    per keyframe.  This provider only changes the transport boundary: FFmpeg
    writes one full-video float32 PCM file, subsequent windows are copied from
    a memory map, and the previous video's PCM is evicted when processing
    advances to another video.  Keeping one decoded video at a time bounds RAM
    and disk while avoiding one FFmpeg process per keyframe.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        ffmpeg: str = "ffmpeg",
        ffprobe: str = "ffprobe",
        *,
        cleanup: bool = True,
    ) -> None:
        super().__init__(ffmpeg=ffmpeg, ffprobe=ffprobe)
        self.cache_dir = Path(cache_dir)
        self.cleanup = cleanup
        self._active_identity: tuple[Path, int, int, int] | None = None
        self._active_pcm: Path | None = None

    @staticmethod
    def _identity(video_path: Path, sample_rate: int) -> tuple[Path, int, int, int]:
        resolved = Path(video_path).resolve()
        stat = resolved.stat()
        return resolved, stat.st_size, stat.st_mtime_ns, sample_rate

    def _pcm_path(self, identity: tuple[Path, int, int, int]) -> Path:
        digest = hashlib.sha256(
            "\0".join(str(value) for value in identity).encode("utf-8")
        ).hexdigest()[:24]
        return self.cache_dir / f"mvr-asr-{digest}.f32"

    def _discard_active(self) -> None:
        previous = self._active_pcm
        self._active_identity = None
        self._active_pcm = None
        if self.cleanup and previous is not None:
            previous.unlink(missing_ok=True)

    def _decode_full_video(self, video_path: Path, sample_rate: int, target: Path) -> None:
        command = [
            self.ffmpeg,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("wb") as stream:
                subprocess.run(
                    command,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.PIPE,
                )
            if temporary.stat().st_size % np.dtype("<f4").itemsize:
                raise AudioDecodeError(
                    f"decoded PCM has invalid byte length: {temporary.stat().st_size}"
                )
            os.replace(temporary, target)
        except (OSError, subprocess.CalledProcessError) as exc:
            temporary.unlink(missing_ok=True)
            stderr = getattr(exc, "stderr", b"") or b""
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise AudioDecodeError(detail or str(exc)) from exc
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _ensure_pcm(self, video_path: Path, sample_rate: int) -> Path:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        identity = self._identity(video_path, sample_rate)
        if identity == self._active_identity and self._active_pcm is not None:
            return self._active_pcm

        target = self._pcm_path(identity)
        if not target.is_file() or target.stat().st_size % np.dtype("<f4").itemsize:
            self._decode_full_video(identity[0], sample_rate, target)

        self._discard_active()
        self._active_identity = identity
        self._active_pcm = target
        return target

    def decode(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        sample_rate: int,
    ) -> np.ndarray:
        if start_seconds < 0 or end_seconds < start_seconds:
            raise ValueError("audio window must satisfy 0 <= start <= end")
        if end_seconds == start_seconds:
            return np.empty(0, dtype=np.float32)
        pcm_path = self._ensure_pcm(video_path, sample_rate)
        sample_count = pcm_path.stat().st_size // np.dtype("<f4").itemsize
        if sample_count == 0:
            return np.empty(0, dtype=np.float32)
        start_sample = min(sample_count, max(0, round(start_seconds * sample_rate)))
        end_sample = min(sample_count, max(start_sample, round(end_seconds * sample_rate)))
        mapped = np.memmap(pcm_path, dtype="<f4", mode="r", shape=(sample_count,))
        try:
            return np.asarray(mapped[start_sample:end_sample], dtype=np.float32).copy()
        finally:
            del mapped

    def close(self) -> None:
        self._discard_active()

    def __enter__(self) -> "FfmpegCachedAudioWindowProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

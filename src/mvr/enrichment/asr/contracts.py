from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class SpeechRegion:
    """A speech interval relative to one keyframe audio window."""

    start_seconds: float
    end_seconds: float
    confidence: float | None = None


@dataclass(frozen=True)
class KeyframeAudioSource:
    """Authoritative keyframe identity plus its original video source."""

    video_id: str
    keyframe_id: str
    frame_idx: int
    timestamp: float
    shot_id: int | str | None
    video_path: Path
    video_duration: float


@dataclass(frozen=True)
class ASRSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ASRArtifact:
    video_id: str
    keyframe_id: str
    frame_idx: int
    timestamp: float
    shot_id: int | str | None
    text: str
    segments: tuple[ASRSegment, ...]

    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "keyframe_id": self.keyframe_id,
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "shot_id": self.shot_id,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }


class VADProvider(Protocol):
    name: str
    model_name: str
    version: str | None

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechRegion]: ...


class ASRProvider(Protocol):
    name: str
    model_name: str
    revision: str | None
    device: str

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...


class AudioWindowProvider(Protocol):
    def has_audio_stream(self, video_path: Path) -> bool: ...

    def decode(
        self,
        video_path: Path,
        start_seconds: float,
        end_seconds: float,
        sample_rate: int,
    ) -> np.ndarray: ...

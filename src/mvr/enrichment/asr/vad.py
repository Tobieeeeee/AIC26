from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .contracts import SpeechRegion


def merge_speech_regions(
    regions: list[SpeechRegion], merge_gap_ms: int = 250
) -> list[SpeechRegion]:
    if merge_gap_ms < 0:
        raise ValueError("merge_gap_ms must be non-negative")
    if not regions:
        return []
    ordered = sorted(regions, key=lambda region: (region.start_seconds, region.end_seconds))
    merged: list[SpeechRegion] = [ordered[0]]
    max_gap = merge_gap_ms / 1000.0
    for region in ordered[1:]:
        previous = merged[-1]
        if region.start_seconds - previous.end_seconds <= max_gap:
            confidences = [
                value
                for value in (previous.confidence, region.confidence)
                if value is not None
            ]
            merged[-1] = SpeechRegion(
                previous.start_seconds,
                max(previous.end_seconds, region.end_seconds),
                max(confidences) if confidences else None,
            )
        else:
            merged.append(region)
    return merged


class SileroVADProvider:
    name = "silero"
    model_name = "silero_vad"

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 150,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if min_speech_duration_ms <= 0 or min_silence_duration_ms <= 0:
            raise ValueError("minimum speech and silence durations must be positive")
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD runtime is unavailable; install the 'asr' extra"
            ) from exc
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self._torch = torch
        self._get_speech_timestamps = get_speech_timestamps
        self._model = load_silero_vad()
        try:
            self.version = version("silero-vad")
        except PackageNotFoundError:
            self.version = None

    def detect(self, audio: np.ndarray, sample_rate: int) -> list[SpeechRegion]:
        if sample_rate not in (8000, 16000):
            raise ValueError("Silero VAD supports only 8000 Hz or 16000 Hz")
        tensor = self._torch.from_numpy(np.asarray(audio, dtype=np.float32))
        timestamps = self._get_speech_timestamps(
            tensor,
            self._model,
            threshold=self.threshold,
            sampling_rate=sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            return_seconds=False,
        )
        return [
            SpeechRegion(
                start_seconds=float(item["start"]) / sample_rate,
                end_seconds=float(item["end"]) / sample_rate,
            )
            for item in timestamps
        ]

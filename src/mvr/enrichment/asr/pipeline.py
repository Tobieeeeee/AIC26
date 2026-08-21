from __future__ import annotations

import json
import logging
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .artifacts import (
    FORMAT,
    artifact_path,
    is_valid_artifact,
    read_failed_keys,
    read_warnings,
    write_artifact,
    write_manifest,
    write_report,
)
from .audio import calculate_audio_window
from .chunkformer import normalize_transcript
from .contracts import (
    ASRArtifact,
    ASRProvider,
    ASRSegment,
    AudioWindowProvider,
    KeyframeAudioSource,
    SpeechRegion,
    VADProvider,
)
from .mapping import load_keyframe_audio_sources
from .vad import merge_speech_regions


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ASRPipelineConfig:
    sample_rate: int = 16000
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    merge_gap_ms: int = 250
    force: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.pre_seconds < 0 or self.post_seconds < 0:
            raise ValueError("pre_seconds and post_seconds must be non-negative")
        if self.merge_gap_ms < 0:
            raise ValueError("merge_gap_ms must be non-negative")


def _empty_artifact(source: KeyframeAudioSource) -> ASRArtifact:
    return ASRArtifact(
        video_id=source.video_id,
        keyframe_id=source.keyframe_id,
        frame_idx=source.frame_idx,
        timestamp=source.timestamp,
        shot_id=source.shot_id,
        text="",
        segments=(),
    )


def _clamp_region(region: SpeechRegion, duration: float) -> SpeechRegion | None:
    start = max(0.0, min(duration, region.start_seconds))
    end = max(0.0, min(duration, region.end_seconds))
    if end <= start:
        return None
    return SpeechRegion(start, end, region.confidence)


def process_keyframe(
    source: KeyframeAudioSource,
    *,
    audio_provider: AudioWindowProvider,
    vad_provider: VADProvider,
    asr_provider: ASRProvider,
    config: ASRPipelineConfig,
) -> tuple[ASRArtifact, int]:
    """Run window -> VAD -> ASR for exactly one authoritative keyframe."""

    window_start, window_end = calculate_audio_window(
        source.timestamp,
        source.video_duration,
        config.pre_seconds,
        config.post_seconds,
    )
    audio = np.asarray(
        audio_provider.decode(
            source.video_path, window_start, window_end, config.sample_rate
        ),
        dtype=np.float32,
    )
    window_duration = len(audio) / config.sample_rate
    detected = [
        valid
        for region in vad_provider.detect(audio, config.sample_rate)
        if (valid := _clamp_region(region, window_duration)) is not None
    ]
    regions = merge_speech_regions(detected, config.merge_gap_ms)
    if not regions:
        return _empty_artifact(source), 0

    segments: list[ASRSegment] = []
    for region in regions:
        start_sample = max(0, round(region.start_seconds * config.sample_rate))
        end_sample = min(len(audio), round(region.end_seconds * config.sample_rate))
        speech_audio = audio[start_sample:end_sample]
        text = normalize_transcript(
            asr_provider.transcribe(speech_audio, config.sample_rate)
        )
        segments.append(
            ASRSegment(
                start=round(window_start + region.start_seconds, 6),
                end=round(window_start + region.end_seconds, 6),
                text=text,
            )
        )
    joined = normalize_transcript(" ".join(segment.text for segment in segments if segment.text))
    return (
        ASRArtifact(
            video_id=source.video_id,
            keyframe_id=source.keyframe_id,
            frame_idx=source.frame_idx,
            timestamp=source.timestamp,
            shot_id=source.shot_id,
            text=joined,
            segments=tuple(segments),
        ),
        len(segments),
    )


def _artifact_counts(
    output_root: Path,
    sources: list[KeyframeAudioSource],
    failed_keys: set[tuple[str, str]],
) -> tuple[int, int, int]:
    with_speech = 0
    without_speech = 0
    segments = 0
    for source in sources:
        if (source.video_id, source.keyframe_id) in failed_keys:
            continue
        path = artifact_path(output_root, source.video_id, source.keyframe_id)
        if not is_valid_artifact(path, source):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["segments"]:
            with_speech += 1
            segments += len(payload["segments"])
        else:
            without_speech += 1
    return with_speech, without_speech, segments


def build_asr_features(
    dataset_root: Path | str,
    output_root: Path | str,
    *,
    audio_provider: AudioWindowProvider,
    vad_provider: VADProvider,
    asr_provider: ASRProvider,
    config: ASRPipelineConfig = ASRPipelineConfig(),
    videos: Iterable[str] | None = None,
) -> dict:
    """Build one canonical JSON artifact per independently processed keyframe."""

    started = time.perf_counter()
    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sources = load_keyframe_audio_sources(dataset_root, videos)
    prior_failed = read_failed_keys(output_root)
    errors: list[dict] = []
    source_keys = {(source.video_id, source.keyframe_id) for source in sources}
    warnings: list[dict] = []
    if not config.force:
        warnings = [
            item
            for item in read_warnings(output_root)
            if (item.get("video_id"), item.get("keyframe_id")) in source_keys
        ]
    vad_calls = 0
    asr_calls = 0
    skipped = 0
    audio_state: dict[Path, bool | Exception] = {}

    for source in sources:
        target = artifact_path(output_root, source.video_id, source.keyframe_id)
        key = (source.video_id, source.keyframe_id)
        if (
            not config.force
            and key not in prior_failed
            and is_valid_artifact(target, source)
        ):
            skipped += 1
            continue
        try:
            if not source.video_path.is_file():
                raise FileNotFoundError(f"video not found: {source.video_path}")
            if source.video_duration <= 0:
                raise ValueError("video duration must be positive")
            if source.video_path not in audio_state:
                try:
                    audio_state[source.video_path] = audio_provider.has_audio_stream(
                        source.video_path
                    )
                except Exception as exc:
                    audio_state[source.video_path] = exc
            state = audio_state[source.video_path]
            if isinstance(state, Exception):
                raise state
            if not state:
                write_artifact(target, _empty_artifact(source))
                warnings.append(
                    {
                        "video_id": source.video_id,
                        "keyframe_id": source.keyframe_id,
                        "code": "NO_AUDIO_STREAM",
                    }
                )
                continue
            vad_calls += 1
            artifact, calls = process_keyframe(
                source,
                audio_provider=audio_provider,
                vad_provider=vad_provider,
                asr_provider=asr_provider,
                config=config,
            )
            asr_calls += calls
            write_artifact(target, artifact)
        except Exception as exc:
            LOGGER.exception(
                "ASR processing failed for %s/%s",
                source.video_id,
                source.keyframe_id,
            )
            # The canonical schema remains stable. The separate errors report
            # preserves PROCESSING_ERROR and makes the artifact retryable.
            write_artifact(target, _empty_artifact(source))
            errors.append(
                {
                    "video_id": source.video_id,
                    "keyframe_id": source.keyframe_id,
                    "code": "PROCESSING_ERROR",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

    failed_keys = {(item["video_id"], item["keyframe_id"]) for item in errors}
    warnings = list(
        {
            (item.get("video_id"), item.get("keyframe_id"), item.get("code")): item
            for item in warnings
        }.values()
    )
    write_report(output_root, errors, warnings)
    with_speech, without_speech, segment_count = _artifact_counts(
        output_root, sources, failed_keys
    )
    elapsed = time.perf_counter() - started
    runtime = {
        "elapsed_seconds": round(elapsed, 6),
        "average_ms_per_keyframe": round(elapsed * 1000 / len(sources), 6),
        "keyframes_vad": vad_calls,
        "asr_calls": asr_calls,
        "keyframes_skipped": skipped,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    docker_identity = {
        key: value
        for key, value in {
            "image_tag": os.getenv("MVR_DOCKER_IMAGE_TAG"),
            "image_id": os.getenv("MVR_DOCKER_IMAGE_ID"),
        }.items()
        if value
    }
    if docker_identity:
        runtime["docker"] = docker_identity
    manifest = {
        "format": FORMAT,
        "dataset": dataset_root.name,
        "vad": {
            "provider": vad_provider.name,
            "model": vad_provider.model_name,
            "version": vad_provider.version,
            "threshold": getattr(vad_provider, "threshold", None),
            "min_speech_duration_ms": getattr(
                vad_provider, "min_speech_duration_ms", None
            ),
            "min_silence_duration_ms": getattr(
                vad_provider, "min_silence_duration_ms", None
            ),
            "merge_gap_ms": config.merge_gap_ms,
        },
        "asr": {
            "provider": asr_provider.name,
            "model": asr_provider.model_name,
            "revision": asr_provider.revision,
            "device": asr_provider.device,
        },
        "audio": {
            "sample_rate": config.sample_rate,
            "channels": 1,
            "pre_seconds": config.pre_seconds,
            "post_seconds": config.post_seconds,
        },
        "counts": {
            "keyframes_total": len(sources),
            "keyframes_with_speech": with_speech,
            "keyframes_without_speech": without_speech,
            "keyframes_failed": len(errors),
            "segments_total": segment_count,
        },
        "runtime": runtime,
        "errors_file": "errors.json",
    }
    write_manifest(output_root, manifest)
    return manifest

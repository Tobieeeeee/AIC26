from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mvr.enrichment.asr.contracts import KeyframeAudioSource, SpeechRegion
from mvr.enrichment.asr.pipeline import ASRPipelineConfig, process_keyframe


class FakeAudio:
    def __init__(self, seconds: float = 6.0) -> None:
        self.audio = np.zeros(round(16000 * seconds), dtype=np.float32)
        self.decode_calls = 0

    def has_audio_stream(self, video_path: Path) -> bool:
        return True

    def decode(self, video_path, start_seconds, end_seconds, sample_rate):
        self.decode_calls += 1
        return self.audio[: round((end_seconds - start_seconds) * sample_rate)]


class FakeVAD:
    name = "fake-vad"
    model_name = "fake-vad-model"
    version = "1"
    threshold = 0.5
    min_speech_duration_ms = 250
    min_silence_duration_ms = 150

    def __init__(self, regions: list[SpeechRegion]) -> None:
        self.regions = regions
        self.calls = 0

    def detect(self, audio, sample_rate):
        self.calls += 1
        return list(self.regions)


class FakeASR:
    name = "fake-asr"
    model_name = "fake-asr-model"
    revision = "abc123"
    device = "cpu"

    def __init__(self, texts: list[str]) -> None:
        self.texts = iter(texts)
        self.calls = 0

    def transcribe(self, audio, sample_rate):
        self.calls += 1
        return next(self.texts)


def _source(tmp_path: Path) -> KeyframeAudioSource:
    video = tmp_path / "V001.mp4"
    video.touch(exist_ok=True)
    return KeyframeAudioSource("V001", "000123", 3690, 5.0, "shot_42", video, 20.0)


def _write_mapping(tmp_path: Path, source: KeyframeAudioSource) -> None:
    (tmp_path / "video_metadata.csv").write_text(
        "video_id,video_path,duration_seconds,fps,frame_count,width,height\n"
        f"{source.video_id},{source.video_path},{source.video_duration},30,600,1,1\n",
        encoding="utf-8",
    )
    (tmp_path / "keyframe_mapping.csv").write_text(
        "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"
        f"{source.keyframe_id},{source.video_id},42,1,{source.frame_idx},{source.timestamp},x.jpg\n",
        encoding="utf-8",
    )


def test_no_speech_writes_empty_result_and_never_calls_asr(tmp_path: Path) -> None:
    vad = FakeVAD([])
    asr = FakeASR([])

    artifact, calls = process_keyframe(
        _source(tmp_path),
        audio_provider=FakeAudio(),
        vad_provider=vad,
        asr_provider=asr,
        config=ASRPipelineConfig(),
    )

    assert artifact.text == ""
    assert artifact.segments == ()
    assert calls == 0
    assert asr.calls == 0


def test_one_segment_converts_relative_time_to_absolute(tmp_path: Path) -> None:
    asr = FakeASR(["  Hôm nay   có tin mới.  "])

    artifact, calls = process_keyframe(
        _source(tmp_path),
        audio_provider=FakeAudio(),
        vad_provider=FakeVAD([SpeechRegion(1.2, 3.6)]),
        asr_provider=asr,
        config=ASRPipelineConfig(),
    )

    assert calls == 1
    assert asr.calls == 1
    assert artifact.text == "Hôm nay có tin mới."
    assert artifact.segments[0].to_dict() == {
        "start": 3.2,
        "end": 5.6,
        "text": "Hôm nay có tin mới.",
    }


def test_multiple_segments_are_chronological_and_joined(tmp_path: Path) -> None:
    artifact, _ = process_keyframe(
        _source(tmp_path),
        audio_provider=FakeAudio(),
        vad_provider=FakeVAD(
            [SpeechRegion(4.0, 4.5), SpeechRegion(0.5, 1.0)]
        ),
        asr_provider=FakeASR(["đầu", "cuối"]),
        config=ASRPipelineConfig(merge_gap_ms=100),
    )

    assert artifact.text == "đầu cuối"
    assert [(segment.start, segment.end) for segment in artifact.segments] == [
        (2.5, 3.0),
        (6.0, 6.5),
    ]


def test_nearby_vad_regions_are_merged_before_asr(tmp_path: Path) -> None:
    asr = FakeASR(["merged"])
    artifact, calls = process_keyframe(
        _source(tmp_path),
        audio_provider=FakeAudio(),
        vad_provider=FakeVAD(
            [SpeechRegion(1.0, 2.0), SpeechRegion(2.2, 3.0)]
        ),
        asr_provider=asr,
        config=ASRPipelineConfig(merge_gap_ms=250),
    )

    assert calls == 1
    assert asr.calls == 1
    assert [(segment.start, segment.end) for segment in artifact.segments] == [
        (3.0, 5.0)
    ]


def test_pipeline_serialization_is_deterministic_and_resume_honors_force(
    tmp_path: Path,
) -> None:
    from mvr.enrichment.asr.pipeline import build_asr_features

    source = _source(tmp_path)
    _write_mapping(tmp_path, source)
    output = tmp_path / "asr"
    audio = FakeAudio()

    build_asr_features(
        tmp_path,
        output,
        audio_provider=audio,
        vad_provider=FakeVAD([]),
        asr_provider=FakeASR([]),
    )
    artifact_path = output / "V001" / "000123.json"
    first = artifact_path.read_bytes()
    assert json.loads(first)["frame_idx"] == 3690
    assert audio.decode_calls == 1

    resumed = build_asr_features(
        tmp_path,
        output,
        audio_provider=audio,
        vad_provider=FakeVAD([]),
        asr_provider=FakeASR([]),
    )
    assert audio.decode_calls == 1
    assert resumed["runtime"]["keyframes_skipped"] == 1

    build_asr_features(
        tmp_path,
        output,
        audio_provider=audio,
        vad_provider=FakeVAD([]),
        asr_provider=FakeASR([]),
        config=ASRPipelineConfig(force=True),
    )
    assert audio.decode_calls == 2
    assert artifact_path.read_bytes() == first


def test_no_audio_stream_is_auditable_no_speech_not_failure(tmp_path: Path) -> None:
    from mvr.enrichment.asr.pipeline import build_asr_features

    source = _source(tmp_path)
    _write_mapping(tmp_path, source)

    class NoAudio(FakeAudio):
        def has_audio_stream(self, video_path: Path) -> bool:
            return False

    output = tmp_path / "asr"
    manifest = build_asr_features(
        tmp_path,
        output,
        audio_provider=NoAudio(),
        vad_provider=FakeVAD([]),
        asr_provider=FakeASR([]),
    )

    assert manifest["counts"]["keyframes_without_speech"] == 1
    assert manifest["counts"]["keyframes_failed"] == 0
    report = json.loads((output / "errors.json").read_text(encoding="utf-8"))
    assert report["warnings"][0]["code"] == "NO_AUDIO_STREAM"


def test_processing_error_is_isolated_and_recorded(tmp_path: Path) -> None:
    from mvr.enrichment.asr.pipeline import build_asr_features

    source = _source(tmp_path)
    _write_mapping(tmp_path, source)

    class BrokenAudio(FakeAudio):
        def decode(self, *args, **kwargs):
            raise RuntimeError("decode failed")

    output = tmp_path / "asr"
    manifest = build_asr_features(
        tmp_path,
        output,
        audio_provider=BrokenAudio(),
        vad_provider=FakeVAD([]),
        asr_provider=FakeASR([]),
    )

    assert manifest["counts"]["keyframes_failed"] == 1
    artifact = json.loads(
        (output / "V001" / "000123.json").read_text(encoding="utf-8")
    )
    assert artifact["text"] == ""
    assert artifact["segments"] == []
    report = json.loads((output / "errors.json").read_text(encoding="utf-8"))
    assert report["errors"][0]["code"] == "PROCESSING_ERROR"

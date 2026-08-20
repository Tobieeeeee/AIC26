from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.getenv("MVR_RUN_ASR_RUNTIME_TESTS") != "1",
    reason="set MVR_RUN_ASR_RUNTIME_TESTS=1 for real Silero/ChunkFormer inference",
)
def test_real_per_keyframe_asr_runtime(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg and ffprobe are required")
    dataset_value = os.getenv("MVR_ASR_TEST_DATASET_ROOT")
    video_id = os.getenv("MVR_ASR_TEST_VIDEO_ID")
    if not dataset_value or not video_id:
        pytest.skip("MVR_ASR_TEST_DATASET_ROOT and MVR_ASR_TEST_VIDEO_ID are required")

    from mvr.enrichment.asr.audio import FfmpegAudioWindowProvider
    from mvr.enrichment.asr.chunkformer import ChunkFormerASRProvider
    from mvr.enrichment.asr.pipeline import build_asr_features
    from mvr.enrichment.asr.vad import SileroVADProvider

    manifest = build_asr_features(
        Path(dataset_value),
        tmp_path / "asr",
        audio_provider=FfmpegAudioWindowProvider(),
        vad_provider=SileroVADProvider(),
        asr_provider=ChunkFormerASRProvider(device="cpu", allow_download=False),
        videos=[video_id],
    )

    assert manifest["format"] == "mvr-asr-keyframe-v1"
    assert manifest["counts"]["keyframes_total"] > 0
    artifacts = list((tmp_path / "asr" / video_id).glob("*.json"))
    assert len(artifacts) == manifest["counts"]["keyframes_total"]
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert set(payload) == {
        "video_id", "keyframe_id", "frame_idx", "timestamp", "shot_id", "text", "segments"
    }

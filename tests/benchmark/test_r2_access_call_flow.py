"""Audit-only execution evidence for the Data-02R R2 access-pattern review.

These tests have no network dependency and intentionally use fake inference
runtimes.  They verify repository call flow only; Cloudflare R2 billing counts
must still come from the bucket analytics API during a real canary.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from mvr.enrichment.asr.pipeline import ASRPipelineConfig, build_asr_features
from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder
from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.offline import run_selected_embedding_providers
from mvr.enrichment.embeddings.perception_encoder import PerceptionEncoderImageEncoder
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.feature_extraction import extract_objects


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPECS = load_provider_specs(REPOSITORY_ROOT / "configs" / "embedding_providers.json")


def _write_asr_mapping(root: Path, video: Path) -> None:
    (root / "video_metadata.csv").write_text(
        "video_id,video_path,duration_seconds,fps,frame_count,width,height\n"
        f"V1,{video},20,30,600,1,1\n",
        encoding="utf-8",
    )
    (root / "keyframe_mapping.csv").write_text(
        "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"
        "K1,V1,1,1,30,1.0,V1/K1.jpg\n"
        "K2,V1,1,2,90,3.0,V1/K2.jpg\n",
        encoding="utf-8",
    )


def test_asr_probes_one_video_once_but_decodes_it_once_per_keyframe(tmp_path: Path) -> None:
    class Audio:
        def __init__(self) -> None:
            self.has_audio_calls = 0
            self.decode_calls = 0

        def has_audio_stream(self, video_path: Path) -> bool:
            self.has_audio_calls += 1
            return True

        def decode(self, video_path: Path, start: float, end: float, rate: int) -> np.ndarray:
            self.decode_calls += 1
            return np.zeros(round((end - start) * rate), dtype=np.float32)

    class VAD:
        name = "audit-vad"
        model_name = "audit-vad"
        version = "1"
        threshold = 0.0
        min_speech_duration_ms = 0
        min_silence_duration_ms = 0

        def detect(self, audio: np.ndarray, sample_rate: int) -> list:
            return []

    class ASR:
        name = "audit-asr"
        model_name = "audit-asr"
        revision = "1"
        device = "cpu"

        def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
            raise AssertionError("VAD returned no speech")

    video = tmp_path / "V1.mp4"
    video.touch()
    _write_asr_mapping(tmp_path, video)
    audio = Audio()

    manifest = build_asr_features(
        tmp_path,
        tmp_path / "asr",
        audio_provider=audio,
        vad_provider=VAD(),
        asr_provider=ASR(),
        config=ASRPipelineConfig(),
    )

    assert manifest["counts"]["keyframes_total"] == 2
    assert audio.has_audio_calls == 1
    assert audio.decode_calls == 2


def test_scene_text_groups_mapping_rows_and_calls_the_video_worker_once_per_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mvr.enrichment.ocr import scene_text

    mapping = tmp_path / "keyframe_mapping.csv"
    rows = [
        {"keyframe_id": "K1", "video_id": "V1", "shot_id": 1, "ordinal": 1, "frame_idx": 30, "timestamp": 1.0, "image_path": "V1/K1.jpg"},
        {"keyframe_id": "K2", "video_id": "V1", "shot_id": 1, "ordinal": 2, "frame_idx": 60, "timestamp": 2.0, "image_path": "V1/K2.jpg"},
        {"keyframe_id": "K3", "video_id": "V2", "shot_id": 1, "ordinal": 1, "frame_idx": 30, "timestamp": 1.0, "image_path": "V2/K3.jpg"},
    ]
    with mapping.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_process_video_ocr(*, video_path: Path, keyframes: list[dict], **_: object) -> dict:
        calls.append((video_path.stem, tuple(item["keyframe_id"] for item in keyframes)))
        return {"json_file_count": len(keyframes), "canonical_text_count": 0, "backend": "audit"}

    monkeypatch.setattr(scene_text, "resolve_video_path", lambda root, video_id: root / f"{video_id}.mp4")
    monkeypatch.setattr(scene_text, "process_video_ocr", fake_process_video_ocr)

    manifest = scene_text.process_dataset_ocr(mapping, tmp_path / "videos", tmp_path / "ocr")

    assert manifest["json_file_count"] == 3
    assert calls == [("V1", ("K1", "K2")), ("V2", ("K3",))]


def test_real_visual_adapters_open_the_same_image_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "fixture.jpg"
    image_module.new("RGB", (2, 2), color="white").save(image_path)

    class FeaturesModel:
        def __call__(self, *, image, only_infer: bool):
            return torch.ones((image.shape[0], 3)), None

        def get_image_features(self, **inputs):
            return torch.ones((inputs["pixel_values"].shape[0], 3))

        def encode_image(self, batch, normalize: bool):
            return torch.ones((batch.shape[0], 3))

    def preprocess(_image):
        return torch.zeros((3, 2, 2))

    def processor(*, images, return_tensors: str):
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((len(images), 3, 2, 2))}

    encoders = {
        "beit3": Beit3ImageEncoder(
            spec=SPECS.get("beit3"),
            checkpoint_path=tmp_path / "unused.pth",
            runtime_loader=lambda *_: (torch, FeaturesModel(), (preprocess, "cpu")),
        ),
        "metaclip2": MetaClip2ImageEncoder(
            spec=SPECS.get("metaclip2"),
            runtime_loader=lambda *_: (torch, FeaturesModel(), processor, "cpu"),
        ),
        "perception_encoder": PerceptionEncoderImageEncoder(
            spec=SPECS.get("perception_encoder"),
            checkpoint_path=tmp_path / "unused.pt",
            runtime_loader=lambda *_: (torch, FeaturesModel(), preprocess, "cpu"),
        ),
    }
    original_open = image_module.open
    opens: list[Path] = []

    def recording_open(path, *args, **kwargs):
        opens.append(Path(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(image_module, "open", recording_open)
    run_selected_embedding_providers(
        ("beit3", "metaclip2", "perception_encoder"),
        {provider: (lambda encoder=encoder: encoder.encode_images([image_path])) for provider, encoder in encoders.items()},
    )

    assert opens == [image_path, image_path, image_path]


def test_objects_dispatches_one_detector_call_per_keyframe(tmp_path: Path) -> None:
    class Detector:
        model_name = "audit-detector"
        score_threshold = 0.5
        nms_threshold = 0.5

        def __init__(self) -> None:
            self.paths: list[Path] = []

        def detect(self, path: Path) -> list[dict]:
            self.paths.append(Path(path))
            return []

    root = tmp_path / "keyframes"
    for name in ("K1.jpg", "K2.jpg"):
        path = root / "V1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    rows = [
        {"keyframe_id": "K1", "video_id": "V1", "shot_id": "1", "ordinal": "1", "frame_idx": "30", "timestamp": "1.0", "image_path": "V1/K1.jpg"},
        {"keyframe_id": "K2", "video_id": "V1", "shot_id": "1", "ordinal": "2", "frame_idx": "60", "timestamp": "2.0", "image_path": "V1/K2.jpg"},
    ]
    detector = Detector()

    extract_objects("audit", rows, detector, tmp_path / "features", root, tmp_path / "mapping.csv")

    assert Counter(detector.paths) == Counter({root / "V1" / "K1.jpg": 1, root / "V1" / "K2.jpg": 1})

"""Offline unit tests for the pinned RF-DETR production adapter."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mvr.enrichment.objects.rfdetr import (
    CHECKPOINT_FILENAME,
    RFDETRCheckpointError,
    RFDETRConfig,
    RFDETRDetector,
    add_keyframe_identity,
    normalize_detections,
    verify_checkpoint,
)


class _Detections:
    def __init__(self) -> None:
        # Intentionally lower-confidence first to exercise deterministic sort.
        self.xyxy = np.asarray(
            [[10.1236, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]],
            dtype=np.float32,
        )
        self.confidence = np.asarray([0.5, 0.9], dtype=np.float32)
        self.class_id = np.asarray([3, 1], dtype=np.int64)
        self.data = {"class_name": np.asarray(["car", "person"], dtype=object)}

    def __len__(self) -> int:
        return len(self.xyxy)


def test_normalization_is_canonical_and_deterministic() -> None:
    records = normalize_detections(_Detections())

    assert [record["label"] for record in records] == ["person", "car"]
    assert records[0]["class_id"] == 1
    assert records[0]["confidence"] == records[0]["score"] == pytest.approx(0.9)
    assert records[1]["x1"] == 10.124
    assert records[1]["bbox"] == [10.124, 20.0, 30.0, 40.0]


def test_keyframe_identity_preserves_original_frame_index() -> None:
    records = add_keyframe_identity(
        normalize_detections(_Detections()),
        keyframe_id="video_S0001_K01",
        video_id="video",
        frame_id=261,
    )

    assert all(record["keyframe_id"] == "video_S0001_K01" for record in records)
    assert all(record["video_id"] == "video" for record in records)
    assert all(record["frame_id"] == record["frame_idx"] == 261 for record in records)


def test_checkpoint_verification_rejects_wrong_size(tmp_path) -> None:
    checkpoint = tmp_path / CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"not the checkpoint")

    with pytest.raises(RFDETRCheckpointError, match="size mismatch"):
        verify_checkpoint(checkpoint)


def test_checkpoint_verification_can_validate_small_fixture(tmp_path) -> None:
    checkpoint = tmp_path / CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"fixture")

    assert verify_checkpoint(
        checkpoint,
        expected_bytes=7,
        expected_sha256="f16d05ec6b29248d2c61adb1e9263f78e4f7bace1b955014a2d17872cfe4064d",
    ) == checkpoint.resolve()


def test_runtime_requires_explicit_local_checkpoint(monkeypatch) -> None:
    monkeypatch.delenv("MVR_RFDETR_CHECKPOINT", raising=False)
    detector = RFDETRDetector(RFDETRConfig())

    with pytest.raises(RFDETRCheckpointError, match="Automatic checkpoint download is disabled"):
        detector._checkpoint_candidate()


def test_loader_reuses_single_model_instance(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / CHECKPOINT_FILENAME
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setattr(
        "mvr.enrichment.objects.rfdetr.verify_checkpoint",
        lambda path: checkpoint,
    )
    monkeypatch.setattr(
        "mvr.enrichment.objects.rfdetr.importlib.metadata.version",
        lambda name: "1.9.3",
    )
    calls: list[dict] = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model_config=SimpleNamespace(resolution=704, device="cuda"),
            class_names=["person", "car"],
        )

    detector = RFDETRDetector(
        RFDETRConfig(checkpoint_path=checkpoint),
        model_factory=factory,
    )
    first = detector._load_model()
    second = detector._load_model()

    assert first is second
    assert len(calls) == 1
    assert calls[0] == {"pretrain_weights": str(checkpoint), "device": "cuda"}

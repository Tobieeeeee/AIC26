"""Pinned, local RF-DETR Large production object detector.

The runtime is deliberately offline and fail-closed: callers must provide the
official checkpoint locally (or set ``MVR_RFDETR_CHECKPOINT``), and its exact
size and SHA-256 are verified before ``rfdetr`` is imported or model weights
are loaded. No alternate variant or detector fallback is attempted.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Union

from PIL import Image


PROVIDER_ID = "rfdetr"
MODEL_FAMILY = "RF-DETR"
MODEL_VARIANT = "large"
MODEL_CLASS = "RFDETRLarge"
PACKAGE_NAME = "rfdetr"
PACKAGE_VERSION = "1.9.3"
UPSTREAM_SOURCE = "https://github.com/roboflow/rf-detr"
UPSTREAM_REVISION = "3f7748b1783533bc2e62804ca5975fe9b9cfc041"
CHECKPOINT_FILENAME = "rf-detr-large-2026.pth"
CHECKPOINT_SOURCE = "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth"
CHECKPOINT_BYTES = 135_954_129
CHECKPOINT_SHA256 = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
LICENSE = "Apache-2.0"
VOCABULARY = "MS COCO 2017 (80 labels; sparse category IDs in 90 slots)"
INPUT_RESOLUTION = 704
DEFAULT_SCORE_THRESHOLD = 0.4
CHECKPOINT_ENV = "MVR_RFDETR_CHECKPOINT"


class RFDETRPinError(RuntimeError):
    """Raised when runtime configuration differs from the production pin."""


class RFDETRCheckpointError(RuntimeError):
    """Raised when the local checkpoint is absent or fails integrity checks."""


class RFDETRRuntimeError(RuntimeError):
    """Raised when the pinned package or model cannot be loaded."""


@dataclass(frozen=True)
class RFDETRConfig:
    """Exact production configuration for the released Large detector."""

    provider_id: str = PROVIDER_ID
    variant: str | None = MODEL_VARIANT
    model_class: str | None = MODEL_CLASS
    checkpoint: str | None = CHECKPOINT_FILENAME
    checkpoint_path: Path | str | None = None
    checkpoint_bytes: int | None = CHECKPOINT_BYTES
    checkpoint_sha256: str | None = CHECKPOINT_SHA256
    package: str | None = PACKAGE_NAME
    package_version: str | None = PACKAGE_VERSION
    revision: str | None = UPSTREAM_REVISION
    license: str | None = LICENSE
    vocabulary: str | None = VOCABULARY
    resolution: int | None = INPUT_RESOLUTION
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    nms_threshold: float | None = None
    device: str = "cuda"
    batch_size: int = 1

    @property
    def is_pinned(self) -> bool:
        """Return whether every schema-critical value matches the lock."""

        return (
            self.provider_id == PROVIDER_ID
            and self.variant == MODEL_VARIANT
            and self.model_class == MODEL_CLASS
            and self.checkpoint == CHECKPOINT_FILENAME
            and self.checkpoint_bytes == CHECKPOINT_BYTES
            and self.checkpoint_sha256 == CHECKPOINT_SHA256
            and self.package == PACKAGE_NAME
            and self.package_version == PACKAGE_VERSION
            and self.revision == UPSTREAM_REVISION
            and self.license == LICENSE
            and self.vocabulary == VOCABULARY
            and self.resolution == INPUT_RESOLUTION
            and self.batch_size == 1
        )

    @property
    def pin_status(self) -> str:
        return "pinned" if self.is_pinned else "pending"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a file SHA-256 without loading the checkpoint into RAM."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    path: Path | str,
    *,
    expected_bytes: int = CHECKPOINT_BYTES,
    expected_sha256: str = CHECKPOINT_SHA256,
) -> Path:
    """Resolve and verify the exact released checkpoint."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise RFDETRCheckpointError(f"RF-DETR checkpoint not found: {resolved}")
    if resolved.name != CHECKPOINT_FILENAME:
        raise RFDETRCheckpointError(
            f"RF-DETR checkpoint filename must be {CHECKPOINT_FILENAME!r}, "
            f"got {resolved.name!r}"
        )
    actual_bytes = resolved.stat().st_size
    if actual_bytes != expected_bytes:
        raise RFDETRCheckpointError(
            f"RF-DETR checkpoint size mismatch: expected {expected_bytes}, "
            f"got {actual_bytes} ({resolved})"
        )
    actual_sha256 = sha256_file(resolved)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RFDETRCheckpointError(
            "RF-DETR checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} ({resolved})"
        )
    return resolved


def normalize_detections(detections: Any) -> list[dict[str, Any]]:
    """Normalize a supervision ``Detections`` result deterministically."""

    labels = detections.data.get("class_name")
    if labels is None:
        raise RFDETRRuntimeError("RF-DETR result is missing class_name metadata")

    normalized: list[dict[str, Any]] = []
    for index in range(len(detections)):
        x1, y1, x2, y2 = (
            round(float(coordinate), 3)
            for coordinate in detections.xyxy[index].tolist()
        )
        confidence = round(float(detections.confidence[index]), 6)
        item = {
            "label": str(labels[index]),
            "class_id": int(detections.class_id[index]),
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            # Preserve the established Task04 envelope representation too.
            "score": confidence,
            "bbox": [x1, y1, x2, y2],
        }
        normalized.append(item)

    normalized.sort(
        key=lambda item: (
            -item["confidence"],
            item["class_id"],
            item["label"],
            item["x1"],
            item["y1"],
            item["x2"],
            item["y2"],
        )
    )
    return normalized


def add_keyframe_identity(
    detections: list[Mapping[str, Any]],
    *,
    keyframe_id: str,
    video_id: str,
    frame_id: int,
) -> list[dict[str, Any]]:
    """Attach canonical mapping identity to every detection row."""

    if not keyframe_id or not video_id:
        raise ValueError("keyframe_id and video_id must be non-empty")
    if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return [
        {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "frame_id": frame_id,
            "frame_idx": frame_id,
            **dict(detection),
        }
        for detection in detections
    ]


class RFDETRDetector:
    """Lazy, reusable adapter for exactly ``RFDETRLarge`` production weights."""

    model_family = MODEL_FAMILY
    model_name = "rfdetr_large"
    nms_enabled = False
    nms_mode = "none"
    nms_implementation = "nms_free_set_prediction"

    def __init__(
        self,
        config: RFDETRConfig | None = None,
        *,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or RFDETRConfig()
        self.score_threshold = self.config.score_threshold
        self.nms_threshold = self.config.nms_threshold
        self.device = self.config.device
        self._model_factory = model_factory
        self._model: Any = None
        self._checkpoint_path: Path | None = None

    def _require_pin(self) -> None:
        if not self.config.is_pinned:
            raise RFDETRPinError(
                "RF-DETR production configuration does not match the locked "
                f"{MODEL_CLASS}/{CHECKPOINT_FILENAME} identity"
            )
        if not 0.0 <= self.config.score_threshold <= 1.0:
            raise RFDETRPinError("RF-DETR score_threshold must be between 0 and 1")
        if (
            self.config.device not in {"cuda", "cpu", "auto"}
            and not self.config.device.startswith("cuda:")
        ):
            raise RFDETRPinError(f"unsupported RF-DETR device: {self.config.device!r}")

    def _checkpoint_candidate(self) -> Path:
        explicit = self.config.checkpoint_path
        if explicit is not None:
            return Path(explicit)
        environment = os.environ.get(CHECKPOINT_ENV)
        if environment:
            return Path(environment)
        raise RFDETRCheckpointError(
            "RF-DETR checkpoint path is required for offline production inference; "
            f"pass RFDETRConfig(checkpoint_path=...) or set {CHECKPOINT_ENV}. "
            "Automatic checkpoint download is disabled."
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        self._require_pin()
        checkpoint_path = verify_checkpoint(self._checkpoint_candidate())

        try:
            installed = importlib.metadata.version(PACKAGE_NAME)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RFDETRRuntimeError(
                f"{PACKAGE_NAME} is not installed; install the project's objects extra"
            ) from exc
        if installed != PACKAGE_VERSION:
            raise RFDETRRuntimeError(
                f"expected {PACKAGE_NAME}=={PACKAGE_VERSION}, found {installed}"
            )
        try:
            if self._model_factory is None:
                from rfdetr import RFDETRLarge

                factory = RFDETRLarge
            else:
                factory = self._model_factory
            device = self.config.device
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            model = factory(
                pretrain_weights=str(checkpoint_path),
                device=device,
            )
        except Exception as exc:
            raise RFDETRRuntimeError(
                f"failed to load pinned {MODEL_CLASS} from {checkpoint_path}: {exc}"
            ) from exc

        if int(model.model_config.resolution) != INPUT_RESOLUTION:
            raise RFDETRRuntimeError(
                f"loaded RF-DETR resolution is {model.model_config.resolution}, "
                f"expected {INPUT_RESOLUTION}"
            )
        self._checkpoint_path = checkpoint_path
        self.device = str(model.model_config.device)
        self._model = model
        return model

    def detect(self, image_path: Union[Path, str]) -> list[dict[str, Any]]:
        """Infer one image (batch=1) and return normalized COCO detections."""

        self._require_pin()
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"keyframe image not found: {path}")
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            raw = self._load_model().predict(
                image,
                threshold=self.config.score_threshold,
            )
        except (RFDETRPinError, RFDETRCheckpointError, RFDETRRuntimeError):
            raise
        except Exception as exc:
            raise RFDETRRuntimeError(f"RF-DETR inference failed for {path}: {exc}") from exc
        return normalize_detections(raw)

    def detect_keyframe(
        self,
        image_path: Union[Path, str],
        *,
        keyframe_id: str,
        video_id: str,
        frame_id: int,
    ) -> list[dict[str, Any]]:
        """Infer one canonical keyframe and preserve its mapping identity."""

        return add_keyframe_identity(
            self.detect(image_path),
            keyframe_id=keyframe_id,
            video_id=video_id,
            frame_id=frame_id,
        )

    @property
    def categories(self) -> list[str]:
        """Return the actual package mapping after load, if available."""

        return [] if self._model is None else list(self._model.class_names)

    @property
    def provider_info(self) -> dict[str, Any]:
        return {
            "provider_id": self.config.provider_id,
            "variant": self.config.variant,
            "model_class": self.config.model_class,
            "checkpoint": self.config.checkpoint,
            "checkpoint_path": (
                str(self._checkpoint_path) if self._checkpoint_path is not None else None
            ),
            "checkpoint_bytes": self.config.checkpoint_bytes,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "package": self.config.package,
            "package_version": self.config.package_version,
            "source": UPSTREAM_SOURCE,
            "revision": self.config.revision,
            "license": self.config.license,
            "vocabulary": self.config.vocabulary,
            "resolution": self.config.resolution,
            "score_threshold": self.config.score_threshold,
            "nms_enabled": self.nms_enabled,
            "pin_status": self.config.pin_status,
            "device": self.device,
            "batch_size": self.config.batch_size,
        }

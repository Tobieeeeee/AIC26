"""RF-DETR object detector — canonical production provider.

RF-DETR is the production-default object detector for AIC26. This module
defines the provider identity and the fail-fast contract for unpinned
runtime variants.

The exact RF-DETR variant, checkpoint, and revision are NOT YET PINNED.
Full object extraction is BLOCKED until these are resolved.

This module provides:
- Provider identity (rfdetr)
- Explicit configuration fields for variant/checkpoint/revision
- Fail-fast if production runtime is invoked without a resolved pin
- Same output contract as existing detectors (joinable by keyframe_id)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

LOGGER = logging.getLogger(__name__)

PROVIDER_ID = "rfdetr"


class RFDETRPinError(RuntimeError):
    """Raised when RF-DETR is invoked without a pinned production model."""


@dataclass(frozen=True)
class RFDETRConfig:
    """RF-DETR provider configuration.

    All fields are explicit. If variant or checkpoint is None, the runtime
    will refuse to run inference and raise RFDETRPinError.
    """
    provider_id: str = PROVIDER_ID
    variant: str | None = None
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    revision: str | None = None
    score_threshold: float = 0.5
    nms_threshold: float = 0.5
    device: str = "auto"

    @property
    def is_pinned(self) -> bool:
        """Whether the variant and checkpoint are fully resolved."""
        return self.variant is not None and self.checkpoint is not None

    @property
    def pin_status(self) -> str:
        if self.is_pinned:
            return "pinned"
        return "pending"


class RFDETRDetector:
    """RF-DETR object detector.

    This is a placeholder implementation that enforces the pinning contract.
    When the exact variant and checkpoint are resolved, the runtime loader
    will be implemented to load the actual model.
    """

    def __init__(self, config: RFDETRConfig | None = None) -> None:
        self.config = config or RFDETRConfig()
        self._model: Any = None

    def _require_pin(self) -> None:
        """Fail fast if the RF-DETR runtime is not pinned."""
        if not self.config.is_pinned:
            raise RFDETRPinError(
                "RF-DETR production runtime invoked without a resolved pin. "
                f"variant={self.config.variant!r}, checkpoint={self.config.checkpoint!r}. "
                "Full object extraction is BLOCKED until the exact RF-DETR "
                "variant and checkpoint are pinned in configs/production.yaml."
            )

    def detect(self, image_path: Union[Path, str]) -> list[dict[str, Any]]:
        """Run object detection on a single image.

        Returns a list of detections, each with:
        - label: str
        - class_id: int
        - confidence: float (rounded to 4 decimals)
        - bbox: [x1, y1, x2, y2] (float, pixel coordinates)

        Raises RFDETRPinError if variant/checkpoint are not pinned.
        """
        self._require_pin()

        # Once pinned, the actual implementation would:
        # 1. Load the RF-DETR model on first call
        # 2. Preprocess the image
        # 3. Run inference
        # 4. Post-process and return detections
        #
        # For now, fail fast to signal that implementation is blocked.
        raise RFDETRPinError(
            "RF-DETR inference is not yet implemented. "
            "Pin the exact variant and checkpoint first."
        )

    @property
    def provider_info(self) -> dict[str, Any]:
        """Return provider metadata for manifests."""
        return {
            "provider_id": self.config.provider_id,
            "variant": self.config.variant,
            "checkpoint": self.config.checkpoint,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "revision": self.config.revision,
            "score_threshold": self.config.score_threshold,
            "nms_threshold": self.config.nms_threshold,
            "pin_status": self.config.pin_status,
        }

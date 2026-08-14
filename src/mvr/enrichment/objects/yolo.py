"""YOLOv8n object-count adapter extracted from Lab 6 cell 10."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from typing import Any


DEFAULT_WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.4


class YoloObjectDetector:
    def __init__(
        self,
        *,
        weights: str = DEFAULT_WEIGHTS,
        confidence: float = DEFAULT_CONFIDENCE,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - optional integration
                raise RuntimeError(
                    "Ultralytics is missing; install the project's objects extra"
                ) from exc
            model_factory = YOLO
        self.confidence = confidence
        self.model = model_factory(weights)

    def detect(self, image: Any) -> dict[str, int]:
        result = self.model.predict(
            image,
            conf=self.confidence,
            verbose=False,
        )[0]
        return dict(Counter(result.names[int(class_id)] for class_id in result.boxes.cls))

"""EasyOCR adapter extracted from Lab 6 cell 6."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


OCR_CONFIDENCE = 0.3
LANGUAGES = ["vi", "en"]


def extract_easyocr_text(
    detections: Sequence[Sequence[Any]],
    confidence_threshold: float = OCR_CONFIDENCE,
) -> str:
    return " ".join(
        str(text)
        for _, text, confidence in detections
        if float(confidence) >= confidence_threshold
    )


class EasyOcrEngine:
    """Initialize the reader once and apply Lab 6's text filtering."""

    def __init__(
        self,
        *,
        gpu: bool | None = None,
        confidence_threshold: float = OCR_CONFIDENCE,
        reader_factory: Callable[..., Any] | None = None,
        cuda_available: Callable[[], bool] | None = None,
    ) -> None:
        if gpu is None:
            if cuda_available is None:
                try:
                    import torch
                except ImportError:  # pragma: no cover - EasyOCR normally supplies torch
                    cuda_available = lambda: False
                else:
                    cuda_available = torch.cuda.is_available
            gpu = cuda_available()
        if reader_factory is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover - optional integration
                raise RuntimeError(
                    "EasyOCR is missing; install the project's ocr extra"
                ) from exc
            reader_factory = easyocr.Reader
        self.confidence_threshold = confidence_threshold
        self.reader = reader_factory(LANGUAGES.copy(), gpu=gpu)

    def read_text(self, image: Any) -> str:
        detections = self.reader.readtext(np.asarray(image))
        return extract_easyocr_text(detections, self.confidence_threshold)

"""PaddleOCR 2.x/3.x compatibility adapter from Lab 6 cells 7-8."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .easyocr_engine import OCR_CONFIDENCE


def make_paddle_reader(factory: Callable[..., Any] | None = None) -> Any:
    if factory is None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError(
                "PaddleOCR is missing; install the project's ocr extra"
            ) from exc
        factory = PaddleOCR
    try:
        return factory(
            lang="vi",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    except (TypeError, ValueError):
        return factory(
            lang="vi",
            use_angle_cls=False,
            show_log=False,
            enable_mkldnn=False,
        )


def extract_paddle_lines(
    result: Any,
    confidence_threshold: float = OCR_CONFIDENCE,
) -> list[tuple[str, float | None]]:
    extracted: list[tuple[str, float | None]] = []

    def add(text: Any, confidence: Any) -> None:
        if text is None:
            return
        value = str(text).strip()
        if not value:
            return
        try:
            score = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            score = None
        if score is None or score >= confidence_threshold:
            extracted.append((value, score))

    def walk(node: Any) -> None:
        if node is None:
            return
        if hasattr(node, "to_dict") and not isinstance(node, dict):
            try:
                walk(node.to_dict())
                return
            except Exception:
                pass
        if isinstance(node, dict):
            texts = node.get("rec_texts")
            scores = node.get("rec_scores")
            if texts is not None:
                text_values = list(texts)
                score_values = list(scores) if scores is not None else [None] * len(text_values)
                for text, score in zip(text_values, score_values):
                    add(text, score)
                return
            if "text" in node:
                add(node.get("text"), node.get("score", node.get("confidence")))
                return
            for key in ("res", "result", "data", "ocr_res"):
                if key in node:
                    walk(node[key])
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and isinstance(node[1], (list, tuple)):
                recognition = node[1]
                if len(recognition) >= 2 and isinstance(recognition[0], str):
                    add(recognition[0], recognition[1])
                    return
            if len(node) >= 2 and isinstance(node[0], str):
                add(node[0], node[1])
                return
            for child in node:
                walk(child)

    walk(result)
    unique: list[tuple[str, float | None]] = []
    seen: set[tuple[str, float | None]] = set()
    for text, score in extracted:
        key = (text, None if score is None else round(score, 8))
        if key not in seen:
            seen.add(key)
            unique.append((text, score))
    return unique


def paddle_predict(reader: Any, image: Any) -> Any:
    array = np.asarray(image)
    if hasattr(reader, "predict"):
        return list(reader.predict(array))
    return reader.ocr(array, cls=False)


class PaddleOcrEngine:
    """Use PaddleOCR as Lab 6's selected final OCR output."""

    def __init__(
        self,
        *,
        reader: Any | None = None,
        confidence_threshold: float = OCR_CONFIDENCE,
        reader_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.reader = reader if reader is not None else make_paddle_reader(reader_factory)
        self.confidence_threshold = confidence_threshold

    def read_text(self, image: Any) -> str:
        lines = extract_paddle_lines(
            paddle_predict(self.reader, image),
            self.confidence_threshold,
        )
        return " ".join(text for text, _ in lines)

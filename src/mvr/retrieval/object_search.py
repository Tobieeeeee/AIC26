"""Flexible COCO object filtering adapted from Lab 6 cells 17 and 28."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any


DEFAULT_INDEX_NAME = "keyframes"
COCO_CLASSES = frozenset(
    (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    )
)


def parse_object_conditions(want: str) -> list[tuple[str, int]]:
    if not want or not want.strip():
        raise ValueError("want must contain at least one class:count condition")
    conditions: list[tuple[str, int]] = []
    seen: set[str] = set()
    for raw_condition in want.split(","):
        condition = raw_condition.strip()
        if ":" not in condition:
            raise ValueError(
                f"Invalid condition '{condition}': expected COCO-class:non-negative-count"
            )
        class_name, count_text = (
            part.strip() for part in condition.rsplit(":", 1)
        )
        if not class_name:
            raise ValueError("Object class name cannot be empty")
        if class_name not in COCO_CLASSES:
            raise ValueError(f"Unknown COCO class '{class_name}'")
        if not count_text.isdigit():
            raise ValueError(
                f"Count for '{class_name}' must be a non-negative integer"
            )
        if class_name in seen:
            raise ValueError(f"Duplicate COCO class '{class_name}'")
        seen.add(class_name)
        conditions.append((class_name, int(count_text)))
    return conditions


def build_object_filters(
    conditions: Iterable[tuple[str, int]],
) -> list[dict[str, Any]]:
    return [
        {"range": {f"objects.{class_name}": {"gte": minimum}}}
        for class_name, minimum in conditions
    ]


def search_objects(
    client: Any,
    want: str,
    *,
    top_k: int = 5,
    index_name: str = DEFAULT_INDEX_NAME,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    filters = build_object_filters(parse_object_conditions(want))
    return client.search(
        index=index_name,
        size=top_k,
        query={"bool": {"filter": filters}},
    )

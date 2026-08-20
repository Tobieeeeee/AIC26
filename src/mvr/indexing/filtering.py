"""Task05A feature gates and deterministic text projection helpers."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .normalization import (
    clean_display_text,
    is_punctuation_or_symbol_only,
    normalize_text,
)


OCR_DROP_REASONS = (
    "empty",
    "invalid_conf",
    "invalid_support",
    "below_absolute_conf",
    "low_support_and_not_high_conf",
    "punctuation_only",
)


@dataclass(frozen=True)
class OcrFilterConfig:
    absolute_min_conf: float = 0.50
    high_conf: float = 0.85
    min_temporal_support: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.absolute_min_conf)):
            raise ValueError("absolute_min_conf must be finite")
        if not math.isfinite(float(self.high_conf)):
            raise ValueError("high_conf must be finite")
        if not 0 <= self.absolute_min_conf <= 1:
            raise ValueError("absolute_min_conf must be in [0, 1]")
        if not 0 <= self.high_conf <= 1:
            raise ValueError("high_conf must be in [0, 1]")
        if not isinstance(self.min_temporal_support, int) or isinstance(
            self.min_temporal_support, bool
        ):
            raise ValueError("min_temporal_support must be an integer")
        if self.min_temporal_support < 1:
            raise ValueError("min_temporal_support must be >= 1")


@dataclass(frozen=True)
class AsrFilterConfig:
    min_conf: float | None = None

    def __post_init__(self) -> None:
        if self.min_conf is not None:
            if not math.isfinite(float(self.min_conf)) or not 0 <= self.min_conf <= 1:
                raise ValueError("ASR min_conf must be null or in [0, 1]")


@dataclass
class FilterResult:
    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped_by_reason: Counter[str] = field(default_factory=Counter)
    dropped_samples: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    kept_special_samples: list[dict[str, Any]] = field(default_factory=list)
    deduplicated_count: int = 0

    @property
    def input_count(self) -> int:
        return len(self.kept) + sum(self.dropped_by_reason.values())

    @property
    def dropped_count(self) -> int:
        return sum(self.dropped_by_reason.values())

    def drop_samples_as_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {reason: list(self.dropped_samples.get(reason, [])) for reason in OCR_DROP_REASONS}


def _sample_entry(entry: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    sample: dict[str, Any] = {"reason": reason}
    for key in (
        "keyframe_id",
        "video_id",
        "text",
        "conf",
        "support_count",
        "frame_idx",
    ):
        if key in entry:
            sample[key] = entry[key]
    return sample


def _record_drop(result: FilterResult, reason: str, entry: Mapping[str, Any]) -> None:
    result.dropped_by_reason[reason] += 1
    if len(result.dropped_samples[reason]) < 5:
        result.dropped_samples[reason].append(_sample_entry(entry, reason=reason))


def filter_ocr_entries(
    entries: Iterable[Mapping[str, Any]],
    config: OcrFilterConfig = OcrFilterConfig(),
) -> FilterResult:
    """Apply the approved Task05A OCR quality and lexical-noise gates."""

    result = FilterResult()
    for raw_entry in entries:
        entry = dict(raw_entry)
        display = clean_display_text(entry.get("text"))
        if not display:
            _record_drop(result, "empty", entry)
            continue

        conf_value = entry.get("conf")
        try:
            conf = float(conf_value)
        except (TypeError, ValueError):
            _record_drop(result, "invalid_conf", entry)
            continue
        if not math.isfinite(conf) or not 0 <= conf <= 1:
            _record_drop(result, "invalid_conf", entry)
            continue

        support_value = entry.get("support_count")
        if (
            isinstance(support_value, bool)
            or not isinstance(support_value, int)
            or support_value < 1
        ):
            _record_drop(result, "invalid_support", entry)
            continue

        if conf < config.absolute_min_conf:
            _record_drop(result, "below_absolute_conf", entry)
            continue
        if (
            support_value < config.min_temporal_support
            and conf < config.high_conf
        ):
            _record_drop(result, "low_support_and_not_high_conf", entry)
            continue
        if is_punctuation_or_symbol_only(display):
            _record_drop(result, "punctuation_only", entry)
            continue

        kept = dict(entry)
        kept["text"] = display
        kept["normalized"] = normalize_text(display)
        kept["conf"] = conf
        kept["support_count"] = support_value
        result.kept.append(kept)
        if (
            conf < config.high_conf
            and support_value >= config.min_temporal_support
            and len(result.kept_special_samples) < 5
        ):
            result.kept_special_samples.append(
                _sample_entry(kept, reason="low_conf_high_support_keep")
            )
    for reason in OCR_DROP_REASONS:
        result.dropped_by_reason.setdefault(reason, 0)
    return result


def deduplicate_ocr_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    anchor_frame_idx: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep one representative for each normalized OCR lexical identity."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_entry in entries:
        entry = dict(raw_entry)
        entry["text"] = clean_display_text(entry.get("text"))
        entry["normalized"] = normalize_text(entry.get("text"))
        grouped[entry["normalized"]].append(entry)

    representatives: list[dict[str, Any]] = []
    for normalized, group in grouped.items():
        def rank(entry: Mapping[str, Any]) -> tuple[Any, ...]:
            frame_idx = entry.get("frame_idx")
            try:
                frame_distance = (
                    abs(int(frame_idx) - anchor_frame_idx)
                    if anchor_frame_idx is not None and frame_idx is not None
                    else 0
                )
            except (TypeError, ValueError):
                frame_distance = 0
            return (
                -int(entry.get("support_count", 0)),
                -float(entry.get("conf", 0.0)),
                frame_distance,
                int(frame_idx) if isinstance(frame_idx, int) else 0,
                str(entry.get("text", "")),
            )

        representatives.append(dict(sorted(group, key=rank)[0]))

    representatives.sort(
        key=lambda entry: (
            str(entry.get("normalized", "")),
            str(entry.get("text", "")),
            -int(entry.get("support_count", 0)),
            -float(entry.get("conf", 0.0)),
        )
    )
    return representatives, max(0, len(entries) - len(representatives))


def filter_asr_entries(
    entries: Iterable[Mapping[str, Any]],
    *,
    config: AsrFilterConfig = AsrFilterConfig(),
    confidence_field: str | None = None,
) -> FilterResult:
    """Filter ASR records without creating confidence values.

    ``confidence_field`` is discovered from the actual artifact schema by the
    loader.  If it is ``None``, records are never confidence-filtered and the
    output contains no synthetic confidence field.
    """

    result = FilterResult()
    for raw_entry in entries:
        entry = dict(raw_entry)
        display = clean_display_text(entry.get("text"))
        if not display:
            result.dropped_by_reason["empty"] += 1
            if len(result.dropped_samples["empty"]) < 5:
                result.dropped_samples["empty"].append(_sample_entry(entry, reason="empty"))
            continue

        if confidence_field is not None and confidence_field in entry:
            try:
                conf = float(entry[confidence_field])
            except (TypeError, ValueError):
                result.dropped_by_reason["invalid_conf"] += 1
                continue
            if not math.isfinite(conf) or not 0 <= conf <= 1:
                result.dropped_by_reason["invalid_conf"] += 1
                continue
            if config.min_conf is not None and conf < config.min_conf:
                result.dropped_by_reason["low_conf"] += 1
                if len(result.dropped_samples["low_conf"]) < 5:
                    result.dropped_samples["low_conf"].append(
                        _sample_entry(entry, reason="low_conf")
                    )
                continue
            entry[confidence_field] = conf

        entry["text"] = display
        entry["normalized"] = normalize_text(display)
        result.kept.append(entry)

    for reason in ("empty", "low_conf", "invalid_conf"):
        result.dropped_by_reason.setdefault(reason, 0)
    return result


def deduplicate_asr_entries(
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate ASR text values within one keyframe only."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_entry in entries:
        entry = dict(raw_entry)
        entry["text"] = clean_display_text(entry.get("text"))
        entry["normalized"] = normalize_text(entry.get("text"))
        grouped[entry["normalized"]].append(entry)

    representatives: list[dict[str, Any]] = []
    for normalized, group in grouped.items():
        representatives.append(
            dict(
                sorted(
                    group,
                    key=lambda entry: (
                        -float(entry.get("conf", 0.0))
                        if "conf" in entry
                        else 0.0,
                        str(entry.get("text", "")),
                    ),
                )[0]
            )
        )
    representatives.sort(key=lambda entry: (str(entry.get("normalized", "")), str(entry.get("text", ""))))
    return representatives, max(0, len(entries) - len(representatives))


def project_object_metadata(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Project Task04 object output without another threshold or NMS pass."""

    raw_objects = payload.get("objects", [])
    counts: Counter[str] = Counter()
    if isinstance(raw_objects, Mapping):
        for raw_label, raw_count in raw_objects.items():
            label = clean_display_text(raw_label)
            if not label:
                raise ValueError("object label must be non-empty")
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                raise ValueError("object counts must be non-negative integers")
            counts[label] = raw_count
    elif isinstance(raw_objects, list):
        for raw_object in raw_objects:
            if not isinstance(raw_object, Mapping):
                raise ValueError("objects entries must be objects")
            label = clean_display_text(raw_object.get("label"))
            if not label:
                raise ValueError("object label must be non-empty")
            counts[label] += 1
    else:
        raise ValueError("objects must be a list or count mapping")

    for label, field_name in (("person", "person_count"), ("car", "car_count")):
        if field_name not in payload:
            continue
        raw_count = payload[field_name]
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise ValueError(f"{field_name} must be a non-negative integer")
        # Task04's explicit person/car counts are authoritative.
        counts[label] = raw_count

    projected_counts = {
        label: count for label, count in sorted(counts.items()) if count > 0
    }
    return (
        {
            "labels": list(projected_counts),
            "counts": projected_counts,
        },
        sum(projected_counts.values()),
    )

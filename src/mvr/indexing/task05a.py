"""Task05A v1 feature filtering and index-document preparation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvr.enrichment.embeddings.artifacts import (
    EmbeddingArtifactError,
    load_provider_artifact,
    sha256_file,
)

from .filtering import (
    AsrFilterConfig,
    OcrFilterConfig,
    deduplicate_asr_entries,
    deduplicate_ocr_entries,
    filter_asr_entries,
    filter_ocr_entries,
    project_object_metadata,
)
from .normalization import clean_display_text
from .validation import (
    Task05AValidationError,
    ensure_finite_json,
    identity_key,
    validate_clip_vector,
    validate_task05a_document,
)


DOCUMENT_SCHEMA_VERSION = "1.0"
MANIFEST_SCHEMA_VERSION = "1.0"

# Branch-level JSON metadata is not a keyframe artifact.  Keep this explicit
# and branch-aware so malformed real keyframe JSON is still validated.
AUXILIARY_FILES_BY_BRANCH: dict[str, set[str]] = {
    "asr": {"manifest.json", "errors.json"},
}


@dataclass(frozen=True)
class Task05AConfig:
    ocr: OcrFilterConfig = field(default_factory=OcrFilterConfig)
    asr: AsrFilterConfig = field(default_factory=AsrFilterConfig)
    clip_expected_dimension: int | None = None
    required_embedding_providers: tuple[str, ...] = ()
    sample_limit: int = 5

    def __post_init__(self) -> None:
        if self.clip_expected_dimension is not None and self.clip_expected_dimension < 1:
            raise ValueError("clip_expected_dimension must be positive")
        if self.sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        if len(set(self.required_embedding_providers)) != len(
            self.required_embedding_providers
        ):
            raise ValueError("required_embedding_providers must be unique")


@dataclass(frozen=True)
class CanonicalKeyframe:
    keyframe_id: str
    video_id: str
    shot_id: int
    frame_idx: int
    timestamp: float
    image_path: str

    @property
    def identity(self) -> tuple[str, str]:
        return identity_key(self.video_id, self.keyframe_id)


@dataclass(frozen=True)
class Task05ABuildResult:
    output_root: Path
    manifest_path: Path
    filtering_report_path: Path
    document_count: int
    manifest: dict[str, Any]
    filtering_report: dict[str, Any]


class _IssueCollector:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add(
        self,
        category: str,
        message: str,
        *,
        branch: str,
        keyframe_id: str | None = None,
        video_id: str | None = None,
        path: Path | None = None,
    ) -> None:
        self.counts[category] += 1
        if len(self.samples[category]) >= self.limit:
            return
        sample: dict[str, Any] = {
            "branch": branch,
            "category": category,
            "message": message,
        }
        if keyframe_id is not None:
            sample["keyframe_id"] = keyframe_id
        if video_id is not None:
            sample["video_id"] = video_id
        if path is not None:
            sample["path"] = str(path)
        self.samples[category].append(sample)

    def all_samples(self) -> list[dict[str, Any]]:
        return [
            sample
            for category in sorted(self.samples)
            for sample in self.samples[category]
        ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_manifest(root: Path | None) -> tuple[dict[str, Any], Path | None]:
    if root is None:
        return {}, None
    path = root / "manifest.json"
    if not path.is_file():
        return {}, None
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise Task05AValidationError(f"manifest must be an object: {path}")
    return payload, path


def _manifest_reference(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"exists": False, "path": None, "sha256": None}
    path = root / "manifest.json"
    if not path.is_file():
        return {"exists": False, "path": None, "sha256": None}
    return {
        "exists": True,
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
    }


def load_canonical_mapping(mapping_csv: Path) -> list[CanonicalKeyframe]:
    """Load and strictly validate the canonical Task05A join mapping."""

    mapping_csv = Path(mapping_csv)
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"mapping CSV not found: {mapping_csv}")
    with mapping_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "keyframe_id",
            "video_id",
            "shot_id",
            "frame_idx",
            "timestamp",
        }
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise Task05AValidationError(
                f"mapping missing required columns: {sorted(missing_columns)}"
            )
        rows = list(reader)
    if not rows:
        raise Task05AValidationError("canonical mapping is empty")

    keyframes: list[CanonicalKeyframe] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        keyframe_id = str(row.get("keyframe_id", "")).strip()
        video_id = str(row.get("video_id", "")).strip()
        if not keyframe_id or not video_id:
            raise Task05AValidationError(
                f"invalid blank identity in mapping line {line_number}"
            )
        try:
            shot_id = int(row["shot_id"])
            frame_idx = int(row["frame_idx"])
            timestamp = float(row["timestamp"])
        except (TypeError, ValueError) as exc:
            raise Task05AValidationError(
                f"invalid numeric identity in mapping line {line_number}"
            ) from exc
        if frame_idx < 0 or not math.isfinite(timestamp) or timestamp < 0:
            raise Task05AValidationError(
                f"invalid frame/timestamp in mapping line {line_number}"
            )
        identity = identity_key(video_id, keyframe_id)
        if identity in seen:
            raise Task05AValidationError(
                f"duplicate canonical identity: {video_id}/{keyframe_id}"
            )
        seen.add(identity)
        keyframes.append(
            CanonicalKeyframe(
                keyframe_id=keyframe_id,
                video_id=video_id,
                shot_id=shot_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                image_path=str(row.get("image_path", "")),
            )
        )
    keyframes.sort(
        key=lambda item: (
            item.video_id,
            item.shot_id,
            item.frame_idx,
            item.keyframe_id,
        )
    )
    return keyframes


def _canonical_lookup(
    keyframes: Iterable[CanonicalKeyframe],
) -> tuple[
    dict[tuple[str, str], CanonicalKeyframe],
    dict[str, CanonicalKeyframe],
]:
    by_identity: dict[tuple[str, str], CanonicalKeyframe] = {}
    by_keyframe_id: dict[str, CanonicalKeyframe] = {}
    for keyframe in keyframes:
        by_identity[keyframe.identity] = keyframe
        if keyframe.keyframe_id in by_keyframe_id:
            raise Task05AValidationError(
                f"keyframe_id is not globally unique: {keyframe.keyframe_id}"
            )
        by_keyframe_id[keyframe.keyframe_id] = keyframe
    return by_identity, by_keyframe_id


def _resolve_canonical(
    payload: Mapping[str, Any],
    by_keyframe_id: Mapping[str, CanonicalKeyframe],
    *,
    branch: str,
    path: Path,
    issues: _IssueCollector,
) -> CanonicalKeyframe | None:
    keyframe_id = str(payload.get("keyframe_id", "")).strip()
    video_id = str(payload.get("video_id", "")).strip()
    if not keyframe_id or not video_id:
        issues.add(
            "invalid_schema",
            "missing keyframe_id/video_id",
            branch=branch,
            path=path,
        )
        return None
    canonical = by_keyframe_id.get(keyframe_id)
    if canonical is None:
        issues.add(
            "extra_identity",
            "artifact identity is not in canonical mapping",
            branch=branch,
            keyframe_id=keyframe_id,
            video_id=video_id,
            path=path,
        )
        return None
    if canonical.video_id != video_id:
        issues.add(
            "identity_mismatch",
            f"video_id {video_id!r} != canonical {canonical.video_id!r}",
            branch=branch,
            keyframe_id=keyframe_id,
            video_id=video_id,
            path=path,
        )
        return None
    return canonical


def _validate_branch_identity(
    payload: Mapping[str, Any],
    canonical: CanonicalKeyframe,
    *,
    branch: str,
    path: Path,
    issues: _IssueCollector,
    frame_field: str,
    timestamp_field: str,
) -> bool:
    comparisons = (
        ("shot_id", canonical.shot_id, int),
        (frame_field, canonical.frame_idx, int),
        (timestamp_field, canonical.timestamp, float),
    )
    for field_name, expected, cast in comparisons:
        if field_name not in payload:
            issues.add(
                "invalid_schema",
                f"missing identity field {field_name}",
                branch=branch,
                keyframe_id=canonical.keyframe_id,
                video_id=canonical.video_id,
                path=path,
            )
            return False
        try:
            actual = cast(payload[field_name])
        except (TypeError, ValueError):
            issues.add(
                "invalid_schema",
                f"invalid identity field {field_name}",
                branch=branch,
                keyframe_id=canonical.keyframe_id,
                video_id=canonical.video_id,
                path=path,
            )
            return False
        matches = (
            abs(actual - expected) <= 1e-6
            if field_name == timestamp_field
            else actual == expected
        )
        if not matches:
            issues.add(
                "identity_mismatch",
                f"{field_name} {actual!r} != canonical {expected!r}",
                branch=branch,
                keyframe_id=canonical.keyframe_id,
                video_id=canonical.video_id,
                path=path,
            )
            return False
    return True


def _json_feature_files(
    root: Path,
    excluded: set[str],
    *,
    branch: str | None = None,
) -> list[Path]:
    branch_excluded = set(excluded)
    if branch is not None:
        branch_excluded.update(AUXILIARY_FILES_BY_BRANCH.get(branch, set()))
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.name not in branch_excluded
    )


def _load_ocr(
    root: Path | None,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    by_identity, by_keyframe_id = _canonical_lookup(canonical)
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    drop_counts: Counter[str] = Counter()
    input_entries = 0
    kept_entries = 0
    deduplicated_entries = 0
    files_processed = 0
    seen: set[tuple[str, str]] = set()
    dropped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    special: dict[str, list[dict[str, Any]]] = {
        "high_conf_support_1": [],
        "low_conf_high_support": [],
        "numeric_only": [],
        "dropped_low_support": [],
    }

    if root is None or not root.is_dir():
        issues.add(
            "missing_branch_artifact",
            "OCR root is absent",
            branch="ocr",
            path=root,
        )
    else:
        for path in _json_feature_files(
            root,
            {"manifest.json"},
        ):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                issues.add("invalid_schema", str(exc), branch="ocr", path=path)
                continue
            if not isinstance(payload, dict):
                issues.add("invalid_schema", "OCR JSON must be an object", branch="ocr", path=path)
                continue
            mapped = _resolve_canonical(
                payload,
                by_keyframe_id,
                branch="ocr",
                path=path,
                issues=issues,
            )
            if mapped is None:
                continue
            if mapped.identity in seen:
                issues.add(
                    "duplicate_identity",
                    "duplicate OCR keyframe artifact",
                    branch="ocr",
                    keyframe_id=mapped.keyframe_id,
                    video_id=mapped.video_id,
                    path=path,
                )
                outputs.pop(mapped.identity, None)
                continue
            seen.add(mapped.identity)
            if not _validate_branch_identity(
                payload,
                mapped,
                branch="ocr",
                path=path,
                issues=issues,
                frame_field="anchor_frame_idx",
                timestamp_field="anchor_timestamp",
            ):
                continue
            texts = payload.get("texts", [])
            if not isinstance(texts, list) or not all(
                isinstance(item, Mapping) for item in texts
            ):
                issues.add(
                    "invalid_schema",
                    "OCR texts must be a list of objects",
                    branch="ocr",
                    keyframe_id=mapped.keyframe_id,
                    video_id=mapped.video_id,
                    path=path,
                )
                continue
            files_processed += 1
            prepared: list[dict[str, Any]] = []
            for raw in texts:
                entry = dict(raw)
                entry["keyframe_id"] = mapped.keyframe_id
                entry["video_id"] = mapped.video_id
                if "video_id" in raw and str(raw["video_id"]) != mapped.video_id:
                    issues.add(
                        "identity_mismatch",
                        "OCR text video_id mismatch",
                        branch="ocr",
                        keyframe_id=mapped.keyframe_id,
                        video_id=mapped.video_id,
                        path=path,
                    )
                    continue
                if "shot_id" in raw:
                    try:
                        if int(raw["shot_id"]) != mapped.shot_id:
                            raise ValueError
                    except (TypeError, ValueError):
                        issues.add(
                            "identity_mismatch",
                            "OCR text shot_id mismatch",
                            branch="ocr",
                            keyframe_id=mapped.keyframe_id,
                            video_id=mapped.video_id,
                            path=path,
                        )
                        continue
                prepared.append(entry)

            filtered = filter_ocr_entries(prepared, config.ocr)
            input_entries += filtered.input_count
            kept_entries += len(filtered.kept)
            drop_counts.update(filtered.dropped_by_reason)
            for reason, samples in filtered.dropped_samples.items():
                for sample in samples:
                    if len(dropped_samples[reason]) < config.sample_limit:
                        dropped_samples[reason].append(sample)
            deduplicated, dedup_count = deduplicate_ocr_entries(
                filtered.kept,
                anchor_frame_idx=mapped.frame_idx,
            )
            deduplicated_entries += dedup_count
            projected: list[dict[str, Any]] = []
            for entry in deduplicated:
                item = {
                    "text": entry["text"],
                    "normalized": entry["normalized"],
                    "conf": float(entry["conf"]),
                    "support_count": int(entry["support_count"]),
                }
                projected.append(item)
                sample = {
                    "keyframe_id": mapped.keyframe_id,
                    **item,
                }
                if (
                    item["support_count"] == 1
                    and item["conf"] >= config.ocr.high_conf
                    and len(special["high_conf_support_1"]) < config.sample_limit
                ):
                    special["high_conf_support_1"].append(sample)
                if (
                    item["conf"] < config.ocr.high_conf
                    and item["support_count"] >= config.ocr.min_temporal_support
                    and len(special["low_conf_high_support"]) < config.sample_limit
                ):
                    special["low_conf_high_support"].append(sample)
                compact = item["text"].replace(" ", "")
                if (
                    compact
                    and all(char.isnumeric() for char in compact)
                    and len(special["numeric_only"]) < config.sample_limit
                ):
                    special["numeric_only"].append(sample)
            outputs[mapped.identity] = projected

    for sample in dropped_samples.get("low_support_and_not_high_conf", []):
        if len(special["dropped_low_support"]) < config.sample_limit:
            special["dropped_low_support"].append(sample)
    missing = set(by_identity) - set(outputs) - {
        key for key in seen if key not in outputs
    }
    # Empty OCR is valid and is represented by an empty list in outputs.
    for key in seen:
        outputs.setdefault(key, [])
    missing = set(by_identity) - set(seen)
    for video_id, keyframe_id in sorted(missing):
        issues.add(
            "missing_branch_artifact",
            "OCR artifact missing for canonical keyframe",
            branch="ocr",
            keyframe_id=keyframe_id,
            video_id=video_id,
        )

    stats = {
        "input_entries": input_entries,
        "kept_entries": kept_entries,
        "dropped_by_reason": {
            reason: int(drop_counts.get(reason, 0))
            for reason in (
                "empty",
                "invalid_conf",
                "invalid_support",
                "below_absolute_conf",
                "low_support_and_not_high_conf",
                "punctuation_only",
            )
        },
        "deduplicated_entries": deduplicated_entries,
        "keyframe_files_processed": files_processed,
        "missing_keyframe_artifacts": len(missing),
    }
    samples = {
        "representative_dropped_ocr_samples": {
            reason: samples[: config.sample_limit]
            for reason, samples in sorted(dropped_samples.items())
        },
        "kept_low_conf_high_support_ocr_samples": special["low_conf_high_support"],
        "high_conf_support_1_samples": special["high_conf_support_1"],
        "numeric_only_samples": special["numeric_only"],
        "dropped_low_support_samples": special["dropped_low_support"],
    }
    return outputs, stats, samples


def _extract_asr_records(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str | None]:
    for container_field in ("segments", "utterances", "results", "texts"):
        value = payload.get(container_field)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)], container_field
    if "transcript" in payload or "text" in payload:
        return [payload], None
    return [], None


def _discover_asr_text_field(records: list[Mapping[str, Any]]) -> str | None:
    for candidate in ("transcript", "text", "utterance"):
        if any(candidate in record for record in records):
            return candidate
    return None


def _discover_asr_confidence_field(
    records: list[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> str | None:
    explicit = manifest.get("confidence_field")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    present = [
        candidate
        for candidate in ("confidence", "conf", "score")
        if any(candidate in record for record in records)
    ]
    return present[0] if len(present) == 1 else None


def _load_asr(
    root: Path | None,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    by_identity, by_keyframe_id = _canonical_lookup(canonical)
    manifest, _ = _load_manifest(root)
    raw_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    kept_example: list[dict[str, Any]] = []

    if root is None or not root.is_dir():
        issues.add(
            "missing_branch_artifact",
            "ASR root is absent; no ASR fields were inferred",
            branch="asr",
            path=root,
        )
    else:
        for path in _json_feature_files(
            root,
            {"manifest.json"},
            branch="asr",
        ):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                issues.add("invalid_schema", str(exc), branch="asr", path=path)
                continue
            if not isinstance(payload, dict):
                issues.add("invalid_schema", "ASR JSON must be an object", branch="asr", path=path)
                continue
            mapped = _resolve_canonical(
                payload,
                by_keyframe_id,
                branch="asr",
                path=path,
                issues=issues,
            )
            if mapped is None:
                continue
            if mapped.identity in seen:
                issues.add(
                    "duplicate_identity",
                    "duplicate ASR keyframe artifact",
                    branch="asr",
                    keyframe_id=mapped.keyframe_id,
                    video_id=mapped.video_id,
                    path=path,
                )
                raw_by_identity.pop(mapped.identity, None)
                continue
            seen.add(mapped.identity)
            frame_field = "anchor_frame_idx" if "anchor_frame_idx" in payload else "frame_idx"
            timestamp_field = (
                "anchor_timestamp" if "anchor_timestamp" in payload else "timestamp"
            )
            if not _validate_branch_identity(
                payload,
                mapped,
                branch="asr",
                path=path,
                issues=issues,
                frame_field=frame_field,
                timestamp_field=timestamp_field,
            ):
                continue
            raw_records, _ = _extract_asr_records(payload)
            prepared: list[dict[str, Any]] = []
            for record in raw_records:
                item = dict(record)
                item["keyframe_id"] = mapped.keyframe_id
                item["video_id"] = mapped.video_id
                prepared.append(item)
            raw_by_identity[mapped.identity] = prepared
            all_records.extend(prepared)

    text_field = _discover_asr_text_field(all_records)
    confidence_field = _discover_asr_confidence_field(all_records, manifest)
    confidence_filter_active = (
        confidence_field is not None and config.asr.min_conf is not None
    )
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    drop_counts: Counter[str] = Counter()
    input_entries = 0
    kept_entries = 0
    deduplicated_entries = 0
    dropped_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for identity, records in sorted(raw_by_identity.items()):
        mapped_records: list[dict[str, Any]] = []
        for record in records:
            item = dict(record)
            item["text"] = record.get(text_field) if text_field is not None else None
            mapped_records.append(item)
        filtered = filter_asr_entries(
            mapped_records,
            config=config.asr,
            confidence_field=confidence_field,
        )
        input_entries += filtered.input_count
        kept_entries += len(filtered.kept)
        drop_counts.update(filtered.dropped_by_reason)
        for reason, samples in filtered.dropped_samples.items():
            for sample in samples:
                if len(dropped_samples[reason]) < config.sample_limit:
                    dropped_samples[reason].append(sample)
        deduplicated, count = deduplicate_asr_entries(filtered.kept)
        deduplicated_entries += count
        projected: list[dict[str, Any]] = []
        for entry in deduplicated:
            item = {
                "text": entry["text"],
                "normalized": entry["normalized"],
            }
            if confidence_field is not None and confidence_field in entry:
                item["conf"] = float(entry[confidence_field])
            projected.append(item)
            if len(kept_example) < config.sample_limit:
                kept_example.append({"keyframe_id": identity[1], **item})
        outputs[identity] = projected

    for identity in seen:
        outputs.setdefault(identity, [])
    missing = set(by_identity) - set(seen)
    for video_id, keyframe_id in sorted(missing):
        issues.add(
            "missing_branch_artifact",
            "ASR artifact missing for canonical keyframe",
            branch="asr",
            keyframe_id=keyframe_id,
            video_id=video_id,
        )
    stats = {
        "input_entries": input_entries,
        "kept_entries": kept_entries,
        "dropped_by_reason": {
            reason: int(drop_counts.get(reason, 0))
            for reason in ("empty", "low_conf", "invalid_conf")
        },
        "deduplicated_entries": deduplicated_entries,
        "confidence_field": confidence_field,
        "confidence_filter_active": confidence_filter_active,
        "transcript_field": text_field,
        "missing_keyframe_artifacts": len(missing),
    }
    samples = {
        "asr_kept_samples": kept_example,
        "asr_dropped_samples": {
            reason: values[: config.sample_limit]
            for reason, values in sorted(dropped_samples.items())
        },
    }
    return outputs, stats, samples


def _load_objects(
    root: Path | None,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_identity, by_keyframe_id = _canonical_lookup(canonical)
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[tuple[str, str]] = set()
    object_entries = 0
    source_example: list[dict[str, Any]] = []
    if root is None or not root.is_dir():
        issues.add(
            "missing_branch_artifact",
            "Object root is absent",
            branch="objects",
            path=root,
        )
    else:
        for path in _json_feature_files(root, {"manifest.json", "shot_summary.json"}):
            try:
                payload = _read_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                issues.add("invalid_schema", str(exc), branch="objects", path=path)
                continue
            if not isinstance(payload, dict):
                issues.add("invalid_schema", "Object JSON must be an object", branch="objects", path=path)
                continue
            mapped = _resolve_canonical(
                payload,
                by_keyframe_id,
                branch="objects",
                path=path,
                issues=issues,
            )
            if mapped is None:
                continue
            if mapped.identity in seen:
                issues.add(
                    "duplicate_identity",
                    "duplicate object keyframe artifact",
                    branch="objects",
                    keyframe_id=mapped.keyframe_id,
                    video_id=mapped.video_id,
                    path=path,
                )
                outputs.pop(mapped.identity, None)
                continue
            seen.add(mapped.identity)
            if not _validate_branch_identity(
                payload,
                mapped,
                branch="objects",
                path=path,
                issues=issues,
                frame_field="frame_idx",
                timestamp_field="timestamp",
            ):
                continue
            try:
                projected, projected_count = project_object_metadata(payload)
            except ValueError as exc:
                issues.add(
                    "invalid_schema",
                    str(exc),
                    branch="objects",
                    keyframe_id=mapped.keyframe_id,
                    video_id=mapped.video_id,
                    path=path,
                )
                continue
            outputs[mapped.identity] = projected
            object_entries += projected_count
            if len(source_example) < config.sample_limit:
                source_example.append(
                    {
                        "keyframe_id": mapped.keyframe_id,
                        "source_person_count": payload.get("person_count"),
                        "source_car_count": payload.get("car_count"),
                        "source_object_entries": (
                            len(payload.get("objects", []))
                            if isinstance(payload.get("objects", []), list)
                            else sum(payload.get("objects", {}).values())
                            if isinstance(payload.get("objects", {}), Mapping)
                            else None
                        ),
                        "projected": projected,
                    }
                )
    missing = set(by_identity) - set(seen)
    for video_id, keyframe_id in sorted(missing):
        issues.add(
            "missing_branch_artifact",
            "Object artifact missing for canonical keyframe",
            branch="objects",
            keyframe_id=keyframe_id,
            video_id=video_id,
        )
    stats = {
        "keyframes_processed": len(outputs),
        "object_entries_projected": object_entries,
        "missing_keyframe_artifacts": len(missing),
        "second_threshold_applied": False,
        "second_nms_applied": False,
    }
    return outputs, stats, {"source_to_projected_samples": source_example}


def _clip_expected_dimension(
    config: Task05AConfig,
    manifest: Mapping[str, Any],
) -> int | None:
    if config.clip_expected_dimension is not None:
        return config.clip_expected_dimension
    raw = manifest.get("embedding_dimension")
    try:
        dimension = int(raw)
    except (TypeError, ValueError):
        return None
    return dimension if dimension > 0 else None


def _load_clip(
    root: Path | None,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_identity, by_keyframe_id = _canonical_lookup(canonical)
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_by_identity: dict[tuple[str, str], str] = {
        identity: "missing_mapping" for identity in by_identity
    }
    invalid_samples: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    mapping_mismatches = 0
    manifest, _ = _load_manifest(root)
    expected_dimension = _clip_expected_dimension(config, manifest)
    dtype_names: set[str] = set()
    mapping_rows = 0
    seen: set[tuple[str, str]] = set()
    arrays: dict[Path, np.ndarray] = {}

    if root is None or not root.is_dir():
        issues.add(
            "missing_branch_artifact",
            "CLIP root is absent",
            branch="clip",
            path=root,
        )
    else:
        mapping_name = str(manifest.get("mapping_file", "clip_feature_mapping.csv"))
        mapping_path = root / mapping_name
        if not mapping_path.is_file():
            issues.add(
                "missing_branch_artifact",
                "CLIP mapping CSV is absent",
                branch="clip",
                path=mapping_path,
            )
        else:
            with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            mapping_rows = len(rows)
            for row_number, row in enumerate(rows, start=2):
                keyframe_id = str(row.get("keyframe_id", "")).strip()
                video_id = str(row.get("video_id", "")).strip()
                canonical_row = by_keyframe_id.get(keyframe_id)
                if canonical_row is None:
                    mapping_mismatches += 1
                    issues.add(
                        "extra_identity",
                        "CLIP mapping identity not in canonical mapping",
                        branch="clip",
                        keyframe_id=keyframe_id,
                        video_id=video_id,
                        path=mapping_path,
                    )
                    continue
                identity = canonical_row.identity
                if identity in seen:
                    mapping_mismatches += 1
                    invalid_by_identity[identity] = "duplicate_identity"
                    outputs.pop(identity, None)
                    issues.add(
                        "duplicate_identity",
                        "duplicate CLIP mapping identity",
                        branch="clip",
                        keyframe_id=keyframe_id,
                        video_id=video_id,
                        path=mapping_path,
                    )
                    continue
                seen.add(identity)
                payload = dict(row)
                payload["keyframe_id"] = keyframe_id
                payload["video_id"] = video_id
                if video_id != canonical_row.video_id or not _validate_branch_identity(
                    payload,
                    canonical_row,
                    branch="clip",
                    path=mapping_path,
                    issues=issues,
                    frame_field="frame_idx",
                    timestamp_field="timestamp",
                ):
                    mapping_mismatches += 1
                    invalid_by_identity[identity] = "mapping_mismatch"
                    continue
                feature_file = str(row.get("feature_file", "")).strip()
                try:
                    feature_index = int(row.get("feature_index", ""))
                except (TypeError, ValueError):
                    invalid_by_identity[identity] = "invalid_row_index"
                    continue
                if not feature_file or feature_index < 0:
                    invalid_by_identity[identity] = "invalid_row_index"
                    continue
                feature_path = (root / feature_file).resolve()
                try:
                    feature_path.relative_to(root.resolve())
                except ValueError:
                    invalid_by_identity[identity] = "invalid_feature_path"
                    continue
                if feature_path not in arrays:
                    try:
                        arrays[feature_path] = np.load(
                            feature_path,
                            mmap_mode="r",
                            allow_pickle=False,
                        )
                    except (OSError, ValueError):
                        invalid_by_identity[identity] = "missing_vector_file"
                        continue
                array = arrays[feature_path]
                dtype_names.add(str(array.dtype))
                if not np.issubdtype(array.dtype, np.number) or array.ndim != 2:
                    invalid_by_identity[identity] = "non_numeric_dtype"
                    continue
                dimension = expected_dimension or int(array.shape[1])
                if expected_dimension is None:
                    expected_dimension = dimension
                if feature_index >= array.shape[0]:
                    invalid_by_identity[identity] = "missing_vector_row"
                    continue
                reason = validate_clip_vector(
                    array[feature_index],
                    expected_dimension=dimension,
                )
                if reason is not None:
                    invalid_by_identity[identity] = reason
                    continue
                outputs[identity] = {
                    "video_id": canonical_row.video_id,
                    "feature_file": feature_file,
                    "row_idx": feature_index,
                }
                invalid_by_identity.pop(identity, None)

    for identity, reason in sorted(invalid_by_identity.items()):
        invalid_reasons[reason] += 1
        if len(invalid_samples) < config.sample_limit:
            invalid_samples.append(
                {
                    "video_id": identity[0],
                    "keyframe_id": identity[1],
                    "reason": reason,
                }
            )
    stats = {
        "vectors_expected": len(canonical),
        "vectors_valid": len(outputs),
        "vectors_invalid": len(canonical) - len(outputs),
        "source_mapping_rows": mapping_rows,
        "dimension": expected_dimension,
        "dtype": sorted(dtype_names),
        "mapping_mismatches": mapping_mismatches,
        "nan_inf": int(invalid_reasons.get("non_finite", 0)),
        "invalid_by_reason": dict(sorted(invalid_reasons.items())),
        "near_duplicate_suppression": False,
    }
    return outputs, stats, {"invalid_vector_samples": invalid_samples}


def _empty_embedding_stats(
    expected: int,
    *,
    required: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "required": required,
        "dimension": None,
        "dtype": None,
        "normalized": None,
        "vectors_expected": expected,
        "vectors_valid": 0,
        "vectors_invalid": expected,
        "missing_ids": expected,
        "extra_ids": 0,
        "duplicate_ids": 0,
        "mapping_mismatches": 0,
        "source_mapping_rows": 0,
        "invalid_by_reason": {status: expected},
    }


def _load_embedding_provider(
    provider_root: Path,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
    *,
    required: bool,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Validate one provider without allowing its failure to affect peers."""

    provider_id = provider_root.name
    by_identity, by_keyframe_id = _canonical_lookup(canonical)
    try:
        artifact = load_provider_artifact(provider_root, verify_checksum=True)
    except (EmbeddingArtifactError, OSError, ValueError) as exc:
        issues.add(
            "invalid_embedding_provider",
            str(exc),
            branch=f"embeddings.{provider_id}",
            path=provider_root,
        )
        stats = _empty_embedding_stats(
            len(canonical), required=required, status="invalid_artifact"
        )
        return {}, stats, [{"provider": provider_id, "reason": str(exc), "path": str(provider_root)}]

    manifest = artifact.manifest
    provider_id = manifest.model_id
    invalid_reasons: Counter[str] = Counter()
    invalid_samples: list[dict[str, Any]] = []
    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    rows_by_identity: dict[tuple[str, str], dict[str, str]] = {}
    duplicate_identities: set[tuple[str, str]] = set()
    extra_ids: set[str] = set()
    mapping_mismatches = 0
    arrays: dict[Path, np.ndarray | None] = {}

    try:
        with artifact.mapping_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required_columns = {
                "keyframe_id",
                "video_id",
                "feature_file",
                "feature_index",
            }
            missing_columns = required_columns - set(reader.fieldnames or [])
            if missing_columns:
                raise EmbeddingArtifactError(
                    f"mapping missing required columns: {sorted(missing_columns)}"
                )
            rows = list(reader)
    except (OSError, ValueError, EmbeddingArtifactError) as exc:
        issues.add(
            "invalid_embedding_mapping",
            str(exc),
            branch=f"embeddings.{provider_id}",
            path=artifact.mapping_path,
        )
        stats = _empty_embedding_stats(
            len(canonical), required=required, status="invalid_mapping"
        )
        stats.update(
            {
                "dimension": manifest.embedding_dimension,
                "dtype": manifest.dtype,
                "normalized": manifest.normalized,
            }
        )
        return {}, stats, [{"provider": provider_id, "reason": str(exc), "path": str(artifact.mapping_path)}]

    if len(rows) != manifest.keyframe_count:
        mapping_mismatches += abs(len(rows) - manifest.keyframe_count) or 1
        issues.add(
            "embedding_count_mismatch",
            f"mapping rows {len(rows)} != manifest keyframe_count {manifest.keyframe_count}",
            branch=f"embeddings.{provider_id}",
            path=artifact.mapping_path,
        )

    for row in rows:
        keyframe_id = str(row.get("keyframe_id", "")).strip()
        video_id = str(row.get("video_id", "")).strip()
        canonical_row = by_keyframe_id.get(keyframe_id)
        if canonical_row is None:
            extra_ids.add(keyframe_id)
            issues.add(
                "extra_identity",
                "embedding mapping identity not in canonical mapping",
                branch=f"embeddings.{provider_id}",
                keyframe_id=keyframe_id,
                video_id=video_id,
                path=artifact.mapping_path,
            )
            continue
        identity = canonical_row.identity
        if identity in rows_by_identity or identity in duplicate_identities:
            duplicate_identities.add(identity)
            rows_by_identity.pop(identity, None)
            outputs.pop(identity, None)
            issues.add(
                "duplicate_identity",
                "duplicate embedding mapping identity",
                branch=f"embeddings.{provider_id}",
                keyframe_id=keyframe_id,
                video_id=video_id,
                path=artifact.mapping_path,
            )
            continue
        rows_by_identity[identity] = dict(row)

    for identity, row in sorted(rows_by_identity.items()):
        canonical_row = by_identity[identity]
        if str(row.get("video_id", "")).strip() != canonical_row.video_id:
            invalid_reasons["mapping_mismatch"] += 1
            mapping_mismatches += 1
            continue
        optional_comparisons = (
            ("frame_id", canonical_row.frame_idx, int),
            ("shot_id", canonical_row.shot_id, int),
            ("timestamp", canonical_row.timestamp, float),
        )
        mismatch = False
        for field_name, expected, cast in optional_comparisons:
            raw = str(row.get(field_name, "")).strip()
            if not raw:
                continue
            try:
                actual = cast(raw)
            except (TypeError, ValueError):
                mismatch = True
                break
            if field_name == "timestamp":
                mismatch = abs(actual - expected) > 1e-6
            else:
                mismatch = actual != expected
            if mismatch:
                break
        if mismatch:
            invalid_reasons["mapping_mismatch"] += 1
            mapping_mismatches += 1
            continue

        feature_file = str(row.get("feature_file", "")).strip()
        try:
            feature_index = int(row.get("feature_index", ""))
            vector_path = artifact.resolve_vector_path(feature_file)
        except (TypeError, ValueError, EmbeddingArtifactError):
            invalid_reasons["invalid_locator"] += 1
            continue
        if feature_index < 0:
            invalid_reasons["invalid_locator"] += 1
            continue
        if vector_path not in arrays:
            try:
                arrays[vector_path] = np.load(
                    vector_path, mmap_mode="r", allow_pickle=False
                )
            except (OSError, ValueError):
                arrays[vector_path] = None
        matrix = arrays[vector_path]
        if matrix is None or matrix.ndim != 2 or feature_index >= matrix.shape[0]:
            invalid_reasons["invalid_locator"] += 1
            continue
        if str(matrix.dtype) != manifest.dtype:
            invalid_reasons["dtype_mismatch"] += 1
            continue
        vector = matrix[feature_index]
        reason = validate_clip_vector(
            vector, expected_dimension=manifest.embedding_dimension
        )
        if reason is not None:
            invalid_reasons[reason] += 1
            continue
        if manifest.normalized and not np.isclose(
            float(np.linalg.norm(vector)), 1.0, rtol=1e-3, atol=1e-3
        ):
            invalid_reasons["not_normalized"] += 1
            continue
        outputs[identity] = {
            "provider": provider_id,
            "video_id": canonical_row.video_id,
            "feature_file": feature_file,
            "row_idx": feature_index,
        }

    missing_identities = set(by_identity) - set(rows_by_identity) - duplicate_identities
    invalid_reasons["missing_mapping"] += len(missing_identities)
    invalid_reasons["duplicate_identity"] += len(duplicate_identities)
    for identity in sorted(set(by_identity) - set(outputs)):
        if len(invalid_samples) >= config.sample_limit:
            break
        reason = "missing_mapping"
        if identity in duplicate_identities:
            reason = "duplicate_identity"
        elif identity in rows_by_identity:
            reason = next(
                (name for name, count in invalid_reasons.items() if count > 0),
                "invalid_vector",
            )
        invalid_samples.append(
            {
                "provider": provider_id,
                "video_id": identity[0],
                "keyframe_id": identity[1],
                "reason": reason,
                "path": str(artifact.mapping_path),
            }
        )

    provider_valid = (
        len(outputs) == len(canonical)
        and not extra_ids
        and not duplicate_identities
        and mapping_mismatches == 0
    )
    stats = {
        "status": "valid" if provider_valid else "invalid",
        "required": required,
        "dimension": manifest.embedding_dimension,
        "dtype": manifest.dtype,
        "normalized": manifest.normalized,
        "vectors_expected": len(canonical),
        "vectors_valid": len(outputs),
        "vectors_invalid": len(canonical) - len(outputs),
        "missing_ids": len(missing_identities),
        "extra_ids": len(extra_ids),
        "duplicate_ids": len(duplicate_identities),
        "mapping_mismatches": mapping_mismatches,
        "source_mapping_rows": len(rows),
        "manifest_keyframe_count": manifest.keyframe_count,
        "mapping_sha256": sha256_file(artifact.mapping_path),
        "invalid_by_reason": dict(sorted(invalid_reasons.items())),
    }
    return outputs, stats, invalid_samples


def _load_embeddings(
    root: Path,
    canonical: list[CanonicalKeyframe],
    config: Task05AConfig,
    issues: _IssueCollector,
) -> tuple[
    dict[str, dict[tuple[str, str], dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    outputs: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    stats: dict[str, dict[str, Any]] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    required = set(config.required_embedding_providers)
    provider_roots = (
        sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if root.is_dir()
        else []
    )
    for provider_root in provider_roots:
        provider_id = provider_root.name
        provider_outputs, provider_stats, provider_samples = _load_embedding_provider(
            provider_root,
            canonical,
            config,
            issues,
            required=provider_id in required,
        )
        outputs[provider_id] = provider_outputs
        stats[provider_id] = provider_stats
        samples[provider_id] = provider_samples
    for provider_id in sorted(required - set(stats)):
        issues.add(
            "missing_required_embedding_provider",
            "required embedding provider is absent",
            branch=f"embeddings.{provider_id}",
            path=root / provider_id,
        )
        outputs[provider_id] = {}
        stats[provider_id] = _empty_embedding_stats(
            len(canonical), required=True, status="missing_provider"
        )
        samples[provider_id] = [
            {
                "provider": provider_id,
                "reason": "missing_provider",
                "path": str(root / provider_id),
            }
        ]
    return outputs, dict(sorted(stats.items())), {
        "invalid_embedding_samples": dict(sorted(samples.items()))
    }


def _prepare_output_root(output_root: Path, protected_roots: Iterable[Path | None]) -> None:
    output_root = output_root.resolve()
    for root in protected_roots:
        if root is None:
            continue
        resolved = root.resolve()
        if (
            output_root == resolved
            or output_root.is_relative_to(resolved)
            or resolved.is_relative_to(output_root)
        ):
            raise ValueError("output root must not contain or equal a Task04 input root")
    output_root.mkdir(parents=True, exist_ok=True)
    documents = output_root / "documents"
    if documents.exists():
        manifest_path = output_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(
                f"refusing to replace unmarked documents directory: {documents}"
            )
        existing_manifest = _read_json(manifest_path)
        if not isinstance(existing_manifest, Mapping) or existing_manifest.get("task") != "task05a":
            raise FileExistsError(
                f"refusing to replace non-Task05A output: {output_root}"
            )
        shutil.rmtree(documents)


def _project_document(
    keyframe: CanonicalKeyframe,
    *,
    ocr: list[dict[str, Any]],
    asr: list[dict[str, Any]],
    objects: dict[str, Any] | None,
    clip: dict[str, Any] | None,
    embeddings: Mapping[str, dict[str, Any] | None],
) -> dict[str, Any]:
    embedding_flags = {
        provider_id: locator is not None
        for provider_id, locator in sorted(embeddings.items())
    }
    indexability = {
        "lexical": bool(ocr or asr),
        "vector": False,
        "object": objects is not None,
        "embeddings": embedding_flags,
    }
    indexability["vector"] = any(indexability["embeddings"].values())
    document = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "keyframe_id": keyframe.keyframe_id,
        "video_id": keyframe.video_id,
        "shot_id": keyframe.shot_id,
        "frame_idx": keyframe.frame_idx,
        "timestamp": keyframe.timestamp,
        "ocr": {
            "texts": ocr,
            "joined": " ".join(item["text"] for item in ocr),
        },
        "asr": {
            "texts": asr,
            "joined": " ".join(item["text"] for item in asr),
        },
        "objects": objects or {"labels": [], "counts": {}},
        "clip": clip
        or {
            "video_id": keyframe.video_id,
            "feature_file": None,
            "row_idx": None,
        },
        "embeddings": dict(sorted(embeddings.items())),
        "indexability": indexability,
    }
    validate_task05a_document(document)
    ensure_finite_json(document)
    return document


def build_index_documents(
    *,
    features_root: Path,
    mapping_csv: Path,
    output_root: Path,
    clip_root: Path | None = None,
    embeddings_root: Path | None = None,
    objects_root: Path | None = None,
    ocr_root: Path | None = None,
    asr_root: Path | None = None,
    dataset: str | None = None,
    config: Task05AConfig = Task05AConfig(),
) -> Task05ABuildResult:
    """Build deterministic Task05A documents from immutable Task04 artifacts."""

    features_root = Path(features_root)
    mapping_csv = Path(mapping_csv)
    output_root = Path(output_root)
    if not features_root.is_dir():
        raise FileNotFoundError(f"features root not found: {features_root}")
    clip_root = Path(clip_root) if clip_root is not None else features_root / "clip"
    embeddings_root = (
        Path(embeddings_root)
        if embeddings_root is not None
        else features_root / "embeddings"
    )
    objects_root = (
        Path(objects_root) if objects_root is not None else features_root / "objects"
    )
    ocr_root = Path(ocr_root) if ocr_root is not None else features_root / "text"
    asr_root = Path(asr_root) if asr_root is not None else features_root / "asr"

    canonical = load_canonical_mapping(mapping_csv)
    issues = _IssueCollector(config.sample_limit)
    ocr, ocr_stats, ocr_samples = _load_ocr(ocr_root, canonical, config, issues)
    asr, asr_stats, asr_samples = _load_asr(asr_root, canonical, config, issues)
    objects, object_stats, object_samples = _load_objects(
        objects_root, canonical, config, issues
    )
    clip, clip_stats, clip_samples = _load_clip(
        clip_root, canonical, config, issues
    )
    embeddings, embedding_stats, embedding_samples = _load_embeddings(
        embeddings_root, canonical, config, issues
    )
    # Compatibility adapter for pre-Task04 CLIP-only artifacts. Modern provider
    # artifacts always take precedence and remain independently addressable.
    if not embedding_stats:
        embeddings = {"clip": clip}
        embedding_stats = {"clip": {**clip_stats, "required": False, "status": "legacy"}}

    _prepare_output_root(
        output_root,
        (
            features_root,
            mapping_csv.parent,
            clip_root,
            embeddings_root,
            objects_root,
            ocr_root,
            asr_root,
        ),
    )
    document_identities: set[tuple[str, str]] = set()
    indexability = Counter()
    for keyframe in canonical:
        identity = keyframe.identity
        document = _project_document(
            keyframe,
            ocr=ocr.get(identity, []),
            asr=asr.get(identity, []),
            objects=objects.get(identity),
            clip=clip.get(identity),
            embeddings={
                provider_id: provider_vectors.get(identity)
                for provider_id, provider_vectors in embeddings.items()
            },
        )
        if identity in document_identities:
            raise Task05AValidationError(f"duplicate output identity: {identity}")
        document_identities.add(identity)
        for branch in ("lexical", "vector", "object"):
            if document["indexability"][branch]:
                indexability[branch] += 1
        for provider_id, enabled in document["indexability"]["embeddings"].items():
            if enabled:
                indexability[f"embeddings.{provider_id}"] += 1
        _atomic_json_write(
            output_root
            / "documents"
            / keyframe.video_id
            / f"{keyframe.keyframe_id}.json",
            document,
        )

    expected_identities = {keyframe.identity for keyframe in canonical}
    if document_identities != expected_identities:
        raise Task05AValidationError("output identity set does not match canonical mapping")

    source_manifests = {
        "clip": _manifest_reference(clip_root),
        "objects": _manifest_reference(objects_root),
        "ocr": _manifest_reference(ocr_root),
        "asr": _manifest_reference(asr_root),
        "embeddings": {
            provider_id: _manifest_reference(embeddings_root / provider_id)
            for provider_id in embedding_stats
            if provider_id != "clip" or (embeddings_root / provider_id).is_dir()
        },
    }
    filter_config = {
        "ocr": {
            "absolute_min_conf": config.ocr.absolute_min_conf,
            "high_conf": config.ocr.high_conf,
            "min_temporal_support": config.ocr.min_temporal_support,
            "punctuation_only": "drop",
            "numeric_only": "preserve",
        },
        "asr": {
            "drop_empty": True,
            "min_conf": config.asr.min_conf,
        },
        "objects": {
            "second_threshold_applied": False,
            "second_nms_applied": False,
        },
        "clip": {
            "require_finite": True,
            "expected_dimension": clip_stats["dimension"],
            "near_duplicate_suppression": False,
        },
    }
    normalization_config = {
        "unicode": "NFC",
        "strip": True,
        "collapse_whitespace": True,
        "normalized_key": "casefold",
        "remove_diacritics": False,
    }
    dataset_identity = clean_display_text(dataset) or features_root.name
    invalid_required_providers = sorted(
        provider_id
        for provider_id in config.required_embedding_providers
        if embedding_stats.get(provider_id, {}).get("status") != "valid"
    )
    embedding_provider_policy = {
        "required": sorted(config.required_embedding_providers),
        "optional": sorted(
            set(embedding_stats) - set(config.required_embedding_providers)
        ),
        "required_satisfied": not invalid_required_providers,
        "invalid_required": invalid_required_providers,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": "task05a",
        "dataset": dataset_identity,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_feature_root": str(features_root.resolve()),
        "source_mapping_path": str(mapping_csv.resolve()),
        "source_mapping_sha256": _sha256_file(mapping_csv),
        "source_branch_roots": {
            "clip": str(clip_root.resolve()),
            "objects": str(objects_root.resolve()),
            "ocr": str(ocr_root.resolve()),
            "asr": str(asr_root.resolve()),
            "embeddings": str(embeddings_root.resolve()),
        },
        "source_branch_manifests": source_manifests,
        "document_count": len(document_identities),
        "filter_config": filter_config,
        "normalization_config": normalization_config,
        "ocr": ocr_stats,
        "asr": asr_stats,
        "objects": object_stats,
        "clip": clip_stats,
        "embeddings": embedding_stats,
        "embedding_provider_policy": embedding_provider_policy,
        "indexability_counts": {
            "lexical": indexability["lexical"],
            "vector": indexability["vector"],
            "object": indexability["object"],
            "embeddings": {
                provider_id: indexability[f"embeddings.{provider_id}"]
                for provider_id in sorted(embedding_stats)
            },
        },
        "validation_issue_counts": dict(sorted(issues.counts.items())),
    }
    report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "task": "task05a",
        "dataset": dataset_identity,
        "aggregate_counts": {
            "canonical_keyframes": len(canonical),
            "documents": len(document_identities),
            "ocr": ocr_stats,
            "asr": asr_stats,
            "objects": object_stats,
            "clip": clip_stats,
            "embeddings": embedding_stats,
            "embedding_provider_policy": embedding_provider_policy,
            "indexability": manifest["indexability_counts"],
        },
        "drop_reasons": {
            "ocr": ocr_stats["dropped_by_reason"],
            "asr": asr_stats["dropped_by_reason"],
        },
        **ocr_samples,
        **asr_samples,
        **object_samples,
        **clip_samples,
        **embedding_samples,
        "identity_schema_error_samples": issues.all_samples(),
        "validation_issue_counts": dict(sorted(issues.counts.items())),
    }
    _atomic_json_write(output_root / "manifest.json", manifest)
    _atomic_json_write(output_root / "filtering_report.json", report)
    return Task05ABuildResult(
        output_root=output_root,
        manifest_path=output_root / "manifest.json",
        filtering_report_path=output_root / "filtering_report.json",
        document_count=len(document_identities),
        manifest=manifest,
        filtering_report=report,
    )

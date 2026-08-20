"""A deterministic, read-only audit for corpus and offline artifacts.

The auditor deliberately observes files and manifests only.  It never invokes
an extractor, model, index client, or repair operation.  The implementation is
stdlib-first; NumPy is used opportunistically for ``.npy``/``.npz`` vectors
when it is installed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


AUDIT_VERSION = "1.0"
READY = "READY"
READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
BLOCKED = "BLOCKED"
PENDING = "PENDING"
NOT_APPLICABLE = "NOT_APPLICABLE"

KNOWN_PENDING_PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg"}
BRANCH_METADATA_FILES = {"manifest.json", "errors.json", "shot_summary.json"}


class AuditConfigurationError(ValueError):
    """Raised when CLI paths or audit configuration are invalid."""


@dataclass(frozen=True)
class AuditConfig:
    dataset_root: Path | None = None
    artifacts_root: Path | None = None
    output_root: Path = Path("reports/data_readiness")
    mapping_path: Path | None = None
    videos_root: Path | None = None
    keyframes_root: Path | None = None
    media_info_root: Path | None = None
    ocr_root: Path | None = None
    asr_root: Path | None = None
    objects_root: Path | None = None
    embeddings_root: Path | None = None
    processed_root: Path | None = None
    indexes_root: Path | None = None
    required_providers: tuple[str, ...] = ()
    run_id: str | None = None

    def normalized(self) -> "AuditConfig":
        def resolve(value: Path | None) -> Path | None:
            return value.expanduser().resolve() if value is not None else None

        return AuditConfig(
            dataset_root=resolve(self.dataset_root),
            artifacts_root=resolve(self.artifacts_root),
            output_root=self.output_root.expanduser().resolve(),
            mapping_path=resolve(self.mapping_path),
            videos_root=resolve(self.videos_root),
            keyframes_root=resolve(self.keyframes_root),
            media_info_root=resolve(self.media_info_root),
            ocr_root=resolve(self.ocr_root),
            asr_root=resolve(self.asr_root),
            objects_root=resolve(self.objects_root),
            embeddings_root=resolve(self.embeddings_root),
            processed_root=resolve(self.processed_root),
            indexes_root=resolve(self.indexes_root),
            required_providers=tuple(
                sorted({item.strip().lower() for item in self.required_providers if item.strip()})
            ),
            run_id=self.run_id,
        )


@dataclass
class Issue:
    severity: str
    category: str
    message: str
    entity_id: str | None = None
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "entity_id": self.entity_id,
            "path": self.path,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class CanonicalRecord:
    keyframe_id: str
    video_id: str
    shot_id: int | str | None
    frame_idx: int | None
    timestamp: float | None
    image_path: str | None
    source_line: int

    @property
    def identity(self) -> tuple[str, str]:
        return self.video_id, self.keyframe_id

    @property
    def output_pair(self) -> tuple[str, int] | None:
        if self.frame_idx is None:
            return None
        return self.video_id, self.frame_idx


@dataclass
class AuditResult:
    config: AuditConfig
    manifest: dict[str, Any]
    summary: dict[str, Any]
    issues: list[Issue]
    coverage: list[dict[str, Any]]
    report_path: Path
    output_dir: Path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_info() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _add(
    issues: list[Issue],
    severity: str,
    category: str,
    message: str,
    *,
    entity_id: str | None = None,
    path: Path | None = None,
    details: Mapping[str, Any] | None = None,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            category=category,
            message=message,
            entity_id=entity_id,
            path=str(path.resolve()) if path is not None else None,
            details=dict(details or {}),
        )
    )


def _status(issues: Iterable[Issue], *, pending: bool = False) -> str:
    severities = {issue.severity for issue in issues}
    if "error" in severities:
        return BLOCKED
    if pending:
        return PENDING
    if "warning" in severities:
        return READY_WITH_WARNINGS
    return READY


def _first_existing(candidates: Iterable[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _provider_key(name: str) -> str:
    compact = "".join(char for char in name.lower() if char.isalnum())
    aliases = {
        "beit3": "beit3",
        "metaclip2": "metaclip2",
        "perceptionencoder": "perception_encoder",
        "clip": "clip",
    }
    return aliases.get(compact, name.lower().replace("-", "_").replace(" ", "_"))


def _discover_roots(config: AuditConfig) -> dict[str, Path | None]:
    dataset = config.dataset_root
    artifacts = config.artifacts_root
    return {
        "mapping": _first_existing(
            [
                config.mapping_path,
                dataset / "keyframe_mapping.csv" if dataset else None,
                dataset / "ingest" / "keyframe_mapping.csv" if dataset else None,
                artifacts / "ingest" / "keyframe_mapping.csv" if artifacts else None,
                artifacts / "keyframe_mapping.csv" if artifacts else None,
            ]
        ),
        "videos": _first_existing(
            [
                config.videos_root,
                dataset / "videos" if dataset else None,
                dataset / "data" / "videos" if dataset else None,
                artifacts / "videos" if artifacts else None,
            ]
        ),
        "keyframes": _first_existing(
            [
                config.keyframes_root,
                dataset / "keyframes" if dataset else None,
                artifacts / "keyframes" if artifacts else None,
                artifacts / "ingest" if artifacts else None,
            ]
        ),
        "media_info": _first_existing(
            [
                config.media_info_root,
                dataset / "media_info" if dataset else None,
                dataset / "video_metadata.csv" if dataset else None,
                artifacts / "ingest" if artifacts else None,
            ]
        ),
        "ocr": _first_existing(
            [
                config.ocr_root,
                dataset / "ocr" if dataset else None,
                artifacts / "features" / "text" if artifacts else None,
                artifacts / "features" / "ocr" if artifacts else None,
            ]
        ),
        "asr": _first_existing(
            [
                config.asr_root,
                dataset / "asr" if dataset else None,
                artifacts / "features" / "asr" if artifacts else None,
            ]
        ),
        "objects": _first_existing(
            [
                config.objects_root,
                dataset / "objects" if dataset else None,
                artifacts / "features" / "objects" if artifacts else None,
            ]
        ),
        "embeddings": _first_existing(
            [
                config.embeddings_root,
                dataset / "embeddings" if dataset else None,
                artifacts / "embeddings" if artifacts else None,
                artifacts / "features" if artifacts else None,
            ]
        ),
        "processed": _first_existing(
            [
                config.processed_root,
                dataset / "processed" if dataset else None,
                artifacts / "processed" if artifacts else None,
            ]
        ),
        "indexes": _first_existing(
            [
                config.indexes_root,
                artifacts / "indexes" if artifacts else None,
                dataset / "indexes" if dataset else None,
            ]
        ),
    }


def _parse_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _load_mapping(path: Path | None, issues: list[Issue]) -> tuple[list[CanonicalRecord], dict[str, Any]]:
    if path is None:
        _add(issues, "error", "missing_mapping", "Canonical mapping was not found.")
        return [], {"path": None, "exists": False, "records": 0}
    if not path.is_file():
        _add(issues, "error", "missing_mapping", "Canonical mapping path is not a file.", path=path)
        return [], {"path": str(path), "exists": False, "records": 0}
    records: list[CanonicalRecord] = []
    seen_keyframes: dict[str, CanonicalRecord] = {}
    seen_pairs: dict[tuple[str, int], CanonicalRecord] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            required = {"keyframe_id", "video_id"}
            missing = sorted(required - fields)
            if missing:
                _add(
                    issues,
                    "error",
                    "mapping_schema",
                    f"Canonical mapping is missing columns: {missing}",
                    path=path,
                )
                return [], {"path": str(path), "exists": True, "records": 0, "columns": sorted(fields)}
            for line, row in enumerate(reader, start=2):
                keyframe_id = str(row.get("keyframe_id", "")).strip()
                video_id = str(row.get("video_id", "")).strip()
                if not keyframe_id or not video_id:
                    _add(
                        issues,
                        "error",
                        "mapping_invalid_identity",
                        "Mapping row has blank keyframe_id or video_id.",
                        path=path,
                        details={"line": line},
                    )
                    continue
                frame_idx = _parse_int(row.get("frame_idx", row.get("frame_id")))
                timestamp = _parse_float(row.get("timestamp", row.get("pts_time")))
                shot_raw = row.get("shot_id")
                shot_id: int | str | None = _parse_int(shot_raw)
                if shot_id is None and str(shot_raw or "").strip():
                    shot_id = str(shot_raw).strip()
                if "frame_idx" in fields or "frame_id" in fields:
                    if frame_idx is None or frame_idx < 0:
                        _add(
                            issues,
                            "error",
                            "mapping_invalid_frame",
                            "Mapping frame index is missing, non-integer, or negative.",
                            entity_id=keyframe_id,
                            path=path,
                            details={"line": line},
                        )
                if "timestamp" in fields or "pts_time" in fields:
                    if timestamp is None or timestamp < 0:
                        _add(
                            issues,
                            "error",
                            "mapping_invalid_timestamp",
                            "Mapping timestamp is missing, non-finite, or negative.",
                            entity_id=keyframe_id,
                            path=path,
                            details={"line": line},
                        )
                record = CanonicalRecord(
                    keyframe_id=keyframe_id,
                    video_id=video_id,
                    shot_id=shot_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    image_path=(str(row.get("image_path", "")).strip() or None),
                    source_line=line,
                )
                if keyframe_id in seen_keyframes:
                    _add(
                        issues,
                        "error",
                        "duplicate_canonical_keyframe_id",
                        "Canonical keyframe_id occurs more than once.",
                        entity_id=keyframe_id,
                        path=path,
                        details={"lines": [seen_keyframes[keyframe_id].source_line, line]},
                    )
                else:
                    seen_keyframes[keyframe_id] = record
                if record.output_pair is not None:
                    if record.output_pair in seen_pairs:
                        _add(
                            issues,
                            "error",
                            "duplicate_output_pair",
                            "Canonical mapping has duplicate (video_id, frame_idx).",
                            entity_id=keyframe_id,
                            path=path,
                            details={
                                "video_id": record.video_id,
                                "frame_idx": record.frame_idx,
                                "previous_keyframe_id": seen_pairs[record.output_pair].keyframe_id,
                            },
                        )
                    else:
                        seen_pairs[record.output_pair] = record
                records.append(record)
    except (OSError, UnicodeError, csv.Error) as exc:
        _add(issues, "error", "mapping_read_error", str(exc), path=path)
    metadata = {
        "path": str(path.resolve()),
        "exists": True,
        "sha256": _sha256(path),
        "records": len(records),
        "unique_keyframe_ids": len(seen_keyframes),
        "unique_output_pairs": len(seen_pairs),
        "columns": sorted(fields) if "fields" in locals() else [],
    }
    return records, metadata


def _iter_files(root: Path | None, suffixes: set[str] | None = None) -> list[Path]:
    if root is None or not root.exists():
        return []
    if root.is_file():
        return [root] if suffixes is None or root.suffix.lower() in suffixes else []
    files: list[Path] = []
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(current) / name).is_symlink()]
        for name in names:
            path = Path(current) / name
            if suffixes is None or path.suffix.lower() in suffixes:
                files.append(path)
    return sorted(files)


def _disk_usage(root: Path | None) -> dict[str, Any]:
    if root is None or not root.exists():
        return {"path": str(root) if root else None, "exists": False, "files": 0, "bytes": 0}
    files = _iter_files(root)
    return {
        "path": str(root.resolve()),
        "exists": True,
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files if path.is_file()),
    }


def _video_inventory(
    root: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
) -> dict[str, Any]:
    issue_start = len(issues)
    files = _iter_files(root, VIDEO_SUFFIXES)
    by_id: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_id[path.stem].append(path)
        if path.stat().st_size == 0:
            _add(issues, "error", "zero_byte_video", "Video file is zero bytes.", entity_id=path.stem, path=path)
        if not os.access(path, os.R_OK):
            _add(issues, "error", "unreadable_video", "Video file is not readable.", entity_id=path.stem, path=path)
    for video_id, matches in by_id.items():
        if len(matches) > 1:
            _add(
                issues,
                "error",
                "duplicate_video_id",
                "Multiple source video files share one video ID.",
                entity_id=video_id,
                details={"paths": [str(item) for item in matches]},
            )
    referenced = {item.video_id for item in canonical}
    present = set(by_id)
    for video_id in sorted(referenced - present):
        _add(
            issues,
            "error",
            "missing_referenced_video",
            "Canonical mapping references a source video that is absent.",
            entity_id=video_id,
            path=root,
        )
    return {
        "status": _status(issues[issue_start:]),
        "root": str(root) if root else None,
        "total_files": len(files),
        "unique_video_ids": len(by_id),
        "referenced_video_ids": len(referenced),
        "missing_referenced_video_ids": sorted(referenced - present),
        "extension_distribution": dict(sorted(Counter(path.suffix.lower() for path in files).items())),
        "disk_usage": _disk_usage(root),
    }


def _resolve_image_path(record: CanonicalRecord, root: Path | None, mapping_path: Path | None) -> Path | None:
    if not record.image_path:
        return None
    raw = Path(record.image_path)
    candidates = [raw] if raw.is_absolute() else []
    if root is not None:
        candidates.append(root / raw)
    if mapping_path is not None:
        candidates.append(mapping_path.parent / raw)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def _keyframe_inventory(
    root: Path | None,
    mapping_path: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    issue_start = len(issues)
    image_files = _iter_files(root, IMAGE_SUFFIXES)
    expected_paths: set[Path] = set()
    present: set[tuple[str, str]] = set()
    resolved_by_path: dict[Path, str] = {}
    for record in canonical:
        path = _resolve_image_path(record, root, mapping_path)
        if path is None:
            _add(
                issues,
                "error",
                "missing_keyframe_file",
                "Canonical mapping has no image_path.",
                entity_id=record.keyframe_id,
            )
            continue
        expected_paths.add(path.resolve())
        if not path.is_file():
            _add(
                issues,
                "error",
                "missing_keyframe_file",
                "Keyframe image referenced by canonical mapping is absent.",
                entity_id=record.keyframe_id,
                path=path,
            )
            continue
        present.add(record.identity)
        resolved = path.resolve()
        if resolved in resolved_by_path:
            _add(
                issues,
                "error",
                "keyframe_file_collision",
                "Multiple canonical rows resolve to one image file.",
                entity_id=record.keyframe_id,
                path=path,
                details={"previous_keyframe_id": resolved_by_path[resolved]},
            )
        resolved_by_path[resolved] = record.keyframe_id
        if path.stat().st_size == 0:
            _add(issues, "error", "zero_byte_keyframe", "Keyframe file is zero bytes.", entity_id=record.keyframe_id, path=path)
        elif path.suffix.lower() in {".jpg", ".jpeg"} and path.read_bytes()[:2] != b"\xff\xd8":
            _add(issues, "warning", "suspicious_keyframe_file", "JPEG does not have a JPEG signature.", entity_id=record.keyframe_id, path=path)
    actual_paths = {path.resolve() for path in image_files}
    orphan_paths = sorted(actual_paths - expected_paths)
    for path in orphan_paths:
        _add(issues, "warning", "orphan_keyframe_file", "Image is not referenced by the canonical mapping.", path=path)
    return (
        {
            "status": _status(issues[issue_start:]),
            "root": str(root) if root else None,
            "expected_canonical_keyframes": len(canonical),
            "present_keyframes": len(present),
            "missing_keyframes": len(canonical) - len(present),
            "orphan_keyframe_files": len(orphan_paths),
            "files": len(image_files),
            "extension_distribution": dict(sorted(Counter(path.suffix.lower() for path in image_files).items())),
            "disk_usage": _disk_usage(root),
        },
        present,
    )


def _media_info(
    root: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
) -> dict[str, Any]:
    issue_start = len(issues)
    files = [
        path
        for path in _iter_files(root, {".csv", ".json"})
        if (
            path.suffix.lower() == ".json"
            or path.name.lower() in {"video_metadata.csv", "media_info.csv"}
            or path.name.lower().startswith("media_info")
        )
    ]
    records: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for path in files:
        if path.name in BRANCH_METADATA_FILES and path.name != "video_metadata.csv":
            continue
        try:
            if path.suffix.lower() == ".csv":
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    rows = list(csv.DictReader(handle))
            else:
                payload = _read_json(path)
                rows = payload if isinstance(payload, list) else [payload]
            for row in rows:
                if not isinstance(row, Mapping):
                    _add(issues, "error", "media_info_schema", "Media-info record is not an object.", path=path)
                    continue
                video_id = str(row.get("video_id", "")).strip()
                if not video_id:
                    _add(issues, "error", "media_info_schema", "Media-info record has no video_id.", path=path)
                    continue
                if video_id in records:
                    duplicate_ids.add(video_id)
                    _add(issues, "error", "duplicate_media_info", "Duplicate media-info record.", entity_id=video_id, path=path)
                records[video_id] = dict(row)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
            _add(issues, "error", "media_info_read_error", str(exc), path=path)
    for video_id, row in records.items():
        duration = _parse_float(row.get("duration_seconds", row.get("duration")))
        fps = _parse_float(row.get("fps", row.get("fps_num")))
        frame_count = _parse_int(row.get("frame_count", row.get("total_frames")))
        if duration is not None and duration < 0:
            _add(issues, "error", "invalid_media_duration", "Duration is negative.", entity_id=video_id)
        if fps is not None and fps <= 0:
            _add(issues, "error", "invalid_media_fps", "FPS is not positive.", entity_id=video_id)
        if frame_count is not None and frame_count < 0:
            _add(issues, "error", "invalid_media_frame_count", "Frame count is negative.", entity_id=video_id)
        if duration is None and "duration_seconds" in row:
            _add(issues, "error", "invalid_media_duration", "Duration is not finite numeric.", entity_id=video_id)
        if fps is None and "fps" in row:
            _add(issues, "error", "invalid_media_fps", "FPS is not finite numeric.", entity_id=video_id)
        if frame_count is None and ("frame_count" in row or "total_frames" in row):
            _add(issues, "error", "invalid_media_frame_count", "Frame count is not an integer.", entity_id=video_id)
    out_of_range = 0
    for record in canonical:
        row = records.get(record.video_id)
        if not row:
            continue
        frame_count = _parse_int(row.get("frame_count", row.get("total_frames")))
        duration = _parse_float(row.get("duration_seconds", row.get("duration")))
        if frame_count is not None and record.frame_idx is not None and record.frame_idx >= frame_count:
            out_of_range += 1
            _add(issues, "error", "canonical_frame_out_of_range", "Canonical frame_idx is outside media frame count.", entity_id=record.keyframe_id, details={"frame_idx": record.frame_idx, "frame_count": frame_count})
        if duration is not None and record.timestamp is not None and record.timestamp > duration + 1.0:
            out_of_range += 1
            _add(issues, "error", "canonical_timestamp_out_of_range", "Canonical timestamp exceeds media duration.", entity_id=record.keyframe_id, details={"timestamp": record.timestamp, "duration": duration})
    missing = sorted({record.video_id for record in canonical} - set(records))
    for video_id in missing:
        _add(issues, "warning", "missing_media_info", "No media-info record for mapped video.", entity_id=video_id, path=root)
    return {
        "status": _status(issues[issue_start:]) if records else NOT_APPLICABLE,
        "root": str(root) if root else None,
        "files": len(files),
        "video_records": len(records),
        "duplicate_video_ids": sorted(duplicate_ids),
        "missing_mapped_videos": missing,
        "out_of_range_keyframes": out_of_range,
        "disk_usage": _disk_usage(root),
    }


def _json_branch(
    name: str,
    root: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    issue_start = len(issues)
    files = [
        path
        for path in _iter_files(root, {".json"})
        if path.name not in BRANCH_METADATA_FILES
    ]
    by_identity: dict[tuple[str, str], Path] = {}
    empty = 0
    valid = 0
    orphan = 0
    malformed = 0
    expected_by_id = {record.keyframe_id: record for record in canonical}
    expected_by_identity = {record.identity: record for record in canonical}
    for path in files:
        try:
            payload = _read_json(path)
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            malformed += 1
            _add(issues, "error", f"{name}_invalid_json", str(exc), path=path)
            continue
        if not isinstance(payload, Mapping):
            malformed += 1
            _add(issues, "error", f"{name}_invalid_schema", "Artifact JSON must be an object.", path=path)
            continue
        keyframe_id = str(payload.get("keyframe_id", "")).strip()
        video_id = str(payload.get("video_id", "")).strip()
        record = expected_by_id.get(keyframe_id)
        if not keyframe_id or not video_id or record is None:
            orphan += 1
            _add(issues, "warning", f"orphan_{name}_artifact", "Artifact identity is absent from canonical mapping.", entity_id=keyframe_id or None, path=path, details={"video_id": video_id})
            continue
        if record.video_id != video_id:
            malformed += 1
            _add(issues, "error", f"{name}_identity_mismatch", "Artifact video_id does not match canonical mapping.", entity_id=keyframe_id, path=path, details={"actual_video_id": video_id, "expected_video_id": record.video_id})
            continue
        identity = record.identity
        if identity in by_identity:
            malformed += 1
            _add(issues, "error", f"duplicate_{name}_artifact", "Multiple artifacts resolve to one canonical keyframe.", entity_id=keyframe_id, path=path, details={"previous_path": str(by_identity[identity])})
            continue
        frame_idx = _parse_int(payload.get("frame_idx"))
        timestamp = _parse_float(payload.get("timestamp"))
        if frame_idx is not None and record.frame_idx is not None and frame_idx != record.frame_idx:
            _add(issues, "error", f"{name}_identity_mismatch", "Artifact frame_idx does not match canonical mapping.", entity_id=keyframe_id, path=path)
            malformed += 1
            continue
        if timestamp is not None and record.timestamp is not None and abs(timestamp - record.timestamp) > 1e-6:
            _add(issues, "error", f"{name}_identity_mismatch", "Artifact timestamp does not match canonical mapping.", entity_id=keyframe_id, path=path)
            malformed += 1
            continue
        valid_schema = True
        if name == "ocr":
            texts = payload.get("texts")
            valid_schema = isinstance(texts, list)
            if valid_schema:
                for item in texts:
                    if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                        valid_schema = False
                        break
                    if "conf" in item and _parse_float(item.get("conf")) is None:
                        _add(issues, "error", "ocr_non_finite_confidence", "OCR confidence is not finite numeric.", entity_id=keyframe_id, path=path)
                        valid_schema = False
            if isinstance(texts, list) and not texts:
                empty += 1
        elif name == "asr":
            valid_schema = (
                isinstance(payload.get("text"), str)
                and isinstance(payload.get("segments"), list)
                and all(
                    isinstance(item, Mapping)
                    and isinstance(item.get("text"), str)
                    and _parse_float(item.get("start")) is not None
                    and _parse_float(item.get("end")) is not None
                    for item in payload["segments"]
                )
            )
            if valid_schema and not str(payload.get("text", "")).strip():
                empty += 1
        elif name == "objects":
            detections = payload.get("objects")
            valid_schema = isinstance(detections, list)
            if valid_schema:
                for item in detections:
                    if not isinstance(item, Mapping) or not isinstance(item.get("label"), str):
                        valid_schema = False
                        break
                    if "score" in item and _parse_float(item.get("score")) is None:
                        _add(issues, "error", "objects_non_finite_score", "Object score is not finite numeric.", entity_id=keyframe_id, path=path)
                        valid_schema = False
                    bbox = item.get("bbox")
                    if bbox is not None and (
                        not isinstance(bbox, list)
                        or len(bbox) != 4
                        or any(_parse_float(value) is None for value in bbox)
                    ):
                        valid_schema = False
                        break
            if isinstance(detections, list) and not detections:
                empty += 1
            for count_field in ("person_count", "car_count"):
                if count_field in payload and (
                    isinstance(payload[count_field], bool)
                    or not isinstance(payload[count_field], int)
                    or payload[count_field] < 0
                ):
                    valid_schema = False
                    _add(issues, "error", "objects_invalid_count", "Object count is not a non-negative integer.", entity_id=keyframe_id, path=path, details={"field": count_field})
        if not valid_schema:
            malformed += 1
            _add(issues, "error", f"{name}_invalid_schema", "Artifact does not satisfy the observed branch schema.", entity_id=keyframe_id, path=path)
            continue
        by_identity[identity] = path
        valid += 1
    missing = set(expected_by_identity) - set(by_identity)
    for video_id, keyframe_id in sorted(missing):
        _add(issues, "warning", f"missing_{name}_artifact", "Canonical keyframe has no branch artifact.", entity_id=keyframe_id, details={"video_id": video_id})
    status = _status(issues[issue_start:]) if files else NOT_APPLICABLE
    return (
        {
            "status": status,
            "root": str(root) if root else None,
            "expected_keyframes": len(canonical),
            "artifact_files": len(files),
            "valid_artifacts": valid,
            "coverage": (valid / len(canonical)) if canonical else 0.0,
            "missing_canonical": len(missing),
            "orphan_artifacts": orphan,
            "malformed_artifacts": malformed,
            "empty_results": empty,
            "metadata_excluded": len(_iter_files(root, {".json"})) - len(files),
            "disk_usage": _disk_usage(root),
        },
        set(by_identity),
    )


def _optional_numpy() -> Any:
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def _embedding_dirs(root: Path | None) -> list[Path]:
    if root is None or not root.exists():
        return []
    if root.is_file():
        return [root.parent]
    candidates: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        manifest = path / "manifest.json"
        is_vector_manifest = False
        if manifest.is_file():
            try:
                payload = _read_json(manifest)
                if isinstance(payload, Mapping):
                    feature_type = str(payload.get("feature_type", "")).lower()
                    is_vector_manifest = (
                        "embedding" in feature_type
                        or "clip" in feature_type
                        or "vector" in feature_type
                        or "embedding_dimension" in payload
                    )
            except (OSError, ValueError, json.JSONDecodeError):
                # Let the provider audit surface a malformed manifest only
                # when the directory otherwise looks like a vector branch.
                is_vector_manifest = any(path.glob("*.npy")) or any(path.glob("*.npz"))
        if is_vector_manifest or any(path.glob("*.npy")) or any(path.glob("*.npz")):
            candidates.append(path)
    return candidates


def _embedding_provider(
    provider_root: Path,
    provider_name: str,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
    required: bool,
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    issue_start = len(issues)
    manifest_path = provider_root / "manifest.json"
    manifest: Mapping[str, Any] = {}
    if manifest_path.is_file():
        try:
            payload = _read_json(manifest_path)
            if isinstance(payload, Mapping):
                manifest = payload
            else:
                _add(issues, "error", "embedding_manifest_schema", "Embedding manifest must be an object.", path=manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(issues, "error", "embedding_manifest_read_error", str(exc), path=manifest_path)
    elif required:
        _add(issues, "error", "required_provider_manifest_missing", "Required provider manifest is absent.", entity_id=provider_name, path=provider_root)
    mapping_path = _first_existing(
        [
            provider_root / str(manifest.get("mapping_file")) if manifest.get("mapping_file") else None,
            provider_root / "mapping.csv",
            provider_root / "clip_feature_mapping.csv",
        ]
    )
    mapping_rows: list[dict[str, Any]] = []
    mapping_ids: set[tuple[str, str]] = set()
    duplicate_mapping_ids: set[tuple[str, str]] = set()
    if mapping_path is not None and mapping_path.is_file():
        try:
            with mapping_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    keyframe_id = str(row.get("keyframe_id", "")).strip()
                    video_id = str(row.get("video_id", "")).strip()
                    if not keyframe_id or not video_id:
                        _add(issues, "error", "embedding_mapping_schema", "Embedding mapping has blank identity.", path=mapping_path)
                        continue
                    identity = (video_id, keyframe_id)
                    if identity in mapping_ids:
                        duplicate_mapping_ids.add(identity)
                        _add(issues, "error", "embedding_duplicate_identity", "Embedding mapping repeats an identity.", entity_id=keyframe_id, path=mapping_path)
                    mapping_ids.add(identity)
                    mapping_rows.append(dict(row))
        except (OSError, UnicodeError, csv.Error) as exc:
            _add(issues, "error", "embedding_mapping_read_error", str(exc), path=mapping_path)
    canonical_ids = {item.identity for item in canonical}
    for identity in sorted(mapping_ids - canonical_ids):
        _add(issues, "warning", "embedding_orphan_identity", "Embedding mapping identity is not canonical.", entity_id=identity[1], path=mapping_path)
    missing_mapping = canonical_ids & mapping_ids
    vector_files = [path for path in _iter_files(provider_root, {".npy", ".npz", ".json"}) if path.name not in BRANCH_METADATA_FILES and path != mapping_path]
    vector_count = 0
    dimensions: set[int] = set()
    dtypes: set[str] = set()
    finite = True
    numpy = _optional_numpy()
    vector_file_errors = 0
    if numpy is None and any(path.suffix.lower() in {".npy", ".npz"} for path in vector_files):
        _add(issues, "warning", "embedding_numpy_unavailable", "NumPy is unavailable; binary vector integrity was not inspected.", entity_id=provider_name, path=provider_root)
    for path in vector_files:
        if path.suffix.lower() == ".json":
            try:
                payload = _read_json(path)
                vector = payload.get("vector", payload.get("embedding")) if isinstance(payload, Mapping) else None
                if vector is not None:
                    values = list(vector)
                    vector_count += 1
                    dimensions.add(len(values))
                    dtypes.add("json")
                    if any(_parse_float(value) is None for value in values):
                        finite = False
                        _add(issues, "error", "embedding_non_finite", "JSON embedding contains NaN, Inf, or non-numeric values.", path=path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                vector_file_errors += 1
                _add(issues, "error", "embedding_read_error", str(exc), path=path)
            continue
        if path.suffix.lower() not in {".npy", ".npz"}:
            continue
        if numpy is None:
            continue
        try:
            payload = numpy.load(path, allow_pickle=False, mmap_mode="r" if path.suffix == ".npy" else None)
            arrays = [payload] if path.suffix == ".npy" else [payload[key] for key in payload.files]
            for array in arrays:
                if array.ndim == 1:
                    array = array.reshape(1, -1)
                if array.ndim != 2:
                    vector_file_errors += 1
                    _add(issues, "error", "embedding_dimension_mismatch", "Embedding array is not a 2-D matrix.", path=path)
                    continue
                vector_count += int(array.shape[0])
                dimensions.add(int(array.shape[1]))
                dtypes.add(str(array.dtype))
                if not numpy.isfinite(array).all():
                    finite = False
                    _add(issues, "error", "embedding_non_finite", "Embedding contains NaN or Inf.", path=path)
        except (OSError, ValueError, TypeError) as exc:
            vector_file_errors += 1
            _add(issues, "error", "embedding_read_error", str(exc), path=path)
    manifest_dimension = _parse_int(manifest.get("embedding_dimension", manifest.get("dimension")))
    if manifest_dimension is not None and dimensions and dimensions != {manifest_dimension}:
        _add(issues, "error", "embedding_dimension_mismatch", "Vector dimensions disagree with provider manifest.", entity_id=provider_name, details={"manifest": manifest_dimension, "actual": sorted(dimensions)})
    expected_vectors = _parse_int(manifest.get("vector_count_total", manifest.get("vector_count")))
    if expected_vectors is not None and vector_count and expected_vectors != vector_count:
        _add(issues, "error", "embedding_count_mismatch", "Vector count disagrees with provider manifest.", entity_id=provider_name, details={"manifest": expected_vectors, "actual": vector_count})
    if mapping_rows and vector_count and len(mapping_rows) != vector_count:
        _add(issues, "error", "embedding_mapping_vector_count_mismatch", "Embedding mapping row count differs from vector count.", entity_id=provider_name, details={"mapping": len(mapping_rows), "vectors": vector_count})
    if manifest and mapping_path is None:
        _add(issues, "warning", "embedding_mapping_missing", "Provider has a manifest but no identity mapping file was found.", entity_id=provider_name, path=provider_root)
    status = _status(issues[issue_start:])
    return (
        {
            "status": status,
            "provider": provider_name,
            "root": str(provider_root.resolve()),
            "manifest_path": str(manifest_path.resolve()) if manifest_path.is_file() else None,
            "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
            "model": manifest.get("model", manifest.get("model_name")),
            "checkpoint": manifest.get("pretrained", manifest.get("checkpoint")),
            "expected_canonical_vectors": len(canonical),
            "mapping_path": str(mapping_path.resolve()) if mapping_path else None,
            "mapping_count": len(mapping_rows),
            "mapping_canonical_count": len(missing_mapping),
            "mapping_orphan_count": len(mapping_ids - canonical_ids),
            "vector_count": vector_count,
            "vector_dimensions": sorted(dimensions),
            "manifest_dimension": manifest_dimension,
            "dtype": sorted(dtypes),
            "normalization": manifest.get("normalization"),
            "finite": finite,
            "vector_file_errors": vector_file_errors,
            "disk_usage": _disk_usage(provider_root),
        },
        mapping_ids & canonical_ids,
    )


def _embedding_audit(
    root: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
    required_providers: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, set[tuple[str, str]]]]:
    issue_start = len(issues)
    discovered: dict[str, Path] = {}
    for path in _embedding_dirs(root):
        discovered[_provider_key(path.name)] = path
    providers: dict[str, dict[str, Any]] = {}
    coverage: dict[str, set[tuple[str, str]]] = {}
    names = sorted(set(discovered) | set(KNOWN_PENDING_PROVIDERS) | set(required_providers))
    pending: list[str] = []
    for name in names:
        if name not in discovered:
            pending.append(name)
            severity = "error" if name in required_providers else "warning"
            _add(
                issues,
                severity,
                "required_provider_missing" if name in required_providers else "provider_pending",
                "Embedding provider artifacts were not found.",
                entity_id=name,
                path=root,
            )
            providers[name] = {
                "status": BLOCKED if name in required_providers else PENDING,
                "provider": name,
                "root": None,
                "expected_canonical_vectors": len(canonical),
                "pending": True,
            }
            coverage[name] = set()
            continue
        providers[name], coverage[name] = _embedding_provider(
            discovered[name], name, canonical, issues, name in required_providers
        )
    return (
        {
            "status": _status(
                issues[issue_start:],
                pending=bool(pending)
                and not any(issue.severity == "error" for issue in issues[issue_start:]),
            ),
            "root": str(root) if root else None,
            "providers": providers,
            "discovered_providers": sorted(discovered),
            "pending_providers": pending,
        },
        coverage,
    )


def _processed_audit(
    root: Path | None,
    canonical: list[CanonicalRecord],
    issues: list[Issue],
) -> tuple[dict[str, Any], set[tuple[str, str]]]:
    issue_start = len(issues)
    if root is None or not root.exists():
        return {"status": NOT_APPLICABLE, "root": str(root) if root else None, "documents": 0}, set()
    documents_root = root / "documents" if root.is_dir() and (root / "documents").is_dir() else root
    paths = [path for path in _iter_files(documents_root, {".json"}) if path.name not in BRANCH_METADATA_FILES]
    canonical_by_id = {item.keyframe_id: item for item in canonical}
    seen: dict[tuple[str, str], Path] = {}
    indexability = Counter()
    valid = 0
    orphan = 0
    for path in paths:
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(issues, "error", "processed_invalid_json", str(exc), path=path)
            continue
        if not isinstance(payload, Mapping):
            _add(issues, "error", "processed_invalid_schema", "Processed document must be an object.", path=path)
            continue
        required_fields = {"keyframe_id", "video_id", "frame_idx", "timestamp"}
        missing_fields = sorted(required_fields - set(payload))
        if missing_fields:
            _add(
                issues,
                "error",
                "processed_invalid_schema",
                "Processed document is missing required identity fields.",
                path=path,
                details={"missing_fields": missing_fields},
            )
            continue
        keyframe_id = str(payload.get("keyframe_id", "")).strip()
        video_id = str(payload.get("video_id", "")).strip()
        record = canonical_by_id.get(keyframe_id)
        if record is None or record.video_id != video_id:
            orphan += 1
            _add(issues, "warning", "orphan_processed_document", "Processed document is not canonical.", entity_id=keyframe_id or None, path=path)
            continue
        identity = record.identity
        if identity in seen:
            _add(issues, "error", "duplicate_processed_document", "Multiple Task05 documents resolve to one canonical keyframe.", entity_id=keyframe_id, path=path, details={"previous_path": str(seen[identity])})
            continue
        seen[identity] = path
        if record.frame_idx is not None and _parse_int(payload.get("frame_idx")) != record.frame_idx:
            _add(issues, "error", "processed_identity_mismatch", "Processed frame_idx does not match canonical mapping.", entity_id=keyframe_id, path=path)
            continue
        if _parse_float(payload.get("timestamp")) is None:
            _add(issues, "error", "processed_invalid_schema", "Processed timestamp is not finite numeric.", entity_id=keyframe_id, path=path)
            continue
        indexability_payload = payload.get("indexability", {})
        if isinstance(indexability_payload, Mapping):
            for branch in ("lexical", "object", "vector"):
                if indexability_payload.get(branch) is True:
                    indexability[branch] += 1
                elif branch in indexability_payload and not isinstance(indexability_payload.get(branch), bool):
                    _add(issues, "error", "processed_invalid_indexability", "Indexability flag is not boolean.", entity_id=keyframe_id, path=path, details={"branch": branch})
        valid += 1
    missing = set(item.identity for item in canonical) - set(seen)
    for video_id, keyframe_id in sorted(missing):
        _add(issues, "warning", "missing_processed_document", "Canonical keyframe has no Task05 processed document.", entity_id=keyframe_id, details={"video_id": video_id})
    return (
        {
            "status": _status(issues[issue_start:]) if paths else NOT_APPLICABLE,
            "root": str(root.resolve()),
            "document_files": len(paths),
            "valid_documents": valid,
            "missing_canonical_documents": len(missing),
            "orphan_documents": orphan,
            "indexability_counts": dict(indexability),
            "manifest_path": str((root / "manifest.json").resolve()) if (root / "manifest.json").is_file() else None,
            "manifest_sha256": _sha256(root / "manifest.json") if (root / "manifest.json").is_file() else None,
            "disk_usage": _disk_usage(root),
        },
        set(seen),
    )


def _index_audit(
    root: Path | None,
    canonical: list[CanonicalRecord],
    mapping_meta: Mapping[str, Any],
    processed_meta: Mapping[str, Any],
    issues: list[Issue],
) -> dict[str, Any]:
    issue_start = len(issues)
    if root is None or not root.exists():
        return {"status": NOT_APPLICABLE, "root": str(root) if root else None, "manifests": []}
    manifests: list[dict[str, Any]] = []
    for path in _iter_files(root, {".json"}):
        if path.name != "manifest.json":
            continue
        try:
            payload = _read_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _add(issues, "error", "index_manifest_read_error", str(exc), path=path)
            continue
        if not isinstance(payload, Mapping):
            _add(issues, "error", "index_manifest_schema", "Index manifest must be an object.", path=path)
            continue
        count = payload.get("indexed_count", payload.get("document_count", payload.get("vector_count")))
        source_hash = payload.get("source_mapping_sha256", payload.get("mapping_sha256"))
        if source_hash and mapping_meta.get("sha256") and source_hash != mapping_meta.get("sha256"):
            _add(issues, "warning", "index_mapping_hash_mismatch", "Index manifest source mapping hash differs from audited mapping.", path=path, details={"manifest": source_hash, "current": mapping_meta.get("sha256")})
        source_count = _parse_int(payload.get("source_document_count"))
        processed_count = _parse_int(processed_meta.get("valid_documents"))
        if source_count is not None and processed_count is not None and source_count != processed_count:
            _add(
                issues,
                "warning",
                "index_source_count_mismatch",
                "Index manifest source document count differs from audited processed documents.",
                path=path,
                details={"manifest": source_count, "current": processed_count},
            )
        manifests.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "index_name": payload.get("index_name"),
                "provider": payload.get("provider", payload.get("model", payload.get("backend"))),
                "count": count,
                "source_mapping_sha256": source_hash,
                "created_at": payload.get("created_at"),
                "source_document_count": payload.get("source_document_count"),
            }
        )
    return {
        "status": _status(issues[issue_start:]) if manifests else NOT_APPLICABLE,
        "root": str(root.resolve()),
        "manifests": manifests,
        "manifest_count": len(manifests),
        "disk_usage": _disk_usage(root),
    }


def _coverage_rows(
    canonical: list[CanonicalRecord],
    keyframes: set[tuple[str, str]],
    branches: Mapping[str, set[tuple[str, str]]],
    providers: Mapping[str, set[tuple[str, str]]],
    processed: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(canonical, key=lambda item: (item.video_id, item.keyframe_id)):
        row: dict[str, Any] = {
            "keyframe_id": record.keyframe_id,
            "video_id": record.video_id,
            "keyframe": "present" if record.identity in keyframes else "missing",
            "ocr": "present" if record.identity in branches.get("ocr", set()) else "missing",
            "asr": "present" if record.identity in branches.get("asr", set()) else "missing",
            "objects": "present" if record.identity in branches.get("objects", set()) else "missing",
            "task05_document": "present" if record.identity in processed else "missing",
        }
        for name, identities in sorted(providers.items()):
            row[name] = "present" if record.identity in identities else "missing"
        rows.append(row)
    return rows


def _write_coverage(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_issues(path: Path, issues: Iterable[Issue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for issue in issues:
            handle.write(json.dumps(issue.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def _render_report(
    path: Path,
    *,
    config: AuditConfig,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    issues: list[Issue],
) -> None:
    issue_counts = Counter(issue.severity for issue in issues)
    lines = [
        "# Data Readiness Audit Run",
        "",
        f"- Audit version: `{AUDIT_VERSION}`",
        f"- Run ID: `{manifest.get('run_id')}`",
        f"- Branch: `{manifest.get('git', {}).get('branch')}`",
        f"- HEAD: `{manifest.get('git', {}).get('commit')}`",
        f"- Git dirty at audit start: `{manifest.get('git', {}).get('dirty')}`",
        f"- Overall status: **{summary.get('overall_status')}**",
        "",
        "## Audited paths",
        "",
    ]
    for name, value in manifest.get("paths", {}).items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "## Section status",
            "",
            "| Section | Status | Key counts |",
            "|---|---|---|",
        ]
    )
    for name, section in summary.get("sections", {}).items():
        counts = []
        for key in (
            "expected_canonical_keyframes",
            "present_keyframes",
            "missing_keyframes",
            "coverage",
            "valid_artifacts",
            "missing_canonical",
            "document_files",
            "pending_providers",
        ):
            if key in section:
                counts.append(f"{key}={section[key]}")
        lines.append(f"| `{name}` | `{section.get('status')}` | {', '.join(counts)} |")
    lines.extend(
        [
            "",
            "## Issues",
            "",
            f"- Errors/blockers: `{issue_counts.get('error', 0)}`",
            f"- Warnings: `{issue_counts.get('warning', 0)}`",
            f"- Pending/info findings: `{issue_counts.get('pending', 0) + issue_counts.get('info', 0)}`",
            "",
            "Only filesystem reads and report writes were performed. No corpus "
            "repair, extraction, model inference, download, or index mutation "
            "is performed by this audit.",
            "",
            "Machine-readable outputs in this run directory: "
            "`manifest.json`, `summary.json`, `issues.jsonl`, and `coverage.csv`.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(config: AuditConfig) -> AuditResult:
    """Run one read-only audit and write a deterministic report bundle."""

    config = config.normalized()
    if config.dataset_root is None and config.artifacts_root is None:
        raise AuditConfigurationError("at least one of dataset_root or artifacts_root is required")
    for name, path in (
        ("dataset_root", config.dataset_root),
        ("artifacts_root", config.artifacts_root),
    ):
        if path is not None and not path.exists():
            raise AuditConfigurationError(f"{name} does not exist: {path}")
    roots = _discover_roots(config)
    issues: list[Issue] = []
    mapping_issue_start = len(issues)
    canonical, mapping_meta = _load_mapping(roots["mapping"], issues)
    mapping_meta["status"] = _status(issues[mapping_issue_start:])
    videos = _video_inventory(roots["videos"], canonical, issues)
    keyframes, keyframe_ids = _keyframe_inventory(
        roots["keyframes"], roots["mapping"], canonical, issues
    )
    media = _media_info(roots["media_info"], canonical, issues)
    branch_sets: dict[str, set[tuple[str, str]]] = {}
    branch_stats: dict[str, dict[str, Any]] = {}
    for branch in ("ocr", "asr", "objects"):
        branch_stats[branch], branch_sets[branch] = _json_branch(
            branch, roots[branch], canonical, issues
        )
    embedding_stats, embedding_sets = _embedding_audit(
        roots["embeddings"], canonical, issues, config.required_providers
    )
    processed_stats, processed_ids = _processed_audit(
        roots["processed"], canonical, issues
    )
    index_stats = _index_audit(
        roots["indexes"], canonical, mapping_meta, processed_stats, issues
    )
    coverage = _coverage_rows(
        canonical, keyframe_ids, branch_sets, embedding_sets, processed_ids
    )
    sections: dict[str, dict[str, Any]] = {
        "canonical_mapping": {
            **mapping_meta,
        },
        "videos": videos,
        "keyframes": keyframes,
        "media_info": media,
        **branch_stats,
        "embeddings": embedding_stats,
        "processed_documents": processed_stats,
        "index_manifests": index_stats,
    }
    has_blocker = any(issue.severity == "error" for issue in issues)
    pending_optional = bool(embedding_stats.get("pending_providers"))
    overall = BLOCKED if has_blocker else (
        READY_WITH_WARNINGS if any(issue.severity == "warning" for issue in issues) or pending_optional else READY
    )
    summary = {
        "audit_version": AUDIT_VERSION,
        "overall_status": overall,
        "canonical_keyframes": len(canonical),
        "coverage_rows": len(coverage),
        "issue_counts": dict(Counter(issue.severity for issue in issues)),
        "sections": sections,
        "disk_usage": {
            name: _disk_usage(path)
            for name, path in roots.items()
            if path is not None
        },
        "read_only": True,
        "full_corpus_status": "NOT_RUN_DATA_UNAVAILABLE",
    }
    now = datetime.now(timezone.utc)
    run_id = config.run_id or now.strftime("%Y%m%dT%H%M%SZ")
    output_dir = config.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    git = _git_info()
    manifest = {
        "audit_version": AUDIT_VERSION,
        "run_id": run_id,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "git": git,
        "paths": {
            "dataset_root": str(config.dataset_root) if config.dataset_root else None,
            "artifacts_root": str(config.artifacts_root) if config.artifacts_root else None,
            **{name: str(path) if path else None for name, path in roots.items()},
            "output_dir": str(output_dir.resolve()),
        },
        "canonical_mapping": mapping_meta,
        "providers": sorted(embedding_stats.get("providers", {})),
        "required_providers": list(config.required_providers),
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "summary.json", summary)
    _write_issues(output_dir / "issues.jsonl", issues)
    _write_coverage(output_dir / "coverage.csv", coverage)
    report_path = output_dir / "report.md"
    _render_report(report_path, config=config, manifest=manifest, summary=summary, issues=issues)
    return AuditResult(
        config=config,
        manifest=manifest,
        summary=summary,
        issues=issues,
        coverage=coverage,
        report_path=report_path,
        output_dir=output_dir,
    )

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .contracts import ASRArtifact, KeyframeAudioSource


FORMAT = "mvr-asr-keyframe-v1"
ARTIFACT_FIELDS = {
    "video_id",
    "keyframe_id",
    "frame_idx",
    "timestamp",
    "shot_id",
    "text",
    "segments",
}


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def artifact_path(output_root: Path, video_id: str, keyframe_id: str) -> Path:
    if Path(video_id).name != video_id or Path(keyframe_id).name != keyframe_id:
        raise ValueError("video_id and keyframe_id must be safe path components")
    return output_root / video_id / f"{keyframe_id}.json"


def write_artifact(path: Path, artifact: ASRArtifact) -> None:
    _atomic_json_write(path, artifact.to_dict())


def is_valid_artifact(
    path: Path, expected: KeyframeAudioSource | None = None
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(payload, dict) or set(payload) != ARTIFACT_FIELDS:
        return False
    if not isinstance(payload["video_id"], str) or not payload["video_id"]:
        return False
    if not isinstance(payload["keyframe_id"], str) or not payload["keyframe_id"]:
        return False
    if not isinstance(payload["frame_idx"], int):
        return False
    if not isinstance(payload["timestamp"], (int, float)):
        return False
    if not isinstance(payload["text"], str) or not isinstance(payload["segments"], list):
        return False
    if expected is not None and (
        payload["video_id"] != expected.video_id
        or payload["keyframe_id"] != expected.keyframe_id
        or payload["frame_idx"] != expected.frame_idx
        or payload["timestamp"] != expected.timestamp
        or payload["shot_id"] != expected.shot_id
    ):
        return False
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "text"}:
            return False
        if not isinstance(segment["start"], (int, float)):
            return False
        if not isinstance(segment["end"], (int, float)):
            return False
        if segment["end"] < segment["start"] or not isinstance(segment["text"], str):
            return False
    return True


def read_failed_keys(output_root: Path) -> set[tuple[str, str]]:
    path = output_root / "errors.json"
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            (str(item["video_id"]), str(item["keyframe_id"]))
            for item in payload.get("errors", [])
        }
    except (OSError, ValueError, TypeError, KeyError):
        return set()


def read_warnings(output_root: Path) -> list[dict]:
    path = output_root / "errors.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        warnings = payload.get("warnings", [])
        return [item for item in warnings if isinstance(item, dict)]
    except (OSError, ValueError, TypeError):
        return []


def write_report(output_root: Path, errors: list[dict], warnings: list[dict]) -> None:
    _atomic_json_write(
        output_root / "errors.json",
        {"errors": errors, "warnings": warnings},
    )


def write_manifest(output_root: Path, manifest: dict) -> None:
    _atomic_json_write(output_root / "manifest.json", manifest)

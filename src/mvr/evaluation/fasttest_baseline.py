"""Deterministic structural regression gate for the five-keyframe fasttest."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mvr.indexing.validation import Task05AValidationError, validate_task05a_document


class FasttestBaselineError(ValueError):
    """Raised when the fasttest config or artifact contract is invalid."""


@dataclass(frozen=True)
class FasttestBaselineConfig:
    fasttest_root: Path
    canonical_mapping: Path
    processed_root: Path
    lexical_manifest: Path
    expected_keyframes: int = 5

    def __post_init__(self) -> None:
        if self.expected_keyframes < 1:
            raise FasttestBaselineError("expected_keyframes must be positive")
        for name in (
            "fasttest_root",
            "canonical_mapping",
            "processed_root",
            "lexical_manifest",
        ):
            value = getattr(self, name)
            if value.is_absolute() or ".." in value.parts:
                raise FasttestBaselineError(
                    f"{name} must be a safe repository-relative path"
                )


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FasttestBaselineError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FasttestBaselineError(f"JSON must be an object: {path}")
    return payload


def load_fasttest_config(path: Path | str) -> FasttestBaselineConfig:
    config_path = Path(path)
    payload = _read_json(config_path)
    required = {
        "fasttest_root",
        "canonical_mapping",
        "processed_root",
        "lexical_manifest",
        "expected_keyframes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise FasttestBaselineError(f"config missing required fields: {missing}")
    try:
        return FasttestBaselineConfig(
            fasttest_root=Path(str(payload["fasttest_root"])),
            canonical_mapping=Path(str(payload["canonical_mapping"])),
            processed_root=Path(str(payload["processed_root"])),
            lexical_manifest=Path(str(payload["lexical_manifest"])),
            expected_keyframes=int(payload["expected_keyframes"]),
        )
    except (TypeError, ValueError) as exc:
        raise FasttestBaselineError(f"invalid fasttest config: {exc}") from exc


def _canonical_ids(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "keyframe_id" not in reader.fieldnames:
                raise FasttestBaselineError(
                    "canonical mapping must contain keyframe_id"
                )
            ids = [str(row.get("keyframe_id", "")).strip() for row in reader]
    except OSError as exc:
        raise FasttestBaselineError(f"cannot read canonical mapping {path}: {exc}") from exc
    if any(not keyframe_id for keyframe_id in ids):
        raise FasttestBaselineError("canonical mapping contains an empty keyframe_id")
    return ids


def verify_fasttest_baseline(
    repository_root: Path | str,
    config: FasttestBaselineConfig,
    *,
    require_lexical: bool = False,
) -> dict[str, Any]:
    """Return a stable report; ``errors`` is empty only when the gate passes."""

    repository_root = Path(repository_root).resolve()
    fasttest_root = repository_root / config.fasttest_root
    mapping_path = fasttest_root / config.canonical_mapping
    processed_root = fasttest_root / config.processed_root
    documents_root = processed_root / "documents"
    processed_manifest_path = processed_root / "manifest.json"
    lexical_manifest_path = fasttest_root / config.lexical_manifest

    canonical_ids = _canonical_ids(mapping_path)
    canonical_set = set(canonical_ids)
    canonical_duplicates = len(canonical_ids) - len(canonical_set)

    document_ids: list[str] = []
    invalid_documents = 0
    if documents_root.is_dir():
        for path in sorted(documents_root.rglob("*.json")):
            try:
                payload = _read_json(path)
                validate_task05a_document(payload)
            except (FasttestBaselineError, Task05AValidationError):
                invalid_documents += 1
                continue
            document_ids.append(str(payload["keyframe_id"]).strip())
    document_set = set(document_ids)
    document_duplicates = len(document_ids) - len(document_set)
    missing_ids = sorted(canonical_set - document_set)
    extra_ids = sorted(document_set - canonical_set)

    processed_manifest = _read_json(processed_manifest_path)
    issue_counts = processed_manifest.get("validation_issue_counts", {})
    if not isinstance(issue_counts, Mapping):
        raise FasttestBaselineError(
            "Task05A manifest validation_issue_counts must be an object"
        )
    invalid_schema = int(issue_counts.get("invalid_schema", 0))
    manifest_document_count = int(processed_manifest.get("document_count", -1))

    lexical_available = lexical_manifest_path.is_file()
    lexical_counts: dict[str, int | None] = {
        "indexed_documents": None,
        "failed": None,
        "missing": None,
        "extra": None,
        "duplicates": None,
    }
    if lexical_available:
        lexical_manifest = _read_json(lexical_manifest_path)
        lexical_counts = {
            "indexed_documents": int(lexical_manifest.get("indexed_count", -1)),
            "failed": int(lexical_manifest.get("failed_count", -1)),
            "missing": int(lexical_manifest.get("missing_id_count", -1)),
            "extra": int(lexical_manifest.get("extra_id_count", -1)),
            "duplicates": int(lexical_manifest.get("duplicate_id_count", -1)),
        }

    errors: list[str] = []
    expected = config.expected_keyframes
    checks = (
        (len(canonical_ids) == expected, "canonical mapping count mismatch"),
        (canonical_duplicates == 0, "duplicate canonical keyframe_id"),
        (len(document_ids) == expected, "Task05A document count mismatch"),
        (document_duplicates == 0, "duplicate Task05A keyframe_id"),
        (not missing_ids, "Task05A documents are missing canonical identities"),
        (not extra_ids, "Task05A documents contain extra identities"),
        (invalid_documents == 0, "Task05A documents contain invalid schema"),
        (invalid_schema == 0, "Task05A manifest reports invalid_schema"),
        (
            manifest_document_count == len(document_ids),
            "Task05A manifest/document count mismatch",
        ),
    )
    errors.extend(message for passed, message in checks if not passed)
    if require_lexical and not lexical_available:
        errors.append("Task05B lexical manifest is required but missing")
    if lexical_available:
        lexical_checks = (
            (
                lexical_counts["indexed_documents"] == expected,
                "Task05B indexed document count mismatch",
            ),
            (lexical_counts["failed"] == 0, "Task05B reports failed documents"),
            (lexical_counts["missing"] == 0, "Task05B reports missing identities"),
            (lexical_counts["extra"] == 0, "Task05B reports extra identities"),
            (
                lexical_counts["duplicates"] == 0,
                "Task05B reports duplicate identities",
            ),
        )
        errors.extend(message for passed, message in lexical_checks if not passed)

    return {
        "schema_version": "1.0",
        "status": "pass" if not errors else "fail",
        "expected_keyframes": expected,
        "canonical": {
            "mapping_rows": len(canonical_ids),
            "unique_keyframe_ids": len(canonical_set),
            "duplicate_id_count": canonical_duplicates,
        },
        "task05a": {
            "documents": len(document_ids),
            "unique_keyframe_ids": len(document_set),
            "invalid_documents": invalid_documents,
            "invalid_schema": invalid_schema,
            "missing_id_count": len(missing_ids),
            "extra_id_count": len(extra_ids),
            "duplicate_id_count": document_duplicates,
        },
        "task05b": {
            "available": lexical_available,
            **lexical_counts,
        },
        "errors": errors,
    }

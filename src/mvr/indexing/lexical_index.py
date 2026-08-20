"""Task05B v1 Elasticsearch lexical-index construction.

Task05A processed documents are the sole input contract.  This module does not
read or reconstruct raw OCR/ASR features.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK05B_SCHEMA_VERSION = "1.0"
SEARCHABLE_FIELDS = ["ocr_text", "asr_text"]
LEXICAL_MAPPINGS: dict[str, Any] = {
    "properties": {
        "keyframe_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "frame_idx": {"type": "integer"},
        "shot_id": {"type": "integer"},
        "timestamp": {"type": "float"},
        "ocr_text": {"type": "text"},
        "asr_text": {"type": "text"},
    }
}


class Task05BValidationError(ValueError):
    """Raised when authoritative Task05A input violates the lexical contract."""


class Task05BBulkError(RuntimeError):
    """Raised when Elasticsearch reports one or more bulk item failures."""

    def __init__(self, failed_count: int, errors: list[Any]) -> None:
        super().__init__(f"Elasticsearch bulk indexing failed for {failed_count} items")
        self.failed_count = failed_count
        self.errors = errors


@dataclass(frozen=True)
class LexicalIndexDocument:
    keyframe_id: str
    video_id: str
    frame_idx: int
    shot_id: int | None
    timestamp: float
    ocr_text: str
    asr_text: str

    def to_source(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "video_id": self.video_id,
            "frame_idx": self.frame_idx,
            "shot_id": self.shot_id,
            "timestamp": self.timestamp,
            "ocr_text": self.ocr_text,
            "asr_text": self.asr_text,
        }


@dataclass(frozen=True)
class Task05ASource:
    processed_root: Path
    manifest_path: Path
    manifest_sha256: str
    source_document_count: int
    lexical_indexable_count: int
    skipped_count: int
    documents: tuple[LexicalIndexDocument, ...]


@dataclass(frozen=True)
class LexicalIndexBuildResult:
    source_document_count: int
    lexical_indexable_count: int
    indexed_count: int
    skipped_count: int
    failed_count: int
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    manifest_path: Path
    manifest: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_nonblank_string(
    document: Mapping[str, Any],
    field: str,
    *,
    path: Path,
) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise Task05BValidationError(f"{path}: {field} must be a non-empty string")
    return value.strip()


def _required_integer(
    document: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    nullable: bool = False,
) -> int | None:
    value = document.get(field)
    if nullable and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise Task05BValidationError(f"{path}: {field} must be an integer")
    if field == "frame_idx" and value < 0:
        raise Task05BValidationError(f"{path}: frame_idx must be non-negative")
    return value


def _required_timestamp(document: Mapping[str, Any], *, path: Path) -> float:
    value = document.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Task05BValidationError(f"{path}: timestamp must be numeric")
    timestamp = float(value)
    if not math.isfinite(timestamp) or timestamp < 0:
        raise Task05BValidationError(f"{path}: timestamp must be finite and non-negative")
    return timestamp


def _joined_text(
    document: Mapping[str, Any],
    section: str,
    *,
    path: Path,
) -> str:
    payload = document.get(section)
    if not isinstance(payload, Mapping):
        raise Task05BValidationError(f"{path}: {section} must be an object")
    joined = payload.get("joined")
    if not isinstance(joined, str):
        raise Task05BValidationError(f"{path}: {section}.joined must be a string")
    return joined


def project_task05a_lexical_document(
    document: Mapping[str, Any],
    *,
    path: Path = Path("<document>"),
) -> LexicalIndexDocument | None:
    """Validate and project one Task05A document.

    ``indexability.lexical`` is authoritative.  A false value always skips the
    document, even when text happens to be non-empty.
    """

    indexability = document.get("indexability")
    if not isinstance(indexability, Mapping):
        raise Task05BValidationError(f"{path}: indexability must be an object")
    lexical = indexability.get("lexical")
    if not isinstance(lexical, bool):
        raise Task05BValidationError(f"{path}: indexability.lexical must be boolean")

    keyframe_id = _required_nonblank_string(document, "keyframe_id", path=path)
    video_id = _required_nonblank_string(document, "video_id", path=path)
    frame_idx = int(_required_integer(document, "frame_idx", path=path))
    shot_id = _required_integer(document, "shot_id", path=path, nullable=True)
    timestamp = _required_timestamp(document, path=path)
    ocr_text = _joined_text(document, "ocr", path=path)
    asr_text = _joined_text(document, "asr", path=path)
    if not lexical:
        return None

    return LexicalIndexDocument(
        keyframe_id=keyframe_id,
        video_id=video_id,
        frame_idx=frame_idx,
        shot_id=shot_id,
        timestamp=timestamp,
        ocr_text=ocr_text,
        asr_text=asr_text,
    )


def load_task05a_lexical_source(processed_root: Path) -> Task05ASource:
    processed_root = Path(processed_root)
    manifest_path = processed_root / "manifest.json"
    documents_root = processed_root / "documents"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Task05A manifest not found: {manifest_path}")
    if not documents_root.is_dir():
        raise FileNotFoundError(f"Task05A documents root not found: {documents_root}")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise Task05BValidationError("Task05A manifest must be an object")
    if manifest.get("task") != "task05a":
        raise Task05BValidationError("source manifest task must be task05a")
    expected_count = manifest.get("document_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise Task05BValidationError("Task05A manifest document_count must be an integer")

    paths = sorted(documents_root.rglob("*.json"))
    if len(paths) != expected_count:
        raise Task05BValidationError(
            "Task05A document file count does not match manifest document_count"
        )

    projected: list[LexicalIndexDocument] = []
    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for path in paths:
        payload = _read_json(path)
        if not isinstance(payload, Mapping):
            raise Task05BValidationError(f"{path}: document must be an object")
        keyframe_id = _required_nonblank_string(payload, "keyframe_id", path=path)
        if keyframe_id in seen_ids:
            duplicate_ids.add(keyframe_id)
        seen_ids.add(keyframe_id)
        item = project_task05a_lexical_document(payload, path=path)
        if item is not None:
            projected.append(item)
    if duplicate_ids:
        raise Task05BValidationError(
            f"duplicate Task05A keyframe IDs: {sorted(duplicate_ids)}"
        )

    projected.sort(key=lambda item: item.keyframe_id)
    declared_lexical = manifest.get("indexability_counts", {}).get("lexical")
    if (
        isinstance(declared_lexical, bool)
        or not isinstance(declared_lexical, int)
        or declared_lexical != len(projected)
    ):
        raise Task05BValidationError(
            "Task05A manifest lexical count does not match authoritative documents"
        )
    return Task05ASource(
        processed_root=processed_root.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256_file(manifest_path),
        source_document_count=len(paths),
        lexical_indexable_count=len(projected),
        skipped_count=len(paths) - len(projected),
        documents=tuple(projected),
    )


def lexical_bulk_actions(
    documents: Iterable[LexicalIndexDocument],
    *,
    index_name: str,
) -> Iterable[dict[str, Any]]:
    for document in documents:
        yield {
            "_index": index_name,
            "_id": document.keyframe_id,
            "_source": document.to_source(),
        }


def _default_bulk_fn(client: Any, actions: Iterable[dict[str, Any]], **kwargs: Any) -> Any:
    try:
        from elasticsearch.helpers import bulk
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError(
            "Elasticsearch client is missing; install the project's search extra"
        ) from exc
    return bulk(client, actions, **kwargs)


def _default_scan_fn(client: Any, *, index: str) -> Iterable[Mapping[str, Any]]:
    try:
        from elasticsearch.helpers import scan
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError(
            "Elasticsearch client is missing; install the project's search extra"
        ) from exc
    return scan(
        client,
        index=index,
        query={"query": {"match_all": {}}},
        _source=False,
    )


def _index_exists(client: Any, index_name: str) -> bool:
    return bool(client.indices.exists(index=index_name))


def create_lexical_index(
    client: Any,
    *,
    index_name: str,
    recreate: bool,
) -> None:
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("index_name must be non-empty")
    exists = _index_exists(client, index_name)
    if recreate and exists:
        client.indices.delete(index=index_name)
        exists = False
    if not exists:
        client.indices.create(index=index_name, mappings=LEXICAL_MAPPINGS)


def _normalize_bulk_result(result: Any) -> tuple[int, list[Any]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("malformed Elasticsearch bulk response")
    success, errors = result
    if isinstance(success, bool) or not isinstance(success, int):
        raise RuntimeError("malformed Elasticsearch bulk success count")
    if errors is None:
        errors = []
    if not isinstance(errors, list):
        raise RuntimeError("malformed Elasticsearch bulk error list")
    return success, errors


def _scan_ids(
    client: Any,
    *,
    index_name: str,
    scan_fn: Callable[..., Iterable[Mapping[str, Any]]],
) -> tuple[set[str], tuple[str, ...]]:
    ids: list[str] = []
    for hit in scan_fn(client, index=index_name):
        if not isinstance(hit, Mapping):
            raise RuntimeError("malformed Elasticsearch identity scan hit")
        raw_id = hit.get("_id")
        if not isinstance(raw_id, str) or not raw_id:
            raise RuntimeError("Elasticsearch identity scan hit is missing _id")
        ids.append(raw_id)
    duplicates = tuple(sorted({item for item in ids if ids.count(item) > 1}))
    return set(ids), duplicates


def build_lexical_index(
    client: Any,
    *,
    processed_root: Path,
    index_name: str,
    recreate: bool = False,
    manifest_path: Path | None = None,
    bulk_fn: Callable[..., Any] | None = None,
    scan_fn: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> LexicalIndexBuildResult:
    source = load_task05a_lexical_source(processed_root)
    create_lexical_index(client, index_name=index_name, recreate=recreate)
    bulk_callable = bulk_fn or _default_bulk_fn
    bulk_result = bulk_callable(
        client,
        lexical_bulk_actions(source.documents, index_name=index_name),
        chunk_size=50,
        request_timeout=120,
        refresh=False,
        raise_on_error=False,
        raise_on_exception=False,
    )
    success_count, errors = _normalize_bulk_result(bulk_result)
    failed_count = len(errors)
    if failed_count:
        raise Task05BBulkError(failed_count, errors)
    if success_count != source.lexical_indexable_count:
        raise RuntimeError(
            "Elasticsearch bulk success count does not match lexical source count"
        )

    client.indices.refresh(index=index_name)
    indexed_count = int(client.count(index=index_name)["count"])
    expected_ids = {document.keyframe_id for document in source.documents}
    actual_ids, duplicate_ids = _scan_ids(
        client,
        index_name=index_name,
        scan_fn=scan_fn or _default_scan_fn,
    )
    missing_ids = tuple(sorted(expected_ids - actual_ids))
    extra_ids = tuple(sorted(actual_ids - expected_ids))
    if (
        indexed_count != source.lexical_indexable_count
        or missing_ids
        or extra_ids
        or duplicate_ids
    ):
        raise RuntimeError(
            "Elasticsearch lexical identity/cardinality audit failed: "
            f"count={indexed_count}, missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}, duplicates={len(duplicate_ids)}"
        )

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else source.processed_root.parent / "indexes" / "lexical" / "manifest.json"
    )
    manifest = {
        "schema_version": TASK05B_SCHEMA_VERSION,
        "task": "task05b",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "backend": "elasticsearch",
        "index_name": index_name,
        "similarity": "BM25",
        "searchable_fields": list(SEARCHABLE_FIELDS),
        "source_processed_root": str(source.processed_root),
        "source_task05a_manifest": str(source.manifest_path),
        "source_task05a_manifest_sha256": source.manifest_sha256,
        "source_document_count": source.source_document_count,
        "lexical_indexable_count": source.lexical_indexable_count,
        "indexed_count": indexed_count,
        "skipped_count": source.skipped_count,
        "failed_count": 0,
        "missing_id_count": 0,
        "extra_id_count": 0,
        "duplicate_id_count": 0,
        "mapping": LEXICAL_MAPPINGS,
        "settings": {
            "similarity": "Elasticsearch default BM25",
            "analysis": "Elasticsearch default text analysis",
            "bulk_chunk_size": 50,
            "refresh_per_document": False,
            "final_refresh": True,
        },
    }
    _atomic_write_json(resolved_manifest_path, manifest)
    return LexicalIndexBuildResult(
        source_document_count=source.source_document_count,
        lexical_indexable_count=source.lexical_indexable_count,
        indexed_count=indexed_count,
        skipped_count=source.skipped_count,
        failed_count=0,
        missing_ids=missing_ids,
        extra_ids=extra_ids,
        duplicate_ids=duplicate_ids,
        manifest_path=resolved_manifest_path,
        manifest=manifest,
    )

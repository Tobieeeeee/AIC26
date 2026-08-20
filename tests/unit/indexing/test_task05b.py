from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.indexing.lexical_index import (
    LEXICAL_MAPPINGS,
    Task05BBulkError,
    Task05BValidationError,
    build_lexical_index,
    lexical_bulk_actions,
    load_task05a_lexical_source,
    project_task05a_lexical_document,
)


def _document(
    keyframe_id: str,
    *,
    lexical: bool = True,
    ocr: str = "TIN CHÍNH",
    asr: str = "đàn bò ở lâm đồng",
) -> dict:
    return {
        "schema_version": "1.0",
        "keyframe_id": keyframe_id,
        "video_id": "L21_V010",
        "frame_idx": 578,
        "shot_id": 7,
        "timestamp": 23.12,
        "ocr": {"joined": ocr, "texts": []},
        "asr": {"joined": asr, "texts": []},
        "indexability": {"lexical": lexical, "vector": True, "object": True},
    }


def _write_processed(root: Path, documents: list[dict]) -> Path:
    processed = root / "processed"
    for document in documents:
        path = (
            processed
            / "documents"
            / document["video_id"]
            / f"{document['keyframe_id']}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "task": "task05a",
        "document_count": len(documents),
        "indexability_counts": {
            "lexical": sum(
                1 for document in documents if document["indexability"]["lexical"]
            )
        },
    }
    (processed / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return processed


def test_task05b_mapping_uses_default_text_fields_without_combined_text() -> None:
    assert LEXICAL_MAPPINGS == {
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
    assert "combined_text" not in LEXICAL_MAPPINGS["properties"]


def test_projection_preserves_identity_and_keeps_ocr_asr_separate() -> None:
    projected = project_task05a_lexical_document(_document("frame-a"))
    assert projected is not None
    assert projected.to_source() == {
        "keyframe_id": "frame-a",
        "video_id": "L21_V010",
        "frame_idx": 578,
        "shot_id": 7,
        "timestamp": 23.12,
        "ocr_text": "TIN CHÍNH",
        "asr_text": "đàn bò ở lâm đồng",
    }
    assert "combined_text" not in projected.to_source()


@pytest.mark.parametrize(
    ("ocr", "asr"),
    [("OCR only", ""), ("", "ASR only"), ("OCR", "ASR")],
)
def test_projection_supports_ocr_only_asr_only_and_both(ocr: str, asr: str) -> None:
    projected = project_task05a_lexical_document(
        _document("frame-a", ocr=ocr, asr=asr)
    )
    assert projected is not None
    assert projected.ocr_text == ocr
    assert projected.asr_text == asr


def test_lexical_false_is_skipped_even_with_nonempty_text() -> None:
    assert (
        project_task05a_lexical_document(
            _document("frame-a", lexical=False, ocr="must not index", asr="also skip")
        )
        is None
    )


def test_malformed_required_identity_is_surfaced() -> None:
    document = _document("frame-a")
    document.pop("frame_idx")
    with pytest.raises(Task05BValidationError, match="frame_idx"):
        project_task05a_lexical_document(document)


def test_source_loader_selects_only_authoritative_lexical_true(tmp_path: Path) -> None:
    processed = _write_processed(
        tmp_path,
        [_document("frame-a"), _document("frame-b", lexical=False)],
    )
    source = load_task05a_lexical_source(processed)
    assert source.source_document_count == 2
    assert source.lexical_indexable_count == 1
    assert source.skipped_count == 1
    assert [document.keyframe_id for document in source.documents] == ["frame-a"]


def test_bulk_action_uses_keyframe_id_as_elasticsearch_id() -> None:
    projected = project_task05a_lexical_document(_document("frame-a"))
    assert projected is not None
    assert list(lexical_bulk_actions([projected], index_name="lexical")) == [
        {
            "_index": "lexical",
            "_id": "frame-a",
            "_source": projected.to_source(),
        }
    ]


class _Indices:
    def __init__(
        self,
        calls: list[tuple[str, dict]],
        *,
        exists: bool = False,
    ) -> None:
        self.calls = calls
        self._exists = exists

    def exists(self, **kwargs):
        self.calls.append(("exists", kwargs))
        return self._exists

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))

    def delete(self, **kwargs):
        self.calls.append(("delete", kwargs))

    def refresh(self, **kwargs):
        self.calls.append(("refresh", kwargs))


class _Client:
    def __init__(self, *, exists: bool = False) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.indices = _Indices(self.calls, exists=exists)

    def count(self, **kwargs):
        self.calls.append(("count", kwargs))
        return {"count": 1}


def test_build_indexes_and_audits_identity(tmp_path: Path) -> None:
    processed = _write_processed(tmp_path, [_document("frame-a")])
    client = _Client()
    captured_actions = []

    def bulk_fn(client, actions, **kwargs):
        captured_actions.extend(actions)
        assert kwargs["refresh"] is False
        return 1, []

    def scan_fn(client, *, index):
        return [{"_id": "frame-a"}]

    result = build_lexical_index(
        client,
        processed_root=processed,
        index_name="lexical",
        recreate=False,
        bulk_fn=bulk_fn,
        scan_fn=scan_fn,
    )
    assert result.indexed_count == 1
    assert result.failed_count == 0
    assert result.manifest["similarity"] == "BM25"
    assert result.manifest["searchable_fields"] == ["ocr_text", "asr_text"]
    assert result.manifest["source_document_count"] == 1
    assert result.manifest["lexical_indexable_count"] == 1
    assert result.manifest["indexed_count"] == 1
    assert captured_actions[0]["_id"] == "frame-a"
    create_call = next(call for call in client.calls if call[0] == "create")
    assert create_call[1]["mappings"] == LEXICAL_MAPPINGS


def test_recreate_deletes_existing_index_before_create(tmp_path: Path) -> None:
    processed = _write_processed(tmp_path, [_document("frame-a")])
    client = _Client(exists=True)
    build_lexical_index(
        client,
        processed_root=processed,
        index_name="lexical",
        recreate=True,
        bulk_fn=lambda *args, **kwargs: (1, []),
        scan_fn=lambda *args, **kwargs: [{"_id": "frame-a"}],
    )
    operation_names = [name for name, _ in client.calls]
    assert operation_names[:3] == ["exists", "delete", "create"]


def test_bulk_partial_failures_are_raised(tmp_path: Path) -> None:
    processed = _write_processed(tmp_path, [_document("frame-a")])
    with pytest.raises(Task05BBulkError, match="failed for 1"):
        build_lexical_index(
            _Client(),
            processed_root=processed,
            index_name="lexical",
            bulk_fn=lambda *args, **kwargs: (0, [{"index": {"status": 500}}]),
            scan_fn=lambda *args, **kwargs: [],
        )

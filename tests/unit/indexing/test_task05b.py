from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.indexing.lexical_index import (
    LEXICAL_MAPPINGS,
    LEXICAL_SETTINGS,
    TASK05B_SCHEMA_VERSION,
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


def test_task05b_mapping_has_exact_and_folded_multifields() -> None:
    """Schema 1.2 keeps exact text and adds folded search-only subfields."""
    assert LEXICAL_MAPPINGS == {
        "properties": {
            "keyframe_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "frame_idx": {"type": "integer"},
            "shot_id": {"type": "integer"},
            "timestamp": {"type": "float"},
            "ocr_text": {
                "type": "text",
                "analyzer": "vi_exact",
                "fields": {
                    "folded": {"type": "text", "analyzer": "vi_folded"}
                },
            },
            "asr_text": {
                "type": "text",
                "analyzer": "vi_exact",
                "fields": {
                    "folded": {"type": "text", "analyzer": "vi_folded"}
                },
            },
        }
    }
    assert TASK05B_SCHEMA_VERSION == "1.2"
    assert LEXICAL_SETTINGS["analysis"]["analyzer"] == {
        "vi_exact": {"tokenizer": "standard", "filter": ["lowercase"]},
        "vi_folded": {
            "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding"],
        },
    }
    assert "combined_text" not in LEXICAL_MAPPINGS["properties"]
    assert not any(
        "boost" in field for field in LEXICAL_MAPPINGS["properties"].values()
    )


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


def test_bulk_action_identifies_a_document_by_video_and_keyframe() -> None:
    """This asserted `_id == "frame-a"` -- the keyframe id alone -- and was
    right while the team generated its own ids: `L21_V001_S0007_K01` spells
    the video into the id, so it could not collide.

    The organizer's ids do not. 166,550 keyframes carry 632 distinct ids
    ("034" is in 842 videos, one per video), so `_id = keyframe_id` would
    index every document and keep 632, each overwritten ~263 times, without
    a single error anywhere.
    """
    projected = project_task05a_lexical_document(_document("frame-a"))
    assert projected is not None
    assert list(lexical_bulk_actions([projected], index_name="lexical")) == [
        {
            "_index": "lexical",
            "_id": "L21_V010/frame-a",
            "_source": projected.to_source(),
        }
    ]


def test_the_same_keyframe_id_in_two_videos_gets_two_documents() -> None:
    """The case the fixture could not produce: `_document` hardcodes
    L21_V010, so every test built one video and a collision had nowhere to
    happen."""
    first = project_task05a_lexical_document(
        {**_document("034"), "video_id": "L21_V001"}
    )
    second = project_task05a_lexical_document(
        {**_document("034"), "video_id": "L21_V002"}
    )
    assert first is not None and second is not None

    ids = [
        action["_id"]
        for action in lexical_bulk_actions([first, second], index_name="lexical")
    ]

    assert ids == ["L21_V001/034", "L21_V002/034"]
    assert len(set(ids)) == 2


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
        return [{"_id": "L21_V010/frame-a"}]

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
    assert result.manifest["schema_version"] == "1.2"
    assert captured_actions[0]["_id"] == "L21_V010/frame-a"
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
        scan_fn=lambda *args, **kwargs: [{"_id": "L21_V010/frame-a"}],
    )
    operation_names = [name for name, _ in client.calls]
    assert operation_names[:3] == ["exists", "delete", "create"]


def test_existing_index_requires_explicit_recreate(tmp_path: Path) -> None:
    processed = _write_processed(tmp_path, [_document("frame-a")])
    client = _Client(exists=True)

    with pytest.raises(FileExistsError, match="recreate=True explicitly"):
        build_lexical_index(
            client,
            processed_root=processed,
            index_name="production-index",
            recreate=False,
            bulk_fn=lambda *args, **kwargs: pytest.fail("bulk must not run"),
            scan_fn=lambda *args, **kwargs: pytest.fail("scan must not run"),
        )

    assert [name for name, _ in client.calls] == ["exists"]


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


def test_index_is_created_with_a_diacritic_folding_vietnamese_analyzer(
    tmp_path: Path,
) -> None:
    """OCR và ASR của corpus là tiếng Việt, và OCR sai dấu rất thường xuyên.
    Không có `asciifolding` thì "Ha Noi" không khớp "Hà Nội", nên một truy vấn
    tiếng Việt đúng vẫn trượt document đúng."""
    from mvr.indexing.lexical_index import LEXICAL_SETTINGS

    processed = _write_processed(tmp_path, [_document("frame-a")])
    client = _Client(exists=False)
    build_lexical_index(
        client,
        processed_root=processed,
        index_name="lexical-v1",
        recreate=False,
        bulk_fn=lambda *args, **kwargs: (1, []),
        scan_fn=lambda *args, **kwargs: [{"_id": "L21_V010/frame-a"}],
    )

    create_call = next(call for call in client.calls if call[0] == "create")
    assert create_call[1]["settings"] == LEXICAL_SETTINGS
    analyzer = LEXICAL_SETTINGS["analysis"]["analyzer"]["vi_folded"]
    assert analyzer["filter"] == ["lowercase", "asciifolding"]
    for field in ("ocr_text", "asr_text"):
        assert create_call[1]["mappings"]["properties"][field] == {
            "type": "text",
            "analyzer": "vi_exact",
            "fields": {
                "folded": {"type": "text", "analyzer": "vi_folded"}
            },
        }


def test_projection_keeps_source_case_diacritics_and_whitespace() -> None:
    original = "  GIÁ   VÀNG TĂNG  "
    projected = project_task05a_lexical_document(
        _document("frame-a", ocr=original, asr="")
    )
    assert projected is not None
    assert projected.to_source()["ocr_text"] == original

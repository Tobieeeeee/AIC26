from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mvr.evaluation.fasttest_baseline import (
    FasttestBaselineConfig,
    FasttestBaselineError,
    load_fasttest_config,
    verify_fasttest_baseline,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _document(index: int) -> dict:
    keyframe_id = f"V{1 + index // 3}_S{index + 1:04d}_K01"
    return {
        "schema_version": "1.0",
        "keyframe_id": keyframe_id,
        "video_id": f"V{1 + index // 3}",
        "shot_id": index + 1,
        "frame_idx": 100 + index,
        "timestamp": 4.0 + index,
        # Required since the Q&A track began reading these documents as
        # answering context: without a path there is no frame to show a
        # VLM or a human reviewer. build_index_documents emits it, so a
        # fixture without one is a document the pipeline cannot produce.
        "image_path": f"keyframes/{keyframe_id}.jpg",
        "ocr": {"texts": [], "joined": ""},
        "asr": {"texts": [], "joined": ""},
        "objects": {"labels": [], "counts": {}},
        "clip": {},
        "indexability": {"lexical": False, "vector": False, "object": False},
    }


def _make_baseline(root: Path, *, lexical: bool = False) -> FasttestBaselineConfig:
    fasttest = root / "artifacts" / "fasttest"
    documents = [_document(index) for index in range(5)]
    mapping = fasttest / "ingest" / "keyframe_mapping.csv"
    mapping.parent.mkdir(parents=True)
    with mapping.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["keyframe_id", "video_id", "frame_idx", "timestamp"],
        )
        writer.writeheader()
        for document in documents:
            writer.writerow(
                {
                    key: document[key]
                    for key in ("keyframe_id", "video_id", "frame_idx", "timestamp")
                }
            )
    for document in documents:
        _write_json(
            fasttest
            / "processed"
            / "documents"
            / document["video_id"]
            / f"{document['keyframe_id']}.json",
            document,
        )
    _write_json(
        fasttest / "processed" / "manifest.json",
        {"document_count": 5, "validation_issue_counts": {}},
    )
    if lexical:
        _write_json(
            fasttest / "indexes" / "lexical" / "manifest.json",
            {
                "indexed_count": 5,
                "failed_count": 0,
                "missing_id_count": 0,
                "extra_id_count": 0,
                "duplicate_id_count": 0,
            },
        )
    return FasttestBaselineConfig(
        fasttest_root=Path("artifacts/fasttest"),
        canonical_mapping=Path("ingest/keyframe_mapping.csv"),
        processed_root=Path("processed"),
        lexical_manifest=Path("indexes/lexical/manifest.json"),
    )


def test_structural_fasttest_passes_without_optional_backend_and_is_deterministic(
    tmp_path: Path,
) -> None:
    config = _make_baseline(tmp_path)

    first = verify_fasttest_baseline(tmp_path, config)
    second = verify_fasttest_baseline(tmp_path, config)

    assert first == second
    assert first["status"] == "pass"
    assert first["canonical"] == {
        "mapping_rows": 5,
        "unique_keyframe_ids": 5,
        "duplicate_id_count": 0,
    }
    assert first["task05a"]["documents"] == 5
    assert first["task05a"]["invalid_schema"] == 0
    assert first["task05b"]["available"] is False


def test_task05b_manifest_is_checked_when_available(tmp_path: Path) -> None:
    config = _make_baseline(tmp_path, lexical=True)

    report = verify_fasttest_baseline(tmp_path, config, require_lexical=True)

    assert report["status"] == "pass"
    assert report["task05b"]["indexed_documents"] == 5
    assert report["task05b"]["failed"] == 0


def test_required_task05b_manifest_missing_fails(tmp_path: Path) -> None:
    config = _make_baseline(tmp_path)

    report = verify_fasttest_baseline(tmp_path, config, require_lexical=True)

    assert report["status"] == "fail"
    assert report["errors"] == ["Task05B lexical manifest is required but missing"]


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("duplicate_mapping", "duplicate canonical keyframe_id"),
        ("missing_document", "Task05A document count mismatch"),
        ("malformed_document", "Task05A documents contain invalid schema"),
        ("invalid_schema_issue", "Task05A manifest reports invalid_schema"),
    ],
)
def test_structural_regressions_fail(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    config = _make_baseline(tmp_path)
    fasttest = tmp_path / config.fasttest_root
    if mutation == "duplicate_mapping":
        mapping = fasttest / config.canonical_mapping
        lines = mapping.read_text(encoding="utf-8").splitlines()
        mapping.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    elif mutation == "missing_document":
        next((fasttest / "processed" / "documents").rglob("*.json")).unlink()
    elif mutation == "malformed_document":
        path = next((fasttest / "processed" / "documents").rglob("*.json"))
        _write_json(path, {"keyframe_id": path.stem})
    else:
        _write_json(
            fasttest / "processed" / "manifest.json",
            {"document_count": 5, "validation_issue_counts": {"invalid_schema": 1}},
        )

    report = verify_fasttest_baseline(tmp_path, config)

    assert report["status"] == "fail"
    assert expected_error in report["errors"]


def test_config_rejects_absolute_or_parent_traversal_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(
        config_path,
        {
            "fasttest_root": "../outside",
            "canonical_mapping": "ingest/keyframe_mapping.csv",
            "processed_root": "processed",
            "lexical_manifest": "indexes/lexical/manifest.json",
            "expected_keyframes": 5,
        },
    )

    with pytest.raises(FasttestBaselineError, match="repository-relative"):
        load_fasttest_config(config_path)

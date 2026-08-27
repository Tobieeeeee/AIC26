"""The emitted document must carry the keyframe's image path.

`CanonicalKeyframe` already reads `image_path` out of the mapping CSV, and
`_project_document` then left it out, so the one field that lets a consumer
*see* the frame was dropped at the last step.

Two consumers need it and neither can recover it from the document alone:

- Q&A ranks up to 100 candidate frames and the flow ends in human or VLM
  review; nobody adjudicates a frame from `frame_idx: 578`.
- The search UI displays results, and a result page cannot render an
  integer.

Nothing else changes: the value is already in hand at that point, so this
is about not discarding it rather than about producing anything new.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.indexing.validation import Task05AValidationError, validate_task05a_document


def _make_image_path_fixture(tmp_path: Path) -> tuple[Path, Path]:
    features_root = tmp_path / "features"
    features_root.mkdir()
    mapping_csv = tmp_path / "ingest" / "keyframe_mapping.csv"
    mapping_csv.parent.mkdir()
    mapping_csv.write_text(
        "keyframe_id,video_id,shot_id,frame_idx,timestamp,image_path\n"
        "V1_S0001_K01,V1,1,10,0.4,keyframes/V1/V1_S0001_K01.jpg\n",
        encoding="utf-8",
    )
    return features_root, mapping_csv


def _document(**overrides):
    document = {
        "schema_version": "1.0",
        "keyframe_id": "L21_V010_S0007_K01",
        "video_id": "L21_V010",
        "shot_id": 7,
        "frame_idx": 578,
        "timestamp": 23.12,
        "image_path": "keyframes/L21_V010/L21_V010_S0007_K01.jpg",
        "ocr": {"texts": [], "joined": ""},
        "asr": {"texts": [], "joined": ""},
        "objects": {"labels": [], "counts": {}},
        "clip": {"video_id": "L21_V010", "feature_file": None, "row_idx": None},
        "indexability": {"lexical": False, "vector": False, "object": False},
    }
    document.update(overrides)
    return document


def test_a_document_carrying_an_image_path_validates() -> None:
    validate_task05a_document(_document())


def test_image_path_is_required() -> None:
    document = _document()
    del document["image_path"]

    with pytest.raises(Task05AValidationError):
        validate_task05a_document(document)


def test_a_blank_image_path_is_rejected() -> None:
    """An empty string would pass a presence check while helping nobody."""
    with pytest.raises(Task05AValidationError):
        validate_task05a_document(_document(image_path="   "))


def test_a_non_string_image_path_is_rejected() -> None:
    with pytest.raises(Task05AValidationError):
        validate_task05a_document(_document(image_path=578))


def test_the_pipeline_emits_the_image_path_from_the_mapping(tmp_path) -> None:
    """End-to-end through the real builder, not a hand-built document."""
    from mvr.indexing.task05a import build_index_documents

    features_root, mapping_csv = _make_image_path_fixture(tmp_path)
    result = build_index_documents(
        mapping_csv=mapping_csv,
        features_root=features_root,
        output_root=tmp_path / "processed",
        dataset="demo",
    )

    document = json.loads(
        (result.output_root / "documents" / "V1" / "V1_S0001_K01.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["image_path"] == "keyframes/V1/V1_S0001_K01.jpg"

import sys
from pathlib import Path
import pytest
import json

# Add worker to sys.path
WORKER_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "docker" / "ocr" / "worker"
sys.path.insert(0, str(WORKER_DIR))

from mvr.enrichment.ocr.docker_ocr import (
    _build_request_payload,
    process_candidates_to_canonical,
)
from json_file_contract import OcrCandidateFile, CandidateObject, write_json_file


def test_request_payload_uses_external_runtime_mount() -> None:
    sample = type(
        "Sample",
        (),
        {
            "keyframe_id": "K1",
            "video_id": "V1",
            "shot_id": 1,
            "frame_idx": 12,
            "timestamp": 0.48,
            "image_path": "temp/V1_K1_12.jpg",
        },
    )()
    anchor = {"frame_idx": 13, "timestamp": 0.52}

    payload = _build_request_payload(sample, anchor)

    assert payload["image_path"] == "/ocr-runtime/frames/V1/temp/V1_K1_12.jpg"

def test_process_candidates_to_canonical(tmp_path):
    # Setup candidate JSON directory
    cand1 = CandidateObject(
        detection_index=0, text="LUCKY",
        det_conf=0.999, rec_conf=0.9987, conf=0.9987,
        polygon=[[100.0, 50.0], [430.0, 50.0], [430.0, 130.0], [100.0, 130.0]],
        bbox=[100.0, 50.0, 430.0, 130.0], crop_strategy="min_area_rect_perspective"
    )

    cand2 = CandidateObject(
        detection_index=0, text="LUCKY",
        det_conf=0.999, rec_conf=0.9987, conf=0.9987,
        polygon=[[100.0, 50.0], [430.0, 50.0], [430.0, 130.0], [100.0, 130.0]],
        bbox=[100.0, 50.0, 430.0, 130.0], crop_strategy="min_area_rect_perspective"
    )

    cfile1 = OcrCandidateFile(
        schema_version="1.0", artifact_type="parseq_candidate_result",
        request_id="A:1", video_id="V1", shot_id=1, keyframe_id="K1",
        anchor_frame_idx=10, anchor_timestamp=1.0, source_frame_idx=9,
        source_timestamp=0.9, status="ok", candidates=[cand1]
    )

    cfile2 = OcrCandidateFile(
        schema_version="1.0", artifact_type="parseq_candidate_result",
        request_id="A:2", video_id="V1", shot_id=1, keyframe_id="K1",
        anchor_frame_idx=10, anchor_timestamp=1.0, source_frame_idx=10,
        source_timestamp=1.0, status="ok", candidates=[cand2]
    )

    cand_dir = tmp_path / "candidates"
    write_json_file(cfile1, cand_dir / "V1" / "K1" / "9.json")
    write_json_file(cfile2, cand_dir / "V1" / "K1" / "10.json")

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    dummy_kf = {"video_id": "V1", "shot_id": 1, "keyframe_id": "K1", "frame_idx": 10, "timestamp": 1.0}
    res = process_candidates_to_canonical(cand_dir, out_dir, [dummy_kf])
    assert res["canonical_text_count"] == 1
    assert res["json_file_count"] == 1

    # Verify Schema 2.0
    json_file = out_dir / "V1" / "K1.json"
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["keyframe_id"] == "K1"
    assert data["video_id"] == "V1"
    assert data["shot_id"] == 1
    assert data["anchor_frame_idx"] == 10
    assert data["anchor_timestamp"] == 1.0

    assert len(data["texts"]) == 1

    text_obj = data["texts"][0]

    assert text_obj["video_id"] == data["video_id"]
    assert text_obj["shot_id"] == data["shot_id"]

    assert text_obj["frame_idx"] == 10
    assert text_obj["timestamp"] == 1.0

    assert text_obj["text"] == "LUCKY"

    assert text_obj["conf"] == text_obj["rec_conf"]
    assert 0.0 <= text_obj["det_conf"] <= 1.0
    assert 0.0 <= text_obj["rec_conf"] <= 1.0

    assert len(text_obj["bbox"]) == 4
    assert text_obj["support_count"] == 2

    assert "polygon" not in text_obj
    assert "crop_strategy" not in text_obj
    assert "detection_index" not in text_obj


def test_process_candidates_to_canonical_writes_empty_detection_result(tmp_path):
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    keyframe = {
        "video_id": "V1",
        "shot_id": 1,
        "keyframe_id": "K1",
        "frame_idx": 10,
        "timestamp": 1.0,
    }

    result = process_candidates_to_canonical(
        candidate_dir, output_dir, [keyframe]
    )

    payload = json.loads((output_dir / "V1" / "K1.json").read_text(encoding="utf-8"))
    assert result == {
        "json_file_count": 1,
        "canonical_text_count": 0,
        "backend": "deepsolo+parseq",
    }
    assert payload["texts"] == []

import pytest
import math
import json
from pathlib import Path
from docker.ocr.worker.json_file_contract import (
    OcrRequest, DetectionObject, OcrDetectionFile,
    CandidateObject, OcrCandidateFile, read_json_file, write_json_file
)

def test_ocr_request_valid():
    req = OcrRequest(
        schema_version="1.0",
        artifact_type="ocr_frame_request",
        request_id="V1_K1:10",
        video_id="V1", shot_id=1, keyframe_id="V1_K1",
        anchor_frame_idx=11, anchor_timestamp=1.1,
        source_frame_idx=10, source_timestamp=1.0,
        image_path="/path.jpg"
    )
    assert req.schema_version == "1.0"

def test_ocr_request_invalid_schema():
    with pytest.raises(ValueError, match="schema_version"):
        OcrRequest(
            schema_version="2.0", artifact_type="ocr_frame_request",
            request_id="V1_K1:10", video_id="V1", shot_id=1, keyframe_id="V1_K1",
            anchor_frame_idx=11, anchor_timestamp=1.1,
            source_frame_idx=10, source_timestamp=1.0, image_path="/path.jpg"
        )

def test_write_read_json_file(tmp_path):
    req = OcrRequest(
        schema_version="1.0", artifact_type="ocr_frame_request",
        request_id="V1_K1:10", video_id="V1", shot_id=1, keyframe_id="V1_K1",
        anchor_frame_idx=11, anchor_timestamp=1.1,
        source_frame_idx=10, source_timestamp=1.0, image_path="/path.jpg"
    )
    file_path = tmp_path / "req.json"
    write_json_file(req, file_path)

    loaded = read_json_file(file_path, OcrRequest)
    assert loaded.request_id == req.request_id

def test_nan_rejection():
    with pytest.raises(ValueError, match="finite"):
        DetectionObject(
            detection_index=0, det_conf=0.9,
            polygon=[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, float('nan')]],
            bbox=[10.0, 10.0, 20.0, 20.0]
        )

def test_bbox_invalid():
    with pytest.raises(ValueError, match="x2 >= x1"):
        DetectionObject(
            detection_index=0, det_conf=0.9,
            polygon=[[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]],
            bbox=[20.0, 10.0, 10.0, 20.0] # x1 > x2
        )

def test_detection_file_zero_detections():
    det = OcrDetectionFile(
        schema_version="1.0", artifact_type="deepsolo_detection_result",
        request_id="req1", video_id="V1", shot_id=1, keyframe_id="K1",
        anchor_frame_idx=1, anchor_timestamp=0.1,
        source_frame_idx=1, source_timestamp=0.1, image_path="path",
        status="ok", detections=[]
    )
    assert len(det.detections) == 0

def test_candidate_file_read_write(tmp_path):
    cand = OcrCandidateFile(
        schema_version="1.0", artifact_type="parseq_candidate_result",
        request_id="req1", video_id="V1", shot_id=1, keyframe_id="K1",
        anchor_frame_idx=1, anchor_timestamp=0.1,
        source_frame_idx=1, source_timestamp=0.1,
        status="ok",
        candidates=[
            CandidateObject(
                detection_index=0, text="HELLO", det_conf=0.9, rec_conf=0.8, conf=0.8,
                polygon=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                bbox=[0.0, 0.0, 10.0, 10.0], crop_strategy="bbox"
            )
        ]
    )
    file_path = tmp_path / "cand.json"
    write_json_file(cand, file_path)

    loaded = read_json_file(file_path, OcrCandidateFile)
    assert len(loaded.candidates) == 1
    assert loaded.candidates[0].text == "HELLO"

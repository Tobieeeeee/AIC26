import json
import math
from dataclasses import dataclass, asdict
from typing import List, Any
from pathlib import Path

@dataclass
class OcrRequest:
    schema_version: str
    artifact_type: str
    request_id: str
    video_id: str
    shot_id: int
    keyframe_id: str
    anchor_frame_idx: int
    anchor_timestamp: float
    source_frame_idx: int
    source_timestamp: float
    image_path: str

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be 1.0")
        if self.artifact_type != "ocr_frame_request":
            raise ValueError("artifact_type must be ocr_frame_request")
        if not self.request_id or not self.keyframe_id or not self.video_id:
            raise ValueError("request_id, keyframe_id, and video_id are required")

@dataclass
class DetectionObject:
    detection_index: int
    det_conf: float
    polygon: List[List[float]]
    bbox: List[float]

    def __post_init__(self):
        if not (0.0 <= self.det_conf <= 1.0):
            raise ValueError("det_conf must be between 0.0 and 1.0")
        if len(self.bbox) != 4:
            raise ValueError("bbox must have exactly 4 values")
        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            raise ValueError("invalid bbox, x2 >= x1 and y2 >= y1 are required")
        for p in self.polygon:
            if len(p) != 2:
                raise ValueError("polygon point must have 2 values")
            if not math.isfinite(p[0]) or not math.isfinite(p[1]):
                raise ValueError("polygon coordinates must be finite")
        for b in self.bbox:
            if not math.isfinite(b):
                raise ValueError("bbox coordinates must be finite")

@dataclass
class OcrDetectionFile:
    schema_version: str
    artifact_type: str
    request_id: str
    video_id: str
    shot_id: int
    keyframe_id: str
    anchor_frame_idx: int
    anchor_timestamp: float
    source_frame_idx: int
    source_timestamp: float
    image_path: str
    status: str
    detections: List[DetectionObject]

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be 1.0")
        if self.artifact_type != "deepsolo_detection_result":
            raise ValueError("artifact_type must be deepsolo_detection_result")
        if not self.request_id:
            raise ValueError("request_id required")

        parsed_dets = []
        for d in self.detections:
            if isinstance(d, dict):
                parsed_dets.append(DetectionObject(**d))
            else:
                parsed_dets.append(d)
        self.detections = parsed_dets

@dataclass
class CandidateObject:
    detection_index: int
    text: str
    det_conf: float
    rec_conf: float
    conf: float
    polygon: List[List[float]]
    bbox: List[float]
    crop_strategy: str

    def __post_init__(self):
        if not (0.0 <= self.rec_conf <= 1.0):
            raise ValueError("rec_conf must be between 0.0 and 1.0")
        if self.conf != self.rec_conf:
            raise ValueError("conf must equal rec_conf")
        if not (0.0 <= self.det_conf <= 1.0):
            raise ValueError("det_conf must be between 0.0 and 1.0")
        x1, y1, x2, y2 = self.bbox
        if x1 > x2 or y1 > y2:
            raise ValueError("invalid bbox, x2 >= x1 and y2 >= y1 are required")

@dataclass
class OcrCandidateFile:
    schema_version: str
    artifact_type: str
    request_id: str
    video_id: str
    shot_id: int
    keyframe_id: str
    anchor_frame_idx: int
    anchor_timestamp: float
    source_frame_idx: int
    source_timestamp: float
    status: str
    candidates: List[CandidateObject]

    def __post_init__(self):
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be 1.0")
        if self.artifact_type != "parseq_candidate_result":
            raise ValueError("artifact_type must be parseq_candidate_result")

        parsed_cands = []
        for c in self.candidates:
            if isinstance(c, dict):
                parsed_cands.append(CandidateObject(**c))
            else:
                parsed_cands.append(c)
        self.candidates = parsed_cands

def read_json_file(path: Path, cls):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return cls(**data)

def write_json_file(item: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(item) if hasattr(item, "__dataclass_fields__") else item

    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        # allow_nan=False raises ValueError if NaNs are present during json dumping
        json.dump(d, f, ensure_ascii=False, allow_nan=False, indent=2)
    temp_path.replace(path)

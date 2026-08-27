import argparse
import pytest
import json
import numpy as np
import csv
from pathlib import Path
from mvr.enrichment.feature_extraction import extract_clip_features, extract_objects, sort_keyframes_deterministically
from scripts.run_feature_extraction import parse_embedding_providers

class FakeEncoder:
    def __init__(self):
        self.model_name = "FakeCLIP"
        self.pretrained = "fake_weights"

    def encode_images(self, paths):
        # Return a dummy vector of dim 10
        return np.ones((1, 10), dtype=np.float32)

class FakeDetector:
    def __init__(self):
        self.calls = 0
        self.model_name = "FakeRCNN"
        self.score_threshold = 0.5
        self.nms_threshold = 0.5
        self.model_family = "FakeRCNN-Family"
        self.nms_enabled = True
        self.nms_mode = "class_aware"
        self.nms_implementation = "built_in_faster_rcnn"

    def detect(self, path):
        self.calls += 1
        # Return 2 persons and 1 car
        return [
            {"label": "person", "class_id": 1, "score": 0.99, "bbox": [0.0, 0.0, 10.0, 10.0]},
            {"label": "person", "class_id": 1, "score": 0.88, "bbox": [10.0, 10.0, 20.0, 20.0]},
            {"label": "car", "class_id": 3, "score": 0.95, "bbox": [20.0, 20.0, 30.0, 30.0]}
        ]


def test_parse_embedding_provider_list():
    assert parse_embedding_providers("beit3, metaclip2,perception_encoder") == (
        "beit3",
        "metaclip2",
        "perception_encoder",
    )


@pytest.mark.parametrize("value", ["", "  ", "beit3,beit3"])
def test_parse_embedding_provider_list_rejects_empty_or_duplicate(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_embedding_providers(value)

def test_extract_clip_features(tmp_path):
    # Setup fake keyframe images and mapping
    keyframe_root = tmp_path / "keyframes"
    video1_dir = keyframe_root / "video1"
    video1_dir.mkdir(parents=True)
    (video1_dir / "frame1.jpg").touch()

    video2_dir = keyframe_root / "video2"
    video2_dir.mkdir(parents=True)
    (video2_dir / "frame1.jpg").touch()

    keyframes = [
        {
            "keyframe_id": "video1_S0001_K01",
            "video_id": "video1",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "10",
            "timestamp": "0.5",
            "image_path": "video1/frame1.jpg"
        },
        {
            "keyframe_id": "video2_S0001_K01",
            "video_id": "video2",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "5",
            "timestamp": "0.2",
            "image_path": "video2/frame1.jpg"
        }
    ]

    output_dir = tmp_path / "features"
    encoder = FakeEncoder()

    extract_clip_features("test_dataset", keyframes, encoder, output_dir, keyframe_root)

    clip_dir = output_dir / "clip"
    assert not (clip_dir / "clip_features.npy").exists()
    assert (clip_dir / "video1.npy").exists()
    assert (clip_dir / "video2.npy").exists()
    assert (clip_dir / "clip_feature_mapping.csv").exists()
    assert (clip_dir / "manifest.json").exists()

    arr1 = np.load(clip_dir / "video1.npy")
    assert arr1.shape == (1, 10)

    arr2 = np.load(clip_dir / "video2.npy")
    assert arr2.shape == (1, 10)

    manifest = json.loads((clip_dir / "manifest.json").read_text())
    assert manifest["vector_count_total"] == 2
    assert manifest["video_count"] == 2
    assert manifest["embedding_dimension"] == 10
    assert manifest["storage_format"] == "one_npy_per_video"

def test_extract_objects(tmp_path):
    keyframe_root = tmp_path / "keyframes"
    video_dir = keyframe_root / "video1"
    video_dir.mkdir(parents=True)
    img_path1 = video_dir / "frame1.jpg"
    img_path1.touch()
    img_path2 = video_dir / "frame2.jpg"
    img_path2.touch()

    keyframes = [
        {
            "keyframe_id": "video1_S0001_K01",
            "video_id": "video1",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "10",
            "timestamp": "0.5",
            "image_path": "video1/frame1.jpg"
        },
        {
            "keyframe_id": "video1_S0001_K02",
            "video_id": "video1",
            "shot_id": "1",
            "ordinal": "2",
            "frame_idx": "20",
            "timestamp": "1.0",
            "image_path": "video1/frame2.jpg"
        }
    ]

    output_dir = tmp_path / "features"
    detector = FakeDetector()
    mapping_source = tmp_path / "mapping.csv"

    extract_objects("test_dataset", keyframes, detector, output_dir, keyframe_root, mapping_source)

    obj_dir = output_dir / "objects"
    assert (obj_dir / "manifest.json").exists()

    # Check JSON 1
    json_path1 = obj_dir / "video1" / "video1_S0001_K01.json"
    assert json_path1.exists()
    data1 = json.loads(json_path1.read_text())
    assert data1["keyframe_id"] == "video1_S0001_K01"
    assert len(data1["objects"]) == 3
    assert data1["person_count"] == 2
    assert data1["car_count"] == 1
    assert data1["video_id"] == "video1"
    assert data1["shot_id"] == 1
    assert data1["frame_id"] == data1["frame_idx"] == 10
    assert all(item["keyframe_id"] == data1["keyframe_id"] for item in data1["objects"])
    assert all(item["video_id"] == data1["video_id"] for item in data1["objects"])
    assert all(item["frame_id"] == data1["frame_idx"] for item in data1["objects"])
    assert all("confidence" in item and "x1" in item for item in data1["objects"])

    staging1 = json.loads(
        (output_dir / "objects_staging" / "video1" / "video1_S0001_K01.json").read_text()
    )
    assert len(staging1) == 3
    assert staging1[0]["keyframe_id"] == "video1_S0001_K01"
    assert staging1[0]["frame_id"] == staging1[0]["frame_idx"] == 10

    # Check JSON 2
    json_path2 = obj_dir / "video1" / "video1_S0001_K02.json"
    assert json_path2.exists()
    data2 = json.loads(json_path2.read_text())
    assert data2["keyframe_id"] == "video1_S0001_K02"
    assert len(data2["objects"]) == 3
    assert data2["person_count"] == 2

    # Check Shot summary
    shot_summary_path = obj_dir / "shot_summary.json"
    assert shot_summary_path.exists()
    summary = json.loads(shot_summary_path.read_text())
    assert len(summary) == 1
    assert summary[0]["video_id"] == "video1"
    assert summary[0]["shot_id"] == 1
    assert summary[0]["keyframe_count"] == 2
    assert summary[0]["person_count_max"] == 2
    assert summary[0]["car_count_max"] == 1

    # Check Manifest
    manifest = json.loads((obj_dir / "manifest.json").read_text())
    assert manifest["json_file_count"] == 2
    assert manifest["keyframe_count"] == 2
    assert manifest["shot_count"] == 1
    assert manifest["model_family"] == "FakeRCNN-Family"
    assert manifest["nms_enabled"] is True
    assert manifest["nms_threshold"] == 0.5
    assert manifest["nms_mode"] == "class_aware"
    assert manifest["nms_implementation"] == "built_in_faster_rcnn"
    staging_manifest = json.loads(
        (output_dir / "objects_staging" / "manifest.json").read_text()
    )
    assert staging_manifest["schema"] == "objects"
    assert staging_manifest["frame_id_contract"] == (
        "frame_id == frame_idx from canonical mapping"
    )

    resumed_detector = FakeDetector()
    extract_objects(
        "test_dataset",
        keyframes,
        resumed_detector,
        output_dir,
        keyframe_root,
        mapping_source,
        resume=True,
    )
    assert resumed_detector.calls == 0

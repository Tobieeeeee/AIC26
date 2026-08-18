import pytest
import json
import numpy as np
import csv
from pathlib import Path
from mvr.enrichment.feature_extraction import extract_clip_features, extract_objects, sort_keyframes_deterministically

class FakeEncoder:
    def __init__(self):
        self.model_name = "FakeCLIP"
        self.pretrained = "fake_weights"
        
    def encode_images(self, paths):
        # Return a dummy vector of dim 10
        return np.ones((1, 10), dtype=np.float32)

class FakeDetector:
    def __init__(self):
        self.model_name = "FakeRCNN"
        self.score_threshold = 0.5
        
    def detect(self, path):
        # Return a dummy detection
        return [{"label": "person", "class_id": 1, "score": 0.99, "bbox": [0.0, 0.0, 10.0, 10.0]}]

def test_extract_clip_features(tmp_path):
    # Setup fake keyframe images and mapping
    keyframe_root = tmp_path / "keyframes"
    video_dir = keyframe_root / "video1"
    video_dir.mkdir(parents=True)
    img_path = video_dir / "frame1.jpg"
    img_path.touch()
    
    keyframes = [
        {
            "keyframe_id": "video1_S0001_K01",
            "video_id": "video1",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "10",
            "timestamp": "0.5",
            "image_path": "video1/frame1.jpg"
        }
    ]
    
    output_dir = tmp_path / "features"
    encoder = FakeEncoder()
    
    extract_clip_features("test_dataset", keyframes, encoder, output_dir, keyframe_root)
    
    clip_dir = output_dir / "clip"
    assert (clip_dir / "clip_features.npy").exists()
    assert (clip_dir / "clip_feature_mapping.csv").exists()
    assert (clip_dir / "manifest.json").exists()
    
    arr = np.load(clip_dir / "clip_features.npy")
    assert arr.shape == (1, 10)
    
    manifest = json.loads((clip_dir / "manifest.json").read_text())
    assert manifest["vector_count"] == 1
    assert manifest["embedding_dimension"] == 10

def test_extract_objects(tmp_path):
    keyframe_root = tmp_path / "keyframes"
    video_dir = keyframe_root / "video1"
    video_dir.mkdir(parents=True)
    img_path = video_dir / "frame1.jpg"
    img_path.touch()
    
    keyframes = [
        {
            "keyframe_id": "video1_S0001_K01",
            "video_id": "video1",
            "shot_id": "1",
            "ordinal": "1",
            "frame_idx": "10",
            "timestamp": "0.5",
            "image_path": "video1/frame1.jpg"
        }
    ]
    
    output_dir = tmp_path / "features"
    detector = FakeDetector()
    mapping_source = tmp_path / "mapping.csv"
    
    extract_objects("test_dataset", keyframes, detector, output_dir, keyframe_root, mapping_source)
    
    obj_dir = output_dir / "objects"
    assert (obj_dir / "manifest.json").exists()
    json_path = obj_dir / "video1" / "video1_S0001_K01.json"
    assert json_path.exists()
    
    data = json.loads(json_path.read_text())
    assert data["keyframe_id"] == "video1_S0001_K01"
    assert len(data["objects"]) == 1
    assert data["objects"][0]["label"] == "person"

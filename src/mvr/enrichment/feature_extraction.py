import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from mvr.enrichment.embeddings.openclip import OpenClipEncoder
from mvr.enrichment.objects.faster_rcnn import FasterRCNN

def sort_keyframes_deterministically(mapping_csv: Path) -> list[dict]:
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        keyframes = list(reader)
        
    # video_id ASC, shot_id ASC, ordinal ASC
    keyframes.sort(key=lambda k: (
        k["video_id"],
        int(k["shot_id"]),
        int(k["ordinal"])
    ))
    return keyframes

def extract_clip_features(
    dataset_name: str,
    keyframes: list[dict],
    encoder: OpenClipEncoder,
    output_dir: Path,
    keyframe_root: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = output_dir / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    
    if (clip_dir / "clip_features.npy").exists():
        raise FileExistsError("clip_features.npy already exists. Delete to overwrite.")
        
    features = []
    mapping_rows = []
    
    for i, frame in enumerate(tqdm(keyframes, desc="CLIP Features")):
        img_path = keyframe_root / frame["image_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Keyframe image not found: {img_path}")
            
        emb = encoder.encode_images([str(img_path)])
        if np.isnan(emb).any() or np.isinf(emb).any():
            raise ValueError(f"NaN or Inf encountered in embedding for {img_path}")
            
        features.append(emb[0])
        mapping_rows.append({
            "feature_index": i,
            "keyframe_id": frame["keyframe_id"],
            "video_id": frame["video_id"],
            "frame_idx": frame["frame_idx"],
            "timestamp": frame["timestamp"],
            "image_path": frame["image_path"],
        })
        
    feature_matrix = np.array(features, dtype=np.float32)
    
    # Save the npy array
    np.save(clip_dir / "clip_features.npy", feature_matrix)
    
    # Save mapping
    mapping_cols = ["feature_index", "keyframe_id", "video_id", "frame_idx", "timestamp", "image_path"]
    with (clip_dir / "clip_feature_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mapping_cols)
        writer.writeheader()
        writer.writerows(mapping_rows)
        
    # Save manifest
    manifest = {
        "schema_version": "1.0",
        "feature_type": "clip_image",
        "model": encoder.model_name,
        "pretrained": encoder.pretrained,
        "embedding_dimension": feature_matrix.shape[1],
        "vector_count": feature_matrix.shape[0],
        "dtype": "float32",
        "normalization": "l2",
        "ordering": "clip_feature_mapping.csv:feature_index",
        "mapping_file": "clip_feature_mapping.csv"
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_objects(
    dataset_name: str,
    keyframes: list[dict],
    detector: FasterRCNN,
    output_dir: Path,
    keyframe_root: Path,
    mapping_source: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = output_dir / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)
    
    json_count = 0
    for frame in tqdm(keyframes, desc="Faster R-CNN Objects"):
        img_path = keyframe_root / frame["image_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Keyframe image not found: {img_path}")
            
        detections = detector.detect(img_path)
        
        video_dir = obj_dir / frame["video_id"]
        video_dir.mkdir(parents=True, exist_ok=True)
        
        target_json = video_dir / f"{frame['keyframe_id']}.json"
        
        payload = {
            "keyframe_id": frame["keyframe_id"],
            "video_id": frame["video_id"],
            "frame_idx": int(frame["frame_idx"]),
            "timestamp": float(frame["timestamp"]),
            "image_path": frame["image_path"],
            "objects": detections
        }
        target_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        json_count += 1
        
    manifest = {
        "schema_version": "1.0",
        "feature_type": "object_detection",
        "storage_format": "one_json_per_keyframe",
        "model_family": "Faster R-CNN",
        "model": detector.model_name,
        "weights": "DEFAULT",
        "score_threshold": detector.score_threshold,
        "coordinate_format": "xyxy",
        "keyframe_count": len(keyframes),
        "json_file_count": json_count,
        "mapping_source": str(mapping_source)
    }
    (obj_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

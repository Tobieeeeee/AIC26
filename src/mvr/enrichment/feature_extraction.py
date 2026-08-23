import csv
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from mvr.enrichment.embeddings.openclip import OpenClipEncoder
from mvr.enrichment.objects.rf_detr import RfDetr

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

from collections import defaultdict
import uuid

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

    # Group by video_id
    video_groups = defaultdict(list)
    for frame in keyframes:
        video_groups[frame["video_id"]].append(frame)

    mapping_rows = []
    vector_count_total = 0
    embedding_dimension = None

    for video_id, group_frames in tqdm(video_groups.items(), desc="CLIP Features (per video)"):
        # Deterministic chronological sort: frame_idx ASC, keyframe_id ASC
        group_frames.sort(key=lambda k: (int(k["frame_idx"]), k["keyframe_id"]))
        
        target_npy = clip_dir / f"{video_id}.npy"
        if target_npy.exists():
            raise FileExistsError(f"{target_npy.name} already exists. Delete to overwrite.")
            
        features = []
        for i, frame in enumerate(group_frames):
            img_path = keyframe_root / frame["image_path"]
            if not img_path.exists():
                raise FileNotFoundError(f"Keyframe image not found: {img_path}")

            emb = encoder.encode_images([str(img_path)])
            if np.isnan(emb).any() or np.isinf(emb).any():
                raise ValueError(f"NaN or Inf encountered in embedding for {img_path}")

            features.append(emb[0])
            mapping_rows.append({
                "video_id": video_id,
                "feature_file": f"{video_id}.npy",
                "feature_index": i,
                "keyframe_id": frame["keyframe_id"],
                "shot_id": int(frame["shot_id"]),
                "frame_idx": int(frame["frame_idx"]),
                "timestamp": float(frame["timestamp"]),
                "image_path": frame["image_path"],
            })

        feature_matrix = np.array(features, dtype=np.float32)
        if embedding_dimension is None:
            embedding_dimension = feature_matrix.shape[1]
        
        # Atomic write
        temp_npy = clip_dir / f".tmp_{uuid.uuid4().hex}_{video_id}.npy"
        np.save(temp_npy, feature_matrix)
        temp_npy.replace(target_npy)
        
        vector_count_total += feature_matrix.shape[0]

    # Sort mapping rows deterministically: video_id ASC, feature_index ASC
    mapping_rows.sort(key=lambda x: (x["video_id"], x["feature_index"]))

    # Save mapping
    mapping_cols = ["video_id", "feature_file", "feature_index", "keyframe_id", "shot_id", "frame_idx", "timestamp", "image_path"]
    with (clip_dir / "clip_feature_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=mapping_cols)
        writer.writeheader()
        writer.writerows(mapping_rows)

    # Save manifest
    manifest = {
        "schema_version": "1.1",
        "feature_type": "clip_image",
        "storage_format": "one_npy_per_video",
        "model": encoder.model_name,
        "pretrained": encoder.pretrained,
        "embedding_dimension": embedding_dimension if embedding_dimension is not None else 0,
        "dtype": "float32",
        "normalization": "l2",
        "video_count": len(video_groups),
        "vector_count_total": vector_count_total,
        "mapping_file": "clip_feature_mapping.csv",
        "ordering_within_video": "frame_idx_asc_then_keyframe_id_asc"
    }
    (clip_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

from collections import defaultdict

def extract_objects(
    dataset_name: str,
    keyframes: list[dict],
    detector: RfDetr,
    output_dir: Path,
    keyframe_root: Path,
    mapping_source: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = output_dir / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)

    json_count = 0
    shot_stats = defaultdict(lambda: {"keyframe_count": 0, "person_count_max": 0, "car_count_max": 0})

    for frame in tqdm(keyframes, desc=f"{detector.model_family} Objects"):
        img_path = keyframe_root / frame["image_path"]
        if not img_path.exists():
            raise FileNotFoundError(f"Keyframe image not found: {img_path}")

        detections = detector.detect(img_path)

        person_count = sum(1 for d in detections if d["label"] == "person")
        car_count = sum(1 for d in detections if d["label"] == "car")

        video_dir = obj_dir / frame["video_id"]
        video_dir.mkdir(parents=True, exist_ok=True)

        target_json = video_dir / f"{frame['keyframe_id']}.json"

        payload = {
            "keyframe_id": frame["keyframe_id"],
            "video_id": frame["video_id"],
            "shot_id": int(frame["shot_id"]),
            "frame_idx": int(frame["frame_idx"]),
            "timestamp": float(frame["timestamp"]),
            "image_path": frame["image_path"],
            "person_count": person_count,
            "car_count": car_count,
            "objects": detections
        }
        target_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        json_count += 1

        shot_key = (frame["video_id"], int(frame["shot_id"]))
        shot_stats[shot_key]["keyframe_count"] += 1
        shot_stats[shot_key]["person_count_max"] = max(shot_stats[shot_key]["person_count_max"], person_count)
        shot_stats[shot_key]["car_count_max"] = max(shot_stats[shot_key]["car_count_max"], car_count)

    shot_summary = []
    for (video_id, shot_id), stats in sorted(shot_stats.items(), key=lambda x: (x[0][0], x[0][1])):
        shot_summary.append({
            "video_id": video_id,
            "shot_id": shot_id,
            "keyframe_count": stats["keyframe_count"],
            "person_count_max": stats["person_count_max"],
            "car_count_max": stats["car_count_max"]
        })

    (obj_dir / "shot_summary.json").write_text(json.dumps(shot_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "feature_type": "object_detection",
        "storage_format": "one_json_per_keyframe",
        "model_family": detector.model_family,
        "model": detector.model_name,
        "weights": "DEFAULT",
        "score_threshold": detector.score_threshold,
        "nms_enabled": detector.nms_enabled,
        "nms_threshold": detector.nms_threshold,
        "nms_mode": detector.nms_mode,
        "nms_implementation": detector.nms_implementation,
        "coordinate_format": "xyxy",
        "keyframe_count": len(keyframes),
        "json_file_count": json_count,
        "shot_summary_file": "shot_summary.json",
        "shot_count": len(shot_summary),
        "mapping_source": str(mapping_source)
    }
    (obj_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

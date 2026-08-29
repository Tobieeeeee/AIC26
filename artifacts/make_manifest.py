from pathlib import Path
import json
import sys

sys.path.insert(0, "src")
from mvr.enrichment.ocr.docker_ocr import process_candidates_to_canonical

temp_dir = Path("artifacts/full_ocr_temp")
output_dir = Path("artifacts/full_ocr_output")
output_dir.mkdir(parents=True, exist_ok=True)

from mvr.enrichment.ocr.scene_text import load_keyframes
keyframes = load_keyframes(Path("artifacts/real_task04_test/ingest/keyframe_mapping.csv"))

res = process_candidates_to_canonical(temp_dir / "candidates", output_dir, keyframes)
print("Candidates processed:", res)

manifest = {
    "schema_version": "2.0",
    "feature_type": "scene_text",
    "storage_format": "one_json_per_keyframe",
    "ocr_backend": res["backend"],
    "detector": "deepsolo",
    "recognizer": "parseq_vi",
    "ocr_conf_threshold": 0.0,
    "temporal_offsets": [-1, 0, 1],
    "text_similarity": "normalized_levenshtein",
    "text_similarity_threshold": 0.80,
    "spatial_similarity": "bbox_iou",
    "bbox_iou_threshold": 0.30,
    "bbox_format": "xyxy_pixels",
    "representative_selection": "highest_conf",
    "keyframe_count": len(keyframes),
    "json_file_count": res["json_file_count"],
    "canonical_text_count": res["canonical_text_count"],
    "mapping_source": "artifacts/real_task04_test/ingest/keyframe_mapping.csv"
}

with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

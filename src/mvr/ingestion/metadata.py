import csv
import json
from pathlib import Path

from .models import Keyframe, Shot, VideoInfo


def write_metadata(output_root: Path, videos: list[VideoInfo], shots: list[Shot], keyframes: list[Keyframe]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "frame_indexing": "zero_based",
        "shot_end_frame": "inclusive",
        "shots": [shot.to_dict() for shot in shots],
        "keyframes": [frame.to_dict() for frame in keyframes],
    }
    (output_root / "shots.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    columns = ["keyframe_id", "video_id", "shot_id", "ordinal", "frame_idx", "timestamp", "image_path"]
    with (output_root / "keyframe_mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(frame.to_dict() for frame in keyframes)

    video_columns = ["video_id", "video_path", "duration_seconds", "fps", "frame_count", "width", "height"]
    with (output_root / "video_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=video_columns)
        writer.writeheader()
        for v in videos:
            writer.writerow({
                "video_id": v.video_id,
                "video_path": str(v.path),
                "duration_seconds": round(v.duration, 6),
                "fps": round(v.fps, 6),
                "frame_count": v.frame_count,
                "width": v.width,
                "height": v.height,
            })
    dataset_name = output_root.name.replace("ingest_", "") if output_root.name.startswith("ingest_") else output_root.name
    manifest = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "video_count": len(videos),
        "shot_count": len(shots),
        "keyframe_count": len(keyframes)
    }
    (output_root / "ingest_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")



def write_benchmark(output_root: Path, rows: list[dict]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    summary = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        duration = sum(float(row["duration_seconds"]) for row in group)
        detect_time = sum(float(row["detection_seconds"]) for row in group)
        total_shots = sum(int(row["shot_count"]) for row in group)
        summary.append({
            "method": method, "video_count": len(group), "total_shots": total_shots,
            "mean_shots_per_video": round(total_shots / len(group), 6),
            "model_load_seconds": round(float(group[0]["model_load_seconds"]), 6),
            "total_detection_seconds": round(detect_time, 6),
            "mean_detection_seconds": round(detect_time / len(group), 6),
            "aggregate_realtime_factor": round(detect_time / duration, 6),
        })
    (output_root / "benchmark.json").write_text(
        json.dumps({"results": rows, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_root / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_root / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

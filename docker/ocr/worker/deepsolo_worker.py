import sys
import os
import time
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

REPOSITORY_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(REPOSITORY_SRC))
from mvr.enrichment.ocr.model_integrity import ModelIntegrityError, verify_model_file
from mvr.enrichment.ocr.worker_timing import (
    format_stage_timing,
    should_report_progress,
)
import cv2
import json
import argparse
from pathlib import Path

# Add the worker directory to sys.path so we can import local files
sys.path.insert(0, str(Path(__file__).resolve().parent))

from json_file_contract import read_json_file, write_json_file, OcrRequest, DetectionObject, OcrDetectionFile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    requests_dir = Path(args.requests_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)
    config_path = args.config

    is_dummy = False
    allow_mock = os.environ.get("MVR_OCR_ALLOW_MOCK") == "1"

    if weights_path.exists() and weights_path.is_file():
        if weights_path.stat().st_size == 0:
            print(f"ERROR: Weights file is empty: {weights_path}", file=sys.stderr)
            sys.exit(1)

        with open(weights_path, "rb") as f:
            header = f.read(10)
            if b"dummy" in header or b"<!DOCTYPE" in header or b"<html>" in header:
                if not allow_mock:
                    print("ERROR: Dummy/HTML weights detected but MVR_OCR_ALLOW_MOCK is not set to 1.", file=sys.stderr)
                    sys.exit(1)
                is_dummy = True
    else:
        print(f"ERROR: Weights file not found: {weights_path}", file=sys.stderr)
        sys.exit(1)

    if not is_dummy:
        try:
            verify_model_file(
                weights_path,
                args.expected_sha256,
                label="deepsolo runtime preflight",
            )
        except ModelIntegrityError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    predictor = None
    if not is_dummy:
        print("Loading real DeepSolo weights...")
        try:
            # Need to append DeepSolo to path for imports to work correctly
            sys.path.insert(0, '/opt/DeepSolo/DeepSolo')
            import detectron2
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            import adet._C
            from adet.config import get_cfg as get_adet_cfg

            cfg = get_adet_cfg()
            cfg.merge_from_file(config_path)
            cfg.MODEL.WEIGHTS = str(weights_path)
            cfg.MODEL.DEVICE = "cuda"
            predictor = DefaultPredictor(cfg)
            print("DeepSolo initialized successfully.")
        except Exception as e:
            print(f"ERROR: DeepSolo initialization failed: {e}", file=sys.stderr)
            sys.exit(1)

    started = time.perf_counter()
    processed = 0
    skipped = 0
    for req_file in sorted(requests_dir.rglob("*.json")):
        req = read_json_file(req_file, OcrRequest)
        out_file = output_dir / req_file.relative_to(requests_dir)
        if args.resume and out_file.is_file():
            try:
                existing = read_json_file(out_file, OcrDetectionFile)
                if (
                    existing.status == "ok"
                    and existing.request_id == req.request_id
                    and existing.video_id == req.video_id
                    and existing.keyframe_id == req.keyframe_id
                    and existing.source_frame_idx == req.source_frame_idx
                    and existing.image_path == req.image_path
                ):
                    skipped += 1
                    continue
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        detections = []
        if is_dummy:
            det = DetectionObject(
                detection_index=0,
                det_conf=0.99,
                polygon=[[10.0, 10.0], [100.0, 10.0], [100.0, 50.0], [10.0, 50.0]],
                bbox=[10.0, 10.0, 100.0, 50.0]
            )
            detections.append(det)
        else:
            if predictor is None:
                continue

            try:
                img_path = Path(req.image_path)
                if not img_path.exists():
                    print(f"Warning: Image not found {img_path}", file=sys.stderr)
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Failed to read image {img_path}", file=sys.stderr)
                    continue

                outputs = predictor(img)
                processed += 1
                if should_report_progress(processed):
                    print(
                        format_stage_timing(
                            "deepsolo",
                            processed=processed,
                            skipped=skipped,
                            seconds=time.perf_counter() - started,
                            final=False,
                        ),
                        flush=True,
                    )
                instances = outputs["instances"]

                # DeepSolo returns polygons, pred_boxes, and scores
                if len(instances) > 0 and hasattr(instances, "bd"):
                    polygons = instances.bd.cpu().numpy() # [N, 25, 4]
                    scores = instances.scores.cpu().numpy()

                    for i in range(len(instances)):
                        # Reshape [25, 4] -> [50, 2]
                        poly_flat = polygons[i].reshape(-1, 2)
                        poly_points = poly_flat.tolist()

                        box = [
                            float(np.min(poly_flat[:, 0])),
                            float(np.min(poly_flat[:, 1])),
                            float(np.max(poly_flat[:, 0])),
                            float(np.max(poly_flat[:, 1]))
                        ]
                        score = float(scores[i])

                        det = DetectionObject(
                            detection_index=i,
                            det_conf=score,
                            polygon=poly_points,
                            bbox=box
                        )
                        detections.append(det)

            except Exception as e:
                print(f"ERROR processing {req_file}: {e}", file=sys.stderr)

        det_result = OcrDetectionFile(
            schema_version="1.0",
            artifact_type="deepsolo_detection_result",
            request_id=req.request_id,
            video_id=req.video_id,
            shot_id=req.shot_id,
            keyframe_id=req.keyframe_id,
            anchor_frame_idx=req.anchor_frame_idx,
            anchor_timestamp=req.anchor_timestamp,
            source_frame_idx=req.source_frame_idx,
            source_timestamp=req.source_timestamp,
            image_path=req.image_path,
            status="ok",
            detections=detections
        )
        write_json_file(det_result, out_file)

    print(
        format_stage_timing(
            "deepsolo",
            processed=processed,
            skipped=skipped,
            seconds=time.perf_counter() - started,
        ),
        flush=True,
    )
    print("DeepSolo worker finished.")

if __name__ == "__main__":
    main()

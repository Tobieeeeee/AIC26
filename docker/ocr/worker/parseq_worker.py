import sys
import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

REPOSITORY_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(REPOSITORY_SRC))
from mvr.enrichment.ocr.model_integrity import ModelIntegrityError, verify_model_file
import cv2
import json
import argparse
from pathlib import Path

# Add the worker directory to sys.path so we can import local files
sys.path.insert(0, str(Path(__file__).resolve().parent))

from json_file_contract import read_json_file, write_json_file, OcrDetectionFile, CandidateObject, OcrCandidateFile
from rectify import rectify_crop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()

    detections_dir = Path(args.detections_dir)
    output_dir = Path(args.output_dir)
    weights_path = Path(args.weights)

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
                label="parseq_vi runtime preflight",
            )
        except ModelIntegrityError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    parseq_model = None
    img_transform = None
    if not is_dummy:
        print("Loading real PARSeq weights...")
        try:
            sys.path.insert(0, '/opt/vietnamese-scenetext-detection-recognition')
            import torch
            from src.parseq.strhub.data.module import SceneTextDataModule
            from src.parseq.strhub.models.utils import load_from_checkpoint
            from src.parseq.parseq_predict import predict_parseq

            device = "cuda" if torch.cuda.is_available() else "cpu"
            parseq_model = load_from_checkpoint(str(weights_path)).eval().to(device)
            img_transform = SceneTextDataModule.get_transform(parseq_model.hparams.img_size)
            print("PARSeq initialized successfully.")
        except Exception as e:
            print(f"ERROR: PARSeq initialization failed: {e}", file=sys.stderr)
            sys.exit(1)

    for det_file in sorted(detections_dir.rglob("*.json")):
        det_result = read_json_file(det_file, OcrDetectionFile)

        candidates = []
        for det in det_result.detections:
            if is_dummy:
                cand = CandidateObject(
                    detection_index=det.detection_index,
                    text="MOCK_VIỆT_NAM",
                    det_conf=det.det_conf,
                    rec_conf=0.95,
                    conf=0.95,
                    polygon=det.polygon,
                    bbox=det.bbox,
                    crop_strategy="min_area_rect_perspective"
                )
                candidates.append(cand)
            else:
                if parseq_model is None:
                    continue

                try:
                    img_path = Path(det_result.image_path)
                    if not img_path.exists():
                        print(f"Warning: Image not found {img_path}", file=sys.stderr)
                        continue

                    # Load image
                    image_array = cv2.imread(str(img_path))
                    if image_array is None:
                        print(f"Warning: Failed to load image {img_path}", file=sys.stderr)
                        continue

                    # Rectify crop using the polygon
                    cropped, crop_strategy = rectify_crop(image_array, det.polygon, det.bbox)
                    if cropped.size == 0:
                        continue

                    # Convert to PIL RGB for PARSeq
                    img_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

                    device = "cuda" if next(parseq_model.parameters()).is_cuda else "cpu"
                    (pred, prob) = predict_parseq(parseq=parseq_model, img_transform=img_transform, image=img_pil, device=device)

                    cand = CandidateObject(
                        detection_index=det.detection_index,
                        text=pred[0],
                        det_conf=det.det_conf,
                        rec_conf=float(prob),
                        conf=float(prob),
                        polygon=det.polygon,
                        bbox=det.bbox,
                        crop_strategy=crop_strategy
                    )
                    candidates.append(cand)
                except Exception as e:
                    print(f"ERROR processing detection {det.detection_index} in {det_file}: {e}", file=sys.stderr)

        out_file = output_dir / det_file.relative_to(detections_dir)
        cand_result = OcrCandidateFile(
            schema_version="1.0",
            artifact_type="parseq_candidate_result",
            request_id=det_result.request_id,
            video_id=det_result.video_id,
            shot_id=det_result.shot_id,
            keyframe_id=det_result.keyframe_id,
            anchor_frame_idx=det_result.anchor_frame_idx,
            anchor_timestamp=det_result.anchor_timestamp,
            source_frame_idx=det_result.source_frame_idx,
            source_timestamp=det_result.source_timestamp,
            status="ok",
            candidates=candidates
        )
        write_json_file(cand_result, out_file)

    print("PARSeq worker finished.")

if __name__ == "__main__":
    main()

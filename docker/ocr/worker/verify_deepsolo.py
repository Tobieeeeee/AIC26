import sys
import os
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '/opt/DeepSolo/DeepSolo')
sys.path.insert(0, '/workspace/src')
from mvr.enrichment.ocr.model_integrity import ModelIntegrityError, verify_model_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--image", required=False, default="")
    args = parser.parse_args()

    try:
        verify_model_file(
            Path(args.weights),
            args.expected_sha256,
            label="deepsolo smoke preflight",
        )
    except ModelIntegrityError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("=== DeepSolo Docker Verification ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
    else:
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    try:
        import detectron2
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        print(f"Detectron2 imported successfully: {detectron2.__version__}")
    except ImportError as e:
        print(f"ERROR: Detectron2 import failed: {e}")
        sys.exit(1)

    try:
        import adet._C
        from adet.config import get_cfg as get_adet_cfg
        print("adet._C imported successfully.")
    except ImportError as e:
        print(f"ERROR: adet._C import failed: {e}")
        sys.exit(1)

    print(f"Loading config from: {args.config}")
    cfg = get_adet_cfg()
    cfg.merge_from_file(args.config)

    print(f"Loading weights from: {args.weights}")
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda"

    try:
        predictor = DefaultPredictor(cfg)
        print("Checkpoint loaded and predictor initialized successfully.")
    except Exception as e:
        print(f"ERROR: Checkpoint load failed: {e}")
        sys.exit(1)

    # Run real GPU inference
    if args.image and os.path.exists(args.image):
        print(f"Loading image from {args.image}")
        img = cv2.imread(args.image)
    else:
        print("No image provided. Using a 800x600 random noisy image for GPU inference...")
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    try:
        print("Running inference...")
        outputs = predictor(img)
        print("Inference completed successfully.")

        # Verify output formats
        instances = outputs["instances"]
        print(f"Detected {len(instances)} instances.")
        if len(instances) > 0:
            if hasattr(instances, "polygons"):
                print("Polygons output exists.")
            if hasattr(instances, "pred_boxes"):
                print("Bounding boxes output exists.")
    except Exception as e:
        print(f"ERROR: GPU inference failed: {e}")
        sys.exit(1)

    print("DeepSolo full verification PASS.")

if __name__ == "__main__":
    main()

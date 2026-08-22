import sys
import argparse
import torch
import cv2
from PIL import Image
from pathlib import Path

sys.path.insert(0, '/opt/vietnamese-scenetext-detection-recognition')
sys.path.insert(0, '/workspace/src')
from mvr.enrichment.ocr.model_integrity import ModelIntegrityError, verify_model_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--image", required=False, default="")
    args = parser.parse_args()

    try:
        verify_model_file(
            Path(args.weights),
            args.expected_sha256,
            label="parseq_vi smoke preflight",
        )
    except ModelIntegrityError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("=== PARSeq-Vietnamese Docker Verification ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if not cuda_avail:
        print("ERROR: CUDA is not available!")
        sys.exit(1)

    try:
        from src.parseq.strhub.data.module import SceneTextDataModule
        from src.parseq.strhub.models.utils import load_from_checkpoint
        from src.parseq.parseq_predict import predict_parseq
    except ImportError as e:
        print(f"Import failed: {e}")
        sys.exit(1)

    print(f"Loading weights from: {args.weights}")
    try:
        parseq = load_from_checkpoint(args.weights).eval().to("cuda")
        img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
        print("Checkpoint loaded and model initialized successfully.")
    except Exception as e:
        print(f"ERROR: Checkpoint load failed: {e}")
        sys.exit(1)

    if args.image:
        print(f"Loading image from {args.image}")
        img = Image.open(args.image).convert("RGB")
    else:
        print("No image provided. Using a 200x50 random noisy image for GPU inference...")
        import numpy as np
        img_np = np.random.randint(0, 255, (50, 200, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)

    try:
        print("Running inference...")
        (pred, prob) = predict_parseq(parseq=parseq, img_transform=img_transform, image=img, device='cuda')
        print(f"Inference completed successfully. Prediction: {pred[0]}, Prob: {prob:.4f}")
    except Exception as e:
        print(f"ERROR: GPU inference failed: {e}")
        sys.exit(1)

    print("PARSeq full verification PASS.")

if __name__ == "__main__":
    main()

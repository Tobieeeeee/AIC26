import argparse
from pathlib import Path
from mvr.enrichment.feature_extraction import (
    sort_keyframes_deterministically,
    extract_clip_features,
    extract_objects
)
from mvr.enrichment.embeddings.openclip import OpenClipEncoder
from mvr.enrichment.objects.faster_rcnn import FasterRCNN


def parse_embedding_providers(value: str) -> tuple[str, ...]:
    providers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not providers:
        raise argparse.ArgumentTypeError("at least one embedding provider is required")
    if len(set(providers)) != len(providers):
        raise argparse.ArgumentTypeError("embedding providers must be unique")
    return providers

def main():
    parser = argparse.ArgumentParser(description="Run Offline Feature Extraction (Stage 2)")
    parser.add_argument("--mapping-csv", type=Path, required=True, help="Path to keyframe_mapping.csv from Stage 1")
    parser.add_argument("--keyframe-root", type=Path, required=True, help="Root directory containing keyframe images (should be parent of paths in mapping.csv)")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for feature artifacts")
    
    parser.add_argument("--clip-model", type=str, default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k", help="CLIP pretrained weights")
    parser.add_argument(
        "--embedding-providers",
        type=parse_embedding_providers,
        default=("clip",),
        help="Comma-separated provider IDs; provider implementations are registered by later tasks",
    )
    parser.add_argument("--rcnn-model", type=str, default="fasterrcnn_resnet50_fpn", help="Faster R-CNN model")
    parser.add_argument("--rcnn-threshold", type=float, default=0.5, help="Faster R-CNN score threshold")
    parser.add_argument("--rcnn-nms-threshold", type=float, default=0.5, help="Faster R-CNN NMS threshold")
    
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")

    args = parser.parse_args()

    unavailable = [provider for provider in args.embedding_providers if provider != "clip"]
    if unavailable:
        parser.error(
            "embedding provider implementation is not available yet: "
            + ", ".join(unavailable)
        )

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")
        
    print(f"Loading mapping from {args.mapping_csv}")
    keyframes = sort_keyframes_deterministically(args.mapping_csv)
    print(f"Loaded {len(keyframes)} keyframes.")
    
    dataset_name = args.output.name.replace("features_", "") if args.output.name.startswith("features_") else args.output.name
    
    # 1. CLIP
    print(f"Initializing OpenClipEncoder ({args.clip_model})...")
    clip_encoder = OpenClipEncoder(model_name=args.clip_model, pretrained=args.clip_pretrained, device=args.device)
    extract_clip_features(dataset_name, keyframes, clip_encoder, args.output, args.keyframe_root)
    
    # 2. Faster R-CNN
    print(f"Initializing Faster R-CNN ({args.rcnn_model})...")
    rcnn_detector = FasterRCNN(model_name=args.rcnn_model, device=args.device, score_threshold=args.rcnn_threshold, nms_threshold=args.rcnn_nms_threshold)
    extract_objects(dataset_name, keyframes, rcnn_detector, args.output, args.keyframe_root, args.mapping_csv)
    
    print(f"Feature extraction completed. Outputs saved to {args.output}")

if __name__ == "__main__":
    main()

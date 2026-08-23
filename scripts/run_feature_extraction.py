import argparse
from pathlib import Path
from mvr.enrichment.feature_extraction import (
    sort_keyframes_deterministically,
    extract_clip_features,
    extract_objects
)
from mvr.enrichment.embeddings.openclip import OpenClipEncoder
from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder
from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.perception_encoder import PerceptionEncoderImageEncoder
from mvr.enrichment.embeddings.artifacts import verify_file_sha256
from mvr.enrichment.embeddings.offline import (
    extract_embedding_artifacts,
    run_selected_embedding_providers,
)
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.objects.rf_detr import RfDetr


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
        help="Comma-separated provider IDs (clip, beit3, metaclip2, perception_encoder)",
    )
    parser.add_argument(
        "--provider-specs",
        type=Path,
        default=Path("configs/embedding_providers.json"),
        help="Locked provider specification JSON",
    )
    parser.add_argument(
        "--beit3-checkpoint",
        type=Path,
        help="Local locked BEiT-3 checkpoint; never downloaded implicitly",
    )
    parser.add_argument(
        "--metaclip2-model",
        type=str,
        help="Pinned local/cache model path; defaults to locked Hugging Face ID",
    )
    parser.add_argument(
        "--allow-model-downloads",
        action="store_true",
        help="Explicitly allow pinned provider downloads (disabled by default)",
    )
    parser.add_argument(
        "--perception-encoder-checkpoint",
        type=Path,
        help="Local locked PE-Core-L14-336 checkpoint; never downloaded implicitly",
    )
    parser.add_argument("--detector-model", type=str, default="rfdetr_medium", help="RF-DETR model variant (rfdetr_nano/small/medium/large)")
    parser.add_argument("--detector-threshold", type=float, default=0.5, help="RF-DETR score threshold")
    parser.add_argument("--detector-nms-threshold", type=float, default=0.5, help="Unused: RF-DETR is NMS-free; kept for interface parity")
    
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Image embedding batch size",
    )
    parser.add_argument(
        "--skip-objects",
        action="store_true",
        help="Run only selected embedding providers",
    )

    args = parser.parse_args()

    if not args.mapping_csv.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {args.mapping_csv}")
        
    print(f"Loading mapping from {args.mapping_csv}")
    keyframes = sort_keyframes_deterministically(args.mapping_csv)
    print(f"Loaded {len(keyframes)} keyframes.")
    
    dataset_name = args.output.name.replace("features_", "") if args.output.name.startswith("features_") else args.output.name
    
    def run_clip():
        print(f"Initializing OpenClipEncoder ({args.clip_model})...")
        clip_encoder = OpenClipEncoder(
            model_name=args.clip_model,
            pretrained=args.clip_pretrained,
            device=args.device,
        )
        extract_clip_features(
            dataset_name, keyframes, clip_encoder, args.output, args.keyframe_root
        )

    def run_beit3():
        if args.beit3_checkpoint is None:
            parser.error("--beit3-checkpoint is required when selecting beit3")
        specs = load_provider_specs(args.provider_specs)
        spec = specs.get("beit3")
        checkpoint_hash = verify_file_sha256(
            args.beit3_checkpoint, spec.checkpoint_sha256, "BEiT-3 checkpoint"
        )
        encoder = Beit3ImageEncoder(
            spec=spec,
            checkpoint_path=args.beit3_checkpoint,
            device=args.device,
        )
        extract_embedding_artifacts(
            keyframes,
            keyframe_root=args.keyframe_root,
            embeddings_root=args.output / "embeddings",
            spec=spec,
            encoder=encoder,
            batch_size=args.embedding_batch_size,
            checkpoint_sha256=checkpoint_hash,
        )

    def run_metaclip2():
        specs = load_provider_specs(args.provider_specs)
        spec = specs.get("metaclip2")
        encoder = MetaClip2ImageEncoder(
            spec=spec,
            model_location=args.metaclip2_model,
            device=args.device,
            allow_download=args.allow_model_downloads,
        )
        extract_embedding_artifacts(
            keyframes,
            keyframe_root=args.keyframe_root,
            embeddings_root=args.output / "embeddings",
            spec=spec,
            encoder=encoder,
            batch_size=args.embedding_batch_size,
        )

    def run_perception_encoder():
        if args.perception_encoder_checkpoint is None:
            parser.error(
                "--perception-encoder-checkpoint is required when selecting "
                "perception_encoder"
            )
        specs = load_provider_specs(args.provider_specs)
        spec = specs.get("perception_encoder")
        checkpoint_hash = verify_file_sha256(
            args.perception_encoder_checkpoint,
            spec.checkpoint_sha256,
            "Perception Encoder checkpoint",
        )
        encoder = PerceptionEncoderImageEncoder(
            spec=spec,
            checkpoint_path=args.perception_encoder_checkpoint,
            device=args.device,
        )
        extract_embedding_artifacts(
            keyframes,
            keyframe_root=args.keyframe_root,
            embeddings_root=args.output / "embeddings",
            spec=spec,
            encoder=encoder,
            batch_size=args.embedding_batch_size,
            checkpoint_sha256=checkpoint_hash,
        )

    run_selected_embedding_providers(
        args.embedding_providers,
        {
            "clip": run_clip,
            "beit3": run_beit3,
            "metaclip2": run_metaclip2,
            "perception_encoder": run_perception_encoder,
        },
    )
    
    # 2. RF-DETR
    if args.skip_objects:
        print("Object extraction skipped.")
        return
    print(f"Initializing RF-DETR ({args.detector_model})...")
    object_detector = RfDetr(model_name=args.detector_model, device=args.device, score_threshold=args.detector_threshold, nms_threshold=args.detector_nms_threshold)
    extract_objects(dataset_name, keyframes, object_detector, args.output, args.keyframe_root, args.mapping_csv)
    
    print(f"Feature extraction completed. Outputs saved to {args.output}")

if __name__ == "__main__":
    main()

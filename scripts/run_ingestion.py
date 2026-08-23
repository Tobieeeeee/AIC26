import argparse
from pathlib import Path
from mvr.ingestion.pipeline import (
    CANONICAL_DETECTOR,
    CANONICAL_LONG_SHOT_SECONDS,
    CANONICAL_SAMPLE_EVERY_SECONDS,
    CANONICAL_SELECTOR_MODE,
    CANONICAL_TRANSNET_THRESHOLD,
    run_pipeline,
)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Video Ingestion Pipeline (Stage 1)")
    parser.add_argument("--input", type=Path, required=True, help="Input directory containing videos")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ingestion artifacts")
    parser.add_argument("--methods", type=str, default=CANONICAL_DETECTOR, help="Comma-separated shot detection methods")
    parser.add_argument("--primary", type=str, default=CANONICAL_DETECTOR, help="Primary method for keyframe extraction")
    parser.add_argument("--pyscene-threshold", type=float, default=27.0, help="Threshold for PySceneDetect")
    parser.add_argument("--transnet-threshold", type=float, default=CANONICAL_TRANSNET_THRESHOLD, help="Threshold for TransNetV2")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--selector-mode", choices=("midpoint", "periodic", "hybrid_temporal_content"), default=CANONICAL_SELECTOR_MODE)
    parser.add_argument("--long-shot-seconds", type=float, default=CANONICAL_LONG_SHOT_SECONDS, help="Periodic threshold")
    parser.add_argument("--sample-every-seconds", type=float, default=CANONICAL_SAMPLE_EVERY_SECONDS, help="Periodic interval")
    parser.add_argument("--candidate-step-sec", type=float, default=1.0)
    parser.add_argument("--max-temporal-gap-sec", type=float, default=8.0)
    parser.add_argument("--clip-knn-k", type=int, default=3)
    parser.add_argument("--clip-neighbor-window-sec", type=float, default=4.0)
    parser.add_argument("--clip-novelty-threshold", type=float, default=0.15)
    parser.add_argument("--clip-rare-threshold", type=float, default=0.25)
    parser.add_argument("--min-event-gap-sec", type=float, default=2.0)
    parser.add_argument("--edge-change-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--edge-change-min-score", type=float, default=0.08)
    parser.add_argument("--midpoint-mandatory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--provider-specs-path", type=Path, default=Path("configs/embedding_providers.json"))
    parser.add_argument("--metaclip2-model")
    parser.add_argument("--allow-model-downloads", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    return parser


def main():
    parser = build_parser()

    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    
    run_pipeline(
        input_path=args.input,
        output_root=args.output,
        methods=methods,
        primary=args.primary,
        pyscene_threshold=args.pyscene_threshold,
        transnet_threshold=args.transnet_threshold,
        device=args.device,
        long_shot_seconds=args.long_shot_seconds,
        sample_every_seconds=args.sample_every_seconds,
        selector_mode=args.selector_mode,
        candidate_step_sec=args.candidate_step_sec,
        max_temporal_gap_sec=args.max_temporal_gap_sec,
        clip_knn_k=args.clip_knn_k,
        clip_neighbor_window_sec=args.clip_neighbor_window_sec,
        clip_novelty_threshold=args.clip_novelty_threshold,
        clip_rare_threshold=args.clip_rare_threshold,
        min_event_gap_sec=args.min_event_gap_sec,
        edge_change_mad_multiplier=args.edge_change_mad_multiplier,
        edge_change_min_score=args.edge_change_min_score,
        midpoint_mandatory=args.midpoint_mandatory,
        provider_specs_path=args.provider_specs_path,
        metaclip2_model=args.metaclip2_model,
        allow_model_downloads=args.allow_model_downloads,
        embedding_batch_size=args.embedding_batch_size,
    )

if __name__ == "__main__":
    main()

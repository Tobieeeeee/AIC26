import argparse
from pathlib import Path
from mvr.ingestion.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run Video Ingestion Pipeline (Stage 1)")
    parser.add_argument("--input", type=Path, required=True, help="Input directory containing videos")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for ingestion artifacts")
    parser.add_argument("--methods", type=str, default="pyscenedetect", help="Comma-separated shot detection methods")
    parser.add_argument("--primary", type=str, default="pyscenedetect", help="Primary method for keyframe extraction")
    parser.add_argument("--pyscene-threshold", type=float, default=27.0, help="Threshold for PySceneDetect")
    parser.add_argument("--transnet-threshold", type=float, default=0.5, help="Threshold for TransNetV2")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--long-shot-seconds", type=float, default=15.0, help="Threshold for long shots")
    parser.add_argument("--sample-every-seconds", type=float, default=5.0, help="Interval for sampling long shots")

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
    )

if __name__ == "__main__":
    main()

import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .shot_detection import create_detector
from .extract import extract_keyframes
from .keyframes import make_keyframes
from .hybrid_selector import HYBRID_SELECTOR_MODE, HybridSelectorConfig
from .metadata import write_benchmark, write_metadata
from .probe import probe_video
from .selector_cache import run_hybrid_selector


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

CANONICAL_DETECTOR = "transnetv2"
CANONICAL_TRANSNET_THRESHOLD = 0.5
CANONICAL_SELECTOR_MODE = "periodic"
CANONICAL_LONG_SHOT_SECONDS = 12.0
CANONICAL_SAMPLE_EVERY_SECONDS = 8.0


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"Không có video trong {input_path}")
    return videos


def run_pipeline(
    input_path: Path,
    output_root: Path,
    methods: list[str] | None = None,
    primary: str = CANONICAL_DETECTOR,
    pyscene_threshold: float = 27.0,
    transnet_threshold: float = CANONICAL_TRANSNET_THRESHOLD,
    device: str = "auto",
    long_shot_seconds: float = CANONICAL_LONG_SHOT_SECONDS,
    sample_every_seconds: float = CANONICAL_SAMPLE_EVERY_SECONDS,
    selector_mode: str = CANONICAL_SELECTOR_MODE,
    candidate_step_sec: float = 1.0,
    max_temporal_gap_sec: float = 8.0,
    clip_knn_k: int = 3,
    clip_neighbor_window_sec: float = 4.0,
    clip_novelty_threshold: float = 0.15,
    clip_rare_threshold: float = 0.25,
    min_event_gap_sec: float = 2.0,
    edge_change_mad_multiplier: float = 3.0,
    edge_change_min_score: float = 0.08,
    midpoint_mandatory: bool = True,
    provider_specs_path: Path = Path("configs/embedding_providers.json"),
    metaclip2_model: str | Path | None = None,
    allow_model_downloads: bool = False,
    embedding_batch_size: int = 16,
) -> None:
    if methods is None:
        methods = [CANONICAL_DETECTOR]
    if primary not in methods:
        raise ValueError("--primary phải nằm trong --methods")
    video_paths = discover_videos(input_path)
    videos = []
    with tqdm(
        video_paths,
        desc="Probe videos",
        unit="video",
        dynamic_ncols=True,
    ) as progress:
        for path in progress:
            progress.set_postfix(file=path.name)
            videos.append(probe_video(path))
    cache = {}
    benchmark = []
    for method in methods:
        threshold = pyscene_threshold if method == "pyscenedetect" else transnet_threshold
        load_start = time.perf_counter()
        detector = create_detector(method, threshold, device)
        load_seconds = time.perf_counter() - load_start
        with tqdm(
            videos,
            desc=f"Detect: {method}",
            unit="video",
            dynamic_ncols=True,
        ) as progress:
            for video in progress:
                progress.set_postfix(file=video.path.name)
                start = time.perf_counter()
                shots = detector.detect(video)
                elapsed = time.perf_counter() - start
                cache[(method, video.video_id)] = shots
                benchmark.append({
                    "method": method,
                    "video_id": video.video_id,
                    "fps": round(video.fps, 6),
                    "frame_count": video.frame_count,
                    "duration_seconds": round(video.duration, 6),
                    "shot_count": len(shots),
                    "model_load_seconds": round(load_seconds, 6),
                    "detection_seconds": round(elapsed, 6),
                    "realtime_factor": round(elapsed / video.duration, 6),
                })
    primary_shots = [shot for video in videos for shot in cache[(primary, video.video_id)]]
    if selector_mode == "periodic":
        keyframes = make_keyframes(
            primary_shots, long_shot_seconds, sample_every_seconds
        )
        selector_manifest = {
            "schema_version": "1.0",
            "selector_mode": "periodic",
            "periodic": {
                "long_threshold_sec": long_shot_seconds,
                "interval_sec": sample_every_seconds,
            },
        }
    elif selector_mode == "midpoint":
        keyframes = make_keyframes(primary_shots, float("inf"), sample_every_seconds)
        selector_manifest = {
            "schema_version": "1.0",
            "selector_mode": "midpoint",
        }
    elif selector_mode == HYBRID_SELECTOR_MODE:
        from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
        from mvr.enrichment.embeddings.provider_specs import load_provider_specs

        config = HybridSelectorConfig(
            candidate_step_sec=candidate_step_sec,
            max_temporal_gap_sec=max_temporal_gap_sec,
            clip_knn_k=clip_knn_k,
            clip_neighbor_window_sec=clip_neighbor_window_sec,
            clip_novelty_threshold=clip_novelty_threshold,
            clip_rare_threshold=clip_rare_threshold,
            min_event_gap_sec=min_event_gap_sec,
            edge_change_mad_multiplier=edge_change_mad_multiplier,
            edge_change_min_score=edge_change_min_score,
            midpoint_mandatory=midpoint_mandatory,
        )
        spec = load_provider_specs(provider_specs_path).get("metaclip2")
        encoder = MetaClip2ImageEncoder(
            spec=spec,
            model_location=metaclip2_model,
            device=device,
            allow_download=allow_model_downloads,
        )
        keyframes = []
        video_manifests = []
        shots_by_video: dict[str, list] = defaultdict(list)
        for shot in primary_shots:
            shots_by_video[shot.video_id].append(shot)
        for video in videos:
            selected, manifest = run_hybrid_selector(
                video=video,
                shots=shots_by_video[video.video_id],
                output_root=output_root,
                config=config,
                spec=spec,
                encoder=encoder,
                embedding_batch_size=embedding_batch_size,
            )
            keyframes.extend(selected)
            video_manifests.append(manifest)
        selector_manifest = {
            "schema_version": "1.0",
            "selector_mode": HYBRID_SELECTOR_MODE,
            "config": config.to_dict(),
            "metaclip_model": spec.checkpoint_or_pretrained,
            "metaclip_revision": spec.model_revision,
            "candidate_count": sum(row["candidate_count"] for row in video_manifests),
            "selected_keyframe_count": len(keyframes),
            "candidate_extraction_seconds": sum(
                row["candidate_extraction_seconds"] for row in video_manifests
            ),
            "candidate_embedding_seconds": sum(
                row["candidate_embedding_seconds"] for row in video_manifests
            ),
            "selector_only_seconds": sum(
                row["selector_only_seconds"] for row in video_manifests
            ),
            "content_event_selection_count": sum(
                row["content_event_selection_count"] for row in video_manifests
            ),
            "temporal_repair_selection_count": sum(
                row["temporal_repair_selection_count"] for row in video_manifests
            ),
            "max_observed_temporal_gap_sec": max(
                row["max_observed_temporal_gap_sec"] for row in video_manifests
            ),
            "videos": video_manifests,
        }
    else:
        raise ValueError(
            "selector_mode must be midpoint, periodic, or hybrid_temporal_content"
        )
    by_video = defaultdict(list)
    for frame in keyframes:
        by_video[frame.video_id].append(frame)
    if selector_mode != HYBRID_SELECTOR_MODE:
        with tqdm(
            videos,
            desc="Extract keyframes",
            unit="video",
            dynamic_ncols=True,
        ) as progress:
            for video in progress:
                progress.set_postfix(file=video.path.name)
                extract_keyframes(
                    video.path,
                    by_video[video.video_id],
                    output_root,
                )
    write_metadata(output_root, videos, primary_shots, keyframes)
    write_benchmark(output_root, benchmark)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "selector_manifest.json").write_text(
        json.dumps(selector_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

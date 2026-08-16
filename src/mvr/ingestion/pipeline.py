import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .shot_detection import create_detector
from .extract import extract_keyframes
from .keyframes import make_keyframes
from .metadata import write_benchmark, write_metadata
from .probe import probe_video


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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
    methods: list[str],
    primary: str,
    pyscene_threshold: float,
    transnet_threshold: float,
    device: str,
    long_shot_seconds: float,
    sample_every_seconds: float,
) -> None:
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
    keyframes = make_keyframes(primary_shots, long_shot_seconds, sample_every_seconds)
    by_video = defaultdict(list)
    for frame in keyframes:
        by_video[frame.video_id].append(frame)
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
    write_metadata(output_root, primary_shots, keyframes)
    write_benchmark(output_root, benchmark)

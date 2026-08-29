"""Run the one-video INGEST-KF02 keyframe-selection pilot.

The script freezes detection once, extracts one shared candidate cache, and
evaluates all selectors against the same shots and manual temporal GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.perception_encoder import PerceptionEncoderImageEncoder
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.keyframes import make_keyframes
from mvr.ingestion.models import Keyframe, Shot
from mvr.ingestion.probe import probe_video
from mvr.ingestion.shot_detection import create_detector

from adaptive_selector import budget_matched_diversity, novelty_threshold_selection


FIELDNAMES = [
    "keyframe_id", "video_id", "shot_id", "ordinal", "frame_idx", "timestamp", "image_path"
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    rank = max(1, int(np.ceil(percentile * len(ordered))))
    return ordered[rank - 1]


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_gt(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["start_frame"] = int(row["start_frame"])
        row["end_frame"] = int(row["end_frame"])
    return rows


def _freeze_shots(video_path: Path, output_root: Path, detector_name: str) -> list[Shot]:
    manifest_path = output_root / "frozen_shots.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = payload.setdefault("detector_revision_config", {})
        if "package_version" not in config:
            package = "scenedetect" if payload["detector"] == "pyscenedetect" else "transnetv2-pytorch"
            config["package_version"] = importlib.metadata.version(package)
            for row in payload["shots"]:
                row["detector_revision_config"] = dict(config)
            manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return [
            Shot(
                row["video_id"], row["detector"], row["shot_id"],
                row["start_frame"], row["end_frame"], row["fps"],
            )
            for row in payload["shots"]
        ]

    video = probe_video(video_path)
    threshold = 27.0 if detector_name == "pyscenedetect" else 0.5
    start = time.perf_counter()
    shots = create_detector(detector_name, threshold, "cpu").detect(video)
    elapsed = time.perf_counter() - start
    package = "scenedetect" if detector_name == "pyscenedetect" else "transnetv2-pytorch"
    detector_config = {
        "threshold": threshold,
        "package_version": importlib.metadata.version(package),
    }
    payload = {
        "schema_version": "1.0",
        "video_id": video.video_id,
        "source_path": str(video_path.resolve()),
        "source_size_bytes": video_path.stat().st_size,
        "source_sha256": _sha256(video_path),
        "detector": detector_name,
        "detector_revision_config": detector_config,
        "detection_seconds": elapsed,
        "shots": [
            {
                "video_id": shot.video_id,
                "shot_id": shot.shot_id,
                "start_frame": shot.start_frame,
                "end_frame": shot.end_frame,
                "start_time": shot.start_frame / shot.fps,
                "end_time": (shot.end_frame + 1) / shot.fps,
                "fps": shot.fps,
                "duration": shot.duration,
                "detector": shot.method,
                "detector_revision_config": detector_config,
            }
            for shot in shots
        ],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return shots


def _candidate_frames(shots: list[Shot], interval_seconds: float) -> list[Keyframe]:
    candidates: list[Keyframe] = []
    for shot in shots:
        midpoint = (shot.start_frame + shot.end_frame) // 2
        step = max(1, round(interval_seconds * shot.fps))
        indices = {midpoint}
        frame = shot.start_frame + step
        while frame < shot.end_frame:
            indices.add(frame)
            frame += step
        for ordinal, frame_idx in enumerate(sorted(indices), start=1):
            candidate_id = f"{shot.video_id}_S{shot.shot_id:04d}_C{ordinal:04d}"
            candidates.append(
                Keyframe(
                    candidate_id, shot.video_id, shot.shot_id, ordinal, frame_idx,
                    frame_idx / shot.fps,
                    f"candidates/{shot.video_id}/{candidate_id}.jpg",
                )
            )
    return candidates


def _write_mapping(path: Path, frames: list[Keyframe]) -> None:
    _write_csv(path, [frame.to_dict() for frame in frames], FIELDNAMES)


def _use_candidate_paths(
    frames: list[Keyframe], candidates: list[Keyframe]
) -> list[Keyframe]:
    paths = {
        (item.video_id, item.shot_id, item.frame_idx): item.image_path
        for item in candidates
    }
    return [
        Keyframe(
            item.keyframe_id,
            item.video_id,
            item.shot_id,
            item.ordinal,
            item.frame_idx,
            item.timestamp,
            paths[(item.video_id, item.shot_id, item.frame_idx)],
        )
        for item in frames
    ]


def _load_or_extract_candidates(
    video_path: Path, shots: list[Shot], output_root: Path, interval_seconds: float
) -> tuple[list[Keyframe], float]:
    candidates = _candidate_frames(shots, interval_seconds)
    mapping = output_root / "candidate_mapping.csv"
    missing = [item for item in candidates if not (output_root / item.image_path).is_file()]
    elapsed = 0.0
    if missing:
        start = time.perf_counter()
        extract_keyframes(video_path, candidates, output_root)
        elapsed = time.perf_counter() - start
    _write_mapping(mapping, candidates)
    timing_path = output_root / "candidate_extraction_timing.json"
    if timing_path.exists() and elapsed == 0.0:
        elapsed = json.loads(timing_path.read_text(encoding="utf-8"))["seconds"]
    else:
        timing_path.write_text(
            json.dumps({"seconds": elapsed, "candidate_count": len(candidates)}, indent=2),
            encoding="utf-8",
        )
    return candidates, elapsed


def _encode_in_batches(encoder, paths: list[Path], batch_size: int) -> np.ndarray:
    batches = []
    for start in range(0, len(paths), batch_size):
        batches.append(encoder.encode_images(paths[start:start + batch_size]))
    return np.concatenate(batches, axis=0)


def _load_or_encode(
    provider: str,
    candidates: list[Keyframe],
    output_root: Path,
    *,
    device: str,
    batch_size: int,
    pe_checkpoint: Path | None,
    allow_model_downloads: bool,
) -> tuple[np.ndarray, float, dict]:
    provider_root = output_root / "selector_embeddings" / provider
    vectors_path = provider_root / "candidate_embeddings.npy"
    manifest_path = provider_root / "manifest.json"
    if vectors_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return np.load(vectors_path), manifest["embedding_seconds"], manifest

    specs = load_provider_specs(Path("configs/embedding_providers.json"))
    spec = specs.get(provider)
    if provider == "metaclip2":
        encoder = MetaClip2ImageEncoder(
            spec=spec, device=device, allow_download=allow_model_downloads
        )
    elif provider == "perception_encoder":
        if pe_checkpoint is None:
            raise ValueError("--pe-checkpoint is required for perception_encoder")
        encoder = PerceptionEncoderImageEncoder(
            spec=spec, checkpoint_path=pe_checkpoint, device=device
        )
    else:
        raise ValueError(f"unsupported selector provider: {provider}")
    paths = [output_root / item.image_path for item in candidates]
    start = time.perf_counter()
    vectors = _encode_in_batches(encoder, paths, batch_size)
    elapsed = time.perf_counter() - start
    provider_root.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    manifest = {
        "provider": provider,
        "model_name": spec.model_name,
        "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
        "model_revision": spec.model_revision,
        "input_resolution": spec.image_preprocessing.get("size"),
        "embedding_dimension": spec.embedding_dimension,
        "normalized_contract": spec.normalized,
        "device": device,
        "batch_size": batch_size,
        "candidate_count": len(candidates),
        "embedding_seconds": elapsed,
        "artifact_bytes": vectors_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return vectors, elapsed, manifest


def _adaptive_strategy(
    name: str,
    shots: list[Shot],
    candidates: list[Keyframe],
    vectors: np.ndarray,
    budgets: dict[int, int] | None = None,
    novelty_threshold: float | None = None,
) -> tuple[list[Keyframe], float]:
    positions_by_shot: dict[int, list[int]] = defaultdict(list)
    for position, candidate in enumerate(candidates):
        positions_by_shot[candidate.shot_id].append(position)
    selected: list[Keyframe] = []
    start = time.process_time()
    for shot in shots:
        positions = positions_by_shot[shot.shot_id]
        indices = [candidates[position].frame_idx for position in positions]
        shot_vectors = vectors[positions]
        midpoint = (shot.start_frame + shot.end_frame) // 2
        if budgets is not None:
            chosen = budget_matched_diversity(
                indices, shot_vectors, midpoint_frame=midpoint, budget=budgets[shot.shot_id]
            )
        else:
            chosen = novelty_threshold_selection(
                indices,
                shot_vectors,
                midpoint_frame=midpoint,
                novelty_threshold=float(novelty_threshold),
                minimum_gap_frames=round(2.0 * shot.fps),
                maximum_keyframes=16,
            )
        for ordinal, frame_idx in enumerate(chosen, start=1):
            keyframe_id = f"{shot.video_id}_S{shot.shot_id:04d}_K{ordinal:02d}"
            selected.append(
                Keyframe(
                    keyframe_id, shot.video_id, shot.shot_id, ordinal, frame_idx,
                    frame_idx / shot.fps,
                    next(
                        item.image_path for item in candidates
                        if item.shot_id == shot.shot_id and item.frame_idx == frame_idx
                    ),
                )
            )
    return selected, time.process_time() - start


def _selection_metrics(
    name: str,
    frames: list[Keyframe],
    shots: list[Shot],
    gt: list[dict],
    output_root: Path,
    *,
    midpoint_count: int,
    candidate_extraction_seconds: float,
    embedding_seconds: float,
    selection_cpu_seconds: float,
) -> dict:
    frame_by_video: dict[str, list[int]] = defaultdict(list)
    for item in frames:
        frame_by_video[item.video_id].append(item.frame_idx)
    covered = [
        any(row["start_frame"] <= frame <= row["end_frame"] for frame in frame_by_video[row["video_id"]])
        for row in gt
    ]
    high_positions = [i for i, row in enumerate(gt) if row["annotation_confidence"] == "high"]
    high_covered = sum(covered[i] for i in high_positions)
    counts = [sum(item.shot_id == shot.shot_id for item in frames) for shot in shots]
    jpeg_bytes = sum((output_root / item.image_path).stat().st_size for item in frames)
    return {
        "strategy": name,
        "gt_covered_all": sum(covered),
        "gt_total_all": len(gt),
        "selection_recall_all": sum(covered) / len(gt),
        "gt_covered_high": high_covered,
        "gt_total_high": len(high_positions),
        "selection_recall_high": high_covered / len(high_positions),
        "total_keyframes": len(frames),
        "keyframes_per_shot": len(frames) / len(shots),
        "keyframes_median": statistics.median(counts),
        "keyframes_p95": _percentile_nearest_rank(counts, 0.95),
        "keyframes_max": max(counts),
        "percent_shots_one_keyframe": 100 * sum(value == 1 for value in counts) / len(counts),
        "percent_shots_multi_keyframe": 100 * sum(value > 1 for value in counts) / len(counts),
        "relative_keyframe_count": len(frames) / midpoint_count,
        "jpeg_bytes": jpeg_bytes,
        "candidate_extraction_seconds_shared": candidate_extraction_seconds,
        "selector_embedding_seconds": embedding_seconds,
        "selector_only_cpu_seconds": selection_cpu_seconds,
        "recall_per_100_keyframes": 100 * sum(covered) / len(frames),
        "missed_query_ids": ";".join(row["query_id"] for row, hit in zip(gt, covered) if not hit),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector", choices=["pyscenedetect", "transnetv2"], default="pyscenedetect")
    parser.add_argument("--candidate-interval", type=float, default=1.0)
    parser.add_argument("--providers", nargs="*", choices=["metaclip2", "perception_encoder"], default=[])
    parser.add_argument("--pe-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-model-downloads", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    shots = _freeze_shots(args.video, args.output, args.detector)
    candidates, extraction_seconds = _load_or_extract_candidates(
        args.video, shots, args.output, args.candidate_interval
    )
    gt = _load_gt(args.groundtruth)

    strategies: dict[str, tuple[list[Keyframe], float, float]] = {}
    midpoint = _use_candidate_paths(make_keyframes(shots, float("inf"), 8.0), candidates)
    strategies["S0-midpoint"] = (midpoint, 0.0, 0.0)
    start = time.process_time()
    s1 = _use_candidate_paths(make_keyframes(shots, 12.0, 8.0), candidates)
    strategies["S1-12s-8s"] = (s1, 0.0, time.process_time() - start)
    start = time.process_time()
    s2 = _use_candidate_paths(make_keyframes(shots, 15.0, 5.0), candidates)
    strategies["S2-15s-5s"] = (s2, 0.0, time.process_time() - start)
    budgets_s1 = defaultdict(int)
    budgets_s2 = defaultdict(int)
    for item in s1:
        budgets_s1[item.shot_id] += 1
    for item in s2:
        budgets_s2[item.shot_id] += 1

    provider_manifests = []
    failures = []
    for provider in args.providers:
        try:
            vectors, embedding_seconds, manifest = _load_or_encode(
                provider, candidates, args.output, device=args.device,
                batch_size=args.batch_size, pe_checkpoint=args.pe_checkpoint,
                allow_model_downloads=args.allow_model_downloads,
            )
            provider_manifests.append(manifest)
            for budget_name, budgets in (("S1", budgets_s1), ("S2", budgets_s2)):
                selected, cpu_seconds = _adaptive_strategy(
                    f"S3-{provider}-budget-{budget_name}", shots, candidates, vectors, budgets
                )
                strategies[f"S3-{provider}-budget-{budget_name}"] = (
                    selected, embedding_seconds, cpu_seconds
                )
            shot_by_id = {shot.shot_id: shot for shot in shots}
            grid2_positions = [
                position
                for position, candidate in enumerate(candidates)
                if candidate.frame_idx
                == (
                    shot_by_id[candidate.shot_id].start_frame
                    + shot_by_id[candidate.shot_id].end_frame
                )
                // 2
                or (
                    candidate.frame_idx - shot_by_id[candidate.shot_id].start_frame
                )
                % max(1, round(2.0 * shot_by_id[candidate.shot_id].fps))
                == 0
            ]
            grid2_candidates = [candidates[position] for position in grid2_positions]
            selected, cpu_seconds = _adaptive_strategy(
                f"S3-{provider}-budget-S1-grid2s",
                shots,
                grid2_candidates,
                vectors[grid2_positions],
                budgets_s1,
            )
            strategies[f"S3-{provider}-budget-S1-grid2s"] = (
                selected,
                embedding_seconds,
                cpu_seconds,
            )
            for tau in (0.05, 0.10, 0.15, 0.20, 0.25):
                selected, cpu_seconds = _adaptive_strategy(
                    f"S4-{provider}-novelty-{tau:.2f}", shots, candidates, vectors,
                    novelty_threshold=tau,
                )
                strategies[f"S4-{provider}-novelty-{tau:.2f}"] = (
                    selected, embedding_seconds, cpu_seconds
                )
        except Exception as exc:  # preserve failed strategy evidence and continue baselines
            failures.append({"provider": provider, "error_type": type(exc).__name__, "message": str(exc)})

    summaries = []
    all_selections = []
    for name, (frames, embedding_seconds, cpu_seconds) in strategies.items():
        strategy_root = args.output / "strategies" / name
        _write_mapping(strategy_root / "keyframe_mapping.csv", frames)
        summaries.append(
            _selection_metrics(
                name, frames, shots, gt, args.output, midpoint_count=len(midpoint),
                candidate_extraction_seconds=extraction_seconds,
                embedding_seconds=embedding_seconds, selection_cpu_seconds=cpu_seconds,
            )
        )
        for item in frames:
            row = item.to_dict()
            row["strategy"] = name
            all_selections.append(row)

    summary_fields = list(summaries[0])
    _write_csv(args.output / "strategy_summary.csv", summaries, summary_fields)
    _write_csv(args.output / "per_shot_selection.csv", all_selections, ["strategy", *FIELDNAMES])
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "video": str(args.video.resolve()),
                "groundtruth": str(args.groundtruth.resolve()),
                "groundtruth_rows": len(gt),
                "groundtruth_high_rows": sum(row["annotation_confidence"] == "high" for row in gt),
                "candidate_interval_seconds": args.candidate_interval,
                "candidate_count": len(candidates),
                "provider_manifests": provider_manifests,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"shots": len(shots), "candidates": len(candidates), "strategies": summaries, "failures": failures}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

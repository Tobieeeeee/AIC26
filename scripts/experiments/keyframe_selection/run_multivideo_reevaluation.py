"""Run the frozen-shot four-video keyframe-selection reevaluation.

Stages are intentionally resumable because the three pinned image/text models
have incompatible optional runtimes.  Generated data stays below the supplied
artifact root and is never a production input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mvr.enrichment.embeddings.beit3 import Beit3ImageEncoder
from mvr.enrichment.embeddings.metaclip2 import MetaClip2ImageEncoder
from mvr.enrichment.embeddings.perception_encoder import PerceptionEncoderImageEncoder
from mvr.enrichment.embeddings.provider_specs import load_provider_specs
from mvr.enrichment.embeddings.query_encoders import (
    Beit3TextEncoder,
    MetaClip2TextEncoder,
    PerceptionEncoderTextEncoder,
)
from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.hybrid_selector import (
    CandidateSignal,
    HybridSelectorConfig,
    candidate_frame_indices,
    local_clip_novelty,
    normalize_embeddings,
    select_hybrid_frame_indices,
)
from mvr.ingestion.keyframes import select_frame_indices
from mvr.ingestion.models import Keyframe, Shot
from mvr.ingestion.probe import probe_video
from mvr.ingestion.selector_cache import compute_edge_change_scores
from mvr.ingestion.shot_detection import create_detector

from scripts.experiments.keyframe_selection.multivideo_eval import (
    first_relevant_rank,
    retrieval_rrf_order,
    robust_adaptive_threshold,
    rrf_scores,
    select_experimental_events,
    stable_descending_ranks,
)


VIDEOS = ("L21_V001", "L21_V029", "L22_V018", "L24_V043")
REGIMES = ("pyscenedetect", "transnetv2")
PROVIDERS = ("metaclip2", "perception_encoder", "beit3")
CONFIG = HybridSelectorConfig()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    fieldnames.extend(
        key for row in rows for key in row if key not in fieldnames
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _video_path(data_root: Path, video_id: str) -> Path:
    return data_root / "raw" / "batch1" / "videos" / video_id.split("_")[0] / f"{video_id}.mp4"


def _shot_payload(shots: Sequence[Shot]) -> list[dict[str, Any]]:
    return [shot.to_dict() for shot in shots]


def _load_shots(path: Path) -> list[Shot]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Shot(
            row["video_id"], row.get("method", payload["detector"]), int(row["shot_id"]),
            int(row["start_frame"]), int(row["end_frame"]), float(row["fps"]),
        )
        for row in payload["shots"]
    ]


def _load_seed_shots(seed_root: Path, video_id: str) -> list[Shot] | None:
    if video_id != "L21_V001":
        return None
    path = seed_root / "shots.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    shots = [
        Shot(
            row["video_id"], row["method"], int(row["shot_id"]),
            int(row["start_frame"]), int(row["end_frame"]), float(row["fps"]),
        )
        for row in payload["shots"]
    ]
    if not shots or any(shot.method != "pyscenedetect" for shot in shots):
        raise RuntimeError("seed shot manifest is not PySceneDetect")
    return shots


def _freeze_one(
    video_path: Path,
    regime: str,
    root: Path,
    detector: Any,
    seed_root: Path | None,
) -> list[Shot]:
    target = root / "frozen_shots" / regime / f"{video_path.stem}.json"
    if target.is_file():
        return _load_shots(target)
    info = probe_video(video_path)
    start = time.perf_counter()
    shots = (
        _load_seed_shots(seed_root, info.video_id)
        if regime == "pyscenedetect" and seed_root is not None
        else None
    )
    source = "validated_kf04_seed" if shots is not None else "detected_now"
    if shots is None:
        shots = detector.detect(info)
    elapsed = time.perf_counter() - start
    package = "scenedetect" if regime == "pyscenedetect" else "transnetv2-pytorch"
    threshold = 27.0 if regime == "pyscenedetect" else 0.5
    _json(
        target,
        {
            "schema_version": "1.0",
            "video_id": info.video_id,
            "source_path": str(video_path.resolve()),
            "source_size_bytes": video_path.stat().st_size,
            "source_sha256": _sha256(video_path),
            "detector": regime,
            "threshold": threshold,
            "package_version": importlib.metadata.version(package),
            "manifest_source": source,
            "detection_seconds": elapsed,
            "shots": _shot_payload(shots),
        },
    )
    return shots


def _regime_candidates(shots: Sequence[Shot]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shot in shots:
        for ordinal, frame_idx in enumerate(candidate_frame_indices(shot), start=1):
            rows.append(
                {
                    "video_id": shot.video_id,
                    "shot_id": shot.shot_id,
                    "candidate_ordinal": ordinal,
                    "frame_idx": frame_idx,
                    "timestamp": round(frame_idx / shot.fps, 6),
                }
            )
    return rows


def _link_seed_jpegs(seed_root: Path | None, video_id: str, frames: Sequence[int], target: Path) -> int:
    if seed_root is None or video_id != "L21_V001":
        return 0
    source = seed_root / "selector_cache" / video_id / "candidates"
    linked = 0
    for frame in frames:
        src = source / f"{frame:012d}.jpg"
        dst = target / f"{frame:012d}.jpg"
        if not src.is_file():
            continue
        if dst.exists():
            linked += 1
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        linked += 1
    return linked


def prepare(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    detectors = {
        "pyscenedetect": create_detector("pyscenedetect", 27.0, "cpu"),
        "transnetv2": create_detector("transnetv2", 0.5, args.detector_device),
    }
    preparation: list[dict[str, Any]] = []
    for video_id in VIDEOS:
        video_path = _video_path(args.data_root, video_id)
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        video_info = probe_video(video_path)
        rows_by_regime: dict[str, list[dict[str, Any]]] = {}
        for regime in REGIMES:
            shots = _freeze_one(video_path, regime, args.output, detectors[regime], args.seed_root)
            rows = _regime_candidates(shots)
            rows_by_regime[regime] = rows
            _csv(args.output / "regime_candidates" / regime / f"{video_id}.csv", rows)
        union_frames = sorted(
            {int(row["frame_idx"]) for rows in rows_by_regime.values() for row in rows}
        )
        image_root = args.output / "candidates" / video_id
        image_root.mkdir(parents=True, exist_ok=True)
        linked = _link_seed_jpegs(args.seed_root, video_id, union_frames, image_root)
        missing = [frame for frame in union_frames if not (image_root / f"{frame:012d}.jpg").is_file()]
        extract_rows = [
            Keyframe(
                f"{video_id}_F{frame:012d}", video_id, 0, position, frame,
                frame / video_info.fps, f"candidates/{video_id}/{frame:012d}.jpg",
            )
            for position, frame in enumerate(missing, start=1)
        ]
        start = time.perf_counter()
        extract_keyframes(video_path, extract_rows, args.output)
        extraction_seconds = time.perf_counter() - start
        mapping = [
            {
                "video_id": video_id,
                "frame_idx": frame,
                "image_path": f"candidates/{video_id}/{frame:012d}.jpg",
            }
            for frame in union_frames
        ]
        _csv(args.output / "candidates" / video_id / "mapping.csv", mapping)
        position = {frame: index for index, frame in enumerate(union_frames)}
        for regime, rows in rows_by_regime.items():
            for row in rows:
                row["global_position"] = position[int(row["frame_idx"])]
            _csv(args.output / "regime_candidates" / regime / f"{video_id}.csv", rows)
        preparation.append(
            {
                "video_id": video_id,
                "union_candidate_count": len(union_frames),
                "seed_jpegs_reused": linked,
                "new_jpegs_extracted": len(missing),
                "extraction_seconds": extraction_seconds,
                "candidate_jpeg_bytes": sum(
                    (image_root / f"{frame:012d}.jpg").stat().st_size for frame in union_frames
                ),
            }
        )
        print(json.dumps(preparation[-1]), flush=True)
    _json(args.output / "preparation_summary.json", preparation)


def _encoder(args: argparse.Namespace, provider: str):
    specs = load_provider_specs(args.provider_specs)
    spec = specs.get(provider)
    if provider == "metaclip2":
        return spec, MetaClip2ImageEncoder(
            spec=spec, model_location=args.metaclip_model, device=args.device, allow_download=False
        )
    if provider == "perception_encoder":
        if args.pe_checkpoint is None:
            raise ValueError("--pe-checkpoint is required")
        return spec, PerceptionEncoderImageEncoder(
            spec=spec, checkpoint_path=args.pe_checkpoint, device=args.device
        )
    if provider == "beit3":
        if args.beit_checkpoint is None:
            raise ValueError("--beit-checkpoint is required")
        return spec, Beit3ImageEncoder(
            spec=spec, checkpoint_path=args.beit_checkpoint, device=args.device
        )
    raise ValueError(provider)


def _seed_meta_vectors(seed_root: Path | None) -> dict[int, np.ndarray]:
    if seed_root is None:
        return {}
    mapping_path = seed_root / "selector_cache" / "L21_V001" / "candidate_mapping.csv"
    vectors_path = seed_root / "selector_cache" / "L21_V001" / "candidate_embeddings.npy"
    if not mapping_path.is_file() or not vectors_path.is_file():
        return {}
    rows = _read_csv(mapping_path)
    vectors = np.load(vectors_path, allow_pickle=False)
    if len(rows) != len(vectors):
        raise RuntimeError("seed MetaCLIP mapping/vector count mismatch")
    return {int(row["frame_idx"]): vectors[index] for index, row in enumerate(rows)}


def embed(args: argparse.Namespace) -> None:
    provider = args.provider
    spec, encoder = _encoder(args, provider)
    for video_id in VIDEOS:
        mapping_path = args.output / "candidates" / video_id / "mapping.csv"
        rows = _read_csv(mapping_path)
        target = args.output / "embeddings" / provider / video_id
        vectors_path = target / "vectors.npy"
        manifest_path = target / "manifest.json"
        mapping_hash = _sha256(mapping_path)
        if manifest_path.is_file() and vectors_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["mapping_sha256"] != mapping_hash or manifest["provider"] != provider:
                raise RuntimeError(f"stale embedding cache: {target}")
            print(json.dumps({"provider": provider, "video_id": video_id, "cache_reused": True}), flush=True)
            continue
        seeded = _seed_meta_vectors(args.seed_root) if provider == "metaclip2" and video_id == "L21_V001" else {}
        vectors = np.empty((len(rows), spec.embedding_dimension), dtype=np.float32)
        missing_positions: list[int] = []
        for position, row in enumerate(rows):
            frame = int(row["frame_idx"])
            if frame in seeded:
                vectors[position] = seeded[frame]
            else:
                missing_positions.append(position)
        start = time.perf_counter()
        for offset in range(0, len(missing_positions), args.batch_size):
            positions = missing_positions[offset : offset + args.batch_size]
            paths = [args.output / rows[position]["image_path"] for position in positions]
            vectors[positions] = encoder.encode_images(paths)
            if offset and offset % (args.batch_size * 20) == 0:
                print(json.dumps({"provider": provider, "video_id": video_id, "encoded": offset, "total_new": len(missing_positions)}), flush=True)
        elapsed = time.perf_counter() - start
        vectors = normalize_embeddings(vectors)
        target.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, vectors, allow_pickle=False)
        _json(
            manifest_path,
            {
                "provider": provider,
                "model_name": spec.model_name,
                "checkpoint_or_pretrained": spec.checkpoint_or_pretrained,
                "model_revision": spec.model_revision,
                "source_revision": spec.source_revision,
                "dimension": spec.embedding_dimension,
                "candidate_count": len(rows),
                "seed_vectors_reused": len(rows) - len(missing_positions),
                "new_vectors_encoded": len(missing_positions),
                "embedding_seconds": elapsed,
                "batch_size": args.batch_size,
                "device": args.device,
                "mapping_sha256": mapping_hash,
                "vectors_sha256": _sha256(vectors_path),
                "vector_bytes": vectors_path.stat().st_size,
            },
        )
        print(json.dumps({"provider": provider, "video_id": video_id, "seconds": elapsed, "new": len(missing_positions)}), flush=True)


def encode_queries(args: argparse.Namespace) -> None:
    provider = args.provider
    specs = load_provider_specs(args.provider_specs)
    spec = specs.get(provider)
    if provider == "metaclip2":
        encoder = MetaClip2TextEncoder(
            spec=spec, model_location=args.metaclip_model, device=args.device, allow_download=False
        )
    elif provider == "perception_encoder":
        encoder = PerceptionEncoderTextEncoder(
            spec=spec, checkpoint_path=args.pe_checkpoint, device=args.device
        )
    elif provider == "beit3":
        encoder = Beit3TextEncoder(
            spec=spec, checkpoint_path=args.beit_checkpoint,
            tokenizer_path=args.beit_tokenizer, device=args.device,
        )
    else:
        raise ValueError(provider)
    gt = _read_csv(args.groundtruth)
    target = args.output / "queries" / provider
    vectors_path = target / "vectors.npy"
    manifest_path = target / "manifest.json"
    gt_hash = _sha256(args.groundtruth)
    if manifest_path.is_file() and vectors_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload["groundtruth_sha256"] != gt_hash:
            raise RuntimeError("query cache groundtruth mismatch")
        print(json.dumps({"provider": provider, "query_cache_reused": True}), flush=True)
        return
    start = time.perf_counter()
    vectors = np.concatenate(
        [np.asarray(encoder.encode_text(row["query_en"]), dtype=np.float32).reshape(1, -1) for row in gt],
        axis=0,
    )
    elapsed = time.perf_counter() - start
    vectors = normalize_embeddings(vectors)
    target.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors, allow_pickle=False)
    _json(
        manifest_path,
        {
            "provider": provider,
            "query_language": "en",
            "query_count": len(gt),
            "groundtruth_sha256": gt_hash,
            "embedding_seconds": elapsed,
            "vectors_sha256": _sha256(vectors_path),
        },
    )
    print(json.dumps({"provider": provider, "query_seconds": elapsed}), flush=True)


def _edge_scores(root: Path, regime: str, video_id: str, rows: Sequence[dict[str, str]]) -> np.ndarray:
    target = root / "edge_scores" / regime / f"{video_id}.npy"
    timing_path = target.with_suffix(".json")
    if target.is_file():
        values = np.load(target, allow_pickle=False)
        if values.shape != (len(rows),):
            raise RuntimeError(f"stale edge cache: {target}")
        if timing_path.is_file():
            return values
        start = time.perf_counter()
        verified = compute_edge_change_scores(
            [root / "candidates" / video_id / f"{int(row['frame_idx']):012d}.jpg" for row in rows],
            [int(row["shot_id"]) for row in rows],
        )
        elapsed = time.perf_counter() - start
        if not np.array_equal(values, verified):
            raise RuntimeError(f"edge cache recomputation mismatch: {target}")
        _json(
            timing_path,
            {"seconds": elapsed, "candidate_count": len(rows), "validated_existing_cache": True},
        )
        return values
    start = time.perf_counter()
    values = compute_edge_change_scores(
        [root / "candidates" / video_id / f"{int(row['frame_idx']):012d}.jpg" for row in rows],
        [int(row["shot_id"]) for row in rows],
    )
    elapsed = time.perf_counter() - start
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, values, allow_pickle=False)
    _json(
        timing_path,
        {"seconds": elapsed, "candidate_count": len(rows), "validated_existing_cache": False},
    )
    return values


def _selected_rows(
    shots: Sequence[Shot], selected: dict[int, Sequence[int]], root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shot in shots:
        for ordinal, frame in enumerate(sorted(set(selected[shot.shot_id])), start=1):
            rows.append(
                {
                    "video_id": shot.video_id,
                    "shot_id": shot.shot_id,
                    "ordinal": ordinal,
                    "frame_idx": frame,
                    "timestamp": round(frame / shot.fps, 6),
                    "image_path": f"candidates/{shot.video_id}/{frame:012d}.jpg",
                }
            )
    return rows


def _content_select(
    shots: Sequence[Shot],
    candidates: Sequence[dict[str, str]],
    vectors_by_provider: dict[str, np.ndarray],
    edges: np.ndarray,
    provider_names: tuple[str, ...],
    *,
    use_edge: bool = False,
    adaptive: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions_by_shot: dict[int, list[int]] = defaultdict(list)
    for position, row in enumerate(candidates):
        positions_by_shot[int(row["shot_id"])].append(position)
    selected: dict[int, Sequence[int]] = {}
    events = repairs = 0
    max_gap = 0.0
    start = time.process_time()
    for shot in shots:
        positions = positions_by_shot[shot.shot_id]
        frames = [int(candidates[position]["frame_idx"]) for position in positions]
        novelties = [
            local_clip_novelty(
                frames, vectors_by_provider[provider][positions], fps=shot.fps,
                neighbor_window_sec=CONFIG.clip_neighbor_window_sec, knn_k=CONFIG.clip_knn_k,
            )
            for provider in provider_names
        ]
        if len(provider_names) == 1 and not adaptive:
            signals = [
                CandidateSignal(
                    frame,
                    float(novelties[0][offset]),
                    float(edges[position]) if use_edge else 0.0,
                )
                for offset, (position, frame) in enumerate(zip(positions, frames, strict=True))
            ]
            result = select_hybrid_frame_indices(shot, signals, CONFIG)
            chosen = result.selected_indices
            event_count = len(result.content_event_indices)
            repair_count = len(result.temporal_repair_indices)
            observed_gap = result.max_observed_gap_sec
        else:
            priority = novelties[0] if len(provider_names) == 1 else rrf_scores(novelties)
            if adaptive:
                thresholds = [robust_adaptive_threshold(values) for values in novelties]
            else:
                thresholds = [CONFIG.clip_novelty_threshold] * len(novelties)
            event_mask = np.any(
                np.stack([values >= threshold for values, threshold in zip(novelties, thresholds, strict=True)]),
                axis=0,
            )
            rare_mask = np.any(
                np.stack([values >= CONFIG.clip_rare_threshold for values in novelties]), axis=0
            )
            result = select_experimental_events(
                shot, frames, priority, event_mask, rare_mask, config=CONFIG
            )
            chosen = result.selected_indices
            event_count = len(result.event_indices)
            repair_count = len(result.repair_indices)
            observed_gap = result.max_gap_sec
        selected[shot.shot_id] = chosen
        events += event_count
        repairs += repair_count
        max_gap = max(max_gap, observed_gap)
    return _selected_rows(shots, selected, Path()), {
        "event_count": events,
        "repair_count": repairs,
        "max_gap_sec": max_gap,
        "selector_cpu_seconds": time.process_time() - start,
    }


def _baseline_select(shots: Sequence[Shot], periodic: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.process_time()
    selected = {
        shot.shot_id: select_frame_indices(
            shot, 12.0 if periodic else float("inf"), 8.0
        )
        for shot in shots
    }
    rows = _selected_rows(shots, selected, Path())
    max_gap = 0.0
    for shot in shots:
        points = [shot.start_frame, *selected[shot.shot_id], shot.end_frame]
        max_gap = max(max_gap, max((b - a) / shot.fps for a, b in zip(points, points[1:])))
    return rows, {
        "event_count": 0,
        "repair_count": 0,
        "max_gap_sec": max_gap,
        "selector_cpu_seconds": time.process_time() - start,
    }


def _selection_metric(gt: Sequence[dict[str, str]], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    frames = [int(row["frame_idx"]) for row in rows]
    covered = [
        any(int(item["start_frame"]) <= frame <= int(item["end_frame"]) for frame in frames)
        for item in gt
    ]
    high = [i for i, item in enumerate(gt) if item["annotation_confidence"].lower() == "high"]
    return {
        "gt_covered": sum(covered),
        "gt_total": len(gt),
        "selection_recall": sum(covered) / len(gt),
        "high_gt_covered": sum(covered[i] for i in high),
        "high_gt_total": len(high),
        "high_selection_recall": sum(covered[i] for i in high) / len(high),
    }


def _retrieval_ranks(
    selected: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, str]],
    vectors: dict[str, np.ndarray],
    query_vectors: dict[str, np.ndarray],
    gt_rows: Sequence[dict[str, str]],
    providers: tuple[str, ...],
) -> list[int | None]:
    global_position = {
        (row["video_id"], int(row["frame_idx"])): int(row["global_position"])
        for row in candidate_rows
    }
    positions = [global_position[(row["video_id"], int(row["frame_idx"]))] for row in selected]
    identities = [(row["video_id"], int(row["frame_idx"])) for row in selected]
    ranks: list[int | None] = []
    for gt_position, gt in enumerate(gt_rows):
        score_lists = [vectors[p][positions] @ query_vectors[p][gt_position] for p in providers]
        order = (
            np.argsort(-score_lists[0], kind="stable")
            if len(providers) == 1
            else retrieval_rrf_order(score_lists)
        )
        ranks.append(
            first_relevant_rank(
                [identities[int(position)] for position in order],
                video_id=gt["video_id"], start_frame=int(gt["start_frame"]),
                end_frame=int(gt["end_frame"]),
            )
        )
    return ranks


def _rank_metrics(ranks: Sequence[int | None]) -> dict[str, float]:
    return {
        "recall_at_1": sum(rank is not None and rank <= 1 for rank in ranks) / len(ranks),
        "recall_at_10": sum(rank is not None and rank <= 10 for rank in ranks) / len(ranks),
        "recall_at_100": sum(rank is not None and rank <= 100 for rank in ranks) / len(ranks),
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / len(ranks),
    }


def _bytes_for_selection(root: Path, video_id: str, rows: Sequence[dict[str, Any]], dimensions: Sequence[int]) -> dict[str, int]:
    jpeg = sum((root / row["image_path"]).stat().st_size for row in rows)
    vector = len(rows) * sum(dimensions) * 4
    # Exact IndexFlatIP serialization size is a fixed 45-byte header plus vectors
    # in the installed FAISS build; calculate by real serialization when possible.
    try:
        import faiss

        index_bytes = 0
        for dimension in dimensions:
            index = faiss.IndexFlatIP(dimension)
            index.add(np.zeros((len(rows), dimension), dtype=np.float32))
            index_bytes += len(faiss.serialize_index(index))
    except ImportError:
        index_bytes = vector + 45 * len(dimensions)
    return {"jpeg_bytes": jpeg, "vector_bytes": vector, "index_bytes": index_bytes}


def evaluate(args: argparse.Namespace) -> None:
    gt_all = _read_csv(args.groundtruth)
    specs = load_provider_specs(args.provider_specs)
    query_vectors = {
        provider: np.load(args.output / "queries" / provider / "vectors.npy", allow_pickle=False)
        for provider in PROVIDERS
    }
    selections: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    selector_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    strategy_providers: dict[str, tuple[str, ...]] = {
        "S0_midpoint": ("metaclip2",),
        "S1_periodic_12s_8s": ("metaclip2",),
        "S2_metaclip2": ("metaclip2",),
        "S3_perception_encoder": ("perception_encoder",),
        "S4_beit3": ("beit3",),
        "S5_metaclip2_edge": ("metaclip2",),
    }
    novelty_summary: list[dict[str, Any]] = []
    for regime in REGIMES:
        for video_id in VIDEOS:
            shots = _load_shots(args.output / "frozen_shots" / regime / f"{video_id}.json")
            candidates = _read_csv(args.output / "regime_candidates" / regime / f"{video_id}.csv")
            union_vectors = {
                provider: np.load(args.output / "embeddings" / provider / video_id / "vectors.npy", allow_pickle=False)
                for provider in PROVIDERS
            }
            positions = [int(row["global_position"]) for row in candidates]
            vectors = {provider: values[positions] for provider, values in union_vectors.items()}
            edges = _edge_scores(args.output, regime, video_id, candidates)
            s0 = _baseline_select(shots, False)
            s1 = _baseline_select(shots, True)
            s2 = _content_select(shots, candidates, vectors, edges, ("metaclip2",))
            s3 = _content_select(shots, candidates, vectors, edges, ("perception_encoder",))
            s4 = _content_select(shots, candidates, vectors, edges, ("beit3",))
            s5 = _content_select(shots, candidates, vectors, edges, ("metaclip2",), use_edge=True)
            for name, value in zip(strategy_providers, (s0, s1, s2, s3, s4, s5), strict=True):
                selections[(regime, video_id, name)], selector_meta[(regime, video_id, name)] = value

            by_shot: dict[int, list[int]] = defaultdict(list)
            for position, row in enumerate(candidates):
                by_shot[int(row["shot_id"])].append(position)
            for shot in shots:
                shot_positions = by_shot[shot.shot_id]
                frames = [int(candidates[p]["frame_idx"]) for p in shot_positions]
                novelty = {
                    provider: local_clip_novelty(
                        frames, vectors[provider][shot_positions], fps=shot.fps,
                        neighbor_window_sec=CONFIG.clip_neighbor_window_sec,
                        knn_k=CONFIG.clip_knn_k,
                    )
                    for provider in PROVIDERS
                }
                for left, right in (("metaclip2", "perception_encoder"), ("metaclip2", "beit3"), ("perception_encoder", "beit3")):
                    left_rank = stable_descending_ranks(novelty[left]).astype(float)
                    right_rank = stable_descending_ranks(novelty[right]).astype(float)
                    correlation = 1.0 if len(frames) == 1 else float(np.corrcoef(left_rank, right_rank)[0, 1])
                    novelty_summary.append(
                        {
                            "regime": regime, "video_id": video_id, "shot_id": shot.shot_id,
                            "provider_pair": f"{left}+{right}", "candidate_count": len(frames),
                            "spearman_rank_correlation": correlation,
                            "same_top1": int(np.argmax(novelty[left]) == np.argmax(novelty[right])),
                        }
                    )

    pass1_rows, pass1_query = _measure(
        args, gt_all, selections, selector_meta, strategy_providers, query_vectors, specs
    )
    _csv(args.output / "pass1_metrics.csv", pass1_rows)
    _csv(args.output / "pass1_per_query.csv", pass1_query)
    _json(args.output / "pass1_complete.json", {"strategies": list(strategy_providers), "rows": len(pass1_rows)})

    # Competitive gate was frozen before results: retain a provider if it wins
    # any detector on macro R@10, pooled R@10, macro selection recall, or the
    # worst-video versions of those metrics. PE+Meta is additionally mandatory
    # because it is a direct task question.
    competitive = _competitive_providers(pass1_rows)
    fusion_sets: list[tuple[str, ...]] = [("perception_encoder", "metaclip2")]
    for pair in (("metaclip2", "beit3"), ("perception_encoder", "beit3")):
        if all(provider in competitive for provider in pair):
            fusion_sets.append(pair)
    if all(provider in competitive for provider in PROVIDERS):
        fusion_sets.append(PROVIDERS)

    for providers in fusion_sets:
        name = "F_RRF_" + "+".join(providers)
        strategy_providers[name] = providers
        for regime in REGIMES:
            for video_id in VIDEOS:
                shots = _load_shots(args.output / "frozen_shots" / regime / f"{video_id}.json")
                candidates = _read_csv(args.output / "regime_candidates" / regime / f"{video_id}.csv")
                positions = [int(row["global_position"]) for row in candidates]
                vectors = {
                    p: np.load(args.output / "embeddings" / p / video_id / "vectors.npy", allow_pickle=False)[positions]
                    for p in providers
                }
                edges = np.zeros(len(candidates), dtype=np.float32)
                result = _content_select(shots, candidates, vectors, edges, providers)
                selections[(regime, video_id, name)], selector_meta[(regime, video_id, name)] = result

    # Pass 3: top two singles by measured macro R@10, fixed tie breaks.
    best_two = _best_two_single_providers(pass1_rows)
    for provider in best_two:
        name = "KNN_ADAPTIVE_" + provider
        strategy_providers[name] = (provider,)
        for regime in REGIMES:
            for video_id in VIDEOS:
                shots = _load_shots(args.output / "frozen_shots" / regime / f"{video_id}.json")
                candidates = _read_csv(args.output / "regime_candidates" / regime / f"{video_id}.csv")
                positions = [int(row["global_position"]) for row in candidates]
                vectors = {provider: np.load(args.output / "embeddings" / provider / video_id / "vectors.npy", allow_pickle=False)[positions]}
                result = _content_select(
                    shots, candidates, vectors, np.zeros(len(candidates)), (provider,), adaptive=True
                )
                selections[(regime, video_id, name)], selector_meta[(regime, video_id, name)] = result

    all_rows, all_query = _measure(
        args, gt_all, selections, selector_meta, strategy_providers, query_vectors, specs
    )
    _csv(args.output / "all_metrics.csv", all_rows)
    _csv(args.output / "all_per_query.csv", all_query)
    fixed_rows, fixed_query = _measure(
        args, gt_all, selections, selector_meta, strategy_providers, query_vectors,
        specs, fixed_retrieval_provider="metaclip2",
    )
    _csv(args.output / "all_metrics_fixed_metaclip2.csv", fixed_rows)
    _csv(args.output / "all_per_query_fixed_metaclip2.csv", fixed_query)
    selection_rows = []
    for (regime, video_id, strategy), selected in selections.items():
        for row in selected:
            selection_rows.append(
                {"regime": regime, "strategy": strategy, **row}
            )
    _csv(args.output / "all_selections.csv", selection_rows)
    _csv(args.output / "provider_novelty_comparison.csv", novelty_summary)
    _json(
        args.output / "evaluation_manifest.json",
        {
            "videos": list(VIDEOS), "groundtruth_rows": len(gt_all),
            "high_confidence_rows": sum(row["annotation_confidence"].lower() == "high" for row in gt_all),
            "query_language": "en", "regimes": list(REGIMES),
            "pass1_strategies": list(strategy_providers)[:6],
            "competitive_gate": "wins any detector on macro/pooled/worst-video R@10 or selection recall",
            "competitive_providers": sorted(competitive),
            "pass2_fusions": ["+".join(value) for value in fusion_sets],
            "pass3_best_two": best_two,
            "rrf_k": 60,
            "adaptive_threshold": "max(0.15, median + 3*MAD) per shot/provider",
            "production_defaults_changed": False,
        },
    )


def _strategy_single_provider(name: str) -> str | None:
    return {
        "S2_metaclip2": "metaclip2",
        "S3_perception_encoder": "perception_encoder",
        "S4_beit3": "beit3",
    }.get(name)


def _measure(
    args: argparse.Namespace,
    gt_all: Sequence[dict[str, str]],
    selections: dict[tuple[str, str, str], list[dict[str, Any]]],
    selector_meta: dict[tuple[str, str, str], dict[str, Any]],
    strategy_providers: dict[str, tuple[str, ...]],
    query_vectors: dict[str, np.ndarray],
    specs: Any,
    fixed_retrieval_provider: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    gt_position = {row["query_id"]: position for position, row in enumerate(gt_all)}
    for regime in REGIMES:
        for strategy, providers in strategy_providers.items():
            selector_providers = () if strategy in (
                "S0_midpoint", "S1_periodic_12s_8s"
            ) else providers
            retrieval_providers = (
                (fixed_retrieval_provider,) if fixed_retrieval_provider else providers
            )
            pooled_selected: list[dict[str, Any]] = []
            pooled_candidates: list[dict[str, str]] = []
            pooled_vectors: dict[str, list[np.ndarray]] = {p: [] for p in retrieval_providers}
            pooled_gt: list[dict[str, str]] = []
            pooled_query: dict[str, list[np.ndarray]] = {p: [] for p in retrieval_providers}
            video_rows: list[dict[str, Any]] = []
            vector_offset = 0
            for video_id in VIDEOS:
                key = (regime, video_id, strategy)
                if key not in selections:
                    continue
                selected = selections[key]
                candidates = _read_csv(args.output / "regime_candidates" / regime / f"{video_id}.csv")
                gt = [row for row in gt_all if row["video_id"] == video_id]
                union_vectors = {
                    p: np.load(args.output / "embeddings" / p / video_id / "vectors.npy", allow_pickle=False)
                    for p in set(providers) | set(retrieval_providers)
                }
                q = {
                    p: query_vectors[p][[gt_position[row["query_id"]] for row in gt]]
                    for p in retrieval_providers
                }
                ranks = _retrieval_ranks(
                    selected, candidates, union_vectors, q, gt, retrieval_providers
                )
                selection_metric = _selection_metric(gt, selected)
                rank_metric = _rank_metrics(ranks)
                shots = _load_shots(args.output / "frozen_shots" / regime / f"{video_id}.json")
                counts = [sum(int(row["shot_id"]) == shot.shot_id for row in selected) for shot in shots]
                dimensions = [specs.get(p).embedding_dimension for p in retrieval_providers]
                byte_metric = _bytes_for_selection(args.output, video_id, selected, dimensions)
                embedding_seconds = sum(
                    json.loads((args.output / "embeddings" / p / video_id / "manifest.json").read_text(encoding="utf-8"))["embedding_seconds"]
                    for p in selector_providers
                )
                retrieval_embedding_seconds = sum(
                    json.loads((args.output / "embeddings" / p / video_id / "manifest.json").read_text(encoding="utf-8"))["embedding_seconds"]
                    for p in retrieval_providers
                )
                prep = next(
                    row for row in json.loads((args.output / "preparation_summary.json").read_text(encoding="utf-8"))
                    if row["video_id"] == video_id
                )
                meta = selector_meta[key]
                edge_seconds = 0.0
                if strategy == "S5_metaclip2_edge":
                    edge_seconds = json.loads(
                        (args.output / "edge_scores" / regime / f"{video_id}.json").read_text(
                            encoding="utf-8"
                        )
                    )["seconds"]
                row = {
                    "scope": "video", "regime": regime, "strategy": strategy,
                    "providers": "+".join(providers), "video_id": video_id,
                    "selector_providers": "+".join(selector_providers) or "none",
                    "retrieval_providers": "+".join(retrieval_providers),
                    "shot_count": len(shots),
                    "shot_duration_min": min(shot.duration for shot in shots),
                    "shot_duration_median": statistics.median(shot.duration for shot in shots),
                    "shot_duration_p95": float(np.quantile([shot.duration for shot in shots], 0.95)),
                    "shot_duration_max": max(shot.duration for shot in shots),
                    "selected_keyframes": len(selected),
                    "keyframes_per_shot": len(selected) / len(shots),
                    **selection_metric, **rank_metric, **meta,
                    "candidate_extraction_seconds_shared": prep["extraction_seconds"],
                    "embedding_seconds_shared": embedding_seconds,
                    "retrieval_embedding_seconds_shared": retrieval_embedding_seconds,
                    "edge_signal_seconds_shared": edge_seconds,
                    **byte_metric,
                }
                rows.append(row)
                video_rows.append(row)
                for gt_row, rank in zip(gt, ranks, strict=True):
                    per_query.append(
                        {
                            "scope": "video", "regime": regime, "strategy": strategy,
                            "video_id": video_id, "query_id": gt_row["query_id"],
                            "annotation_confidence": gt_row["annotation_confidence"],
                            "first_relevant_rank": "" if rank is None else rank,
                        }
                    )

                pooled_selected.extend(selected)
                for candidate in candidates:
                    copied = dict(candidate)
                    copied["global_position"] = vector_offset + int(candidate["global_position"])
                    pooled_candidates.append(copied)
                for p in retrieval_providers:
                    pooled_vectors[p].append(union_vectors[p])
                    pooled_query[p].extend(q[p])
                pooled_gt.extend(gt)
                vector_offset += len(next(iter(union_vectors.values())))

            if not video_rows:
                continue
            pooled_vector_arrays = {p: np.concatenate(values, axis=0) for p, values in pooled_vectors.items()}
            pooled_query_arrays = {p: np.stack(values) for p, values in pooled_query.items()}
            pooled_ranks = _retrieval_ranks(
                pooled_selected, pooled_candidates, pooled_vector_arrays,
                pooled_query_arrays, pooled_gt, retrieval_providers,
            )
            macro = {
                key: float(np.mean([row[key] for row in video_rows]))
                for key in ("selection_recall", "high_selection_recall", "recall_at_1", "recall_at_10", "recall_at_100", "mrr")
            }
            worst = {
                f"worst_{key}": min(row[key] for row in video_rows)
                for key in ("selection_recall", "high_selection_recall", "recall_at_1", "recall_at_10", "recall_at_100", "mrr")
            }
            rows.append(
                {
                    "scope": "aggregate", "regime": regime, "strategy": strategy,
                    "providers": "+".join(providers), "video_id": "ALL4",
                    "selector_providers": "+".join(selector_providers) or "none",
                    "retrieval_providers": "+".join(retrieval_providers),
                    "shot_count": sum(row["shot_count"] for row in video_rows),
                    "shot_duration_min": min(row["shot_duration_min"] for row in video_rows),
                    "shot_duration_median": "", "shot_duration_p95": "",
                    "shot_duration_max": max(row["shot_duration_max"] for row in video_rows),
                    "selected_keyframes": sum(row["selected_keyframes"] for row in video_rows),
                    "keyframes_per_shot": sum(row["selected_keyframes"] for row in video_rows) / sum(row["shot_count"] for row in video_rows),
                    "gt_covered": sum(row["gt_covered"] for row in video_rows), "gt_total": len(gt_all),
                    "selection_recall": macro["selection_recall"],
                    "pooled_selection_recall": sum(row["gt_covered"] for row in video_rows) / len(gt_all),
                    "high_gt_covered": sum(row["high_gt_covered"] for row in video_rows),
                    "high_gt_total": sum(row["high_gt_total"] for row in video_rows),
                    "high_selection_recall": macro["high_selection_recall"],
                    "recall_at_1": macro["recall_at_1"], "recall_at_10": macro["recall_at_10"],
                    "recall_at_100": macro["recall_at_100"], "mrr": macro["mrr"],
                    "event_count": sum(row["event_count"] for row in video_rows),
                    "repair_count": sum(row["repair_count"] for row in video_rows),
                    "max_gap_sec": max(row["max_gap_sec"] for row in video_rows),
                    "selector_cpu_seconds": sum(row["selector_cpu_seconds"] for row in video_rows),
                    "candidate_extraction_seconds_shared": sum(row["candidate_extraction_seconds_shared"] for row in video_rows),
                    "embedding_seconds_shared": sum(row["embedding_seconds_shared"] for row in video_rows),
                    "retrieval_embedding_seconds_shared": sum(row["retrieval_embedding_seconds_shared"] for row in video_rows),
                    "edge_signal_seconds_shared": sum(row["edge_signal_seconds_shared"] for row in video_rows),
                    "jpeg_bytes": sum(row["jpeg_bytes"] for row in video_rows),
                    "vector_bytes": sum(row["vector_bytes"] for row in video_rows),
                    "index_bytes": sum(row["index_bytes"] for row in video_rows),
                    **worst,
                    **{f"pooled_{key}": value for key, value in _rank_metrics(pooled_ranks).items()},
                }
            )
            for gt_row, rank in zip(pooled_gt, pooled_ranks, strict=True):
                per_query.append(
                    {
                        "scope": "pooled", "regime": regime, "strategy": strategy,
                        "video_id": gt_row["video_id"], "query_id": gt_row["query_id"],
                        "annotation_confidence": gt_row["annotation_confidence"],
                        "first_relevant_rank": "" if rank is None else rank,
                    }
                )
    return rows, per_query


def _aggregate_pass1(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["scope"] == "aggregate"]


def _competitive_providers(rows: Sequence[dict[str, Any]]) -> set[str]:
    aggregate = [row for row in _aggregate_pass1(rows) if _strategy_single_provider(row["strategy"])]
    competitive: set[str] = set()
    for regime in REGIMES:
        regime_rows = [row for row in aggregate if row["regime"] == regime]
        for metric in ("recall_at_10", "pooled_recall_at_10", "selection_recall", "worst_recall_at_10", "worst_selection_recall"):
            best = max(row[metric] for row in regime_rows)
            competitive.update(
                _strategy_single_provider(row["strategy"])
                for row in regime_rows if row[metric] == best
            )
    return {provider for provider in competitive if provider is not None}


def _best_two_single_providers(rows: Sequence[dict[str, Any]]) -> list[str]:
    aggregate = [row for row in _aggregate_pass1(rows) if _strategy_single_provider(row["strategy"])]
    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate:
        scores[_strategy_single_provider(row["strategy"])].append(row)
    ordered = sorted(
        scores,
        key=lambda provider: (
            -float(np.mean([row["recall_at_10"] for row in scores[provider]])),
            -float(np.mean([row["pooled_recall_at_10"] for row in scores[provider]])),
            -float(np.mean([row["selection_recall"] for row in scores[provider]])),
            provider,
        ),
    )
    return ordered[:2]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("stage", choices=["prepare", "embed", "queries", "evaluate"])
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--groundtruth", type=Path)
    value.add_argument("--provider", choices=PROVIDERS)
    value.add_argument("--provider-specs", type=Path, default=Path("configs/embedding_providers.json"))
    value.add_argument("--device", default="cuda")
    value.add_argument("--detector-device", default="cuda")
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--seed-root", type=Path)
    value.add_argument("--metaclip-model", type=Path)
    value.add_argument("--pe-checkpoint", type=Path)
    value.add_argument("--beit-checkpoint", type=Path)
    value.add_argument("--beit-tokenizer", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.stage == "prepare":
        if args.data_root is None:
            raise ValueError("--data-root is required")
        prepare(args)
    elif args.stage == "embed":
        if args.provider is None:
            raise ValueError("--provider is required")
        embed(args)
    elif args.stage == "queries":
        if args.provider is None or args.groundtruth is None:
            raise ValueError("--provider and --groundtruth are required")
        encode_queries(args)
    else:
        if args.groundtruth is None:
            raise ValueError("--groundtruth is required")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

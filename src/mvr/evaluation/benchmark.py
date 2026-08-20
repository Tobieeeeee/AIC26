"""Real-data KIS benchmark: query records, dataset-level summary statistics,
and an AIC-organizer-data adapter that lets the existing FAISS index builder
be reused unchanged.

This module is pure orchestration/aggregation: it calls the existing KIS
evaluator (Task 03) rather than reimplementing scoring, and does not itself
run retrieval -- callers inject a `search_fn`.
"""
from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .aic_kis import KIS_CUTOFFS, KisAnswer, KisGroundTruth, evaluate_kis


@dataclass(frozen=True)
class KisBenchmarkQuery:
    """One team-curated benchmark query. `label` must stay "internal" unless
    the query set is organizer-provided official GT."""

    query_id: str
    query_text: str
    video_id: str
    start_frame: int
    end_frame: int
    label: str = "internal"


@dataclass(frozen=True)
class KisBenchmarkRecord:
    query_id: str
    config_name: str
    r_at_k: dict[int, float]
    final_score: float
    latency_seconds: float


def run_kis_benchmark_query(
    query: KisBenchmarkQuery,
    *,
    config_name: str,
    search_fn: Callable[[str], Sequence[Any]],
) -> KisBenchmarkRecord:
    """Time one query through `search_fn` and score it with the official evaluator.

    `search_fn` must return an ordered (rank 1 first) sequence of objects
    exposing `.video_id` and `.frame_id` (e.g. `KisCandidate`).
    """
    gt = KisGroundTruth(
        video_id=query.video_id, start_frame=query.start_frame, end_frame=query.end_frame
    )
    started = time.perf_counter()
    candidates = search_fn(query.query_text)
    latency = time.perf_counter() - started
    answers = [KisAnswer(video_id=c.video_id, frame_id=c.frame_id) for c in candidates]
    score = evaluate_kis(answers, gt)
    return KisBenchmarkRecord(
        query_id=query.query_id,
        config_name=config_name,
        r_at_k=dict(score.recall_at_k),
        final_score=score.final_score,
        latency_seconds=latency,
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = fraction * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _summarize_one_config(records: Sequence[KisBenchmarkRecord]) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"query_count": len(records)}
    for cutoff in KIS_CUTOFFS:
        values = [record.r_at_k[cutoff] for record in records]
        summary[f"mean_recall@{cutoff}"] = statistics.fmean(values)
    final_scores = [record.final_score for record in records]
    summary["mean_final_score"] = statistics.fmean(final_scores)
    summary["median_final_score"] = statistics.median(final_scores)
    latencies = sorted(record.latency_seconds for record in records)
    summary["mean_latency_seconds"] = statistics.fmean(latencies)
    summary["p95_latency_seconds"] = _percentile(latencies, 0.95)
    return summary


def summarize_kis_benchmark(
    records: Sequence[KisBenchmarkRecord],
) -> dict[str, dict[str, float | int]]:
    """Group records by `config_name` and compute mean/median/p95 dataset
    statistics for each configuration. Labeled per-query "mean per-query ..."
    per the task's own naming rule -- callers must not present this as the
    organizer's official aggregate score."""
    if not records:
        raise ValueError("cannot summarize an empty benchmark run")
    by_config: dict[str, list[KisBenchmarkRecord]] = {}
    for record in records:
        by_config.setdefault(record.config_name, []).append(record)
    return {name: _summarize_one_config(group) for name, group in by_config.items()}


#: The columns `indexing.task05a.load_canonical_mapping` requires, plus the
#: image path every downstream consumer needs to see the frame.
KEYFRAME_MAPPING_COLUMNS = (
    "keyframe_id",
    "video_id",
    "shot_id",
    "frame_idx",
    "timestamp",
    "image_path",
)


def _organizer_keyframe_dirs(dataset_root: Path, video_id: str) -> list[Path]:
    """Where a video's keyframes may live, normalized layout first.

    A normalized root holds `<root>/keyframes/<video_id>/`. The raw organizer
    download splits them across `Keyframes_L21/`, `Keyframes_L26_a/`, `_b/`,
    ... so both shapes are searched rather than requiring a copy of 166,550
    images into one tree.
    """
    candidates = [dataset_root / "keyframes" / video_id]
    candidates.extend(
        path / "keyframes" / video_id for path in sorted(dataset_root.glob("Keyframes_*"))
    )
    return [path for path in candidates if path.is_dir()]


def _organizer_mapping_csv(dataset_root: Path, video_id: str) -> Path | None:
    """The video's `map_keyframes` CSV under either layout, or `None`.

    A normalized root holds `<root>/map_keyframes/`; the raw download nests
    it as `<root>/map-keyframes-aic25-b1/map-keyframes/`. Missing the second
    shape produced zero rows against the real corpus while reporting success.
    """
    candidates = [dataset_root / "map_keyframes" / f"{video_id}.csv"]
    candidates.extend(
        path / "map-keyframes" / f"{video_id}.csv"
        for path in sorted(dataset_root.glob("map-keyframes*"))
    )
    return next((path for path in candidates if path.is_file()), None)


def _shot_id_for(frame_idx: int, boundaries: Sequence[tuple[int, int]] | None) -> str:
    """Which shot contains this frame, or "" when nothing says.

    Blank rather than a guess: the organizer ships no shot identity, and
    `indexing.lexical_index` already accepts a null `shot_id`. Inventing one
    would be unauditable, and the value is metadata -- it is not searched on.
    """
    if not boundaries:
        return ""
    for shot_id, (start, end) in enumerate(boundaries):
        if start <= frame_idx <= end:
            return str(shot_id)
    return ""


def build_aic_keyframe_mapping_csv(
    dataset_root: Path | str,
    output_path: Path | str,
    video_ids: Sequence[str],
    *,
    shot_boundaries: Mapping[str, Sequence[tuple[int, int]]] | None = None,
) -> Path:
    """Adapt AIC organizer keyframe data into the canonical `keyframe_mapping.csv`.

    Everything downstream reads keyframe identity from this one file --
    `task05a.load_canonical_mapping`, `ocr.docker_ocr`, and the semantic
    index builder -- so adopting the organizer's keyframes as the team's
    canonical identity means emitting this CSV, not refactoring a pipeline.

    `keyframe_id` is the **JPEG stem** ("034"), not the CSV `n` column
    ("34"). The stem is what exists on disk and what `image_path` must agree
    with; the two spellings do not join, and a mismatch yields an empty
    result silently rather than an error. `frame_idx` is taken from the
    organizer's own column rather than derived from fps and timestamp: it is
    the number the competition scores on, they state it, and re-deriving it
    would only introduce a rounding disagreement.

    `shot_boundaries` optionally assigns `shot_id` from the team's own shot
    detection, letting TransNetV2 supply boundaries while the organizer
    supplies identity. Without it `shot_id` is blank.

    Rows whose keyframe JPEG does not exist locally, and videos with no
    mapping CSV, are skipped rather than guessed -- 23 of 873 videos ship no
    keyframes, and a fabricated `frame_idx` would poison the one number the
    competition scores on.
    """
    dataset_root = Path(dataset_root)
    rows: list[dict[str, str]] = []
    mappings_found = 0
    for video_id in video_ids:
        csv_path = _organizer_mapping_csv(dataset_root, video_id)
        if csv_path is None:
            continue
        mappings_found += 1

        # Index the video's JPEGs by their numeric value rather than assuming a
        # zero-padding width, matching AicDatasetRegistry.keyframe_path().
        images_by_number: dict[int, Path] = {}
        for keyframe_dir in _organizer_keyframe_dirs(dataset_root, video_id):
            for image_path in keyframe_dir.glob("*.jpg"):
                try:
                    images_by_number.setdefault(int(image_path.stem), image_path)
                except ValueError:
                    continue

        boundaries = None if shot_boundaries is None else shot_boundaries.get(video_id)
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                number = int(row["n"].strip())
                image_path = images_by_number.get(number)
                if image_path is None:
                    continue
                frame_idx = int(row["frame_idx"].strip())
                rows.append(
                    {
                        "keyframe_id": image_path.stem,
                        "video_id": video_id,
                        "shot_id": _shot_id_for(frame_idx, boundaries),
                        "frame_idx": str(frame_idx),
                        "timestamp": row["pts_time"],
                        "image_path": str(image_path),
                    }
                )

    if video_ids and mappings_found == 0:
        # Not finding a single mapping CSV means the dataset root is wrong --
        # a video merely missing its downloaded keyframes still has one, and
        # legitimately yields no rows. An empty CSV would be accepted happily
        # by everything downstream: OCR, Task05A and the index builder would
        # each report "0 documents" as if that were the data. Against the real
        # corpus this went unnoticed until the row count was inspected by hand.
        raise ValueError(
            f"no map_keyframes CSV found under {dataset_root} for any of "
            f"{len(video_ids)} video(s); check the dataset root"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(KEYFRAME_MAPPING_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return output_path

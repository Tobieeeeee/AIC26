import csv
import json
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mvr.ingestion.extract import extract_keyframes
from mvr.ingestion.models import Keyframe


@dataclass
class OcrCandidate:
    video_id: str
    shot_id: int
    keyframe_id: str
    anchor_frame_idx: int
    source_frame_idx: int
    source_timestamp: float
    text: str
    conf: float
    bbox: list[float]  # [x1, y1, x2, y2]

    # New fields for Docker OCR
    request_id: str = ""
    anchor_timestamp: float = 0.0
    detection_index: int = 0
    det_conf: float = 0.0
    rec_conf: float = 0.0
    polygon: list[list[float]] = field(default_factory=list)
    crop_strategy: str = ""
    support_count: int = 1


def normalize_text(text: str) -> str:
    """Normalize text for matching only."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def levenshtein_distance(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[len(a)][len(b)]


def text_similarity(a: str, b: str) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    dist = levenshtein_distance(a_norm, b_norm)
    return 1.0 - dist / max(len(a_norm), len(b_norm))


def bbox_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def group_candidates(
    candidates: list[OcrCandidate],
    similarity_threshold: float = 0.80,
    iou_threshold: float = 0.30,
) -> list[OcrCandidate]:
    sorted_cands = sorted(
        candidates,
        key=lambda c: (
            c.source_frame_idx,
            c.bbox[1],
            c.bbox[0],
            c.text,
        ),
    )

    groups: list[list[OcrCandidate]] = []

    for cand in sorted_cands:
        matched_group = None
        for group in groups:
            rep = group[0]
            if text_similarity(cand.text, rep.text) >= similarity_threshold and bbox_iou(cand.bbox, rep.bbox) >= iou_threshold:
                matched_group = group
                break

        if matched_group is not None:
            matched_group.append(cand)
        else:
            groups.append([cand])

    representatives = []
    for group in groups:
        group_sorted = sorted(
            group,
            key=lambda c: (
                -c.conf,
                abs(c.source_frame_idx - c.anchor_frame_idx),
                c.source_frame_idx,
                c.text,
            )
        )
        rep = group_sorted[0]
        unique_frames = {c.source_frame_idx for c in group}

        canonical = OcrCandidate(
            request_id=rep.request_id,
            video_id=rep.video_id,
            shot_id=rep.shot_id,
            keyframe_id=rep.keyframe_id,
            anchor_frame_idx=rep.anchor_frame_idx,
            anchor_timestamp=rep.anchor_timestamp,
            source_frame_idx=rep.source_frame_idx,
            source_timestamp=rep.source_timestamp,
            detection_index=rep.detection_index,
            text=rep.text,
            det_conf=rep.det_conf,
            rec_conf=rep.rec_conf,
            conf=rep.conf,
            polygon=rep.polygon,
            bbox=rep.bbox,
            crop_strategy=rep.crop_strategy,
            support_count=len(unique_frames),
        )
        representatives.append(canonical)

    return representatives


def resolve_video_path(video_root: Path, video_id: str) -> Path | None:
    """The video's file under either layout, or None.

    A normalized root holds `<root>/<video_id>.mp4`. The raw organizer
    download nests them two levels and splits them across 14 directories:
    `<root>/Videos_L21_a/video/L21_V001.mp4`. Only the flat shape was
    searched, so pointing this at the real download matched nothing --
    and the caller merely warned and continued, so the run reported
    success having processed zero frames.
    """
    video_root = Path(video_root)
    direct = video_root / f"{video_id}.mp4"
    if direct.is_file():
        return direct
    for nested in sorted(video_root.glob("Videos_*")):
        candidate = nested / "video" / f"{video_id}.mp4"
        if candidate.is_file():
            return candidate
    return None


def get_shot_boundaries(shots_json_path: Path) -> dict[tuple[str, int], tuple[int, int]]:
    """Frame range of each shot, keyed on (video_id, shot_id).

    Returns ``{}`` when the file does not exist. Having no shot detection is
    an ordinary state -- the organizer ships keyframes and no shots.json --
    and `process_dataset_ocr` called this unguarded, so a run over organizer
    data died with `FileNotFoundError` before reading a single frame.

    `shot_id` alone is not a key. Every video numbers its shots from 1, so
    ``boundaries[1]`` held whichever video was parsed last, and every other
    video's keyframes were then clipped against a range belonging to an
    unrelated video.
    """
    if not shots_json_path.is_file():
        return {}
    boundaries: dict[tuple[str, int], tuple[int, int]] = {}
    with shots_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        for shot in data.get("shots", []):
            identity = (str(shot["video_id"]), int(shot["shot_id"]))
            boundaries[identity] = (shot["start_frame"], shot["end_frame"])
    return boundaries


def load_keyframes(mapping_csv: Path) -> list[dict]:
    keyframes = []
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            keyframes.append({
                "keyframe_id": row["keyframe_id"],
                "video_id": row["video_id"],
                "shot_id": int(row["shot_id"]),
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
                "image_path": row["image_path"],
            })
    return keyframes


def generate_sampling_frames(
    keyframes: list[dict],
    shot_boundaries: dict[tuple[str, int], tuple[int, int]],
    offsets: list[int] = [-1, 0, 1]
) -> tuple[list[Keyframe], dict[tuple[str, str], list[Keyframe]]]:
    """Frames to extract, and the samples belonging to each keyframe.

    The **boundary lookup** is the live fix: `shot_boundaries` is built once
    from the whole `shots.json` and handed to every video, so on `shot_id`
    alone `boundaries[1]` was one arbitrary video's first shot and every
    other video's keyframes were clipped against a range that had nothing to
    do with them.

    The **returned mapping** being keyed on `(video_id, keyframe_id)` is
    hardening, not a bug report: today both callers pass one video's
    keyframes at a time -- `process_dataset_ocr` groups by video, and
    `docker_ocr` never reads this mapping at all -- so nothing was merging.
    It is keyed correctly so that stops being a thing anyone must remember.
    """
    all_frames_to_extract = []
    keyframe_to_samples = defaultdict(list)

    fps = 25.0 # We approximate fps from timestamp or use 25.0 as default, or fetch from shot boundaries

    for kf in keyframes:
        anchor_idx = kf["frame_idx"]
        shot_id = kf["shot_id"]
        identity = (kf["video_id"], kf["keyframe_id"])
        start_frame, end_frame = shot_boundaries.get(
            (kf["video_id"], shot_id), (0, float("inf"))
        )

        # Approximate FPS from one of the frames if not 0
        if anchor_idx > 0 and kf["timestamp"] > 0:
            fps = anchor_idx / kf["timestamp"]

        for offset in offsets:
            source_idx = anchor_idx + offset
            if start_frame <= source_idx <= end_frame:
                # video_id in the filename is hardening too. Both callers
                # extract into a per-video directory today, so "034_1745.jpg"
                # never collided; naming it fully means a caller that batches
                # videos cannot make it collide either.
                image_path_offset = (
                    f"temp/{kf['video_id']}_{kf['keyframe_id']}_{source_idx}.jpg"
                )
                sample = Keyframe(
                    keyframe_id=kf["keyframe_id"],
                    video_id=kf["video_id"],
                    shot_id=shot_id,
                    ordinal=0, # not used here
                    frame_idx=source_idx,
                    timestamp=source_idx / fps if fps > 0 else 0.0,
                    image_path=image_path_offset,
                )
                all_frames_to_extract.append(sample)
                keyframe_to_samples[identity].append(sample)

    return all_frames_to_extract, dict(keyframe_to_samples)


def process_video_ocr(
    video_path: Path,
    keyframes: list[dict],
    shot_boundaries: dict[int, tuple[int, int]],
    output_dir: Path,
    device: str = "cpu",
    ocr_conf_threshold: float = 0.50,
    text_similarity_threshold: float = 0.80,
    bbox_iou_threshold: float = 0.30,
) -> dict:

    # Generate sampling frames
    frames_to_extract, samples_by_keyframe = generate_sampling_frames(keyframes, shot_boundaries)

    # Only keep unique frame indices to extract, to avoid redundant extraction, but extract_keyframes handles it

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)

        # Extract frames to temp directory
        extract_keyframes(video_path, frames_to_extract, temp_root)

        try:
            import easyocr
            import torch
            gpu = torch.cuda.is_available() and device != "cpu"
            reader = easyocr.Reader(["vi", "en"], gpu=gpu)
            backend_name = "easyocr"
        except ImportError:
            raise RuntimeError("EasyOCR is required for this pipeline")

        video_id = keyframes[0]["video_id"] if keyframes else ""
        canonical_text_count = 0
        json_file_count = 0

        video_out_dir = output_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)

        for kf in keyframes:
            anchor_idx = kf["frame_idx"]
            anchor_timestamp = kf["timestamp"]
            keyframe_id = kf["keyframe_id"]

            samples = samples_by_keyframe.get((kf["video_id"], keyframe_id), [])

            raw_candidates = []

            for sample in samples:
                img_path = temp_root / sample.image_path
                if not img_path.exists():
                    continue

                import cv2
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                results = reader.readtext(img)
                # results: list of (bbox, text, prob)
                # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]

                for res in results:
                    bbox_points, text, conf = res
                    if conf < ocr_conf_threshold:
                        continue

                    xs = [pt[0] for pt in bbox_points]
                    ys = [pt[1] for pt in bbox_points]
                    bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

                    cand = OcrCandidate(
                        video_id=video_id,
                        shot_id=kf["shot_id"],
                        keyframe_id=keyframe_id,
                        anchor_frame_idx=anchor_idx,
                        source_frame_idx=sample.frame_idx,
                        source_timestamp=sample.timestamp,
                        text=text,
                        conf=float(conf),
                        bbox=bbox,
                        support_count=1
                    )
                    raw_candidates.append(cand)

            canonical_groups = group_candidates(raw_candidates, text_similarity_threshold, bbox_iou_threshold)

            out_json_path = video_out_dir / f"{keyframe_id}.json"

            texts_output = []
            for c in canonical_groups:
                texts_output.append({
                    "video_id": c.video_id,
                    "shot_id": c.shot_id,
                    "keyframe_id": c.keyframe_id,
                    "frame_idx": c.source_frame_idx,
                    "timestamp": round(c.source_timestamp, 6),
                    "text": c.text,
                    "conf": round(c.conf, 4),
                    "bbox": c.bbox,
                    "support_count": c.support_count
                })
                canonical_text_count += 1

            out_data = {
                "keyframe_id": keyframe_id,
                "video_id": video_id,
                "shot_id": kf["shot_id"],
                "anchor_frame_idx": anchor_idx,
                "anchor_timestamp": anchor_timestamp,
                "texts": texts_output
            }

            with out_json_path.open("w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)

            json_file_count += 1

    return {
        "json_file_count": json_file_count,
        "canonical_text_count": canonical_text_count,
        "backend": backend_name,
    }


def process_dataset_ocr(
    mapping_csv: Path,
    video_root: Path,
    output_dir: Path,
    device: str = "cpu",
    ocr_conf_threshold: float = 0.50,
    text_similarity_threshold: float = 0.80,
    bbox_iou_threshold: float = 0.30,
) -> dict:

    # We load mapping csv to group by video_id
    keyframes = load_keyframes(mapping_csv)
    by_video = defaultdict(list)
    for kf in keyframes:
        by_video[kf["video_id"]].append(kf)

    total_canonical_text = 0
    total_json_count = 0
    backend_name = "easyocr"

    # shots.json is located in the same directory as mapping_csv
    shots_json = mapping_csv.with_name("shots.json")
    shot_boundaries = get_shot_boundaries(shots_json)

    missing_videos: list[str] = []
    for video_id, video_kfs in by_video.items():
        video_path = resolve_video_path(video_root, video_id)
        if video_path is None:
            print(f"Warning: Video {video_id}.mp4 not found under {video_root}. Skipping.")
            missing_videos.append(video_id)
            continue

        res = process_video_ocr(
            video_path=video_path,
            keyframes=video_kfs,
            shot_boundaries=shot_boundaries,
            output_dir=output_dir,
            device=device,
            ocr_conf_threshold=ocr_conf_threshold,
            text_similarity_threshold=text_similarity_threshold,
            bbox_iou_threshold=bbox_iou_threshold
        )
        total_json_count += res["json_file_count"]
        total_canonical_text += res["canonical_text_count"]
        backend_name = res["backend"]

    # Finding no video at all means the root is wrong, and the old code
    # reported that as success: it warned once per video and returned a
    # manifest saying zero keyframes were processed, which every downstream
    # step accepts as "the data". Against the raw organizer download -- where
    # videos are nested as Videos_L21_a/video/ -- that is what would have
    # happened for all 873.
    if by_video and len(missing_videos) == len(by_video):
        raise FileNotFoundError(
            f"no video found under {video_root} for any of {len(by_video)} "
            f"video(s); expected <root>/<video_id>.mp4 or "
            f"<root>/Videos_*/video/<video_id>.mp4"
        )

    manifest = {
      "schema_version": "1.0",
      "feature_type": "scene_text",
      "storage_format": "one_json_per_keyframe",
      "ocr_backend": backend_name,
      "detector": backend_name,
      "recognizer": backend_name,
      "ocr_conf_threshold": ocr_conf_threshold,
      "temporal_offsets": [-1, 0, 1],
      "text_similarity": "normalized_levenshtein",
      "text_similarity_threshold": text_similarity_threshold,
      "spatial_similarity": "bbox_iou",
      "bbox_iou_threshold": bbox_iou_threshold,
      "bbox_format": "xyxy_pixels",
      "representative_selection": "highest_conf",
      "keyframe_count": len(keyframes),
      "json_file_count": total_json_count,
      "canonical_text_count": total_canonical_text,
      "mapping_source": str(mapping_csv)
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest

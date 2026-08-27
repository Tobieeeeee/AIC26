import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from statistics import median
import tempfile
import sys

from mvr.ingestion.extract import extract_keyframes
from mvr.enrichment.ocr.model_integrity import (
    model_manifest_provenance,
    verify_model_directory,
)
from mvr.enrichment.ocr.scene_text import (
    OcrCandidate,
    group_candidates,
    load_keyframes,
    get_shot_boundaries,
    generate_sampling_frames,
    resolve_video_path,
)


def _get_field(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _build_anchor_lookup(keyframes: list) -> dict:
    """Anchor metadata per keyframe, keyed on (video_id, keyframe_id).

    Built over the whole corpus at once -- `process_dataset_docker_ocr`
    passes every video's keyframes -- so on `keyframe_id` alone the entry
    for "034" was whichever of 842 videos was read last. Every other
    video's OCR request then carried that video's `anchor_frame_idx` and
    `anchor_timestamp`: wrong numbers, written confidently, and the frame
    index is the one the competition scores on.
    """
    return {
        (_get_field(kf, "video_id"), _get_field(kf, "keyframe_id")): kf
        for kf in keyframes
    }


def _build_request_payload(sample, anchor, container_temp_root: str = "/ocr-runtime") -> dict:
    return {
        "schema_version": "1.0",
        "artifact_type": "ocr_frame_request",
        "request_id": f"{sample.keyframe_id}:{sample.frame_idx}",
        "video_id": sample.video_id,
        "shot_id": sample.shot_id,
        "keyframe_id": sample.keyframe_id,
        "anchor_frame_idx": _get_field(anchor, "frame_idx"),
        "anchor_timestamp": _get_field(anchor, "timestamp"),
        "source_frame_idx": sample.frame_idx,
        "source_timestamp": sample.timestamp,
        "image_path": (
            f"{container_temp_root}/frames/{sample.video_id}/{sample.image_path}"
        ),
    }


def _vertical_overlap_ratio(a, b) -> float:
    ay1, ay2 = float(a.bbox[1]), float(a.bbox[3])
    by1, by2 = float(b.bbox[1]), float(b.bbox[3])
    overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    min_height = max(1e-6, min(ay2 - ay1, by2 - by1))
    return overlap / min_height


def _same_line(a, b) -> bool:
    if _vertical_overlap_ratio(a, b) >= 0.50:
        return True

    ay1, ay2 = float(a.bbox[1]), float(a.bbox[3])
    by1, by2 = float(b.bbox[1]), float(b.bbox[3])
    a_center = (ay1 + ay2) / 2.0
    b_center = (by1 + by2) / 2.0
    max_height = max(ay2 - ay1, by2 - by1, 1e-6)
    return abs(a_center - b_center) <= 0.35 * max_height


def _merge_phrase_members(members: list):
    if len(members) == 1:
        return members[0]

    members = sorted(members, key=lambda c: (float(c.bbox[0]), float(c.bbox[1])))
    first = members[0]
    bbox = [
        min(float(c.bbox[0]) for c in members),
        min(float(c.bbox[1]) for c in members),
        max(float(c.bbox[2]) for c in members),
        max(float(c.bbox[3]) for c in members),
    ]

    # Conservative phrase confidence: the phrase is only as reliable as
    # its weakest token. This also gives temporal representative selection
    # a meaningful phrase-level score.
    conf = min(float(c.conf) for c in members)
    det_conf = min(float(c.det_conf) for c in members)
    rec_conf = min(float(c.rec_conf) for c in members)

    return OcrCandidate(
        request_id=first.request_id,
        video_id=first.video_id,
        shot_id=first.shot_id,
        keyframe_id=first.keyframe_id,
        anchor_frame_idx=first.anchor_frame_idx,
        anchor_timestamp=first.anchor_timestamp,
        source_frame_idx=first.source_frame_idx,
        source_timestamp=first.source_timestamp,
        text=" ".join(c.text.strip() for c in members if c.text.strip()),
        conf=conf,
        bbox=bbox,
        detection_index=-1,
        det_conf=det_conf,
        rec_conf=rec_conf,
        polygon=[],
        crop_strategy="merged_line",
        support_count=1,
    )


def _merge_frame_candidates(candidates: list, max_gap_height_ratio: float = 1.5) -> list:
    """Merge word-level OCR candidates into spatial phrases within one source frame."""
    if not candidates:
        return []

    ordered = sorted(
        candidates,
        key=lambda c: (((float(c.bbox[1]) + float(c.bbox[3])) / 2.0), float(c.bbox[0])),
    )

    lines = []
    for cand in ordered:
        best_line = None
        best_overlap = -1.0
        for line in lines:
            # Compare against every member so a slightly slanted line can still cluster.
            overlap = max(_vertical_overlap_ratio(cand, member) for member in line)
            if any(_same_line(cand, member) for member in line) and overlap > best_overlap:
                best_line = line
                best_overlap = overlap
        if best_line is None:
            lines.append([cand])
        else:
            best_line.append(cand)

    merged = []
    for line in lines:
        line = sorted(line, key=lambda c: float(c.bbox[0]))
        heights = [max(1.0, float(c.bbox[3]) - float(c.bbox[1])) for c in line]
        max_gap = max_gap_height_ratio * median(heights)

        phrase = [line[0]]
        for cand in line[1:]:
            prev = phrase[-1]
            gap = float(cand.bbox[0]) - float(prev.bbox[2])
            if gap <= max_gap:
                phrase.append(cand)
            else:
                merged.append(_merge_phrase_members(phrase))
                phrase = [cand]
        merged.append(_merge_phrase_members(phrase))

    return sorted(
        merged,
        key=lambda c: (((float(c.bbox[1]) + float(c.bbox[3])) / 2.0), float(c.bbox[0])),
    )


def _candidate_to_canonical_text(c) -> dict:
    return {
        "video_id": c.video_id,
        "shot_id": c.shot_id,
        "frame_idx": c.source_frame_idx,
        "timestamp": c.source_timestamp,
        "text": c.text,
        "conf": round(c.conf, 4),
        "det_conf": round(c.det_conf, 4),
        "rec_conf": round(c.rec_conf, 4),
        "bbox": [round(x, 1) for x in c.bbox],
        "support_count": c.support_count,
    }

def run_docker_pipeline(
    frames_to_extract: list,
    temp_dir: Path,
    output_dir: Path,
    keyframes: list,
    device: str,
    models: dict,
    weights_root: Path,
) -> dict:

    temp_dir_abs = temp_dir.resolve()
    workspace_root = Path.cwd().resolve()
    container_temp_root = "/ocr-runtime"

    requests_dir = temp_dir_abs / "requests"
    detections_dir = temp_dir_abs / "detections"
    candidates_dir = temp_dir_abs / "candidates"
    model_mount = f"{weights_root.resolve()}:/models:ro"
    deepsolo = models["deepsolo"]
    parseq_vi = models["parseq_vi"]

    # Write requests to per-file JSON. Neighbor frames keep the original
    # keyframe as anchor and only vary source_frame_idx/source_timestamp.
    anchor_by_keyframe = _build_anchor_lookup(keyframes)
    for kf in frames_to_extract:
        req_path = requests_dir / kf.video_id / kf.keyframe_id / f"{kf.frame_idx}.json"
        req_path.parent.mkdir(parents=True, exist_ok=True)
        anchor = anchor_by_keyframe.get((kf.video_id, kf.keyframe_id))
        if anchor is None:
            raise KeyError(
                f"Missing anchor keyframe metadata for {kf.video_id}/{kf.keyframe_id}"
            )
        req = _build_request_payload(kf, anchor, container_temp_root)
        with req_path.open("w", encoding="utf-8") as f:
            json.dump(req, f, ensure_ascii=False)

    # Run DeepSolo
    subprocess.run([
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{workspace_root}:/workspace:ro",
        "--mount", f"type=bind,src={temp_dir_abs},dst={container_temp_root}",
        "-v", model_mount,
        "mvr-ocr-deepsolo:task04b",
        "python", "/workspace/docker/ocr/worker/deepsolo_worker.py",
        "--requests-dir",
        f"{container_temp_root}/requests",
        "--output-dir",
        f"{container_temp_root}/detections",
        "--weights",
        str(deepsolo["runtime_path"]),
        "--config",
        str(deepsolo["runtime_config"]),
        "--expected-sha256",
        str(deepsolo["sha256"]),
        "--resume",
    ], check=True)

    request_count = len(list(requests_dir.rglob("*.json")))
    detection_count = len(list(detections_dir.rglob("*.json")))

    if detection_count != request_count:
        raise RuntimeError(
            "DeepSolo output incomplete: "
            f"requests={request_count}, detections={detection_count}"
    )

    # Run PARSeq
    subprocess.run([
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{workspace_root}:/workspace:ro",
        "--mount", f"type=bind,src={temp_dir_abs},dst={container_temp_root}",
        "-v", model_mount,
        "mvr-ocr-parseq-vi:task04b",
        "python", "/workspace/docker/ocr/worker/parseq_worker.py",
        "--detections-dir", f"{container_temp_root}/detections",
        "--output-dir", f"{container_temp_root}/candidates",
        "--weights", str(parseq_vi["runtime_path"]),
        "--expected-sha256", str(parseq_vi["sha256"]),
        "--resume",
    ], check=True)

    candidate_count = len(list(candidates_dir.rglob("*.json")))
    if candidate_count != detection_count:
        raise RuntimeError(
            "PARSeq output incomplete: "
            f"detections={detection_count}, candidates={candidate_count}"
        )

    return process_candidates_to_canonical(candidates_dir, output_dir, keyframes)

def process_candidates_to_canonical(
    candidates_dir: Path,
    output_dir: Path,
    keyframes: list,
    similarity_threshold: float = 0.80,
    iou_threshold: float = 0.30
) -> dict:
    worker_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "docker" / "ocr" / "worker"
    if str(worker_dir) not in sys.path:
        sys.path.insert(0, str(worker_dir))
    from json_file_contract import read_json_file, OcrCandidateFile

    raw_candidates = []

    if candidates_dir.exists():
        for cand_file in candidates_dir.rglob("*.json"):
            cand_obj = read_json_file(cand_file, OcrCandidateFile)
            for cand in cand_obj.candidates:
                hcand = OcrCandidate(
                    request_id=cand_obj.request_id,
                    video_id=cand_obj.video_id,
                    shot_id=cand_obj.shot_id,
                    keyframe_id=cand_obj.keyframe_id,
                    anchor_frame_idx=cand_obj.anchor_frame_idx,
                    anchor_timestamp=cand_obj.anchor_timestamp,
                    source_frame_idx=cand_obj.source_frame_idx,
                    source_timestamp=cand_obj.source_timestamp,
                    text=cand.text,
                    conf=cand.conf,
                    bbox=cand.bbox,
                    detection_index=cand.detection_index,
                    det_conf=cand.det_conf,
                    rec_conf=cand.rec_conf,
                    polygon=cand.polygon,
                    crop_strategy=cand.crop_strategy,
                    support_count=1
                )
                raw_candidates.append(hcand)

    # First merge word-level detections spatially within each sampled source
    # frame. Temporal consolidation then compares phrase-to-phrase across
    # F-1/F/F+1, so representative frame/timestamp remain meaningful.
    by_source_frame = defaultdict(list)
    for c in raw_candidates:
        by_source_frame[(c.video_id, c.keyframe_id, c.source_frame_idx)].append(c)

    merged_candidates = []
    for frame_candidates in by_source_frame.values():
        merged_candidates.extend(_merge_frame_candidates(frame_candidates))

    # Group by keyframe for temporal consolidation, on the full identity.
    #
    # On keyframe_id alone this merged the OCR of every video that numbers a
    # keyframe the same way -- 842 videos have a "034" -- into one group,
    # then took video_id from an anchor that was itself the wrong video's,
    # and wrote a single output/<that video>/034.json for all of them. Text
    # read off one broadcast would have been filed against another.
    by_kf = defaultdict(list)
    for c in merged_candidates:
        by_kf[(c.video_id, c.keyframe_id)].append(c)

    canonical_text_count = 0
    json_file_count = 0
    anchor_lookup = _build_anchor_lookup(keyframes)

    for (video_id, keyframe_id), cands in by_kf.items():
        canonical_groups = group_candidates(cands, similarity_threshold, iou_threshold)

        anchor_meta = anchor_lookup.get((video_id, keyframe_id))
        if anchor_meta is None:
            raise KeyError(
                f"Missing anchor keyframe metadata for {video_id}/{keyframe_id}"
            )

        shot_id = _get_field(anchor_meta, "shot_id")
        anchor_idx = _get_field(anchor_meta, "frame_idx")
        anchor_timestamp = _get_field(anchor_meta, "timestamp")

        video_out_dir = output_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)
        out_json_path = video_out_dir / f"{keyframe_id}.json"

        texts_output = []
        for c in canonical_groups:
            texts_output.append(_candidate_to_canonical_text(c))
            canonical_text_count += 1

        out_data = {
            "keyframe_id": keyframe_id,
            "video_id": video_id,
            "shot_id": shot_id,
            "anchor_frame_idx": anchor_idx,
            "anchor_timestamp": anchor_timestamp,
            "texts": texts_output
        }

        with out_json_path.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        json_file_count += 1

    # Also write empty JSONs for keyframes with no candidates
    for kf in keyframes:
        k_id = kf["keyframe_id"] if isinstance(kf, dict) else kf.keyframe_id
        v_id = kf["video_id"] if isinstance(kf, dict) else kf.video_id
        s_id = kf["shot_id"] if isinstance(kf, dict) else kf.shot_id
        f_idx = kf["frame_idx"] if isinstance(kf, dict) else kf.frame_idx
        t_stamp = kf["timestamp"] if isinstance(kf, dict) else kf.timestamp

        # Membership must use the same key `by_kf` is built with. Checking
        # the bare id against a tuple-keyed dict is always a miss, so this
        # pass would rewrite every keyframe's real OCR as an empty file.
        if (v_id, k_id) not in by_kf:
            video_out_dir = output_dir / v_id
            video_out_dir.mkdir(parents=True, exist_ok=True)
            out_json_path = video_out_dir / f"{k_id}.json"

            out_data = {
                "keyframe_id": k_id,
                "video_id": v_id,
                "shot_id": s_id,
                "anchor_frame_idx": f_idx,
                "anchor_timestamp": t_stamp,
                "texts": []
            }
            with out_json_path.open("w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            json_file_count += 1

    return {
        "json_file_count": json_file_count,
        "canonical_text_count": canonical_text_count,
        "backend": "deepsolo+parseq",
    }

def process_dataset_docker_ocr(
    mapping_csv: Path,
    video_root: Path,
    output_dir: Path,
    temp_dir: Path,
    model_lock_path: Path | None = None,
    weights_root: Path | None = None,
) -> dict:
    repository_root = Path(__file__).resolve().parents[4]
    model_lock_path = model_lock_path or repository_root / "docker" / "ocr" / "model_sources.lock.json"
    weights_root = weights_root or repository_root / "artifacts" / "_ocr_models"
    models = verify_model_directory(model_lock_path, weights_root)
    keyframes = load_keyframes(mapping_csv)
    shots_json = mapping_csv.with_name("shots.json")
    shot_boundaries = get_shot_boundaries(shots_json)

    frames_to_extract, samples_by_keyframe = generate_sampling_frames(keyframes, shot_boundaries)

    # Extract
    temp_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = temp_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Group extraction requests by video
    by_video = defaultdict(list)

    for kf in frames_to_extract:
        by_video[kf.video_id].append(kf)

    for video_id, v_frames in by_video.items():
        # Searches the raw organizer layout too, where videos are nested as
        # Videos_L21_a/video/<video_id>.mp4 across 14 directories. Raising
        # here was already right -- unlike scene_text, which warned and
        # continued -- but the flat-only lookup meant it raised on the real
        # download for every video.
        video_path = resolve_video_path(video_root, video_id)

        if video_path is None:
            raise FileNotFoundError(
                f"Missing source video: {video_id}.mp4 under {video_root}"
            )

        video_frames_dir = frames_dir / video_id
        video_frames_dir.mkdir(parents=True, exist_ok=True)

        missing_frames = [
            frame
            for frame in v_frames
            if not (video_frames_dir / frame.image_path).is_file()
        ]
        extract_keyframes(
            video_path,
            missing_frames,
            video_frames_dir,
        )
    # # Group extraction by video
    # by_video = defaultdict(list)
    # for kf in frames_to_extract:
    #     by_video[kf.video_id].append(kf)
    #
    # for video_id, v_frames in by_video.items():
    #     video_path = video_root / f"{video_id}.mp4"
    #     if video_path.exists():
    #         extract_keyframes(video_path, v_frames, frames_dir)

    expected_frame_count = len(frames_to_extract)

    actual_frame_count = len(
        list(frames_dir.rglob("*.jpg"))
    )

    if actual_frame_count != expected_frame_count:
        raise RuntimeError(
        "Frame extraction incomplete: "
        f"expected={expected_frame_count}, "
        f"actual={actual_frame_count}"
    )

    actual_frame_count = len(
        list(frames_dir.rglob("*.jpg"))
    )

    if actual_frame_count != expected_frame_count:
        raise RuntimeError(
            "Frame extraction incomplete: "
            f"expected={expected_frame_count}, "
            f"actual={actual_frame_count}"
        )

    res = run_docker_pipeline(
        frames_to_extract,
        temp_dir,
        output_dir,
        keyframes,
        "cuda",
        models,
        weights_root,
    )

    manifest = {
      "schema_version": "2.0",
      "feature_type": "scene_text",
      "storage_format": "one_json_per_keyframe",
      "ocr_backend": res["backend"],
      "detector": "deepsolo",
      "recognizer": "parseq_vi",
      "model_provenance": {
          "detector": model_manifest_provenance(models["deepsolo"]),
          "recognizer": model_manifest_provenance(models["parseq_vi"]),
      },
      "ocr_pipeline_version": "docker-ocr-2.1",
      "ocr_conf_threshold": 0.0, # Handled by worker logic if any
      "temporal_offsets": [-1, 0, 1],
      "text_similarity": "normalized_levenshtein",
      "text_similarity_threshold": 0.80,
      "spatial_similarity": "bbox_iou",
      "bbox_iou_threshold": 0.30,
      "bbox_format": "xyxy_pixels",
      "representative_selection": "highest_conf",
      "keyframe_count": len(keyframes),
      "json_file_count": res["json_file_count"],
      "canonical_text_count": res["canonical_text_count"],
      "mapping_source": str(mapping_csv)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--videos", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temp", required=True)
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=Path("docker/ocr/model_sources.lock.json"),
    )
    parser.add_argument(
        "--weights-root",
        type=Path,
        default=Path("artifacts/_ocr_models"),
    )
    args = parser.parse_args()

    process_dataset_docker_ocr(
        Path(args.mapping),
        Path(args.videos),
        Path(args.output),
        Path(args.temp),
        args.model_lock,
        args.weights_root,
    )

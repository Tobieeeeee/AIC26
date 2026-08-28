"""Build a Q&A semantic index from the organizer's precomputed CLIP features.

The organizer ships CLIP ViT-B/32 features for **all 873 videos**
(`clip-features-32/<video_id>.npy`), so an index covering the whole corpus
can be assembled in minutes instead of the ~15 CPU-hours re-encoding every
keyframe would take on this machine.

**Weight compatibility is the critical detail.** The organizer's features
come from *OpenAI* CLIP ViT-B/32 (their FiftyOne baseline uses
`clip-vit-base32-torch`), not the `laion2b_s34b_b79k` weights this repo
encodes with by default. The two are the same architecture and dimension
(512) but different weight spaces: mixing them produces plausible-looking
cosine scores that are meaningless, and fails silently. The manifest
written here therefore records `pretrained: "openai"`, and every consumer
in this repo builds its text encoder from the manifest
(`run_qna_pipeline.py`, `mvr.api.app`), so the matching text encoder is
selected automatically.

Row-order contract, verified against real data (L24_V011: 204 vectors,
204 keyframe JPEGs, 204 CSV rows, all aligned): row *i* of the `.npy`
corresponds to the *i*-th keyframe in **sorted filename order**, which is
also `map_keyframes` row *i* (column `n`, 1-based). This is the ordering
the organizer's own notebook documents.

Example:
    python scripts/build_qna_index_from_organizer_features.py \\
        --dataset-root "<your-dataset-root>" \\
        --output "artifacts\\indexes\\qna_organizer_all"
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mvr.indexing.faiss_index import relative_image_path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# OpenAI's CLIP uses QuickGELU activations. In open_clip, pairing plain
# "ViT-B-32" with pretrained="openai" builds the model with the WRONG
# activation and only emits a UserWarning -- text embeddings then sit in a
# slightly different space from the organizer's image features. Measured on
# a real query, the mismatched pairing scored 0.3527 against 0.3642 for the
# correct one, i.e. silently degraded rather than broken. Always name the
# quickgelu variant for these weights.
ORGANIZER_MODEL_NAME = "ViT-B-32-quickgelu"
ORGANIZER_PRETRAINED = "openai"


@dataclass(frozen=True)
class VideoAssets:
    video_id: str
    features_path: Path
    keyframe_dir: Path
    mapping_csv: Path


def discover_videos(dataset_root: Path) -> list[VideoAssets]:
    """Find every video that has features, keyframes and a mapping CSV.

    The organizer splits keyframes across several `Keyframes_L*` folders,
    so keyframe directories are discovered by scanning rather than assumed
    to live under one root.
    """
    features_root = dataset_root / "clip-features-32-aic25-b1" / "clip-features-32"
    mapping_root = dataset_root / "map-keyframes-aic25-b1" / "map-keyframes"

    keyframe_dirs: dict[str, Path] = {}
    for parent in sorted(dataset_root.glob("Keyframes_*")):
        for candidate in sorted((parent / "keyframes").glob("*")):
            if candidate.is_dir():
                keyframe_dirs.setdefault(candidate.name, candidate)

    assets: list[VideoAssets] = []
    for features_path in sorted(features_root.glob("*.npy")):
        video_id = features_path.stem
        keyframe_dir = keyframe_dirs.get(video_id)
        mapping_csv = mapping_root / f"{video_id}.csv"
        if keyframe_dir is None or not mapping_csv.is_file():
            continue
        assets.append(
            VideoAssets(
                video_id=video_id,
                features_path=features_path,
                keyframe_dir=keyframe_dir,
                mapping_csv=mapping_csv,
            )
        )
    return assets


def load_video(
    assets: VideoAssets, *, image_root: Path
) -> tuple[np.ndarray, list[dict]]:
    """Load one video's vectors plus aligned per-keyframe metadata.

    Raises when the three sources disagree on length rather than truncating
    to the shortest -- a silent misalignment would attach every vector to
    the wrong frame.

    `image_root` is required, not optional. Paths were absolutised here
    until portability was added, which pinned the built index to one disk:
    the review sheet an operator opens by eye reads image_path, so nobody
    else could use the index. A default would quietly restore that.
    """
    features = np.load(assets.features_path)
    keyframes = sorted(assets.keyframe_dir.glob("*.jpg"))
    with assets.mapping_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not (len(features) == len(keyframes) == len(rows)):
        raise ValueError(
            f"{assets.video_id}: length mismatch -- {len(features)} vectors, "
            f"{len(keyframes)} keyframes, {len(rows)} mapping rows"
        )

    metadata = [
        {
            "frame_id": keyframe.stem,
            "video_id": assets.video_id,
            "timestamp": float(row["pts_time"]),
            "image_path": relative_image_path(keyframe, image_root),
        }
        for keyframe, row in zip(keyframes, rows, strict=True)
    ]
    return renormalize(features.astype(np.float32, copy=False)), metadata


def renormalize(features: np.ndarray) -> np.ndarray:
    """Restore exact unit norm after the organizer's float16 storage.

    The shipped vectors are already normalized CLIP embeddings, but float16
    rounding leaves their norms at ~0.9995 -- outside ``FaissIndex``'s 1e-4
    tolerance. Rescaling to exact unit length restores the intended value
    rather than altering it, and cannot change ranking: inner product over
    unit vectors is cosine similarity, which is invariant to per-vector
    scale.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    # Guard against a degenerate all-zero row rather than emitting NaN.
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(features / norms, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Q&A index from organizer CLIP features")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping-csv", type=Path, default=None,
        help=(
            "keyframe_mapping.csv this index will be queried with; recorded "
            "so a mismatched pairing is caught instead of silently dropping rows"
        ),
    )
    parser.add_argument(
        "--videos", nargs="*", default=None,
        help="Limit to these video ids (default: every video with complete assets)",
    )
    args = parser.parse_args()

    from mvr.indexing.faiss_index import FaissIndex, mapping_fingerprint

    assets = discover_videos(args.dataset_root)
    if args.videos:
        wanted = set(args.videos)
        assets = [a for a in assets if a.video_id in wanted]
        missing = wanted - {a.video_id for a in assets}
        if missing:
            raise SystemExit(f"no complete assets for: {sorted(missing)}")
    if not assets:
        raise SystemExit(f"no videos with complete assets under {args.dataset_root}")

    print(f"videos: {len(assets)}")

    all_features: list[np.ndarray] = []
    all_metadata: list[dict] = []
    skipped: list[str] = []
    for position, video in enumerate(assets, start=1):
        try:
            features, metadata = load_video(video, image_root=args.dataset_root)
        except ValueError as exc:
            skipped.append(str(exc))
            continue
        all_features.append(features)
        all_metadata.extend(metadata)
        if position % 100 == 0 or position == len(assets):
            print(f"  [{position}/{len(assets)}] {len(all_metadata)} vectors so far")

    if skipped:
        print(f"\nskipped {len(skipped)} video(s) with inconsistent assets:")
        for line in skipped[:10]:
            print(f"  {line}")

    embeddings = np.ascontiguousarray(np.vstack(all_features), dtype=np.float32)
    del all_features
    print(f"\nstacked: {embeddings.shape}")

    index = FaissIndex.build(
        embeddings,
        all_metadata,
        manifest={
            "embedding_model": ORGANIZER_MODEL_NAME,
            "pretrained": ORGANIZER_PRETRAINED,
            "source": "organizer_precomputed_clip_features",
            "image_root": args.dataset_root,
            # The index holds frame_id, never the competition frame_idx, so
            # every submission row resolves back through this CSV. Binding
            # them here turns a silently short submission into an error.
            **(mapping_fingerprint(args.mapping_csv) if args.mapping_csv else {}),
        },
    )
    args.output.mkdir(parents=True, exist_ok=True)
    index.save(args.output)
    print(f"wrote index to {args.output}")
    print(f"IMPORTANT: query this index with pretrained={ORGANIZER_PRETRAINED!r} "
          "(consumers read this from the manifest automatically)")


if __name__ == "__main__":
    main()

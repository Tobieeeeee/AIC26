"""Real-video verification for exact-frame refinement (Task 05).

Opt-in only: proves that refine_frame(), using the real FFmpeg-based
extractor against a real original video, returns a frame_id whose decoded
image content actually matches that exact original-video frame index -- not
an organizer keyframe, not a timestamp approximation.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MVR_RUN_FRAME_REFINEMENT_REAL_TESTS") != "1",
    reason="set MVR_RUN_FRAME_REFINEMENT_REAL_TESTS=1 to run real FFmpeg/video decode verification",
)


def test_refine_frame_matches_independently_decoded_original_frame(tmp_path) -> None:
    import cv2
    import numpy as np

    from mvr.ingestion.probe import probe_video
    from mvr.retrieval.frame_refinement import RefinementWindowConfig, refine_frame

    dataset_root = os.environ.get("MVR_AIC_DATASET_ROOT", "").strip()
    if not dataset_root:
        pytest.skip("set MVR_AIC_DATASET_ROOT to run this real-data test")
    video_path = Path(dataset_root) / "videos" / "L21_V001.mp4"
    if not video_path.is_file():
        pytest.skip(f"real video not found at {video_path}")

    info = probe_video(video_path)
    coarse_frame_id = 261  # verified real example from AIC_DATA_HANDOFF.md

    class _IndexScoringEncoder:
        """Deliberately trivial: scores each frame by its own index so the
        expected winner is predictable without needing real CLIP weights.
        This test verifies DECODE correctness, not embedding quality."""

        def encode_images(self, image_paths, batch_size: int = 32):
            return np.array(
                [[int(Path(p).stem)] for p in image_paths], dtype=np.float32
            )

    result = refine_frame(
        video_path=video_path,
        video_id="L21_V001",
        coarse_frame_id=coarse_frame_id,
        query_vector=np.array([1.0]),
        encoder=_IndexScoringEncoder(),
        fps=info.fps,
        frame_count=info.frame_count,
        window=RefinementWindowConfig(frame_radius=2),
        tmp_dir=tmp_path,
    )

    # Index-scoring encoder always prefers the highest original frame index
    # in the window.
    assert result.video_id == "L21_V001"
    assert result.frame_id == coarse_frame_id + 2

    refined_image_path = tmp_path / f"{result.frame_id}.jpg"
    assert refined_image_path.is_file()
    assert refined_image_path.stat().st_size > 0

    # Independently decode that exact frame index via OpenCV -- a completely
    # separate decode path from extract_keyframes' FFmpeg pipe -- and confirm
    # it visually matches what refine_frame() produced.
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, result.frame_id)
    ok, independently_decoded = capture.read()
    capture.release()
    assert ok, "OpenCV failed to independently decode the target frame"

    decoded_by_refiner = cv2.imread(str(refined_image_path))
    assert decoded_by_refiner is not None
    assert decoded_by_refiner.shape == independently_decoded.shape

    diff = np.abs(
        decoded_by_refiner.astype(np.int16) - independently_decoded.astype(np.int16)
    )
    mean_diff = float(diff.mean())
    assert mean_diff < 5.0, (
        f"refine_frame's decoded frame does not match the independently "
        f"decoded original frame {result.frame_id} (mean pixel diff={mean_diff})"
    )

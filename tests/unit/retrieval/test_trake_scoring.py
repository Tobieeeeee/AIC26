import numpy as np

from mvr.retrieval.kis import KeyframeFrameIndexResolver
from mvr.retrieval.trake_scoring import build_video_score_matrix


class _FakeFaissIndexCore:
    """Mimics the ``faiss.IndexFlatIP`` attribute FaissIndex wraps: supports
    row-index reconstruction, nothing else."""

    def __init__(self, vectors: dict[int, np.ndarray]) -> None:
        self._vectors = vectors

    def reconstruct(self, row_index: int) -> np.ndarray:
        return self._vectors[row_index]


class _FakeFaissIndex:
    """Mimics FaissIndex's public shape: ``.index.reconstruct(i)`` +
    ``.metadata`` (parallel list of dicts, same order as FAISS row ids)."""

    def __init__(self, vectors: list[np.ndarray], metadata: list[dict]) -> None:
        self.index = _FakeFaissIndexCore(dict(enumerate(vectors)))
        self.metadata = metadata


def test_build_video_score_matrix_scores_match_manual_dot_product() -> None:
    # Rows deliberately out of timestamp order, and include one row from a
    # DIFFERENT video to prove per-video filtering works.
    vectors = [
        np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),  # row 0: V1, ts=3.0
        np.array([0.0, 5.0, 0.0, 0.0], dtype=np.float32),  # row 1: V1, ts=1.0
        np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),  # row 2: V1, ts=2.0
        np.array([9.0, 9.0, 9.0, 9.0], dtype=np.float32),  # row 3: a different video, must be excluded
    ]
    metadata = [
        {"video_id": "V1", "frame_id": "kfA", "timestamp": 3.0},
        {"video_id": "V1", "frame_id": "kfB", "timestamp": 1.0},
        {"video_id": "V1", "frame_id": "kfC", "timestamp": 2.0},
        {"video_id": "V2", "frame_id": "kfOther", "timestamp": 1.0},
    ]
    faiss_index = _FakeFaissIndex(vectors, metadata)
    resolver = KeyframeFrameIndexResolver({"kfA": 300, "kfB": 100, "kfC": 200, "kfOther": 999})
    event_text_vectors = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
    )

    matrix = build_video_score_matrix(
        "V1", event_text_vectors, faiss_index, aic_registry=None, pipeline_resolver=resolver
    )

    assert matrix is not None
    assert matrix.video_id == "V1"
    assert matrix.timestamps == (1.0, 2.0, 3.0)
    assert matrix.original_frame_ids == (100, 200, 300)
    expected = np.array([[0.0, 1.0, 2.0], [5.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(matrix.scores, expected)


def test_build_video_score_matrix_excludes_unresolvable_frames() -> None:
    vectors = [
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ]
    metadata = [
        {"video_id": "V1", "frame_id": "known", "timestamp": 1.0},
        {"video_id": "V1", "frame_id": "unknown-pipeline-format", "timestamp": 2.0},
    ]
    faiss_index = _FakeFaissIndex(vectors, metadata)
    resolver = KeyframeFrameIndexResolver({"known": 42})  # "unknown-pipeline-format" is not mapped
    event_text_vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    matrix = build_video_score_matrix(
        "V1", event_text_vectors, faiss_index, aic_registry=None, pipeline_resolver=resolver
    )

    assert matrix is not None
    assert matrix.original_frame_ids == (42,)
    assert matrix.timestamps == (1.0,)
    assert matrix.scores.shape == (1, 1)


def test_build_video_score_matrix_returns_none_for_video_with_no_indexed_frames() -> None:
    faiss_index = _FakeFaissIndex(
        [np.array([1.0, 0.0], dtype=np.float32)],
        [{"video_id": "OTHER_VIDEO", "frame_id": "kf", "timestamp": 1.0}],
    )
    resolver = KeyframeFrameIndexResolver({"kf": 1})
    event_text_vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    matrix = build_video_score_matrix(
        "V1", event_text_vectors, faiss_index, aic_registry=None, pipeline_resolver=resolver
    )

    assert matrix is None


def test_build_video_score_matrix_returns_none_when_all_frames_unresolvable() -> None:
    faiss_index = _FakeFaissIndex(
        [np.array([1.0, 0.0], dtype=np.float32)],
        [{"video_id": "V1", "frame_id": "unmapped", "timestamp": 1.0}],
    )
    resolver = KeyframeFrameIndexResolver({})  # nothing resolves
    event_text_vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    matrix = build_video_score_matrix(
        "V1", event_text_vectors, faiss_index, aic_registry=None, pipeline_resolver=resolver
    )

    assert matrix is None

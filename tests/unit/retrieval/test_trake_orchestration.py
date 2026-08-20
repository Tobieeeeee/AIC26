import numpy as np

import mvr.retrieval.trake as trake_module
from mvr.evaluation.trake import TrakeChainPrediction
from mvr.retrieval.kis import KisCandidate
from mvr.retrieval.trake import search_trake
from mvr.retrieval.trake_events import TrakeEventQuery, TrakeGapConstraint, TrakeQuery
from mvr.retrieval.trake_scoring import TrakeVideoScoreMatrix


class _FakeTextEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def encode_text(self, texts):
        self.calls.append(list(texts))
        return self._vectors


class _FakeSemanticService:
    def __init__(self, vectors: np.ndarray, index_sentinel=None) -> None:
        self.text_encoder = _FakeTextEncoder(vectors)
        self.index = index_sentinel if index_sentinel is not None else "FAKE_FAISS_INDEX"


class _FakeAlignmentStrategy:
    def __init__(self, chains_by_video_id: dict) -> None:
        self._chains_by_video_id = chains_by_video_id
        self.calls: list[tuple] = []

    def align(self, matrix, gaps):
        self.calls.append((matrix.video_id, gaps))
        return self._chains_by_video_id.get(matrix.video_id, [])


def _query(k: int = 3) -> TrakeQuery:
    events = tuple(TrakeEventQuery(i, f"event {i}") for i in range(1, k + 1))
    gaps = tuple(TrakeGapConstraint(0.0, 60.0) for _ in range(k - 1))
    return TrakeQuery(events=events, gaps=gaps)


def _candidate(video_id: str) -> KisCandidate:
    return KisCandidate(rank=1, video_id=video_id, frame_id=1, score=0.5, keyframe_id="kf")


def _stub_matrix(video_id: str, *_args, **_kwargs) -> TrakeVideoScoreMatrix:
    return TrakeVideoScoreMatrix(video_id, (0.0,), (1,), np.zeros((3, 1)))


# -- happy path -----------------------------------------------------------------


def test_happy_path_shortlists_k_times_and_scores_each_unique_video_once(monkeypatch) -> None:
    query = _query(k=3)
    semantic_service = _FakeSemanticService(np.zeros((3, 4)))

    search_kis_calls: list[str] = []

    def fake_search_kis(text, **_kwargs):
        search_kis_calls.append(text)
        return [_candidate("V1"), _candidate("V2")]  # same two videos every event

    matrix_calls: list[str] = []

    def fake_build_matrix(video_id, *args, **kwargs):
        matrix_calls.append(video_id)
        return _stub_matrix(video_id)

    monkeypatch.setattr(trake_module, "search_kis", fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", fake_build_matrix)

    prediction_v1 = TrakeChainPrediction("V1", (1, 2, 3), score=0.9)
    prediction_v2 = TrakeChainPrediction("V2", (4, 5, 6), score=0.5)
    strategy = _FakeAlignmentStrategy({"V1": [prediction_v1], "V2": [prediction_v2]})

    result = search_trake(
        query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None
    )

    assert len(search_kis_calls) == 3
    assert sorted(matrix_calls) == ["V1", "V2"]  # once per UNIQUE video, not once per event
    assert result == [prediction_v1, prediction_v2]  # sorted by score descending


# -- top_k truncation -------------------------------------------------------------


def test_top_k_truncates_after_merging_across_videos(monkeypatch) -> None:
    query = _query(k=2)
    semantic_service = _FakeSemanticService(np.zeros((2, 4)))
    monkeypatch.setattr(
        trake_module, "search_kis", lambda text, **kw: [_candidate("V1"), _candidate("V2"), _candidate("V3")]
    )
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _stub_matrix)

    chains = {
        "V1": [TrakeChainPrediction("V1", (1, 2), score=0.9)],
        "V2": [TrakeChainPrediction("V2", (3, 4), score=0.8)],
        "V3": [TrakeChainPrediction("V3", (5, 6), score=0.95)],
    }
    strategy = _FakeAlignmentStrategy(chains)

    result = search_trake(
        query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None, top_k=2
    )

    assert len(result) == 2
    assert [chain.video_id for chain in result] == ["V3", "V1"]


# -- per-event failure isolation --------------------------------------------------


def test_one_event_search_kis_failure_is_skipped_not_raised(monkeypatch, caplog) -> None:
    query = _query(k=3)
    semantic_service = _FakeSemanticService(np.zeros((3, 4)))

    def fake_search_kis(text, **_kwargs):
        if text == "event 2":
            raise RuntimeError("boom")
        return [_candidate("V1")]

    monkeypatch.setattr(trake_module, "search_kis", fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _stub_matrix)
    strategy = _FakeAlignmentStrategy({"V1": [TrakeChainPrediction("V1", (1, 2, 3), score=1.0)]})

    with caplog.at_level("ERROR"):
        result = search_trake(
            query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None
        )

    assert len(result) == 1
    assert any("event" in record.message.lower() for record in caplog.records)


def test_all_events_failing_returns_empty_list_not_exception(monkeypatch, caplog) -> None:
    query = _query(k=2)
    semantic_service = _FakeSemanticService(np.zeros((2, 4)))

    def always_fails(text, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trake_module, "search_kis", always_fails)
    strategy = _FakeAlignmentStrategy({})

    with caplog.at_level("ERROR"):
        result = search_trake(
            query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None
        )

    assert result == []
    assert any("all" in record.message.lower() for record in caplog.records)


# -- per-video failure isolation --------------------------------------------------


def test_one_video_matrix_failure_is_skipped_other_videos_still_come_through(monkeypatch, caplog) -> None:
    query = _query(k=2)
    semantic_service = _FakeSemanticService(np.zeros((2, 4)))
    monkeypatch.setattr(trake_module, "search_kis", lambda text, **kw: [_candidate("V1"), _candidate("V2")])

    def fake_build_matrix(video_id, *args, **kwargs):
        if video_id == "V1":
            raise RuntimeError("boom")
        return _stub_matrix(video_id)

    monkeypatch.setattr(trake_module, "build_video_score_matrix", fake_build_matrix)
    strategy = _FakeAlignmentStrategy({"V2": [TrakeChainPrediction("V2", (1, 2), score=1.0)]})

    with caplog.at_level("ERROR"):
        result = search_trake(
            query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None
        )

    assert len(result) == 1
    assert result[0].video_id == "V2"


def test_video_with_none_matrix_is_silently_skipped(monkeypatch) -> None:
    query = _query(k=2)
    semantic_service = _FakeSemanticService(np.zeros((2, 4)))
    monkeypatch.setattr(trake_module, "search_kis", lambda text, **kw: [_candidate("V1"), _candidate("V2")])

    def fake_build_matrix(video_id, *args, **kwargs):
        return None if video_id == "V1" else _stub_matrix(video_id)

    monkeypatch.setattr(trake_module, "build_video_score_matrix", fake_build_matrix)
    strategy = _FakeAlignmentStrategy({"V2": [TrakeChainPrediction("V2", (1, 2), score=1.0)]})

    result = search_trake(
        query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None
    )

    assert len(result) == 1
    assert result[0].video_id == "V2"


# -- strategy interchangeability ---------------------------------------------------


def test_swapping_alignment_strategy_changes_output(monkeypatch) -> None:
    query = _query(k=2)
    semantic_service = _FakeSemanticService(np.zeros((2, 4)))
    monkeypatch.setattr(trake_module, "search_kis", lambda text, **kw: [_candidate("V1")])
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _stub_matrix)

    strategy_a = _FakeAlignmentStrategy({"V1": [TrakeChainPrediction("V1", (1, 2), score=0.5)]})
    strategy_b = _FakeAlignmentStrategy({"V1": [TrakeChainPrediction("V1", (9, 9), score=0.9)]})

    result_a = search_trake(
        query, alignment_strategy=strategy_a, semantic_service=semantic_service, aic_registry=None
    )
    result_b = search_trake(
        query, alignment_strategy=strategy_b, semantic_service=semantic_service, aic_registry=None
    )

    assert result_a[0].score == 0.5
    assert result_b[0].score == 0.9
    assert result_a != result_b


# -- text encoding batching --------------------------------------------------------


def test_text_encoding_happens_once_per_query_not_per_event_per_video(monkeypatch) -> None:
    query = _query(k=3)
    semantic_service = _FakeSemanticService(np.zeros((3, 4)))
    monkeypatch.setattr(trake_module, "search_kis", lambda text, **kw: [_candidate("V1"), _candidate("V2")])
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _stub_matrix)
    strategy = _FakeAlignmentStrategy({})

    search_trake(query, alignment_strategy=strategy, semantic_service=semantic_service, aic_registry=None)

    assert len(semantic_service.text_encoder.calls) == 1
    assert semantic_service.text_encoder.calls[0] == ["event 1", "event 2", "event 3"]


# -- dense scoring always uses semantic_service's own index -----------------------


def test_dense_scoring_uses_semantic_service_index_even_with_multimodal_given(monkeypatch) -> None:
    query = _query(k=2)
    sentinel_index = object()
    multimodal_sentinel = object()
    semantic_service = _FakeSemanticService(np.zeros((2, 4)), index_sentinel=sentinel_index)
    monkeypatch.setattr(trake_module, "search_kis", lambda text, **kw: [_candidate("V1")])

    received_index = []

    def fake_build_matrix(video_id, vectors, faiss_index, aic_registry, pipeline_resolver):
        received_index.append(faiss_index)
        return _stub_matrix(video_id)

    monkeypatch.setattr(trake_module, "build_video_score_matrix", fake_build_matrix)
    strategy = _FakeAlignmentStrategy({})

    search_trake(
        query,
        alignment_strategy=strategy,
        semantic_service=semantic_service,
        aic_registry=None,
        multimodal_service=multimodal_sentinel,
    )

    assert received_index == [sentinel_index]

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import mvr.retrieval.trake as trake_module
from mvr.api.app import create_app
from mvr.retrieval.kis import KisCandidate
from mvr.retrieval.trake_scoring import TrakeVideoScoreMatrix


class _FakeTextEncoder:
    def encode_text(self, texts):
        return np.zeros((len(texts), 4))


class _FakeTrakeRetriever:
    """What LazyFaissSemanticService.retriever() returns: something with
    .text_encoder/.index, distinct from the plain .search()-only fake used
    for /search/kis tests."""

    def __init__(self) -> None:
        self.text_encoder = _FakeTextEncoder()
        self.index = "FAKE_FAISS_INDEX"


class FakeTrakeSemanticService:
    def retriever(self):
        return _FakeTrakeRetriever()


def _fake_search_kis(query, **_kwargs):
    return [KisCandidate(rank=1, video_id="L21_V018", frame_id=1, score=0.5, keyframe_id="kf")]


def _fake_build_video_score_matrix(video_id, *_args, **_kwargs):
    # 2 events, 3 frames; frame_ids offset from column positions on purpose.
    scores = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])
    return TrakeVideoScoreMatrix(
        video_id=video_id,
        timestamps=(0.0, 1.0, 2.0),
        original_frame_ids=(100, 200, 300),
        scores=scores,
    )


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    return TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )


_VALID_REQUEST = {
    "events": [{"text": "event one"}, {"text": "event two"}],
    "gap_phrases": ["rồi"],
}


def test_trake_search_happy_path_returns_ranked_chains(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post("/search/trake", json=_VALID_REQUEST)

    assert response.status_code == 200
    payload = response.json()
    assert payload["event_count"] == 2
    assert payload["alignment_strategy"] == "atlas"
    assert payload["count"] >= 1
    first = payload["results"][0]
    assert first["rank"] == 1
    assert first["video_id"] == "L21_V018"
    assert isinstance(first["frames"], list)
    assert isinstance(first["score"], float)


def test_trake_search_rejects_mismatched_gap_count(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/search/trake",
        json={"events": [{"text": "a"}, {"text": "b"}], "gap_phrases": []},  # needs exactly 1
    )

    assert response.status_code == 422


def test_trake_search_rejects_single_event(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/search/trake", json={"events": [{"text": "only one"}], "gap_phrases": []}
    )

    assert response.status_code == 422


def test_trake_search_rejects_unknown_alignment_strategy_name(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/search/trake", json={**_VALID_REQUEST, "alignment_strategy": "not-a-real-strategy"}
    )

    assert response.status_code == 422


def test_trake_search_rejects_blank_event_text(monkeypatch) -> None:
    client = _client(monkeypatch)

    response = client.post(
        "/search/trake", json={"events": [{"text": "  "}, {"text": "b"}], "gap_phrases": ["rồi"]}
    )

    assert response.status_code == 422


def test_trake_search_strategy_override_is_honored(monkeypatch) -> None:
    client = _client(monkeypatch)

    default_response = client.post("/search/trake", json=_VALID_REQUEST)
    override_response = client.post(
        "/search/trake", json={**_VALID_REQUEST, "alignment_strategy": "ucese"}
    )

    assert default_response.json()["alignment_strategy"] == "atlas"
    assert override_response.json()["alignment_strategy"] == "ucese"


def test_trake_search_none_markers_serialize_as_json_null(monkeypatch) -> None:
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)

    def fake_matrix_with_gap(video_id, *_args, **_kwargs):
        # event1's row never clears any reasonable min_sim -> Lucifer skips it.
        scores = np.array([[0.9, 0.1, 0.1], [0.01, 0.01, 0.01]])
        return TrakeVideoScoreMatrix(
            video_id=video_id,
            timestamps=(0.0, 1.0, 2.0),
            original_frame_ids=(100, 200, 300),
            scores=scores,
        )

    monkeypatch.setattr(trake_module, "build_video_score_matrix", fake_matrix_with_gap)
    client = TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )

    response = client.post(
        "/search/trake", json={**_VALID_REQUEST, "alignment_strategy": "lucifer"}
    )

    assert response.status_code == 200
    frames = response.json()["results"][0]["frames"]
    assert None in frames


def test_trake_search_returns_503_when_semantic_service_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=None,
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )

    response = client.post("/search/trake", json=_VALID_REQUEST)

    assert response.status_code == 503


def test_trake_search_per_event_retrieval_failure_does_not_crash_route(monkeypatch) -> None:
    def failing_search_kis(query, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trake_module, "search_kis", failing_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )

    response = client.post("/search/trake", json=_VALID_REQUEST)

    # search_trake() already guarantees this at the orchestration layer
    # (Phase 05): all events failing -> empty list, not an exception.
    assert response.status_code == 200
    assert response.json()["count"] == 0

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import mvr.retrieval.trake as trake_module
from mvr.api.app import create_app
from mvr.retrieval.kis import KeyframeFrameIndexResolver, KisCandidate
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


def test_trake_search_accepts_a_plain_retriever_shaped_service(monkeypatch) -> None:
    """A plain SemanticRetriever already exposes .text_encoder/.index -- the
    exact shape TRAKE needs, and what the real-data integration tests
    construct. The route must accept it directly, not only the production
    LazyFaissSemanticService's lazy retriever() wrapper."""
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=_FakeTrakeRetriever(),  # no retriever(); has .text_encoder/.index
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        )
    )

    response = client.post("/search/trake", json=_VALID_REQUEST)

    assert response.status_code == 200
    assert response.json()["count"] >= 1


def test_trake_search_returns_503_not_500_for_a_search_only_service(monkeypatch) -> None:
    """Regression: a service exposing only .search() (e.g. an older or custom
    injection) previously produced an opaque AttributeError -> HTTP 500. It
    must fail as a clear, actionable 503 instead."""

    class _SearchOnlyService:
        def search(self, query, top_k=5):
            return []

    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=_SearchOnlyService(),
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
        ),
        raise_server_exceptions=False,
    )

    response = client.post("/search/trake", json=_VALID_REQUEST)

    assert response.status_code == 503
    assert "dense scoring" in response.json()["detail"]


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


def test_trake_search_uses_an_injected_query_translator(monkeypatch) -> None:
    """POST /search/trake honors a caller-injected trake_query_translator,
    the same create_app() extension point qna_answer_reasoner already uses.
    Verifies the TRANSLATED text (not the raw request text) is what
    actually reaches search_kis/dense scoring."""

    class _FakeTranslator:
        def translate(self, text: str) -> str:
            return f"[en] {text}"

    seen_queries: list[str] = []

    def recording_search_kis(query, **_kwargs):
        seen_queries.append(query)
        return [KisCandidate(rank=1, video_id="L21_V018", frame_id=1, score=0.5, keyframe_id="kf")]

    monkeypatch.setattr(trake_module, "search_kis", recording_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    from mvr.retrieval.query_preprocessing import QueryPreprocessingService
    client = TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=None,
            kis_aic_registry=None,
            query_preprocessing=QueryPreprocessingService(
                translator_factory=lambda: _FakeTranslator()
            ),
        )
    )

    response = client.post("/search/trake", json=_VALID_REQUEST)

    assert response.status_code == 200
    assert seen_queries == ["[en] event one", "[en] event two"]


def test_trake_search_translation_is_off_by_default(monkeypatch) -> None:
    """Without MVR_TRAKE_TRANSLATE_QUERIES set and without an explicit
    trake_query_translator override, event text must pass through
    unchanged -- the documented, benchmarked default behavior."""
    monkeypatch.delenv("MVR_TRAKE_TRANSLATE_QUERIES", raising=False)
    seen_queries: list[str] = []

    def recording_search_kis(query, **_kwargs):
        seen_queries.append(query)
        return [KisCandidate(rank=1, video_id="L21_V018", frame_id=1, score=0.5, keyframe_id="kf")]

    monkeypatch.setattr(trake_module, "search_kis", recording_search_kis)
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

    assert response.status_code == 200
    assert seen_queries == ["event one", "event two"]


def test_ui_search_trake_mode_reaches_search_trake(monkeypatch) -> None:
    """`/api/v1/search` is a SECOND call site into search_trake(), separate
    from `/search/trake`. Nothing else in this suite exercises it, so a
    signature change on search_trake() reaches the UI unnoticed: the two live
    in different files, Git reports no conflict, and the break surfaces only
    once a real request arrives with the encoder and index already loaded."""
    client = _client(monkeypatch)

    response = client.post(
        "/api/v1/search",
        json={
            "task_mode": "TRAKE",
            "query": "a diver enters the water",
            "trake_events": [
                "a diver stands at the end of the board",
                "the diver breaks the surface of the water",
            ],
            "top_k": 5,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["total_results"] >= 1
    assert body["results"][0]["video_id"] == "L21_V018"


class _FakeFrameMapping:
    def __init__(self, frame_idx: int, timestamp: float) -> None:
        self.frame_idx = frame_idx
        self.timestamp = timestamp
        self.fps = 25.0


class _FakeRegistry:
    """Answers the one question the UI needs to timestamp a chain."""

    def __init__(self, table: dict) -> None:
        self._table = table

    def resolve_frame(self, video_id: str, keyframe_ref: str):
        try:
            return self._table[(video_id, str(keyframe_ref))]
        except KeyError as exc:
            raise LookupError(f"no mapping for {video_id!r}/{keyframe_ref!r}") from exc


def _client_with_mapping(monkeypatch) -> TestClient:
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    return TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=KeyframeFrameIndexResolver(
                {("L21_V018", "007"): 100, ("L21_V018", "021"): 300}
            ),
            kis_aic_registry=_FakeRegistry(
                {
                    ("L21_V018", "007"): _FakeFrameMapping(100, 4.2),
                    ("L21_V018", "021"): _FakeFrameMapping(300, 12.6),
                }
            ),
        )
    )


_UI_TRAKE_REQUEST = {
    "task_mode": "TRAKE",
    "query": "two events",
    "trake_events": ["the first event", "the second event"],
    "top_k": 1,
}


def test_ui_search_trake_thumbnail_uses_the_keyframe_id_not_the_frame_index(
    monkeypatch,
) -> None:
    """Keyframe files are named by keyframe id -- 001.jpg, 002.jpg -- so a URL
    built from the frame index (5674.jpg) names a file that does not exist and
    404s in the browser. The KIS branch passes the keyframe id and its URLs
    resolve; TRAKE has to do the same."""
    response = _client_with_mapping(monkeypatch).post("/api/v1/search", json=_UI_TRAKE_REQUEST)

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["frame_id"] == 100
    assert result["thumbnail_url"] == "/keyframes/L21_V018/007.jpg"


def test_ui_search_trake_reports_the_real_timestamp(monkeypatch) -> None:
    """Every TRAKE chain used to report 00:00:00, because the branch passed
    timestamp=None. A reviewer scrubbing to that point lands at the start of
    the video every time."""
    response = _client_with_mapping(monkeypatch).post("/api/v1/search", json=_UI_TRAKE_REQUEST)

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["timestamp_sec"] == 4.2
    assert result["timestamp"] == "00:00:04"


def test_ui_search_trake_survives_a_frame_with_no_mapping(monkeypatch) -> None:
    """An unmapped frame must degrade to the old behaviour, not raise: TRAKE
    already tolerates per-event failures rather than aborting the batch."""
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=KeyframeFrameIndexResolver({("L21_V018", "999"): 55555}),
            kis_aic_registry=_FakeRegistry({}),
        )
    )

    response = client.post("/api/v1/search", json=_UI_TRAKE_REQUEST)

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["timestamp_sec"] == 0.0
    assert result["thumbnail_url"] == "/keyframes/L21_V018/100.jpg"


def test_ui_search_trake_drops_a_timestamp_from_a_disagreeing_mapping(monkeypatch) -> None:
    """keyframe_id comes from keyframe_mapping.csv, the timestamp from the
    organizer's map-keyframes -- two files. When they disagree about which
    frame an id sits at, the timestamp belongs to a different moment, and a
    reviewer scrubbing to it sees the wrong shot with no indication anything
    is wrong. Showing nothing is the honest answer."""
    monkeypatch.setattr(trake_module, "search_kis", _fake_search_kis)
    monkeypatch.setattr(trake_module, "build_video_score_matrix", _fake_build_video_score_matrix)
    client = TestClient(
        create_app(
            semantic_service=FakeTrakeSemanticService(),
            multimodal_service=None,
            kis_pipeline_resolver=KeyframeFrameIndexResolver({("L21_V018", "007"): 100}),
            # The registry puts keyframe 007 at frame 999, not 100.
            kis_aic_registry=_FakeRegistry({("L21_V018", "007"): _FakeFrameMapping(999, 41.5)}),
        )
    )

    response = client.post("/api/v1/search", json=_UI_TRAKE_REQUEST)

    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["timestamp_sec"] == 0.0
    assert result["timestamp"] == "00:00:00"
    # The thumbnail still resolves: the id itself is not in dispute, only
    # which frame it sits at.
    assert result["thumbnail_url"] == "/keyframes/L21_V018/007.jpg"


def test_ui_search_emits_a_video_url_when_only_the_registry_can_serve(monkeypatch) -> None:
    """/videos is no longer a bare static mount -- it falls back to the dataset
    registry -- so withholding the URL whenever MVR_VIDEO_DIR happens to be
    unset hides a link that would in fact resolve."""
    monkeypatch.delenv("MVR_VIDEO_DIR", raising=False)
    response = _client_with_mapping(monkeypatch).post("/api/v1/search", json=_UI_TRAKE_REQUEST)

    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["video_url"] == "/videos/L21_V018.mp4"

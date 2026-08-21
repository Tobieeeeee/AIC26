from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from mvr.enrichment.qna_answer import AnswerReasoner
from mvr.retrieval.kis import KeyframeFrameIndexResolver
from mvr.retrieval.qna import QnaCandidate, search_qna


def _resolver(mapping, videos=("L21_V001",)):
    """Fixtures pre-dating the pair key: one keyframe id per video.

    `KeyframeFrameIndexResolver` now keys on `(video_id, keyframe_id)`,
    because a bare keyframe id does not identify a keyframe -- the
    organizer's 166,550 keyframes carry only 632 distinct ids. These
    fixtures use synthetic ids that were meant to resolve whatever video
    asked, so registering each id under every video the test uses keeps
    exactly the behaviour they were written to check.
    """
    return KeyframeFrameIndexResolver(
        {(video, kf): idx for video in videos for kf, idx in mapping.items()}
    )




@dataclass
class _FakeSemanticResult:
    frame_id: str
    video_id: str
    score: float
    timestamp: float
    image_path: str = "/tmp/frame.jpg"

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "video_id": self.video_id,
            "score": self.score,
            "timestamp": self.timestamp,
            "image_path": self.image_path,
        }


class _FakeMultimodalService:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits
        self.calls: list[tuple[str, int, int]] = []

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int):
        self.calls.append((query, top_k, rrf_k))
        return self._hits[:top_k]


def _hit(video_id: str, keyframe_id: str, score: float, timestamp: float = 1.0) -> dict:
    return {
        "video_id": video_id,
        "frame_id": keyframe_id,
        "score": score,
        "timestamp": timestamp,
        "image_path": f"/tmp/{keyframe_id}.jpg",
    }


class _ScriptedLlmBackend:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._outputs.pop(0)


def _reasoner(answer: str, source: str) -> AnswerReasoner:
    return AnswerReasoner(_ScriptedLlmBackend([json.dumps({"answer": answer, "source": source})] * 200))


class _FakeOcrEngine:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls: list[object] = []

    def read_text(self, image: object) -> str:
        self.calls.append(image)
        return self._text


# -- validation ---------------------------------------------------------


def test_search_qna_rejects_blank_event_description() -> None:
    with pytest.raises(ValueError):
        search_qna(
            "   ", "how many?", top_k=5,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=_reasoner("1", "context-grounded"),
        )


def test_search_qna_rejects_blank_question() -> None:
    with pytest.raises(ValueError):
        search_qna(
            "an event", "   ", top_k=5,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=_reasoner("1", "context-grounded"),
        )


def test_search_qna_rejects_top_k_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        search_qna(
            "event", "question", top_k=0,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=_reasoner("1", "context-grounded"),
        )
    with pytest.raises(ValueError):
        search_qna(
            "event", "question", top_k=101,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=_reasoner("1", "context-grounded"),
        )


def test_search_qna_requires_at_least_one_retrieval_backend() -> None:
    with pytest.raises(RuntimeError):
        search_qna(
            "event", "question", top_k=5,
            semantic_service=None, multimodal_service=None,
            answer_reasoner=_reasoner("1", "context-grounded"),
        )


def test_search_qna_requires_an_answer_reasoner() -> None:
    with pytest.raises(RuntimeError):
        search_qna(
            "event", "question", top_k=5,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=None,
        )


def test_search_qna_rejects_ocr_engine_without_image_loader() -> None:
    with pytest.raises(ValueError):
        search_qna(
            "event", "question", top_k=5,
            semantic_service=None, multimodal_service=_FakeMultimodalService([]),
            answer_reasoner=_reasoner("1", "context-grounded"),
            ocr_engine=_FakeOcrEngine("hi"), image_loader=None,
        )


# -- correctness ----------------------------------------------------------


def test_search_qna_returns_answered_ranked_candidates() -> None:
    resolver = _resolver({"kf-1": 261})
    hits = [_hit("L21_V001", "kf-1", score=0.9)]
    service = _FakeMultimodalService(hits)
    reasoner = _reasoner("5", "context-grounded")

    results = search_qna(
        "an event with people", "how many people?", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=reasoner,
    )

    assert len(results) == 1
    assert results[0] == QnaCandidate(
        rank=1, video_id="L21_V001", frame_id=261, score=0.9,
        keyframe_id="kf-1", timestamp=1.0, answer="5", answer_source="context-grounded",
        image_path="/tmp/kf-1.jpg",
    )


def test_search_qna_wires_ocr_and_objects_into_context() -> None:
    resolver = _resolver({"kf-1": 261})
    hits = [_hit("L21_V001", "kf-1", score=0.9)]
    service = _FakeMultimodalService(hits)
    ocr = _FakeOcrEngine("EXIT 5B")
    loaded_images: list[str] = []

    def image_loader(path: str):
        loaded_images.append(path)
        return object()

    class _CapturingBackend:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return json.dumps({"answer": "EXIT 5B", "source": "context-grounded"})

    backend = _CapturingBackend()
    reasoner = AnswerReasoner(backend)

    results = search_qna(
        "a sign", "what does the sign say?", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=reasoner,
        ocr_engine=ocr, image_loader=image_loader,
    )

    assert results[0].answer == "EXIT 5B"
    assert loaded_images == ["/tmp/kf-1.jpg"]
    assert ocr.calls  # OCR ran on the loaded image
    assert "EXIT 5B" in backend.prompts[0]


def test_search_qna_preserves_image_path_for_human_review() -> None:
    """A human/VLM reviewing a Q&A candidate must be able to see the frame,
    so the retrieval hit's image path has to survive into the candidate."""
    resolver = _resolver({"kf-1": 261})
    service = _FakeMultimodalService([_hit("L21_V001", "kf-1", score=0.9)])

    results = search_qna(
        "event", "question", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner("1", "context-grounded"),
    )

    assert results[0].image_path == "/tmp/kf-1.jpg"


def test_search_qna_tolerates_hits_without_an_image_path() -> None:
    resolver = _resolver({"kf-1": 261})
    hit = {"video_id": "L21_V001", "frame_id": "kf-1", "score": 0.9, "timestamp": 1.0}
    service = _FakeMultimodalService([hit])

    results = search_qna(
        "event", "question", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner("1", "context-grounded"),
    )

    assert results[0].image_path is None


class _RecordingSemanticService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, top_k: int):
        self.queries.append(query)
        return []


def test_search_qna_translates_only_the_retrieval_query() -> None:
    """CLIP's text tower is English-trained, so the event description is
    translated before retrieval -- but the question must reach the answer
    reasoner untranslated, or the very detail being asked about can be
    corrupted in transit."""
    from mvr.enrichment.qna_answer import AnswerReasoner

    class _Translator:
        def translate(self, text: str) -> str:
            return "TRANSLATED: " + text

    resolver = _resolver({"kf-1": 261})
    service = _FakeMultimodalService([_hit("L21_V001", "kf-1", score=0.9)])

    class _CapturingBackend:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return json.dumps({"answer": "x", "source": "context-grounded"})

    backend = _CapturingBackend()

    search_qna(
        "một đoạn mô tả", "câu hỏi gốc", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=AnswerReasoner(backend),
        query_translator=_Translator(),
    )

    assert service.calls[0][0] == "TRANSLATED: một đoạn mô tả"
    assert "câu hỏi gốc" in backend.prompts[0]
    assert "TRANSLATED: câu hỏi gốc" not in backend.prompts[0]


def test_search_qna_without_translator_sends_the_original_query() -> None:
    service = _RecordingSemanticService()

    search_qna(
        "một đoạn mô tả", "câu hỏi", top_k=5,
        semantic_service=service, multimodal_service=None,
        answer_reasoner=_reasoner("1", "context-grounded"),
    )

    assert service.queries == ["một đoạn mô tả"]


def test_search_qna_falls_back_to_original_when_translation_fails() -> None:
    class _Broken:
        def translate(self, text: str) -> str:
            raise RuntimeError("down")

    service = _RecordingSemanticService()

    search_qna(
        "một đoạn mô tả", "câu hỏi", top_k=5,
        semantic_service=service, multimodal_service=None,
        answer_reasoner=_reasoner("1", "context-grounded"),
        query_translator=_Broken(),
    )

    assert service.queries == ["một đoạn mô tả"]


def test_search_qna_drops_unresolvable_candidates_without_guessing() -> None:
    resolver = _resolver({"kf-a": 10})  # kf-b unknown
    hits = [_hit("L21_V001", "kf-a", score=0.9), _hit("L21_V001", "kf-b", score=0.8)]
    service = _FakeMultimodalService(hits)

    results = search_qna(
        "event", "question", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner("1", "context-grounded"),
    )

    assert len(results) == 1
    assert results[0].frame_id == 10


def test_search_qna_deduplicates_same_video_and_frame_id() -> None:
    resolver = _resolver({"kf-a": 10, "kf-b": 10})
    hits = [_hit("L21_V001", "kf-a", score=0.9), _hit("L21_V001", "kf-b", score=0.8)]
    service = _FakeMultimodalService(hits)

    results = search_qna(
        "event", "question", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner("1", "context-grounded"),
    )

    assert len(results) == 1
    assert results[0].keyframe_id == "kf-a"


def test_search_qna_top_100_works() -> None:
    mapping = {f"kf-{i}": i for i in range(150)}
    resolver = _resolver(mapping)
    hits = [_hit("L21_V001", f"kf-{i}", score=1.0 - i / 1000) for i in range(150)]
    service = _FakeMultimodalService(hits)
    reasoner = _reasoner("1", "context-grounded")

    results = search_qna(
        "event", "question", top_k=100,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=reasoner, overfetch=150,
    )

    assert len(results) == 100
    assert [r.rank for r in results] == list(range(1, 101))


def test_qna_candidate_to_answer_matches_evaluator_identity() -> None:
    from mvr.evaluation.aic_qna import QnaAnswer

    candidate = QnaCandidate(
        rank=1, video_id="L21_V001", frame_id=261, score=0.9,
        keyframe_id="003", answer="5", answer_source="context-grounded",
    )

    assert candidate.to_answer() == QnaAnswer(video_id="L21_V001", frame_id=261, answer="5")


def test_search_qna_objects_path_resolver_is_wired_per_candidate() -> None:
    resolver = _resolver({"kf-1": 261})
    hits = [_hit("L21_V001", "kf-1", score=0.9)]
    service = _FakeMultimodalService(hits)
    calls: list[tuple[str, str]] = []

    def objects_path_resolver(video_id: str, keyframe_id: str):
        calls.append((video_id, keyframe_id))
        return None  # organizer file not present -- must not be guessed

    search_qna(
        "event", "question", top_k=5,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner("1", "context-grounded"),
        objects_path_resolver=objects_path_resolver,
    )

    assert calls == [("L21_V001", "kf-1")]

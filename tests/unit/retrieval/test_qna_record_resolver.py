"""Feeding `search_qna` from Task05A documents instead of live enrichment.

`frame_context_from_record` exists but nothing called it, so the reader was
unfinished work. This wires it in the same shape as `objects_path_resolver`:
an injected callable returning a record for one `(video_id, keyframe_id)`,
or `None` when there is none.

When a record is available it supersedes the live path entirely -- no
image load, no OCR call. That is the point: the enrichment already ran
offline, and redoing it at query time cost ~6.7s per candidate on CPU.

Coverage will be partial for a long time (Task05A has to be run over the
corpus first), so a missing record falls back to the live path rather than
producing an empty context. A frame with genuinely nothing on it and a
frame nobody has processed yet must not look the same.
"""
from __future__ import annotations

import json

from mvr.enrichment.qna_answer import AnswerReasoner
from mvr.retrieval.kis import KeyframeFrameIndexResolver
from mvr.retrieval.qna import search_qna


def _resolver(mapping, videos=("V", "A", "B")):
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




class _Service:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query=None):
        return self._hits[:top_k]


class _CapturingBackend:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps({"answer": "x", "source": "context-grounded"})


def _hit(keyframe_id: str = "kf-1") -> dict:
    return {
        "video_id": "V",
        "frame_id": keyframe_id,
        "score": 0.9,
        "timestamp": 1.0,
        "image_path": "/tmp/kf-1.jpg",
    }


def _record(frame_idx: int = 261) -> dict:
    return {
        "video_id": "V",
        "keyframe_id": "kf-1",
        "frame_idx": frame_idx,
        "timestamp": 10.4,
        "image_path": "/frames/V/kf-1.jpg",
        "ocr": {"texts": [], "joined": "Xã Giang Ly"},
        "asr": {"texts": [], "joined": "hai câu thơ bất hủ"},
        "objects": {"labels": ["person"], "counts": {"person": 2}},
    }


def _run(backend: _CapturingBackend, **kwargs):
    return search_qna(
        "event", "question", top_k=5,
        semantic_service=None,
        multimodal_service=_Service([_hit()]),
        pipeline_resolver=_resolver({"kf-1": 261}),
        answer_reasoner=AnswerReasoner(backend),
        **kwargs,
    )


def test_a_record_supplies_ocr_objects_and_asr() -> None:
    backend = _CapturingBackend()

    results = _run(backend, record_resolver=lambda v, k: _record())

    assert results[0].frame_id == 261
    prompt = backend.prompts[0]
    assert "Xã Giang Ly" in prompt
    assert "hai câu thơ bất hủ" in prompt
    assert "person: 2" in prompt


def test_a_record_skips_the_live_ocr_path_entirely() -> None:
    """The enrichment already ran offline; redoing it is the cost being avoided."""
    backend = _CapturingBackend()
    loaded: list[str] = []

    class _Ocr:
        def __init__(self) -> None:
            self.calls = 0

        def read_text(self, image: object) -> str:
            self.calls += 1
            return "SHOULD NOT APPEAR"

    ocr = _Ocr()

    _run(
        backend,
        record_resolver=lambda v, k: _record(),
        ocr_engine=ocr,
        image_loader=lambda path: loaded.append(path) or object(),
    )

    assert ocr.calls == 0
    assert loaded == []
    assert "SHOULD NOT APPEAR" not in backend.prompts[0]


def test_a_missing_record_falls_back_to_the_live_path() -> None:
    """Task05A coverage will be partial for a long time. A frame nobody has
    processed must not look like a frame with nothing on it."""
    backend = _CapturingBackend()

    class _Ocr:
        def read_text(self, image: object) -> str:
            return "LIVE OCR TEXT"

    _run(
        backend,
        record_resolver=lambda v, k: None,
        ocr_engine=_Ocr(),
        image_loader=lambda path: object(),
    )

    assert "LIVE OCR TEXT" in backend.prompts[0]


def test_the_record_is_looked_up_by_the_candidate_keyframe() -> None:
    asked: list[tuple[str, str]] = []

    def resolver(video_id: str, keyframe_id: str):
        asked.append((video_id, keyframe_id))
        return _record()

    _run(_CapturingBackend(), record_resolver=resolver)

    assert asked == [("V", "kf-1")]


def test_a_record_path_on_disk_is_accepted(tmp_path) -> None:
    path = tmp_path / "kf-1.json"
    path.write_text(json.dumps(_record(), ensure_ascii=False), encoding="utf-8")
    backend = _CapturingBackend()

    results = _run(backend, record_resolver=lambda v, k: path)

    assert results[0].frame_id == 261
    assert "Xã Giang Ly" in backend.prompts[0]


def test_the_record_supplies_the_image_path_for_review() -> None:
    results = _run(_CapturingBackend(), record_resolver=lambda v, k: _record())

    assert results[0].image_path == "/frames/V/kf-1.jpg"


def test_a_record_disagreeing_with_the_resolver_is_not_silently_preferred() -> None:
    """Both claim to know the competition frame_id. A mismatch is a real
    defect -- one of the two mappings is wrong -- so it must not be papered
    over by picking a winner."""
    import pytest

    with pytest.raises(ValueError):
        _run(_CapturingBackend(), record_resolver=lambda v, k: _record(frame_idx=999))

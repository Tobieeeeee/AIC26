"""Reading a Task05A document as Q&A answering context.

Until now Q&A stitched its own context together at query time: load the
keyframe JPEG, run EasyOCR on it, read the organizer objects JSON. That
cost ~6.7s per candidate on CPU -- the reason every measurement before this
ran with `--no-ocr` -- and it had no access to speech at all.

Task05A already emits one document per keyframe carrying OCR, ASR, objects
and the competition `frame_idx`. Reading it replaces all of that work with
a file read, and brings in ASR, which Q&A has never had. That matters
concretely: of the three verified queries, p1-19 asks what two lines of
poetry are, and the poetry is *recited*, not shown.

`frame_idx` coming from the document also removes `resolve_original_frame_id`
from this path -- and with it the class of bug where an unresolvable
mapping silently produced `frame_id=None` for every candidate.
"""
from __future__ import annotations

import json

import pytest

from mvr.enrichment.qna_context import FrameContext, frame_context_from_record


def _record(**overrides):
    record = {
        "schema_version": "1.0",
        "keyframe_id": "L30_V072_S0007_K01",
        "video_id": "L30_V072",
        "shot_id": 7,
        "frame_idx": 1745,
        "timestamp": 69.8,
        "image_path": "keyframes/L30_V072/L30_V072_S0007_K01.jpg",
        "ocr": {
            "texts": [
                {"text": "Xã Giang Ly", "normalized": "xa giang ly", "conf": 0.31},
                {"text": "04.08.2024", "normalized": "04.08.2024", "conf": 0.72},
            ],
            "joined": "Xã Giang Ly 04.08.2024",
        },
        "asr": {
            "texts": [{"text": "giá vàng trong nước tiếp tục tăng"}],
            "joined": "giá vàng trong nước tiếp tục tăng",
        },
        "objects": {"labels": ["person", "tv"], "counts": {"person": 2, "tv": 1}},
        "clip": {"video_id": "L30_V072", "feature_file": None, "row_idx": 123},
        "indexability": {"lexical": True, "vector": True, "object": True},
    }
    record.update(overrides)
    return record


def test_reads_identity_ocr_objects_and_asr() -> None:
    context = frame_context_from_record(_record())

    assert context.video_id == "L30_V072"
    assert context.frame_id == 1745
    assert context.ocr_text == "Xã Giang Ly 04.08.2024"
    assert context.asr_text == "giá vàng trong nước tiếp tục tăng"
    assert context.object_counts == {"person": 2, "tv": 1}


def test_the_image_path_survives_for_human_review() -> None:
    context = frame_context_from_record(_record())

    assert context.image_path == "keyframes/L30_V072/L30_V072_S0007_K01.jpg"


def test_empty_sections_are_empty_not_missing() -> None:
    context = frame_context_from_record(
        _record(
            ocr={"texts": [], "joined": ""},
            asr={"texts": [], "joined": ""},
            objects={"labels": [], "counts": {}},
        )
    )

    assert context.ocr_text == ""
    assert context.asr_text == ""
    assert context.object_counts == {}


def test_a_record_from_a_file_is_accepted(tmp_path) -> None:
    path = tmp_path / "L30_V072_S0007_K01.json"
    path.write_text(json.dumps(_record(), ensure_ascii=False), encoding="utf-8")

    assert frame_context_from_record(path).frame_id == 1745


def test_a_malformed_record_is_not_silently_accepted() -> None:
    """Same policy as the objects reader: bad data is a defect, not a blank."""
    bad = _record()
    del bad["frame_idx"]

    with pytest.raises(ValueError):
        frame_context_from_record(bad)


def test_a_negative_frame_idx_is_rejected() -> None:
    with pytest.raises(ValueError):
        frame_context_from_record(_record(frame_idx=-1))


# -- the context has to reach the reasoner ------------------------------


def test_asr_reaches_the_answer_prompt() -> None:
    """ASR is useless if it stops at the dataclass."""
    from mvr.enrichment.qna_answer import AnswerReasoner

    class _CapturingBackend:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return json.dumps({"answer": "x", "source": "context-grounded"})

    backend = _CapturingBackend()
    context = frame_context_from_record(_record())

    AnswerReasoner(backend).answer("Hai câu thơ đó là gì?", context)

    assert "giá vàng trong nước tiếp tục tăng" in backend.prompts[0]
    assert "Xã Giang Ly" in backend.prompts[0]


def test_a_context_without_asr_says_so_rather_than_going_blank() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    class _CapturingBackend:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return json.dumps({"answer": "x", "source": "context-grounded"})

    backend = _CapturingBackend()
    context = FrameContext(
        video_id="V", frame_id=1, ocr_text="TIN CHÍNH", objects=None
    )

    AnswerReasoner(backend).answer("q?", context)

    assert "TIN CHÍNH" in backend.prompts[0]
    assert "no speech" in backend.prompts[0].lower()

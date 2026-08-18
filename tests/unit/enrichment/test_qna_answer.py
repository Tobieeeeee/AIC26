import json

import pytest

from mvr.enrichment.qna_context import FrameContext


def _context(ocr_text: str = "", counts: dict[str, int] | None = None) -> FrameContext:
    from mvr.enrichment.objects.aic_objects import FrameObjects, ObjectDetection

    detections = tuple(
        ObjectDetection(label=label, confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))
        for label, count in (counts or {}).items()
        for _ in range(count)
    )
    objects = FrameObjects(video_id="L01_V001", frame_ref="1500", detections=detections) if detections else None
    return FrameContext(video_id="L01_V001", frame_id=1500, ocr_text=ocr_text, objects=objects)


class _ScriptedBackend:
    def __init__(self, outputs: list[str]) -> None:
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._outputs.pop(0)


def test_answer_parses_valid_json_output() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    backend = _ScriptedBackend([json.dumps({"answer": "5", "source": "context-grounded"})])
    reasoner = AnswerReasoner(backend)

    result = reasoner.answer("how many people?", _context(counts={"Person": 5}))

    assert result.text == "5"
    assert result.source == "context-grounded"
    assert result.raw_output == json.dumps({"answer": "5", "source": "context-grounded"})
    assert "Person: 5" in backend.prompts[0]


def test_answer_includes_ocr_text_in_prompt() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    backend = _ScriptedBackend([json.dumps({"answer": "EXIT 5B", "source": "context-grounded"})])
    reasoner = AnswerReasoner(backend)

    reasoner.answer("what does the sign say?", _context(ocr_text="EXIT 5B"))

    assert "EXIT 5B" in backend.prompts[0]


def test_answer_retries_once_on_malformed_json_then_succeeds() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    backend = _ScriptedBackend(
        [
            "not json at all",
            json.dumps({"answer": "blue", "source": "context-grounded"}),
        ]
    )
    reasoner = AnswerReasoner(backend)

    result = reasoner.answer("what color?", _context())

    assert result.text == "blue"
    assert len(backend.prompts) == 2


def test_answer_raises_after_two_malformed_attempts() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner, MalformedAnswerError

    backend = _ScriptedBackend(["not json", "still not json"])
    reasoner = AnswerReasoner(backend)

    with pytest.raises(MalformedAnswerError):
        reasoner.answer("what color?", _context())


def test_answer_rejects_invalid_source_value() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner, MalformedAnswerError

    backend = _ScriptedBackend(
        [
            json.dumps({"answer": "blue", "source": "guessed"}),
            json.dumps({"answer": "blue", "source": "guessed"}),
        ]
    )
    reasoner = AnswerReasoner(backend)

    with pytest.raises(MalformedAnswerError):
        reasoner.answer("what color?", _context())


def test_answer_rejects_blank_answer_text() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner, MalformedAnswerError

    backend = _ScriptedBackend(
        [
            json.dumps({"answer": "  ", "source": "context-grounded"}),
            json.dumps({"answer": "  ", "source": "context-grounded"}),
        ]
    )
    reasoner = AnswerReasoner(backend)

    with pytest.raises(MalformedAnswerError):
        reasoner.answer("what color?", _context())


def test_answer_rejects_blank_question() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    reasoner = AnswerReasoner(_ScriptedBackend([]))

    with pytest.raises(ValueError):
        reasoner.answer("   ", _context())


def test_answer_accepts_knowledge_grounded_source() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner

    backend = _ScriptedBackend(
        [json.dumps({"answer": "France", "source": "knowledge-grounded"})]
    )
    reasoner = AnswerReasoner(backend)

    result = reasoner.answer("which country is famous for Gaulois roosters?", _context())

    assert result.source == "knowledge-grounded"


def test_huggingface_backend_fails_closed_without_cached_weights_or_download_opt_in() -> None:
    """Real (non-mocked) verification: no network model download happens,
    no local model is loaded -- construction must refuse rather than hang
    or silently download, matching the qna extra's fail-closed contract."""
    pytest.importorskip("transformers")
    from mvr.enrichment.qna_answer import AnswerReasonerUnavailable, HuggingFaceCausalLmBackend

    with pytest.raises(AnswerReasonerUnavailable):
        HuggingFaceCausalLmBackend(
            model_name="definitely-not-a-real-org/definitely-not-a-real-model-xyz"
        )

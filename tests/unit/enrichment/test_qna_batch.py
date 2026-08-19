import json

import pytest

from mvr.retrieval.qna import QnaCandidate


def _candidate(rank: int, answer: str = "STUB", source: str = "context-grounded") -> QnaCandidate:
    return QnaCandidate(
        rank=rank,
        video_id="L21_V001",
        frame_id=100 + rank,
        score=1.0 - rank / 100,
        keyframe_id=f"kf-{rank}",
        answer=answer,
        answer_source=source,
        timestamp=float(rank),
    )


# -- CollectingBackend ------------------------------------------------------


def test_collecting_backend_records_prompts_and_returns_parseable_stub() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner
    from mvr.enrichment.qna_batch import CollectingBackend
    from mvr.enrichment.qna_context import FrameContext

    backend = CollectingBackend()
    reasoner = AnswerReasoner(backend)
    context = FrameContext(video_id="L21_V001", frame_id=100, ocr_text="hi", objects=None)

    result = reasoner.answer("what?", context)

    # The stub must parse cleanly so the pipeline completes without the
    # retry path firing -- otherwise prompt count would not match candidates.
    assert result.text
    assert len(backend.prompts) == 1
    assert "what?" in backend.prompts[0]


def test_collecting_backend_emits_exactly_one_prompt_per_call() -> None:
    from mvr.enrichment.qna_answer import AnswerReasoner
    from mvr.enrichment.qna_batch import CollectingBackend
    from mvr.enrichment.qna_context import FrameContext

    backend = CollectingBackend()
    reasoner = AnswerReasoner(backend)
    for i in range(5):
        reasoner.answer(f"q{i}", FrameContext("L21_V001", i, "", None))

    assert len(backend.prompts) == 5


# -- prompt batch round trip ------------------------------------------------


def test_prompt_batch_round_trip(tmp_path) -> None:
    from mvr.enrichment.qna_batch import read_prompt_batch, write_prompt_batch

    candidates = [_candidate(1), _candidate(2)]
    prompts = ["PROMPT-A", "PROMPT-B"]
    path = tmp_path / "prompts.json"

    write_prompt_batch(
        path,
        query_id="query-p1-19-qa",
        event_description="event",
        question="question",
        candidates=candidates,
        prompts=prompts,
    )
    batch = read_prompt_batch(path)

    assert batch.query_id == "query-p1-19-qa"
    assert batch.question == "question"
    assert [item.prompt for item in batch.items] == prompts
    assert [item.video_id for item in batch.items] == ["L21_V001", "L21_V001"]
    assert [item.frame_id for item in batch.items] == [101, 102]


def test_write_prompt_batch_rejects_length_mismatch(tmp_path) -> None:
    from mvr.enrichment.qna_batch import QnaBatchError, write_prompt_batch

    with pytest.raises(QnaBatchError):
        write_prompt_batch(
            tmp_path / "p.json",
            query_id="q",
            event_description="e",
            question="q",
            candidates=[_candidate(1), _candidate(2)],
            prompts=["only-one"],
        )


def test_read_prompt_batch_rejects_unknown_format(tmp_path) -> None:
    from mvr.enrichment.qna_batch import QnaBatchError, read_prompt_batch

    path = tmp_path / "p.json"
    path.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")

    with pytest.raises(QnaBatchError):
        read_prompt_batch(path)


# -- merging answers back ---------------------------------------------------


def _write_answers(path, items, query_id="query-p1-19-qa") -> None:
    payload = {
        "format": "mvr-qna-answer-batch-v1",
        "query_id": query_id,
        "model": "test-model",
        "items": items,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_merge_answers_into_candidates(tmp_path) -> None:
    from mvr.enrichment.qna_batch import merge_answer_batch

    path = tmp_path / "answers.json"
    _write_answers(
        path,
        [
            {"index": 0, "raw_output": json.dumps({"answer": "Kiên Giang", "source": "context-grounded"})},
            {"index": 1, "raw_output": json.dumps({"answer": "5", "source": "mixed"})},
        ],
    )

    merged = merge_answer_batch([_candidate(1), _candidate(2)], path, query_id="query-p1-19-qa")

    assert [c.answer for c in merged] == ["Kiên Giang", "5"]
    assert [c.answer_source for c in merged] == ["context-grounded", "mixed"]
    # Retrieval identity must be preserved exactly -- only the answer changes.
    assert [c.frame_id for c in merged] == [101, 102]
    assert [c.rank for c in merged] == [1, 2]
    assert [c.score for c in merged] == [0.99, 0.98]


def test_merge_rejects_query_id_mismatch(tmp_path) -> None:
    from mvr.enrichment.qna_batch import QnaBatchError, merge_answer_batch

    path = tmp_path / "answers.json"
    _write_answers(path, [{"index": 0, "raw_output": "{}"}], query_id="a-different-query")

    with pytest.raises(QnaBatchError):
        merge_answer_batch([_candidate(1)], path, query_id="query-p1-19-qa")


def test_merge_drops_candidates_whose_answer_is_malformed(tmp_path) -> None:
    """A model that returned unparseable output must not silently become a
    fabricated answer -- the candidate is dropped and ranks are reassigned."""
    from mvr.enrichment.qna_batch import merge_answer_batch

    path = tmp_path / "answers.json"
    _write_answers(
        path,
        [
            {"index": 0, "raw_output": "not json at all"},
            {"index": 1, "raw_output": json.dumps({"answer": "ok", "source": "mixed"})},
        ],
    )

    merged = merge_answer_batch([_candidate(1), _candidate(2)], path, query_id="query-p1-19-qa")

    assert len(merged) == 1
    assert merged[0].answer == "ok"
    assert merged[0].rank == 1  # ranks reassigned contiguously
    assert merged[0].frame_id == 102  # but retrieval identity preserved


def test_merge_rejects_missing_index(tmp_path) -> None:
    from mvr.enrichment.qna_batch import QnaBatchError, merge_answer_batch

    path = tmp_path / "answers.json"
    _write_answers(path, [{"index": 0, "raw_output": "{}"}])

    with pytest.raises(QnaBatchError):
        merge_answer_batch([_candidate(1), _candidate(2)], path, query_id="query-p1-19-qa")

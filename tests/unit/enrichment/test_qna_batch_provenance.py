"""Temporal-expansion provenance has to survive the GPU round-trip.

With the full 100-answer budget and expansion on, most rows in a batch are
hedges rather than retrieval hits. A batch that does not say which is which
cannot be debugged after the fact -- and the batch file is the only record,
since the GPU host never sees the retrieval run.

Old batches (written before expansion existed) must still load: several
already sit in ``artifacts/`` and re-running retrieval to regenerate them
would be pure waste.
"""
from __future__ import annotations

import json

import pytest

from mvr.enrichment.qna_batch import (
    merge_answer_batch,
    read_prompt_batch,
    write_prompt_batch,
)
from mvr.retrieval.qna import QnaCandidate


def _candidate(
    rank: int, keyframe_id: str, *, offset: int = 0, expanded_from: str | None = None
) -> QnaCandidate:
    return QnaCandidate(
        rank=rank,
        video_id="L30_V072",
        frame_id=1700 + rank,
        score=0.9,
        keyframe_id=keyframe_id,
        answer="PENDING",
        answer_source="mixed",
        temporal_offset=offset,
        expanded_from=expanded_from,
    )


def test_prompt_batch_round_trips_expansion_provenance(tmp_path) -> None:
    candidates = [
        _candidate(1, "0030"),
        _candidate(2, "0034", offset=4, expanded_from="0030"),
    ]
    path = write_prompt_batch(
        tmp_path / "b.json",
        query_id="q",
        event_description="e",
        question="q?",
        candidates=candidates,
        prompts=["p1", "p2"],
    )

    batch = read_prompt_batch(path)

    assert (batch.items[0].temporal_offset, batch.items[0].expanded_from) == (0, None)
    assert (batch.items[1].temporal_offset, batch.items[1].expanded_from) == (4, "0030")


def test_reading_a_batch_written_before_expansion_existed(tmp_path) -> None:
    """Backwards compatibility with the batches already on disk."""
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "format": "mvr-qna-prompt-batch-v1",
                "query_id": "q",
                "event_description": "e",
                "question": "q?",
                "items": [
                    {
                        "index": 0,
                        "video_id": "L30_V072",
                        "frame_id": 1745,
                        "keyframe_id": "0030",
                        "score": 0.9,
                        "timestamp": 69.8,
                        "prompt": "p",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    batch = read_prompt_batch(path)

    assert batch.items[0].temporal_offset == 0
    assert batch.items[0].expanded_from is None


def test_merging_answers_preserves_expansion_provenance(tmp_path) -> None:
    candidates = [_candidate(1, "0034", offset=4, expanded_from="0030")]
    answers = tmp_path / "a.json"
    answers.write_text(
        json.dumps(
            {
                "format": "mvr-qna-answer-batch-v1",
                "query_id": "q",
                "items": [
                    {
                        "index": 0,
                        "raw_output": json.dumps(
                            {"answer": "Giang Ly", "source": "context-grounded"}
                        ),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    merged = merge_answer_batch(candidates, answers, query_id="q")

    assert merged[0].answer == "Giang Ly"
    assert merged[0].temporal_offset == 4
    assert merged[0].expanded_from == "0030"


@pytest.mark.parametrize("offset", [-3, 0, 7])
def test_offsets_survive_in_both_directions(tmp_path, offset: int) -> None:
    path = write_prompt_batch(
        tmp_path / "b.json",
        query_id="q",
        event_description="e",
        question="q?",
        candidates=[_candidate(1, "0030", offset=offset, expanded_from="0027")],
        prompts=["p"],
    )

    assert read_prompt_batch(path).items[0].temporal_offset == offset

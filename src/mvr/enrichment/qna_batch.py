"""Offline batch Q&A answering: run retrieval/context locally, generate
answers on separate GPU hardware, merge the results back.

Motivation (recorded so the indirection is not mistaken for over-design):
the development machine has 3.8GB RAM and no GPU -- a 1.5B-parameter model
fails to load at all, and a 0.5B model produces unusable answers (verified:
it echoed the question back). Free GPU hosts (Kaggle, Colab) can run a
7B-14B model, but Kaggle's ~20-minute interactive idle timeout makes a
live inference server impractical there; its "Save & Run All" batch mode
is the natural fit.

The exchange is deliberately **text-only**: only prompts leave the local
machine, never video or keyframes. A full local run over the organizer's
data produces a few hundred KB of prompts.

Flow::

    [local]  search_qna(..., answer_reasoner=AnswerReasoner(CollectingBackend()))
               -> candidates (stub answers) + prompts, same order
             write_prompt_batch(...)              -> prompts.json
    [GPU]    run the model over each prompt       -> answers.json
    [local]  merge_answer_batch(candidates, ...)  -> candidates with real answers

Retrieval is NOT re-run on merge: answers are substituted into the
already-computed candidates, so the expensive local OCR pass happens once.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mvr.enrichment.qna_answer import MalformedAnswerError, _parse_answer

PROMPT_BATCH_FORMAT = "mvr-qna-prompt-batch-v1"
ANSWER_BATCH_FORMAT = "mvr-qna-answer-batch-v1"

#: Returned by ``CollectingBackend`` so ``AnswerReasoner`` parses it cleanly
#: and never triggers its retry path -- one prompt per candidate exactly.
_STUB_OUTPUT = json.dumps({"answer": "PENDING_BATCH_ANSWER", "source": "mixed"})


class QnaBatchError(ValueError):
    """A batch artifact is malformed or inconsistent with its candidates."""


class CollectingBackend:
    """Records prompts instead of generating; returns a parseable stub.

    Used for the local pass that produces the prompt batch. The stub answer
    is deliberately an obvious placeholder so it can never be mistaken for
    a real answer if a batch is left unmerged.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return _STUB_OUTPUT


class ReplayBackend:
    """Replays pre-generated raw outputs in order.

    Lets a full ``search_qna`` run be reproduced end-to-end without any
    model, e.g. to re-verify a batch result deterministically.
    """

    def __init__(self, raw_outputs: Sequence[str]) -> None:
        self._outputs = list(raw_outputs)
        self._position = 0

    def generate(self, prompt: str) -> str:
        if self._position >= len(self._outputs):
            raise QnaBatchError(
                f"replay exhausted after {len(self._outputs)} outputs; "
                "the batch does not match this retrieval run"
            )
        output = self._outputs[self._position]
        self._position += 1
        return output


@dataclass(frozen=True)
class PromptItem:
    index: int
    video_id: str
    frame_id: int
    keyframe_id: str
    score: float
    prompt: str
    timestamp: float | None = None
    #: Expansion provenance, carried so a batch stays debuggable after the
    #: GPU host has replaced every answer. Defaulted, so batches written
    #: before temporal expansion existed still load unchanged.
    temporal_offset: int = 0
    expanded_from: str | None = None


@dataclass(frozen=True)
class PromptBatch:
    query_id: str
    event_description: str
    question: str
    items: tuple[PromptItem, ...]


def write_prompt_batch(
    path: Path | str,
    *,
    query_id: str,
    event_description: str,
    question: str,
    candidates: Sequence[Any],
    prompts: Sequence[str],
) -> Path:
    """Serialize one query's prompts for offline generation."""
    if len(candidates) != len(prompts):
        raise QnaBatchError(
            f"candidate/prompt count mismatch: {len(candidates)} candidates, "
            f"{len(prompts)} prompts -- the collecting run did not emit exactly "
            "one prompt per candidate"
        )
    payload = {
        "format": PROMPT_BATCH_FORMAT,
        "query_id": query_id,
        "event_description": event_description,
        "question": question,
        "items": [
            {
                "index": index,
                "video_id": candidate.video_id,
                "frame_id": candidate.frame_id,
                "keyframe_id": candidate.keyframe_id,
                "score": candidate.score,
                "timestamp": candidate.timestamp,
                "temporal_offset": getattr(candidate, "temporal_offset", 0),
                "expanded_from": getattr(candidate, "expanded_from", None),
                "prompt": prompt,
            }
            for index, (candidate, prompt) in enumerate(zip(candidates, prompts, strict=True))
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_prompt_batch(path: Path | str) -> PromptBatch:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != PROMPT_BATCH_FORMAT:
        raise QnaBatchError(f"unexpected prompt batch format: {payload.get('format')!r}")
    return PromptBatch(
        query_id=payload["query_id"],
        event_description=payload["event_description"],
        question=payload["question"],
        items=tuple(
            PromptItem(
                index=row["index"],
                video_id=row["video_id"],
                frame_id=row["frame_id"],
                keyframe_id=row["keyframe_id"],
                score=row["score"],
                timestamp=row.get("timestamp"),
                temporal_offset=row.get("temporal_offset", 0),
                expanded_from=row.get("expanded_from"),
                prompt=row["prompt"],
            )
            for row in payload["items"]
        ),
    )


def merge_answer_batch(
    candidates: Sequence[Any], path: Path | str, *, query_id: str
) -> list[Any]:
    """Substitute externally-generated answers into local candidates.

    Retrieval identity (``video_id``/``frame_id``/``score``/``keyframe_id``)
    is preserved exactly -- only ``answer``/``answer_source`` change.

    A candidate whose raw output does not parse is **dropped**, never
    replaced with a fabricated answer, matching the pipeline's existing
    "drop rather than guess" policy for unresolvable frames. Remaining
    ranks are reassigned contiguously from 1.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != ANSWER_BATCH_FORMAT:
        raise QnaBatchError(f"unexpected answer batch format: {payload.get('format')!r}")
    if payload.get("query_id") != query_id:
        raise QnaBatchError(
            f"answer batch is for query {payload.get('query_id')!r}, expected {query_id!r}"
        )

    by_index = {row["index"]: row["raw_output"] for row in payload["items"]}
    missing = [i for i in range(len(candidates)) if i not in by_index]
    if missing:
        raise QnaBatchError(
            f"answer batch is missing {len(missing)} of {len(candidates)} indices "
            f"(first missing: {missing[0]})"
        )

    merged: list[Any] = []
    for index, candidate in enumerate(candidates):
        try:
            answer = _parse_answer(by_index[index])
        except MalformedAnswerError:
            continue
        merged.append(
            replace(
                candidate,
                rank=len(merged) + 1,
                answer=answer.text,
                answer_source=answer.source,
            )
        )
    return merged

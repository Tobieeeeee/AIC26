"""Temporal neighbours by position in time, not by arithmetic on an id.

`shift_keyframe_id` bumps the trailing number of the id string. That worked
on organizer keyframes, whose ids are one running sequence per video
("027" -> "028"). It breaks completely on the ids this repo's own pipeline
produces:

    L21_V010_S0007_K01
             ^^^^ ^^^
             shot  index WITHIN that shot

`make_keyframes` takes the shot midpoint and only samples extra frames when
a shot runs past `--long-shot-seconds` (default 12s). Ordinary shots are
3-8 seconds, so almost every shot yields exactly `K01` -- and `+1` asks for
`K02`, which does not exist. Expansion would return nothing, silently, for
most frames. The real next keyframe is `S0008_K01`.

`frame_idx` is the honest ordering: it is the frame's position in the
original video, it is already in every Task05A document, and it does not
care how the id happens to be spelled.

Deliberately NOT stopping at shot boundaries. It sounds right -- a
neighbour in another shot is another scene -- but the one measured case
argues against it: p1-15's answer sat 4 keyframes past the retrieval hit,
roughly 9 seconds of broadcast news, which almost certainly crosses several
shots. Stopping at the boundary would have thrown away the only hit the
system has ever scored.
"""
from __future__ import annotations

import json

from mvr.retrieval.qna import KeyframeNeighbourIndex


def _resolver(mapping, videos=("V", "A", "B")):
    """Fixtures pre-dating the pair key: one keyframe id per video.

    `KeyframeFrameIndexResolver` now keys on `(video_id, keyframe_id)`,
    because a bare keyframe id does not identify a keyframe -- the
    organizer's 166,550 keyframes carry only 632 distinct ids. These
    fixtures use synthetic ids that were meant to resolve whatever video
    asked, so registering each id under every video the test uses keeps
    exactly the behaviour they were written to check.
    """
    from mvr.retrieval.kis import KeyframeFrameIndexResolver

    return KeyframeFrameIndexResolver(
        {(video, kf): idx for video in videos for kf, idx in mapping.items()}
    )




def _record(video_id: str, keyframe_id: str, frame_idx: int) -> dict:
    return {"video_id": video_id, "keyframe_id": keyframe_id, "frame_idx": frame_idx}


def _shot_style_records() -> list[dict]:
    return [
        _record("L21_V010", "L21_V010_S0007_K01", 578),
        _record("L21_V010", "L21_V010_S0008_K01", 655),
        _record("L21_V010", "L21_V010_S0009_K01", 702),
    ]


def test_the_case_string_shifting_cannot_handle() -> None:
    index = KeyframeNeighbourIndex.from_records(_shot_style_records())

    assert index.neighbour("L21_V010", "L21_V010_S0007_K01", 1) == "L21_V010_S0008_K01"
    assert index.neighbour("L21_V010", "L21_V010_S0009_K01", -2) == "L21_V010_S0007_K01"


def test_offset_zero_is_the_keyframe_itself() -> None:
    index = KeyframeNeighbourIndex.from_records(_shot_style_records())

    assert index.neighbour("L21_V010", "L21_V010_S0008_K01", 0) == "L21_V010_S0008_K01"


def test_running_off_either_end_returns_none() -> None:
    index = KeyframeNeighbourIndex.from_records(_shot_style_records())

    assert index.neighbour("L21_V010", "L21_V010_S0007_K01", -1) is None
    assert index.neighbour("L21_V010", "L21_V010_S0009_K01", 1) is None


def test_records_are_ordered_by_frame_idx_not_by_arrival() -> None:
    shuffled = list(reversed(_shot_style_records()))
    index = KeyframeNeighbourIndex.from_records(shuffled)

    assert index.neighbour("L21_V010", "L21_V010_S0007_K01", 2) == "L21_V010_S0009_K01"


def test_videos_do_not_bleed_into_each_other() -> None:
    """The same keyframe_id in two videos must stay separate."""
    index = KeyframeNeighbourIndex.from_records(
        [
            _record("A", "001", 10),
            _record("A", "002", 20),
            _record("B", "001", 99),
        ]
    )

    assert index.neighbour("A", "001", 1) == "002"
    assert index.neighbour("B", "001", 1) is None


def test_an_unknown_video_or_keyframe_returns_none_rather_than_guessing() -> None:
    index = KeyframeNeighbourIndex.from_records(_shot_style_records())

    assert index.neighbour("NOPE", "L21_V010_S0007_K01", 1) is None
    assert index.neighbour("L21_V010", "not-a-keyframe", 1) is None


def test_organizer_style_sequential_ids_still_work() -> None:
    """The old shape must keep working -- the measured index uses it."""
    index = KeyframeNeighbourIndex.from_records(
        [_record("L30_V072", f"{n:03d}", n * 75) for n in range(27, 36)]
    )

    assert index.neighbour("L30_V072", "030", 4) == "034"


def test_built_from_a_directory_of_task05a_documents(tmp_path) -> None:
    for keyframe_id, frame_idx in (("V_S0001_K01", 10), ("V_S0002_K01", 40)):
        path = tmp_path / "V" / f"{keyframe_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"video_id": "V", "keyframe_id": keyframe_id, "frame_idx": frame_idx}
            ),
            encoding="utf-8",
        )

    index = KeyframeNeighbourIndex.from_documents(tmp_path)

    assert index.neighbour("V", "V_S0001_K01", 1) == "V_S0002_K01"


# -- wiring into search_qna ---------------------------------------------


def _search(**kwargs):
    from mvr.enrichment.qna_answer import AnswerReasoner
    from mvr.retrieval.kis import KeyframeFrameIndexResolver
    from mvr.retrieval.qna import search_qna

    class _Backend:
        def generate(self, prompt: str) -> str:
            return json.dumps({"answer": "x", "source": "context-grounded"})

    class _Service:
        def __init__(self, hits):
            self._hits = hits

        def search_hybrid(self, query, *, top_k, rrf_k):
            return self._hits[:top_k]

    hits = [
        {
            "video_id": "V",
            "frame_id": "V_S0007_K01",
            "score": 0.9,
            "timestamp": 1.0,
            "image_path": "/tmp/a.jpg",
        }
    ]
    return search_qna(
        "event",
        "question",
        top_k=10,
        semantic_service=None,
        multimodal_service=_Service(hits),
        pipeline_resolver=_resolver(
            {"V_S0007_K01": 578, "V_S0008_K01": 655, "V_S0009_K01": 702}
        ),
        answer_reasoner=AnswerReasoner(_Backend()),
        temporal_radius=2,
        **kwargs,
    )


def test_search_qna_expands_through_the_neighbour_index() -> None:
    index = KeyframeNeighbourIndex.from_records(
        [
            _record("V", "V_S0007_K01", 578),
            _record("V", "V_S0008_K01", 655),
            _record("V", "V_S0009_K01", 702),
        ]
    )

    results = _search(neighbour_index=index)

    assert [c.keyframe_id for c in results] == [
        "V_S0007_K01",
        "V_S0008_K01",
        "V_S0009_K01",
    ]
    assert [c.frame_id for c in results] == [578, 655, 702]


def test_without_the_index_shot_style_ids_expand_to_nothing() -> None:
    """The regression this class exists to prevent, stated as a test."""
    results = _search()

    assert [c.keyframe_id for c in results] == ["V_S0007_K01"]


def test_timestamp_orders_the_same_when_frame_idx_is_absent() -> None:
    """The persisted FAISS index stores timestamp but not frame_idx.

    Both increase monotonically within a video, so they induce the same
    ordering -- accepting either lets the index be built from what is
    actually on disk instead of requiring a re-encode.
    """
    index = KeyframeNeighbourIndex.from_records(
        [
            {"video_id": "V", "frame_id": "030", "timestamp": 60.8},
            {"video_id": "V", "frame_id": "034", "timestamp": 69.8},
            {"video_id": "V", "frame_id": "027", "timestamp": 55.36},
        ]
    )

    assert index.neighbour("V", "027", 2) == "034"
    assert index.neighbour("V", "034", -1) == "030"


def test_a_record_with_neither_ordering_key_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        KeyframeNeighbourIndex.from_records([{"video_id": "V", "frame_id": "001"}])

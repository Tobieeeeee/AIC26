"""Temporal expansion: when a video ranks highly, also predict its
neighbouring keyframes.

Motivated by a measured failure, not a hunch. Against the human-verified
ground truth (docs/AIC_QNA_IMPLEMENTATION_NOTES.md 3g), query p1-15
retrieved the CORRECT video at ranks #1 and #2 but keyframes 27 and 30,
where the answer lives at keyframes 34-35 -- roughly ten seconds off. The
organizer requires ``frame_id`` to fall inside the ground-truth interval,
so that near-miss scores exactly the same as a completely wrong video.
Video-level retrieval already works; frame-level precision is the failure.
"""
from __future__ import annotations

import json

import pytest

from mvr.enrichment.qna_answer import AnswerReasoner
from mvr.retrieval.kis import KeyframeFrameIndexResolver
from mvr.retrieval.qna import search_qna, shift_keyframe_id


def _resolver(mapping, videos=("V", "W", "X")):
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




class _FakeMultimodalService:
    def __init__(self, hits: list[dict]) -> None:
        self._hits = hits

    def search_hybrid(self, query: str, *, top_k: int, rrf_k: int, lexical_query=None):
        return self._hits[:top_k]


def _hit(video_id: str, keyframe_id: str, score: float) -> dict:
    return {
        "video_id": video_id,
        "frame_id": keyframe_id,
        "score": score,
        "timestamp": 1.0,
        "image_path": f"/tmp/{keyframe_id}.jpg",
    }


class _ScriptedLlmBackend:
    def generate(self, prompt: str) -> str:
        return json.dumps({"answer": "x", "source": "context-grounded"})


def _reasoner() -> AnswerReasoner:
    return AnswerReasoner(_ScriptedLlmBackend())


# -- shift_keyframe_id --------------------------------------------------
#
# Neighbour ids are derived from the hit's own id string rather than
# rebuilt from a bare number, so the zero-padding and any prefix the
# indexing layer chose survive -- a resolver keyed on the exact string
# ("0028", not "28") must still find them.


def test_shift_keyframe_id_preserves_zero_padding() -> None:
    assert shift_keyframe_id("0027", 1) == "0028"
    assert shift_keyframe_id("0027", -2) == "0025"


def test_shift_keyframe_id_preserves_a_prefix() -> None:
    assert shift_keyframe_id("L21_V001_027", 3) == "L21_V001_030"


def test_shift_keyframe_id_widens_when_the_number_outgrows_the_padding() -> None:
    assert shift_keyframe_id("099", 1) == "100"


def test_shift_keyframe_id_handles_unpadded_ids() -> None:
    assert shift_keyframe_id("kf-1", 1) == "kf-2"


def test_shift_keyframe_id_refuses_to_go_negative() -> None:
    assert shift_keyframe_id("0000", -1) is None


def test_shift_keyframe_id_returns_none_without_a_trailing_number() -> None:
    assert shift_keyframe_id("keyframe", 1) is None


def test_shift_keyframe_id_by_zero_is_identity() -> None:
    assert shift_keyframe_id("0027", 0) == "0027"


# -- validation ---------------------------------------------------------


def test_search_qna_rejects_a_negative_temporal_radius() -> None:
    with pytest.raises(ValueError):
        search_qna(
            "event", "question", top_k=5, temporal_radius=-1,
            semantic_service=None,
            multimodal_service=_FakeMultimodalService([_hit("V", "kf-1", 0.9)]),
            answer_reasoner=_reasoner(),
        )


def test_search_qna_rejects_a_non_positive_neighbour_cost() -> None:
    """A zero or negative cost would rank neighbours above the retrieval
    ranking that produced them, which is not an ordering at all."""
    with pytest.raises(ValueError):
        search_qna(
            "event", "question", top_k=5, temporal_radius=2, neighbour_cost=0.0,
            semantic_service=None,
            multimodal_service=_FakeMultimodalService([_hit("V", "kf-1", 0.9)]),
            answer_reasoner=_reasoner(),
        )


# -- behaviour ----------------------------------------------------------


def test_temporal_radius_zero_is_todays_behaviour() -> None:
    resolver = _resolver({"kf-1": 100, "kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9)])

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=0,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert [(c.video_id, c.frame_id) for c in results] == [("V", 200)]
    assert results[0].temporal_offset == 0
    assert results[0].expanded_from is None


def test_temporal_expansion_emits_resolvable_neighbours() -> None:
    resolver = _resolver({"kf-1": 100, "kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9)])

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert [c.frame_id for c in results] == [200, 300, 100]


def test_temporal_expansion_drops_neighbours_that_do_not_resolve() -> None:
    """The same 'never guess a frame_id' rule the primary hits obey: a
    keyframe the resolver has never heard of does not exist."""
    resolver = _resolver({"kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9)])

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=2,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    # kf-1, kf-0 and kf-4 are unknown -- only kf-3 joins kf-2.
    assert [c.frame_id for c in results] == [200, 300]


def test_neighbours_record_where_they_came_from() -> None:
    """A neighbour is a hedge, not a retrieval hit. It carries the parent's
    score because it has none of its own, so the provenance fields are what
    keep that from being a silent claim."""
    resolver = _resolver({"kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.87)])

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    neighbour = results[1]
    assert neighbour.temporal_offset == 1
    assert neighbour.expanded_from == "kf-2"
    assert neighbour.keyframe_id == "kf-3"
    assert neighbour.score == 0.87
    assert neighbour.timestamp is None


def test_expansion_never_demotes_the_top_retrieval_hit() -> None:
    """R@1 is one fifth of the final score; expansion may only spend slots
    the ranking was not already using."""
    resolver = _resolver({f"kf-{n}": n * 10 for n in range(1, 10)})
    service = _FakeMultimodalService([_hit("V", "kf-5", 0.9), _hit("W", "kf-2", 0.8)])

    results = search_qna(
        "event", "question", top_k=20, temporal_radius=3,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert (results[0].video_id, results[0].keyframe_id) == ("V", "kf-5")
    assert results[0].temporal_offset == 0


def test_a_neighbour_colliding_with_another_hit_is_not_duplicated() -> None:
    resolver = _resolver({"kf-1": 100, "kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-1", 0.9), _hit("V", "kf-2", 0.8)])

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    frames = [c.frame_id for c in results]
    assert sorted(frames) == [100, 200, 300]
    assert len(frames) == len(set(frames))
    # The real hit wins the slot, not the neighbour that landed on it.
    assert results[1].temporal_offset == 0


def test_a_high_neighbour_cost_places_every_retrieval_hit_first() -> None:
    resolver = _resolver({f"kf-{n}": n * 10 for n in range(1, 10)})
    service = _FakeMultimodalService(
        [_hit("V", "kf-2", 0.9), _hit("W", "kf-5", 0.8), _hit("X", "kf-8", 0.7)]
    )

    results = search_qna(
        "event", "question", top_k=12, temporal_radius=2, neighbour_cost=50.0,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert [c.temporal_offset for c in results[:3]] == [0, 0, 0]
    assert [c.video_id for c in results[:3]] == ["V", "W", "X"]
    assert all(c.temporal_offset != 0 for c in results[3:])


def test_a_low_neighbour_cost_exhausts_the_top_hit_before_the_next_one() -> None:
    resolver = _resolver({f"kf-{n}": n * 10 for n in range(1, 10)})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9), _hit("W", "kf-8", 0.8)])

    results = search_qna(
        "event", "question", top_k=12, temporal_radius=2, neighbour_cost=0.1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert [c.video_id for c in results[:4]] == ["V", "V", "V", "V"]


def test_expansion_still_honours_top_k() -> None:
    resolver = _resolver({f"kf-{n}": n * 10 for n in range(1, 30)})
    service = _FakeMultimodalService([_hit("V", f"kf-{n}", 1.0 / n) for n in (5, 15, 25)])

    results = search_qna(
        "event", "question", top_k=4, temporal_radius=4,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
    )

    assert len(results) == 4
    assert [c.rank for c in results] == [1, 2, 3, 4]


def test_neighbour_image_path_comes_from_the_injected_resolver() -> None:
    """Neighbours are not retrieval hits, so no backend supplied an image
    for them -- and OCR is the dominant Q&A signal, so a neighbour without
    its frame on disk is a far weaker prediction."""
    resolver = _resolver({"kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9)])
    asked: list[tuple[str, str]] = []

    def keyframe_path_resolver(video_id: str, keyframe_id: str) -> str | None:
        asked.append((video_id, keyframe_id))
        return f"/frames/{video_id}/{keyframe_id}.jpg"

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
        keyframe_path_resolver=keyframe_path_resolver,
    )

    assert results[0].image_path == "/tmp/kf-2.jpg"  # the hit's own, untouched
    assert results[1].image_path == "/frames/V/kf-3.jpg"
    assert ("V", "kf-3") in asked


def test_keyframe_path_resolver_backfills_a_hit_with_no_image() -> None:
    resolver = _resolver({"kf-2": 200})
    hit = {"video_id": "V", "frame_id": "kf-2", "score": 0.9, "timestamp": 1.0}
    service = _FakeMultimodalService([hit])

    results = search_qna(
        "event", "question", top_k=5, temporal_radius=0,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=_reasoner(),
        keyframe_path_resolver=lambda v, k: f"/frames/{v}/{k}.jpg",
    )

    assert results[0].image_path == "/frames/V/kf-2.jpg"


def test_expansion_reruns_the_reasoner_per_frame() -> None:
    """Each neighbour gets its own OCR/object context, so it must get its
    own answer -- copying the parent's answer would defeat the point of
    predicting a different frame."""
    resolver = _resolver({"kf-2": 200, "kf-3": 300})
    service = _FakeMultimodalService([_hit("V", "kf-2", 0.9)])

    class _CountingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str) -> str:
            self.calls += 1
            return json.dumps({"answer": f"a{self.calls}", "source": "context-grounded"})

    backend = _CountingBackend()

    results = search_qna(
        "event", "question", top_k=10, temporal_radius=1,
        semantic_service=None, multimodal_service=service,
        pipeline_resolver=resolver, answer_reasoner=AnswerReasoner(backend),
    )

    assert backend.calls == 2
    assert [c.answer for c in results] == ["a1", "a2"]

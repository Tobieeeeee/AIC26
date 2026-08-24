"""Score retrieval against the answers a person verified after round 1.

The organizer published no round-1 scores, so this set is the only way to
tell whether a round-2 change helped. It holds 17 queries whose answers the
operator confirmed by opening the frames.

**Measure at the video level.** The operator described the answers as
"đúng hoặc tương đối đúng" -- correct or nearly so -- and looked at one
frame, not at the organizer's ``[start_frame, end_frame]`` interval, which
was never published either. So the video id is knowledge; the exact frame
is one sample from an interval of unknown width. Scoring a frame against
an interval invented here would measure the invention.

Video level is also where the system actually fails: a Vietnamese query
scored 0.0000 not because the frame was off by a few hundred, but because
the right video never appeared at all.

The scoring rule itself is not reimplemented. ``evaluate_kis`` already
encodes the official R@1/5/20/50/100 and Final Score, and a video-level
ground truth is exactly that rule over an interval covering the whole
video::

    KisGroundTruth(video_id, 0, WHOLE_VIDEO)

``frame_window`` narrows that interval to ``frame_idx +/- window`` for a
stricter secondary reading. It is a **proxy** and must never be reported
as the organizer's score: the real interval is unknown, so the window is a
guess about how much slack a hit is allowed.

Aggregating across queries follows ``aic_kis``'s own warning -- the mean
below is a *mean per-query* score, not an official overall competition
score, which the organizer has not defined.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .aic_kis import KIS_CUTOFFS, KisAnswer, KisGroundTruth, evaluate_kis

#: Upper bound of the "anywhere in this video" interval. Larger than any
#: frame index a 30-minute video at 30fps can hold, by a wide margin.
WHOLE_VIDEO = 1 << 31

#: Where `scripts/build_round1_ground_truth.py` writes its output.
DEFAULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "ground_truth"
    / "round1_verified.json"
)


@dataclass(frozen=True)
class VerifiedQuery:
    """One query whose answer a person confirmed by looking at the frame."""

    name: str
    kind: str
    video_id: str
    frame_idx: int
    query_vi: str
    query_en: str | None = None
    answer: str | None = None
    frames: tuple[int, ...] = ()

    def text(self, *, language: str) -> str:
        """The query as it should be sent to the encoder.

        ``en`` falls back to the Vietnamese rather than raising: a missing
        English companion is a real state of the pack (query 16 has none),
        and refusing to score it would hide the query instead of scoring
        it honestly at what it retrieves.
        """
        if language == "vi":
            return self.query_vi
        if language == "en":
            return self.query_en or self.query_vi
        raise ValueError(f"language must be 'vi' or 'en', got {language!r}")

    def ground_truth(self, *, frame_window: int | None = None) -> KisGroundTruth:
        if frame_window is None:
            return KisGroundTruth(self.video_id, 0, WHOLE_VIDEO)
        if frame_window < 0:
            raise ValueError("frame_window must not be negative")
        return KisGroundTruth(
            self.video_id,
            max(0, self.frame_idx - frame_window),
            self.frame_idx + frame_window,
        )


@dataclass(frozen=True)
class QueryResult:
    """What one query scored, and where the right video actually landed."""

    query: VerifiedQuery
    #: 1-based rank of the first row naming the right video; None if the
    #: video is absent from the whole submission. This is the number to
    #: read when a score is 0.0 -- "rank 137" and "not found at all" are
    #: different failures with different fixes.
    video_rank: int | None
    recall_at_k: dict[int, float]
    final_score: float


@dataclass(frozen=True)
class SetResult:
    results: tuple[QueryResult, ...]

    @property
    def mean_final_score(self) -> float:
        """Mean per-query Final Score. NOT an official overall score."""
        if not self.results:
            return 0.0
        return sum(r.final_score for r in self.results) / len(self.results)

    def mean_recall_at(self, cutoff: int) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall_at_k[cutoff] for r in self.results) / len(self.results)

    @property
    def found(self) -> int:
        """How many queries retrieved the right video anywhere at all."""
        return sum(1 for r in self.results if r.video_rank is not None)


def load_verified_set(
    path: Path | str | None = None, *, kinds: Iterable[str] | None = None
) -> tuple[VerifiedQuery, ...]:
    """Read the ground truth file.

    ``kinds`` filters by track. TRAKE is included by default because its
    video id is verified like any other, but its retrieval belongs to
    another owner and a chain is not a KIS answer -- callers measuring
    this system pass ``kinds=("kis", "qa")``.
    """
    path = Path(path) if path is not None else DEFAULT_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    wanted = set(kinds) if kinds is not None else None

    queries = []
    for name, entry in payload["queries"].items():
        if wanted is not None and entry["kind"] not in wanted:
            continue
        queries.append(
            VerifiedQuery(
                name=name,
                kind=entry["kind"],
                video_id=entry["video_id"],
                frame_idx=int(entry["frame_idx"]),
                query_vi=entry["query_vi"],
                query_en=entry.get("query_en"),
                answer=entry.get("answer"),
                frames=tuple(int(f) for f in entry.get("frames", ())),
            )
        )
    return tuple(queries)


def score_query(
    query: VerifiedQuery,
    rows: Sequence[tuple[str, int]],
    *,
    frame_window: int | None = None,
) -> QueryResult:
    """Score one ranked submission for one query.

    ``rows`` is ``(video_id, frame_idx)`` in rank order, rank 1 first --
    the same shape ``answer_queries.Searcher.search`` produces.
    """
    answers = [KisAnswer(video_id, int(frame_idx)) for video_id, frame_idx in rows]
    score = evaluate_kis(answers, query.ground_truth(frame_window=frame_window))
    # Ranks past 100 cannot score, but they still say how far off the
    # system was, which is the difference between "nearly" and "never".
    video_rank = next(
        (
            rank
            for rank, answer in enumerate(answers, start=1)
            if answer.video_id == query.video_id
        ),
        None,
    )
    return QueryResult(
        query=query,
        video_rank=video_rank,
        recall_at_k=score.recall_at_k,
        final_score=score.final_score,
    )


def score_set(scored: Iterable[QueryResult]) -> SetResult:
    return SetResult(results=tuple(scored))


def format_report(result: SetResult, *, label: str = "") -> str:
    """The operator-facing table."""
    if not result.results:
        return "không có câu nào để chấm."

    lines = []
    if label:
        lines.append(label)
    lines.append(f"{'câu':<20} {'video đúng':<12} {'hạng':>6}  {'Final':>6}")
    lines.append("-" * 52)
    for item in sorted(result.results, key=lambda r: -r.final_score):
        rank = "-" if item.video_rank is None else str(item.video_rank)
        mark = "OK" if item.final_score > 0 else ("~" if item.video_rank else "x")
        lines.append(
            f"{item.query.name:<20} {item.query.video_id:<12} {rank:>6}  "
            f"{item.final_score:>6.4f}  {mark}"
        )
    lines.append("-" * 52)
    total = len(result.results)
    for cutoff in KIS_CUTOFFS:
        lines.append(f"  R@{cutoff:<4} {result.mean_recall_at(cutoff):.4f}")
    lines.append(f"  Mean Final Score  {result.mean_final_score:.4f}   (n={total})")
    lines.append(
        f"  tìm ra đúng video {result.found}/{total} câu "
        f"(kể cả ở hạng > 100, không tính điểm)"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_PATH",
    "QueryResult",
    "SetResult",
    "VerifiedQuery",
    "WHOLE_VIDEO",
    "format_report",
    "load_verified_set",
    "score_query",
    "score_set",
]

"""Official AIC TRAKE scoring rule.

Implements the documented competition R-Score / Mean-Top-k-R exactly,
matching the reference formulas in the course lab's ``trake_align.py``
(kept outside this repository; see docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md).

TRAKE correctness is deliberately NOT the same shape as KIS's
(``mvr.evaluation.aic_kis``): KIS is binary (exact video, frame inside an
inclusive interval); TRAKE gives PARTIAL credit per chain -- exact video
match, then the fraction of a chain's k markers that each individually fall
within a frame-count tolerance of their own target frame. A marker with no
resolved prediction (``None``) never counts as correct, consistent with the
"never guess" rule used throughout the KIS work.

If a future benchmark averages ``TrakeScore.final_score`` across multiple
queries, that aggregate must be labeled ``mean per-query TRAKE Final Score``
and must not be presented as the organizer's official overall competition
score unless the organizer explicitly defines that aggregation (same
caveat as ``aic_kis``).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

TRAKE_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class TrakeGroundTruth:
    """One target frame per event, in order E1..Ek.

    ``tolerance`` is a frame-COUNT tolerance (matches trake_align.py's
    ``r_score(..., tol=1)`` default), not a time duration -- it must never
    be converted to/from seconds. The real organizer value may differ from
    1; this stays configurable rather than hard-coded as guaranteed correct.
    """

    video_id: str
    frames: tuple[int, ...]
    tolerance: int = 1


@dataclass(frozen=True)
class TrakeChainPrediction:
    """One candidate chain: one predicted frame per event (``None`` =
    unresolved, never a guess), plus the alignment strategy's own score for
    that chain.

    ``score`` is opaque outside the strategy that produced it (ATLAS/Lucifer
    sum similarity minus gap penalties; U-CESE uses ``events_covered +
    0.001 * total_similarity`` -- see ``mvr.retrieval.trake_align``) and is
    used only to RANK chains before evaluation, never compared across
    strategies on an absolute scale. It plays no role in ``trake_r_score``/
    ``evaluate_trake`` correctness -- those depend only on ``frames`` vs.
    ground truth, trusting the caller's rank order.
    """

    video_id: str
    frames: tuple[int | None, ...]
    score: float


@dataclass(frozen=True)
class TrakeScore:
    recall_at_k: dict[int, float]
    final_score: float

    def to_dict(self) -> dict[str, float]:
        payload = {f"recall@{k}": v for k, v in self.recall_at_k.items()}
        payload["final_score"] = self.final_score
        return payload


def trake_r_score(prediction: TrakeChainPrediction, gt: TrakeGroundTruth) -> float:
    """R-Score TRAKE (qualifying round): wrong video = 0; correct video =
    fraction of markers within ``gt.tolerance`` frames of their target."""
    if len(prediction.frames) != len(gt.frames):
        raise ValueError(
            f"prediction has {len(prediction.frames)} markers, ground truth "
            f"has {len(gt.frames)}; a TRAKE chain must predict exactly one "
            "frame per event"
        )
    if prediction.video_id != gt.video_id:
        return 0.0
    correct = sum(
        1
        for predicted, target in zip(prediction.frames, gt.frames)
        if predicted is not None and abs(predicted - target) <= gt.tolerance
    )
    return correct / len(gt.frames)


def evaluate_trake(
    ranked_predictions: Sequence[TrakeChainPrediction], gt: TrakeGroundTruth
) -> TrakeScore:
    """Score up to 100 ranked chain predictions against one ground truth.

    ``ranked_predictions`` must already be rank-ordered (rank 1 first).
    Ranks beyond 100 are dropped before scoring so they can never affect the
    result, matching the official rule regardless of how many predictions
    the caller passes in. Each cutoff's value is the MAX (not sum/mean) of
    ``trake_r_score`` among the top-k predictions, since only the single
    best-ranked correct chain within a cutoff counts for that cutoff.
    """
    considered = list(ranked_predictions[:100])
    recall_at_k: dict[int, float] = {}
    for cutoff in TRAKE_CUTOFFS:
        top = considered[:cutoff]
        recall_at_k[cutoff] = max(
            (trake_r_score(prediction, gt) for prediction in top), default=0.0
        )
    final_score = sum(recall_at_k[cutoff] for cutoff in TRAKE_CUTOFFS) / len(TRAKE_CUTOFFS)
    return TrakeScore(recall_at_k=recall_at_k, final_score=final_score)


def trake_final_round_score(
    t_seconds: float, wrong_markers: int, *, t_limit: float = 300.0
) -> float:
    """DRES-style live final-round score (matches trake_align.py's
    ``final_score(t_sec, wrong, t_limit=300.0)``).

    This is a human submission-timing formula for one already-judged chain
    in the live final round -- NOT a per-chain correctness measure (use
    ``trake_r_score``/``evaluate_trake`` for that). Faster submission and
    fewer wrong markers score higher; floored at 0, never negative.
    """
    return max(0.0, 50.0 + 50.0 * (1.0 - t_seconds / t_limit) - 10.0 * wrong_markers)

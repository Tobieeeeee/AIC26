"""Official AIC Textual KIS scoring rule.

Implements the documented competition R-Score / R@k / Final Score exactly.
This is intentionally NOT Recall@K, MRR, or mAP over an ID set: KIS
correctness is defined by an exact video match plus original-video frame
falling inside an inclusive ``[start_frame, end_frame]`` ground-truth
interval, not by set membership.

If a future benchmark (Task 06) averages this Final Score across multiple
queries, that aggregate must be labeled ``mean per-query KIS Final Score`` and
must not be presented as the organizer's official overall competition score
unless the organizer explicitly defines that aggregation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

KIS_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class KisGroundTruth:
    video_id: str
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class KisAnswer:
    video_id: str
    frame_id: int


@dataclass(frozen=True)
class KisScore:
    recall_at_k: dict[int, float]
    final_score: float

    def to_dict(self) -> dict[str, float]:
        payload = {f"recall@{k}": v for k, v in self.recall_at_k.items()}
        payload["final_score"] = self.final_score
        return payload


def kis_r_score(answer: KisAnswer, gt: KisGroundTruth) -> float:
    if answer.video_id != gt.video_id:
        return 0.0
    if gt.start_frame <= answer.frame_id <= gt.end_frame:
        return 1.0
    return 0.0


def evaluate_kis(answers: Sequence[KisAnswer], gt: KisGroundTruth) -> KisScore:
    """Score up to 100 ranked answers against one ground-truth interval.

    ``answers`` must already be rank-ordered (rank 1 first). Ranks beyond 100
    are dropped before scoring so they can never affect the result, matching
    the official rule regardless of how many answers the caller passes in.
    """
    considered = list(answers[:100])
    recall_at_k: dict[int, float] = {}
    for cutoff in KIS_CUTOFFS:
        top = considered[:cutoff]
        recall_at_k[cutoff] = max((kis_r_score(a, gt) for a in top), default=0.0)
    final_score = sum(recall_at_k[cutoff] for cutoff in KIS_CUTOFFS) / len(KIS_CUTOFFS)
    return KisScore(recall_at_k=recall_at_k, final_score=final_score)

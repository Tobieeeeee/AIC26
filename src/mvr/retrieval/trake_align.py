"""TRAKE chain alignment: given a dense per-video score matrix and
inter-event gap constraints, produce ranked candidate chains for that video.

Three interchangeable strategies port the course lab's three real AIC 2025
team algorithms exactly: ATLAS (``dp_align``), Lucifer (``dp_align_soft``),
U-CESE (``two_pointer_clips``). The course lab's own ``trake_align.py`` is
kept outside this repository (course material, not ours to redistribute);
every fact this module relies on from it is restated in
docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md and
docs/AIC_TRAKE_BENCHMARK_RESULTS.md, so no local copy is needed to read,
maintain, or verify this code.

Each strategy operates on ONE video's ``TrakeVideoScoreMatrix`` at a time --
video shortlisting already happened via ``search_kis()`` in orchestration
(Phase 05), so the lab reference's own multi-video scanning helpers
(``video_spans()``, ``candidate_videos()``) are not needed here; only their
per-video-span core DP/window logic is ported. The lab's ``pre_topk``
parameter (which candidate_videos() used to shortlist videos) has no
equivalent in these per-video strategies -- its role is played by Phase 05
orchestration's own ``shortlist_top_k`` parameter instead.

``TrakeChainPrediction`` is defined in ``mvr.evaluation.trake``. This module
never imports it at module level (only inside ``TYPE_CHECKING`` for
annotations, and lazily inside each function that constructs one) so the
retrieval layer does not depend on the evaluation layer at import time --
mirrors ``mvr.retrieval.kis.KisCandidate.to_answer()``'s own lazy import of
``KisAnswer``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .trake_events import TrakeGapConstraint
from .trake_scoring import TrakeVideoScoreMatrix

if TYPE_CHECKING:
    from mvr.evaluation.trake import TrakeChainPrediction


class TrakeAlignmentStrategy(Protocol):
    def align(
        self, matrix: TrakeVideoScoreMatrix, gaps: Sequence[TrakeGapConstraint]
    ) -> "list[TrakeChainPrediction]": ...


def _penalty(
    dt: np.ndarray, gap: tuple[float, float] | None, alpha: float, beta: float
) -> np.ndarray:
    """Two-sided penalty (ATLAS/Lucifer): too early is penalized, too late is
    penalized. Ported unchanged from trake_align.py's ``_penalty()``."""
    if gap is None:
        return np.zeros_like(dt)
    lo, hi = gap
    return alpha * np.maximum(0.0, dt - hi) ** 2 + beta * np.maximum(0.0, lo - dt) ** 2


def first_touch(
    scores: Sequence[float], after: int, slack: float = 0.05, floor: float = 0.15
) -> int:
    """Earliest index after ``after`` whose score is within ``slack`` of the
    max remaining score (and at least ``floor``) -- the "pick the first
    plausible frame, not necessarily the peak" rule (lecture slide 19/42).

    Ported unchanged from trake_align.py's ``first_touch()``. NOTE: the lab
    reference itself does not call this from ``two_pointer_clips()`` (or any
    other alignment function) -- it is a standalone utility, exposed here
    for the same purpose, not wired into any strategy below.
    """
    values = np.asarray(scores, dtype="float64")
    candidates = np.arange(len(values))
    threshold = max(values[after + 1 :].max() - slack, floor)
    mask = (candidates > after) & (values >= threshold)
    matches = np.flatnonzero(mask)
    return int(matches[0]) if len(matches) else int(values[after + 1 :].argmax() + after + 1)


class AtlasAlignment:
    """ATLAS: DP with two-sided [min,max] gap penalty, exact order enforced.
    Ported from trake_align.py's ``dp_align()``. A video with no valid
    full-length chain produces an empty result (never a guess).
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.01,
        min_sim: float = 0.0,
        top_n: int = 5,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.min_sim = min_sim
        self.top_n = top_n

    def align(
        self, matrix: TrakeVideoScoreMatrix, gaps: Sequence[TrakeGapConstraint]
    ) -> "list[TrakeChainPrediction]":
        from mvr.evaluation.trake import TrakeChainPrediction

        n_events, n_frames = matrix.scores.shape
        if n_frames < n_events:
            return []

        scores = matrix.scores.astype("float64")
        timestamps = np.asarray(matrix.timestamps, dtype="float64")
        ok = scores >= self.min_sim
        neg = -1e18

        dp = np.where(ok[0], scores[0], neg)
        back = np.zeros((n_events, n_frames), dtype=np.int32)
        dt = timestamps[:, None] - timestamps[None, :]
        order = dt > 0
        for event_index in range(1, n_events):
            gap = gaps[event_index - 1] if gaps else None
            gap_bounds = (gap.min_gap_seconds, gap.max_gap_seconds) if gap is not None else None
            trans = np.where(
                order, dp[None, :] - _penalty(dt, gap_bounds, self.alpha, self.beta), neg
            )
            best_prev = trans.max(axis=1)
            back[event_index] = trans.argmax(axis=1)
            dp = np.where(ok[event_index], scores[event_index] + best_prev, neg)

        # N-best by final-event column: the lab reference keeps only argmax
        # (one chain per video), which wastes the competition's 100-prediction
        # budget and gives the correct video exactly one attempt. Backtracking
        # from each of the top-N end columns yields N distinct chains (they
        # differ at least in their final frame), best first.
        predictions: list[TrakeChainPrediction] = []
        for best_column in np.argsort(-dp)[: self.top_n]:
            end_column = int(best_column)
            if dp[end_column] <= neg / 2:
                continue
            chain_score = float(dp[end_column])
            hits = [0] * n_events
            hits[n_events - 1] = end_column
            column = end_column
            for event_index in range(n_events - 1, 0, -1):
                column = int(back[event_index, column])
                hits[event_index - 1] = column
            frames = tuple(matrix.original_frame_ids[c] for c in hits)
            predictions.append(
                TrakeChainPrediction(
                    video_id=matrix.video_id, frames=frames, score=chain_score
                )
            )
        return predictions


class LuciferAlignment:
    """Lucifer: same DP shape as ATLAS but allows skipping an event with no
    viable candidate (``min_sim`` floor) instead of failing the whole video --
    a skipped event's frame is ``None`` (never a guess), and the chain score
    is multiplied by ``lam`` per skipped event. Ported from trake_align.py's
    ``dp_align_soft()``.

    Two deliberate fixes vs. the lab reference (both recorded in
    ``docs/LAB_REUSE_MAP.md`` per AGENTS.md 2.4/18):

    1. **Consistent ``lam``.** The lab's internal state-selection key
       (``_soft_key``) hard-codes the multiplier ``0.8`` regardless of the
       ``lam`` argument passed to ``dp_align_soft()``, while the FINAL score
       correctly uses the passed ``lam`` -- an internal inconsistency that is
       invisible at the lab's own default ``lam=0.8`` (which is why the lab's
       self-check never surfaces it) but would silently misrank DP states for
       any other configured ``lam``. This port threads ``lam`` through the
       state key consistently. Verified NOT to change behavior at the default
       ``lam=0.8`` (see the reference self-check test).

    2. **Accumulated gaps across skipped events.** The lab indexes the gap
       constraint by the CURRENT event (``gaps[j - 1]``) even when the
       preceding events were skipped, so the surviving time span is judged
       against only the last hop's budget. Because a skipped event still
       consumes real elapsed time, this over-penalizes exactly the case this
       strategy exists to handle. Measured example: with
       ``gaps = ([0,10], [0,10])``, E1 invisible, E0 at t=0 and E2 at t=18
       (inside the combined 20s budget), the lab formula charges
       ``(18 - 10)^2`` and the DP then drops the perfectly findable E0
       entirely, returning ``(None, None, f2)`` instead of ``(f0, None, f2)``.
       This port sums the bounds of every gap spanned since the last MATCHED
       event, so the constraint matches the real elapsed interval. The lab's
       own self-check passes gaps=None and therefore never exercises this.

    3. **``min_sim`` default raised from ``0.05`` to ``0.20``.** Measured real
       CLIP cosine scores on the AIC L21 index span roughly -0.05..0.31, with
       correct target frames scoring 0.21..0.31, so a 0.05 floor sits below
       essentially every frame and the skip mechanism -- this strategy's
       entire reason to exist -- can never trigger. At 0.05 this class
       produced output byte-identical to ``AtlasAlignment`` on all five real
       benchmark queries, i.e. it was not being measured at all.

    **WARNING -- ``min_sim`` is a cliff, not a plateau.** Measured on the real
    benchmark: 0.05 -> 0.660, 0.15 -> 0.660 (both identical to ATLAS: no skip
    ever fires), 0.20 -> 0.670, 0.22 -> 0.530, 0.25 -> 0.473, 0.28 -> 0.403.
    The single good value sits immediately next to a sharp drop, and it was
    tuned on only five queries, so treat 0.20 as fragile rather than
    validated. This sensitivity is the main reason ``AtlasAlignment`` -- whose
    own ``top_n`` plateau is flat from 5 to 20 -- is the recommended
    production default despite Lucifer's marginally higher peak. See
    docs/AIC_TRAKE_BENCHMARK_RESULTS.md.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 0.01,
        lam: float = 0.8,
        min_sim: float = 0.20,
        top_n: int = 5,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.min_sim = min_sim
        self.top_n = top_n

    def _state_key(self, total: float, skipped: int) -> float:
        return total * (self.lam**skipped)

    @staticmethod
    def _accumulated_gap(
        gaps: Sequence[TrakeGapConstraint], from_event: int, to_event: int
    ) -> tuple[float, float] | None:
        """Combined [min,max] budget spanning every event step from the last
        MATCHED event to the one now being matched.

        With no skipped events in between this is exactly ``gaps[to_event-1]``,
        identical to the lab reference. With ``s`` skipped events in between it
        sums the ``s+1`` gaps actually traversed, because a skipped event still
        consumes real elapsed time (see the class docstring)."""
        if not gaps or from_event < 0 or to_event <= from_event:
            return None
        spanned = gaps[from_event:to_event]
        return (
            sum(gap.min_gap_seconds for gap in spanned),
            sum(gap.max_gap_seconds for gap in spanned),
        )

    def align(
        self, matrix: TrakeVideoScoreMatrix, gaps: Sequence[TrakeGapConstraint]
    ) -> "list[TrakeChainPrediction]":
        from mvr.evaluation.trake import TrakeChainPrediction

        n_events, n_frames = matrix.scores.shape
        scores = matrix.scores.astype("float64")
        timestamps = np.asarray(matrix.timestamps, dtype="float64")

        # states: {last_column_or_-1: (total_score, skip_count, trail, last_matched_event)}
        # ``last_matched_event`` (-1 = nothing matched yet) is what lets the gap
        # budget span skipped events correctly; the lab reference tracks only
        # the column and therefore loses this.
        states: dict[int, tuple[float, int, list[int | None], int]] = {-1: (0.0, 0, [], -1)}
        for event_index in range(n_events):
            next_states: dict[int, tuple[float, int, list[int | None], int]] = {}
            for last_column, (total, skipped, trail, last_event) in states.items():
                # option 1: skip this event (last_event is unchanged -- the gap
                # budget keeps accumulating across this skipped step)
                skip_candidate = (total, skipped + 1, trail + [None], last_event)
                skip_key = last_column
                if skip_key not in next_states or self._state_key(
                    *skip_candidate[:2]
                ) > self._state_key(*next_states[skip_key][:2]):
                    next_states[skip_key] = skip_candidate

                # option 2: match this event to some later frame
                for column in range(n_frames):
                    if scores[event_index, column] < self.min_sim:
                        continue
                    if last_column >= 0:
                        if timestamps[column] <= timestamps[last_column]:
                            continue
                        gap_bounds = self._accumulated_gap(gaps, last_event, event_index)
                        dt = np.array([[timestamps[column] - timestamps[last_column]]])
                        penalty = float(
                            _penalty(dt, gap_bounds, self.alpha, self.beta)[0, 0]
                        )
                    else:
                        penalty = 0.0
                    match_candidate = (
                        total + scores[event_index, column] - penalty,
                        skipped,
                        trail + [column],
                        event_index,
                    )
                    if column not in next_states or self._state_key(
                        *match_candidate[:2]
                    ) > self._state_key(*next_states[column][:2]):
                        next_states[column] = match_candidate
            states = next_states

        # N-best over surviving DP states (see AtlasAlignment for why one
        # chain per video is not enough). States that skipped every event are
        # dropped, as in the lab reference.
        ranked_states = sorted(
            states.values(), key=lambda state: self._state_key(state[0], state[1]), reverse=True
        )
        predictions: list[TrakeChainPrediction] = []
        for total, skipped, trail, _ in ranked_states:
            if skipped == n_events:
                continue
            frames = tuple(
                None if column is None else matrix.original_frame_ids[column] for column in trail
            )
            predictions.append(
                TrakeChainPrediction(
                    video_id=matrix.video_id,
                    frames=frames,
                    score=total * (self.lam**skipped),
                )
            )
            if len(predictions) >= self.top_n:
                break
        return predictions


class UceseAlignment:
    """U-CESE / Nomial: sliding time-window two-pointer scan. Takes the
    FIRST frame within each window that clears an event's per-event top-k
    threshold, ranks windows by (events_covered, total_score). Does NOT
    enforce chronological order among events within a window -- deliberate,
    per the lab reference's own commentary: fast, for generating
    manual-review suggestions, not the primary scored strategy. Ported from
    trake_align.py's ``two_pointer_clips()``.

    An event never covered by the winning window is left ``None`` (never a
    guess).

    ``topk`` default lowered from the lab's ``300`` to ``5``: in the lab,
    ``topk`` selected an event's strongest frames across the WHOLE corpus,
    but this port is per-video (see the module docstring), where a typical
    AIC video contributes only ~200-320 indexed frames. At ``topk=300``
    essentially every frame of the video qualifies as a "hit" for every
    event, so the first-touch rule collapses to "earliest frame in the
    window" and the strategy returns the SAME frame for all k events --
    measured on real data: ``(8820, 8820, 8820, 8820)``, scoring 0.000,
    while a small ``topk`` on the identical input scores 0.500. The lab's own
    value is therefore not transferable to per-video scope. The default 10 is
    the measured best across the real benchmark (topk 1 -> 0.190, 3 -> 0.300,
    5 -> 0.380, 10 -> 0.413, 20 -> 0.253). See
    docs/AIC_TRAKE_BENCHMARK_RESULTS.md.
    """

    def __init__(self, t_win: float = 20.0, topk: int = 10, top_n: int = 5) -> None:
        self.t_win = t_win
        self.topk = topk
        self.top_n = top_n

    def align(
        self, matrix: TrakeVideoScoreMatrix, gaps: Sequence[TrakeGapConstraint]
    ) -> "list[TrakeChainPrediction]":
        from mvr.evaluation.trake import TrakeChainPrediction

        n_events = matrix.scores.shape[0]
        timestamps = matrix.timestamps

        hits_by_frame: dict[int, list[int]] = {}
        for event_index in range(n_events):
            row = matrix.scores[event_index]
            top_columns = np.argsort(-row)[: self.topk]
            for column in top_columns:
                hits_by_frame.setdefault(int(column), []).append(event_index)

        # Column indices double as time order since matrix columns are
        # already sorted by timestamp ascending (build_video_score_matrix's
        # own guarantee).
        frames_with_hits = sorted(hits_by_frame.keys())
        if not frames_with_hits:
            return []

        # Collect every window's chain, then keep the N best distinct ones
        # (adjacent windows frequently yield an identical first-touch chain,
        # so dedupe before truncating -- see AtlasAlignment for why one chain
        # per video is not enough).
        seen: set[tuple[int | None, ...]] = set()
        candidates: list[tuple[float, tuple[int | None, ...]]] = []
        low = 0
        for high in range(len(frames_with_hits)):
            while timestamps[frames_with_hits[high]] - timestamps[frames_with_hits[low]] > self.t_win:
                low += 1
            window = frames_with_hits[low : high + 1]
            first_hit: dict[int, int] = {}
            for column in window:
                for event_index in hits_by_frame[column]:
                    if event_index not in first_hit:
                        first_hit[event_index] = column
            covered = len(first_hit)
            total = sum(matrix.scores[event_index, column] for event_index, column in first_hit.items())
            frames = tuple(
                None
                if event_index not in first_hit
                else matrix.original_frame_ids[first_hit[event_index]]
                for event_index in range(n_events)
            )
            if frames in seen:
                continue
            seen.add(frames)
            candidates.append((covered + 0.001 * total, frames))

        # Tie-break deterministically by frame tuple, mapping an unresolved
        # (None) marker to -1 so mixed None/int tuples stay comparable --
        # sorting raw tuples raises TypeError as soon as two tied chains
        # differ at a position where one skipped the event.
        candidates.sort(
            key=lambda item: (
                -item[0],
                tuple(-1 if frame is None else frame for frame in item[1]),
            )
        )
        return [
            TrakeChainPrediction(video_id=matrix.video_id, frames=frames, score=score)
            for score, frames in candidates[: self.top_n]
        ]

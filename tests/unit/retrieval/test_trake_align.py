import numpy as np
import pytest

from mvr.retrieval.trake_align import (
    AtlasAlignment,
    LuciferAlignment,
    UceseAlignment,
    first_touch,
)
from mvr.retrieval.trake_events import TrakeGapConstraint
from mvr.retrieval.trake_scoring import TrakeVideoScoreMatrix


def _matrix(video_id: str, scores: np.ndarray, timestamps: tuple, frame_id_offset: int):
    n_frames = scores.shape[1]
    frame_ids = tuple(frame_id_offset + i for i in range(n_frames))
    return TrakeVideoScoreMatrix(
        video_id=video_id,
        timestamps=timestamps,
        original_frame_ids=frame_ids,
        scores=scores,
    )


# -- fixtures reproducing trake_align.py's own __main__ self-check ----------
# (S0/vid0/ts0, split into one per-video matrix since align() operates
# per-video -- see trake_align.py's module docstring for why).

_TS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)


def _self_check_matrices() -> dict[str, TrakeVideoScoreMatrix]:
    s0 = np.full((3, 6), 0.01)
    s0[0, 1], s0[1, 3], s0[2, 5] = 0.8, 0.7, 0.6  # video 0: valid order, total 2.1

    s1 = np.full((3, 6), 0.01)
    s1[0, 3], s1[1, 1], s1[2, 5] = 0.95, 0.9, 0.9  # video 1: E2 peak precedes E1 peak in time

    s2 = np.full((3, 6), 0.01)
    s2[0, 0], s2[1, 4], s2[2, 5] = 0.85, 0.85, 0.75  # video 2: valid order, total 2.45, 8s first gap

    return {
        "0": _matrix("0", s0, _TS, frame_id_offset=1000),
        "1": _matrix("1", s1, _TS, frame_id_offset=2000),
        "2": _matrix("2", s2, _TS, frame_id_offset=3000),
    }


# -- ATLAS --------------------------------------------------------------------


def test_atlas_order_only_reproduces_lab_reference_winner_and_video0_chain() -> None:
    """Reproduces: dp_align(S0, vid0, ts0); top[0][1]==2 and top[0][2]==(12,16,17);
    dict(...)[0]==(1,3,5)."""
    matrices = _self_check_matrices()
    strategy = AtlasAlignment()

    result0 = strategy.align(matrices["0"], gaps=())
    result1 = strategy.align(matrices["1"], gaps=())
    result2 = strategy.align(matrices["2"], gaps=())

    assert len(result0) == len(result1) == len(result2) == 1
    # video 0's own valid chain, exact score (no penalty applied, gaps=()).
    assert result0[0].frames == (1001, 1003, 1005)
    assert result0[0].score == pytest.approx(2.1)
    # video 2 is the overall winner: highest order-respecting total.
    assert result2[0].frames == (3000, 3004, 3005)
    assert result2[0].score == pytest.approx(2.45)
    assert result2[0].score > result0[0].score > result1[0].score


def test_atlas_tight_max_gap_penalizes_video2_more_than_video0() -> None:
    """Reproduces: gaps=[(0,4),(0,4)], alpha=beta=0.05 -> video 0 wins
    (top[0][1]==0, top[0][2]==(1,3,5)) instead of video 2."""
    matrices = _self_check_matrices()
    strategy = AtlasAlignment(alpha=0.05, beta=0.05)
    gaps = (TrakeGapConstraint(0.0, 4.0), TrakeGapConstraint(0.0, 4.0))

    result0 = strategy.align(matrices["0"], gaps=gaps)
    result2 = strategy.align(matrices["2"], gaps=gaps)

    # video 0's real gaps (4s, 4s) sit exactly at the cap -> zero penalty.
    assert result0[0].frames == (1001, 1003, 1005)
    assert result0[0].score == pytest.approx(2.1)
    # video 2's first real gap is 8s, doubling the 4s cap -> heavily penalized,
    # dropping it below video 0 (the lab reference's own conclusion).
    assert result0[0].score > result2[0].score


def test_atlas_min_gap_penalizes_video0_more_than_video2() -> None:
    """Reproduces: gaps=[(5,20),(0,20)], alpha=beta=0.05 -> video 2 wins
    (top[0][1]==2)."""
    matrices = _self_check_matrices()
    strategy = AtlasAlignment(alpha=0.05, beta=0.05)
    gaps = (TrakeGapConstraint(5.0, 20.0), TrakeGapConstraint(0.0, 20.0))

    result0 = strategy.align(matrices["0"], gaps=gaps)
    result2 = strategy.align(matrices["2"], gaps=gaps)

    # video 0's real first gap is 4s, below the 5s minimum -> penalized.
    # video 2's gaps (8s, 2s) both clear their bounds -> unpenalized 2.45.
    assert result2[0].score == pytest.approx(2.45)
    assert result2[0].score > result0[0].score


def test_atlas_returns_empty_when_video_too_short_for_event_count() -> None:
    # 3 events requested against a video with only 2 frames -> impossible.
    short_matrix = _matrix("v", np.full((3, 2), 0.9), (0.0, 1.0), frame_id_offset=0)

    assert AtlasAlignment().align(short_matrix, gaps=()) == []


# -- Lucifer (invisible-event scenario) ---------------------------------------


def test_lucifer_skips_invisible_event_and_matches_lab_reference_score() -> None:
    """Reproduces: S1 = S0.copy(); S1[1] = 0.01 (event index 1 vanishes from
    every video); dp_align(..., min_sim=0.05) == [] for all videos;
    dp_align_soft(..., min_sim=0.05): winning video is video 1, E2 (index 1)
    is None, score == 1.85 * 0.8."""
    matrices = _self_check_matrices()
    for matrix in matrices.values():
        matrix.scores[1, :] = 0.01  # event index 1 (E2) invisible everywhere

    atlas = AtlasAlignment(min_sim=0.05)
    for video_id, matrix in matrices.items():
        assert atlas.align(matrix, gaps=()) == [], f"video {video_id} should have no hard chain"

    lucifer = LuciferAlignment(min_sim=0.05)
    result0 = lucifer.align(matrices["0"], gaps=())
    result1 = lucifer.align(matrices["1"], gaps=())
    result2 = lucifer.align(matrices["2"], gaps=())

    assert result1[0].frames == (2003, None, 2005)
    assert result1[0].score == pytest.approx(1.85 * 0.8)
    # video 1 (0.95 + 0.9 peaks) beats video 0 (0.8 + 0.6) and video 2 (0.85 + 0.75).
    assert result1[0].score > result0[0].score
    assert result1[0].score > result2[0].score


def test_lucifer_soft_key_fix_does_not_change_default_lambda_behavior() -> None:
    """The deliberate fix to LuciferAlignment's internal state key (threading
    `lam` through instead of the lab's hard-coded 0.8) must be invisible at
    the lab's own default lam=0.8 -- verified by the exact score match above
    already; this test pins that guarantee explicitly for lam=0.8."""
    matrices = _self_check_matrices()
    matrices["1"].scores[1, :] = 0.01

    default_lambda = LuciferAlignment(min_sim=0.05)
    explicit_default = LuciferAlignment(min_sim=0.05, lam=0.8)

    result_a = default_lambda.align(matrices["1"], gaps=())
    result_b = explicit_default.align(matrices["1"], gaps=())

    assert result_a[0].score == pytest.approx(result_b[0].score)
    assert result_a[0].score == pytest.approx(1.85 * 0.8)


def test_lucifer_accumulates_gap_budget_across_a_skipped_event() -> None:
    """Regression for a real defect inherited from the lab reference: with
    E1 invisible, gaps ([0,10], [0,10]), E0 at t=0 and E2 at t=18, the real
    E0->E2 span (18s) fits the COMBINED 20s budget. Indexing the gap by the
    current event (the lab's ``gaps[j-1]``) charges (18-10)^2 instead, and
    the DP then drops the perfectly findable E0 entirely -- returning
    (None, None, f2) rather than (f0, None, f2)."""
    scores = np.zeros((3, 3))
    scores[0, 0] = 0.9  # E0 visible at t=0
    scores[1, :] = 0.0  # E1 invisible everywhere
    scores[2, 2] = 0.9  # E2 visible at t=18
    matrix = _matrix("v", scores, (0.0, 9.0, 18.0), frame_id_offset=100)
    gaps = (TrakeGapConstraint(0.0, 10.0), TrakeGapConstraint(0.0, 10.0))

    result = LuciferAlignment(min_sim=0.05, alpha=1.0, beta=1.0).align(matrix, gaps=gaps)

    assert result[0].frames == (100, None, 102)
    # No penalty: 18s is inside the combined [0, 20] budget. One skip -> * lam.
    assert result[0].score == pytest.approx((0.9 + 0.9) * 0.8)


def test_lucifer_accumulated_gap_equals_single_gap_when_nothing_is_skipped() -> None:
    """The accumulation fix must be a no-op when no event is skipped -- the
    budget for E0->E1 is then exactly gaps[0], same as the lab reference."""
    scores = np.array([[0.9, 0.01, 0.01], [0.01, 0.9, 0.01]])
    matrix = _matrix("v", scores, (0.0, 8.0, 20.0), frame_id_offset=0)
    gaps = (TrakeGapConstraint(0.0, 10.0),)

    result = LuciferAlignment(min_sim=0.05, alpha=1.0, beta=1.0).align(matrix, gaps=gaps)

    assert result[0].frames == (0, 1)  # 8s span, inside [0,10] -> unpenalized
    assert result[0].score == pytest.approx(1.8)


def test_lucifer_returns_empty_when_every_event_must_be_skipped() -> None:
    scores = np.full((2, 3), 0.01)  # nothing clears any reasonable min_sim
    matrix = _matrix("v", scores, (0.0, 1.0, 2.0), frame_id_offset=0)
    strategy = LuciferAlignment(min_sim=0.05)

    assert strategy.align(matrix, gaps=()) == []


# -- U-CESE ---------------------------------------------------------------------


def test_ucese_produces_out_of_order_winning_chain() -> None:
    """Qualitatively reproduces trake_align.py's own two_pointer_clips
    self-check claim (a chain whose events are NOT in time order can still
    win, because the strategy deliberately does not enforce order) --
    NOT a numeric reproduction of the lab's own tp[0] values: the lab's
    `topk` is applied GLOBALLY across all shortlisted videos in one call,
    while this port's `topk` is scoped to one video at a time (per Phase 04's
    per-video redesign, since video shortlisting now happens in Phase 05's
    search_kis() calls, not inside alignment). topk=1 here isolates each
    event's single genuine peak within this one video's synthetic matrix,
    which is what actually demonstrates the out-of-order property cleanly.
    """
    matrices = _self_check_matrices()
    strategy = UceseAlignment(t_win=20.0, topk=1)

    result = strategy.align(matrices["1"], gaps=())

    assert len(result) == 1
    prediction = result[0]
    # event0's frame (offset+3, ts=6) is chronologically AFTER event1's frame
    # (offset+1, ts=2) -- the chain covers all 3 events but is NOT time-ordered.
    assert prediction.frames == (2003, 2001, 2005)
    assert prediction.frames[1] < prediction.frames[0]  # explicitly out of order
    assert prediction.score == pytest.approx(3 + 0.001 * (0.95 + 0.9 + 0.9))


def test_ucese_leaves_uncovered_event_as_none() -> None:
    # event0 peaks at col0 (ts=0), event1 peaks at col2 (ts=2). With topk=1
    # each event has exactly one hit column, but T_win=1.0 is too small to
    # bridge ts=0 to ts=2 into one window -- the best single-event window
    # (event0, higher score) wins over covering both, leaving event1 None.
    scores = np.array([[0.9, 0.01, 0.01], [0.01, 0.01, 0.5]])
    matrix = _matrix("v", scores, (0.0, 1.0, 2.0), frame_id_offset=500)
    strategy = UceseAlignment(t_win=1.0, topk=1)

    result = strategy.align(matrix, gaps=())

    assert result[0].frames[0] == 500  # event0 covered
    assert result[0].frames[1] is None  # event1 left uncovered, not guessed


def test_ucese_returns_empty_when_topk_is_zero() -> None:
    scores = np.array([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
    matrix = _matrix("v", scores, (0.0, 1.0, 2.0), frame_id_offset=0)
    strategy = UceseAlignment(t_win=20.0, topk=0)  # no frame ever qualifies as a hit

    assert strategy.align(matrix, gaps=()) == []


# -- first_touch ----------------------------------------------------------------


def test_first_touch_matches_lab_reference_example() -> None:
    """Reproduces: sc = [0.1, 0.1, 0.5, 0.1, 0.52, 0.1]; first_touch(sc, after=0) == 2."""
    scores = [0.1, 0.1, 0.5, 0.1, 0.52, 0.1]

    assert first_touch(scores, after=0) == 2


# -- gap-violation penalty (isolated, hand-verifiable case) -------------------


def test_atlas_prefers_gap_compliant_frame_over_higher_raw_score() -> None:
    # event0 only has a real candidate at frame 0. event1 has two candidates:
    # frame 1 (gap=1s, within [0,2]) scoring 0.5, and frame 3 (gap=10s,
    # violating [0,2] badly) scoring higher at 0.9. A strong alpha must make
    # the compliant-but-lower-scoring frame win.
    scores = np.array(
        [
            [0.5, 0.01, 0.01, 0.01],
            [0.01, 0.5, 0.01, 0.9],
        ]
    )
    matrix = _matrix("v", scores, (0.0, 1.0, 2.0, 10.0), frame_id_offset=0)
    strategy = AtlasAlignment(alpha=1.0, beta=1.0)
    gaps = (TrakeGapConstraint(0.0, 2.0),)

    result = strategy.align(matrix, gaps=gaps)

    assert result[0].frames == (0, 1)  # frame_id 0 (event0), frame_id 1 (event1) -- NOT frame_id 3
    assert result[0].score == pytest.approx(1.0)  # 0.5 + 0.5, no penalty


# -- column-position-to-frame-id translation ----------------------------------


@pytest.mark.parametrize("strategy_factory", [
    lambda: AtlasAlignment(),
    lambda: LuciferAlignment(min_sim=0.0),
    lambda: UceseAlignment(t_win=100.0, topk=3),
])
def test_strategies_translate_column_position_to_real_frame_id(strategy_factory) -> None:
    scores = np.array([[0.9, 0.1, 0.1], [0.1, 0.1, 0.9]])
    # original_frame_ids deliberately NOT equal to column positions (0,1,2).
    matrix = TrakeVideoScoreMatrix(
        video_id="v",
        timestamps=(0.0, 1.0, 2.0),
        original_frame_ids=(7000, 7004, 7009),
        scores=scores,
    )
    strategy = strategy_factory()

    result = strategy.align(matrix, gaps=(TrakeGapConstraint(0.0, 300.0),))

    assert len(result) == 1
    for frame in result[0].frames:
        if frame is not None:
            assert frame in (7000, 7004, 7009), (
                f"got raw column position {frame} instead of a real frame_id"
            )


# -- deterministic tie-breaking ------------------------------------------------


@pytest.mark.parametrize("strategy_factory", [
    lambda: AtlasAlignment(),
    lambda: LuciferAlignment(min_sim=0.0),
    lambda: UceseAlignment(t_win=100.0, topk=3),
])
def test_strategies_are_deterministic_across_repeated_runs(strategy_factory) -> None:
    matrices = _self_check_matrices()
    strategy = strategy_factory()
    gaps = ()

    first_run = strategy.align(matrices["2"], gaps=gaps)
    second_run = strategy.align(matrices["2"], gaps=gaps)

    assert first_run == second_run

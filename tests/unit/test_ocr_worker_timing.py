"""The OCR workers must report what they cost, per stage.

Detection and recognition run as two containers inside one timed step, so the
runner's ``ocr:all_pending`` figure cannot say which of them is slow. Both
workers batch at size one today; deciding which is worth rewriting needs the
split, not the total.
"""
from __future__ import annotations

import pytest

from mvr.enrichment.ocr.worker_timing import format_stage_timing


def test_the_line_carries_the_stage_the_count_and_the_rate() -> None:
    line = format_stage_timing("deepsolo", processed=200, skipped=181, seconds=392.0)

    assert line == (
        "OCR_TIMING stage=deepsolo processed=200 skipped=181 "
        "seconds=392.000 per_item=1.960000 final=yes"
    )


def test_a_stage_that_processed_nothing_reports_no_rate() -> None:
    """A resumed run skips everything; dividing by that would crash the worker.

    The worker prints this line on the way out, after the real work is done and
    written, so a ZeroDivisionError here would destroy a finished stage's
    output for the sake of a diagnostic.
    """
    line = format_stage_timing("parseq_vi", processed=0, skipped=54510, seconds=1.5)

    assert "per_item=none" in line
    assert "processed=0" in line


def test_the_rate_resolves_work_faster_than_a_millisecond() -> None:
    """Recognition crops are small; a coarse rate would round them to zero."""
    line = format_stage_timing("parseq_vi", processed=1000, seconds=0.4)

    assert "per_item=0.000400" in line


@pytest.mark.parametrize("stage", ["", "   "])
def test_an_unnamed_stage_is_refused(stage: str) -> None:
    with pytest.raises(ValueError):
        format_stage_timing(stage, processed=1, seconds=1.0)


def test_negative_work_is_refused_rather_than_reported() -> None:
    with pytest.raises(ValueError):
        format_stage_timing("deepsolo", processed=-1, seconds=1.0)


def test_a_progress_line_is_marked_as_not_final() -> None:
    """A mid-run reading must not be mistaken for the stage's total.

    The stage this measures runs for tens of hours, so the only useful reading
    arrives long before it ends. Someone grepping the log then would otherwise
    read a partial count as the finished one and plan against it.
    """
    line = format_stage_timing("deepsolo", processed=500, seconds=980.0, final=False)

    assert "final=no" in line
    assert "per_item=1.960000" in line


def test_the_closing_line_says_so() -> None:
    line = format_stage_timing("deepsolo", processed=500, seconds=980.0)

    assert "final=yes" in line


@pytest.mark.parametrize(
    ("processed", "every", "expected"),
    [(0, 500, False), (1, 500, False), (500, 500, True), (999, 500, False), (1000, 500, True)],
)
def test_progress_is_reported_on_the_interval_and_not_at_zero(
    processed: int, every: int, expected: bool
) -> None:
    """Reporting at zero would print a rate before any work was measured."""
    from mvr.enrichment.ocr.worker_timing import should_report_progress

    assert should_report_progress(processed, every) is expected


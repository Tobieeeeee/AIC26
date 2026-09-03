"""One greppable line per OCR stage, so a slow run can name what was slow.

Detection and recognition run as two containers inside a single timed step, so
``ocr:all_pending`` in the enrichment report gives their sum and nothing else.
Both workers currently infer one item at a time -- a whole keyframe per
DeepSolo call, a single text crop per PARSeq call -- and choosing which to
rewrite first is guesswork without knowing where the seconds go.
"""
from __future__ import annotations


PREFIX = "OCR_TIMING"
PROGRESS_EVERY = 500


def format_stage_timing(
    stage: str,
    *,
    processed: int,
    seconds: float,
    skipped: int = 0,
    final: bool = True,
) -> str:
    """Return the summary line a worker prints as it exits.

    ``processed`` counts items that ran inference; ``skipped`` counts those a
    resumed run found already written. Only the former sets the rate, or a
    resumed run would look arbitrarily fast.
    """
    if not stage.strip():
        raise ValueError("stage must be named")
    if processed < 0 or skipped < 0:
        raise ValueError(f"item counts cannot be negative: {processed}, {skipped}")
    if seconds < 0:
        raise ValueError(f"elapsed seconds cannot be negative: {seconds}")

    # A resumed stage processes nothing, and this line is printed after the
    # real work is written; a ZeroDivisionError here would throw away a
    # finished stage for the sake of a diagnostic.
    rate = f"{seconds / processed:.6f}" if processed else "none"
    return (
        f"{PREFIX} stage={stage} processed={processed} skipped={skipped} "
        f"seconds={seconds:.3f} per_item={rate} final={'yes' if final else 'no'}"
    )


def should_report_progress(processed: int, every: int = PROGRESS_EVERY) -> bool:
    """Whether this item count lands on a reporting interval.

    A stage over the full corpus runs for tens of hours, and its closing line
    arrives far too late to decide anything. Zero is excluded because a rate
    printed before any work was measured says nothing.
    """
    return processed > 0 and every > 0 and processed % every == 0

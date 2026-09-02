"""One greppable line per OCR stage, so a slow run can name what was slow.

Detection and recognition run as two containers inside a single timed step, so
``ocr:all_pending`` in the enrichment report gives their sum and nothing else.
Both workers currently infer one item at a time -- a whole keyframe per
DeepSolo call, a single text crop per PARSeq call -- and choosing which to
rewrite first is guesswork without knowing where the seconds go.
"""
from __future__ import annotations


PREFIX = "OCR_TIMING"


def format_stage_timing(
    stage: str,
    *,
    processed: int,
    seconds: float,
    skipped: int = 0,
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
        f"seconds={seconds:.3f} per_item={rate}"
    )

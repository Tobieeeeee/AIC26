"""TRAKE prediction contract validation, never-blank fallback, and local
debug export.

Mirrors ``mvr.submission.kis_export``'s shape and local-verification-export
caveat (this is NOT the official organizer submission file format).

Two distinct "never guess" rules apply at different stages and must not be
conflated:

- Retrieval time (``search_kis()``, alignment): never fabricate a frame
  mapping with no evidence -- an unresolved candidate is ``None``.
- Export time (this module): a chain about to be submitted with a ``None``
  marker gets a documented interpolation/extrapolation fallback instead,
  because at this stage the alternative is an empty submission slot that
  can score no credit at all (TRAKE gives partial credit per marker).

``TrakeChainPrediction``/``TrakeGapConstraint`` are imported lazily (inside
function bodies, or only under ``TYPE_CHECKING`` for annotations) so the
submission layer does not depend on the evaluation/retrieval layers at
import time -- mirrors ``mvr.retrieval.kis.KisCandidate.to_answer()``.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mvr.evaluation.trake import TrakeChainPrediction
    from mvr.retrieval.trake_events import TrakeGapConstraint

LOCAL_EXPORT_FORMAT_LABEL = "local_verification_export"


class InvalidTrakePredictionError(ValueError):
    """A TRAKE prediction row violates the documented competition contract."""


@dataclass(frozen=True)
class TrakePredictionRow:
    """One validated, fully-resolved TRAKE chain: ``<video_id>, f_E1..f_Ek``.

    ``is_interpolated[i]`` is ``True`` iff ``frames[i]`` was filled in by
    ``apply_never_blank_fallback()`` rather than found by retrieval --
    visible to any reviewer, never hidden.
    """

    video_id: str
    frames: tuple[int, ...]
    is_interpolated: tuple[bool, ...]


def validate_trake_prediction(row: TrakePredictionRow, k: int) -> None:
    """Raise ``InvalidTrakePredictionError`` on any contract violation:
    wrong marker count, non-strictly-increasing frames, or a negative frame
    number. Never silently drops or coerces bad data."""
    if len(row.frames) != k or len(row.is_interpolated) != k:
        raise InvalidTrakePredictionError(
            f"expected {k} markers, got frames={len(row.frames)} "
            f"is_interpolated={len(row.is_interpolated)}"
        )
    for frame in row.frames:
        if frame < 0:
            raise InvalidTrakePredictionError(
                f"frame numbers must be non-negative, got {frame} in {row.frames}"
            )
    for earlier, later in zip(row.frames, row.frames[1:]):
        if later <= earlier:
            raise InvalidTrakePredictionError(
                f"frames must be strictly increasing by time, got {row.frames}"
            )


def _gap_width_seconds(gap: "TrakeGapConstraint") -> float:
    return gap.max_gap_seconds - gap.min_gap_seconds


def _gap_midpoint_seconds(gap: "TrakeGapConstraint") -> float:
    return (gap.min_gap_seconds + gap.max_gap_seconds) / 2.0


def _frame_from_seconds(seconds: float, fps: float) -> int:
    return max(0, round(seconds * fps))


def _none_runs(frames: Sequence[int | None]) -> list[tuple[int, int]]:
    """Maximal contiguous runs of ``None`` as half-open ``[start, end)``
    index ranges. Runs are always bounded by a genuinely-resolved (not
    interpolated) frame on any side that exists, since a run is maximal."""
    runs: list[tuple[int, int]] = []
    index = 0
    length = len(frames)
    while index < length:
        if frames[index] is None:
            start = index
            while index < length and frames[index] is None:
                index += 1
            runs.append((start, index))
        else:
            index += 1
    return runs


def apply_never_blank_fallback(
    prediction: "TrakeChainPrediction",
    gaps: Sequence["TrakeGapConstraint"],
    video_fps: float,
) -> TrakePredictionRow:
    """Fill every ``None`` marker in ``prediction`` with a documented,
    inspectable fallback value, flagged in the output.

    - Interior run (resolved markers on both sides): the real elapsed time
      between the two resolved neighbors is split across the missing
      markers in proportion to each spanned hop's declared
      ``TrakeGapConstraint`` width -- never re-derived from the gap table
      alone, since the two real neighbors are strictly better information
      than a generic heuristic prior.
    - Leading/trailing run (no resolved marker on one side): extrapolated
      from the nearest resolved marker using each spanned gap's own
      midpoint, since no real bracketing data exists for this case.

    Raises ``InvalidTrakePredictionError`` if EVERY marker is unresolved --
    fabricating an entire chain from nothing is rejected outright rather
    than guessed.
    """
    frames: list[int | None] = list(prediction.frames)
    k = len(frames)
    if all(frame is None for frame in frames):
        raise InvalidTrakePredictionError(
            "cannot submit a chain where every marker is unresolved -- "
            "rejecting rather than fabricating an entire chain from nothing"
        )

    is_interpolated = [False] * k
    for start, end in _none_runs(frames):
        has_prev = start > 0
        has_next = end < k
        run_length = end - start

        if has_prev and has_next:
            prev_index, next_index = start - 1, end
            prev_t = frames[prev_index] / video_fps
            next_t = frames[next_index] / video_fps
            total_span = next_t - prev_t
            hop_gaps = gaps[prev_index:next_index]  # length run_length + 1
            widths = [max(_gap_width_seconds(gap), 0.0) for gap in hop_gaps]
            total_width = sum(widths)
            cumulative = 0.0
            for offset, idx in enumerate(range(start, end)):
                if total_width > 0:
                    cumulative += widths[offset]
                    fraction = cumulative / total_width
                else:
                    fraction = (offset + 1) / (run_length + 1)
                frames[idx] = _frame_from_seconds(prev_t + total_span * fraction, video_fps)
                is_interpolated[idx] = True

        elif has_prev:  # trailing run: no resolved marker after it
            cursor_t = frames[start - 1] / video_fps
            for idx in range(start, end):
                cursor_t += _gap_midpoint_seconds(gaps[idx - 1])
                frames[idx] = _frame_from_seconds(cursor_t, video_fps)
                is_interpolated[idx] = True

        else:  # leading run: no resolved marker before it
            cursor_t = frames[end] / video_fps
            for idx in range(end - 1, start - 1, -1):
                cursor_t -= _gap_midpoint_seconds(gaps[idx])
                frames[idx] = _frame_from_seconds(cursor_t, video_fps)
                is_interpolated[idx] = True

    return TrakePredictionRow(
        video_id=prediction.video_id,
        frames=tuple(frames),  # type: ignore[arg-type]  -- every entry is resolved by this point
        is_interpolated=tuple(is_interpolated),
    )


def export_local_trake_verification_csv(
    rows: Sequence[TrakePredictionRow], output_path: Path | str, *, query_id: str
) -> Path:
    """Write predictions as a LOCAL VERIFICATION EXPORT (CSV). Not the
    official organizer submission file format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["format", "query_id", "rank", "video_id", "frames", "is_interpolated"])
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    LOCAL_EXPORT_FORMAT_LABEL,
                    query_id,
                    rank,
                    row.video_id,
                    ";".join(str(frame) for frame in row.frames),
                    ";".join(str(flag) for flag in row.is_interpolated),
                ]
            )
    return output_path


def export_local_trake_verification_json(
    rows: Sequence[TrakePredictionRow], output_path: Path | str, *, query_id: str
) -> Path:
    """Write predictions as a LOCAL VERIFICATION EXPORT (JSON). Same caveat
    as the CSV export: not the official organizer format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": LOCAL_EXPORT_FORMAT_LABEL,
        "note": (
            "This is a local debug export for internal verification only. "
            "It is NOT the official organizer submission format."
        ),
        "query_id": query_id,
        "predictions": [
            {
                "rank": rank,
                "video_id": row.video_id,
                "frames": list(row.frames),
                "is_interpolated": list(row.is_interpolated),
            }
            for rank, row in enumerate(rows, start=1)
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


#: Official AIC26 result-row cap, one CSV per query
#: (https://sotuyenaic.oj.io.vn/, "Yeu cau ket qua": "toi da 100 dong").
OFFICIAL_MAX_ROWS_PER_QUERY = 100


def export_official_trake_submission_csv(
    rows: Sequence[TrakePredictionRow], output_path: Path | str
) -> Path:
    """Write the REAL AIC26 organizer TRAKE submission CSV -- NOT the local
    verification export above.

    Exact format from https://sotuyenaic.oj.io.vn/ (fetched and verified
    2026-08-21, not guessed): one row per ranked chain, no header row,
    comma-delimited, UTF-8, ``<video_name>, <Frame ID_1>, ..., <Frame
    ID_N>``. ``video_name`` carries no ``.mp4`` suffix (matches this
    codebase's ``video_id`` throughout -- never had one). Frame IDs are
    plain integers, never quoted (the organizer's quoting rules apply only
    to Q&A's free-text answer column, which TRAKE does not have).

    Every row MUST already be null-free (``apply_never_blank_fallback()``)
    and validated (``validate_trake_prediction()``) before reaching this
    function -- it does not call either itself, so a caller cannot forget
    to and silently submit a row with a gap.

    Raises ``InvalidTrakePredictionError`` if ``rows`` is empty or exceeds
    the organizer's 100-row cap, or if rows disagree on marker count (the
    organizer compares Frame ID count against the query's own declared
    event count, so a mixed-k file is already broken before upload).
    """
    if not rows:
        raise InvalidTrakePredictionError("cannot submit an empty TRAKE result file")
    if len(rows) > OFFICIAL_MAX_ROWS_PER_QUERY:
        raise InvalidTrakePredictionError(
            f"{len(rows)} rows exceeds the organizer's {OFFICIAL_MAX_ROWS_PER_QUERY}-row cap"
        )
    k = len(rows[0].frames)
    for row in rows:
        if len(row.frames) != k:
            raise InvalidTrakePredictionError(
                f"mixed marker counts in one submission file: {k} vs. {len(row.frames)} "
                f"({row.video_id})"
            )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow([row.video_id, *row.frames])
    return output_path


def load_local_trake_verification_json(input_path: Path | str) -> list[TrakePredictionRow]:
    """Round-trip reader for ``export_local_trake_verification_json()``'s output."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if payload.get("format") != LOCAL_EXPORT_FORMAT_LABEL:
        raise InvalidTrakePredictionError(
            f"unexpected local export format: {payload.get('format')!r}"
        )
    return [
        TrakePredictionRow(
            video_id=row["video_id"],
            frames=tuple(row["frames"]),
            is_interpolated=tuple(row["is_interpolated"]),
        )
        for row in payload["predictions"]
    ]

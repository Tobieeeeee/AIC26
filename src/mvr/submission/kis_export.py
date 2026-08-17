"""KIS prediction contract validation and local debug export.

Validates ranked KIS predictions against the documented competition
contract -- ``<video_id>, <frame_id>``, at most 100 per query -- without
inventing an organizer transport format that has not been specified. The
current source does not document enough detail to assume a particular
final file transport (CSV header rules, JSON schema, server API); the local
export here is explicitly a LOCAL VERIFICATION EXPORT, not the official
submission file format.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

MAX_PREDICTIONS_PER_QUERY = 100
LOCAL_EXPORT_FORMAT_LABEL = "local_verification_export"


class InvalidKisPredictionError(ValueError):
    """A prediction violates the documented KIS contract."""


@dataclass(frozen=True)
class KisPrediction:
    """One validated, ordered KIS prediction: ``<video_id>, <frame_id>``."""

    rank: int
    video_id: str
    frame_id: int


class SubmissionAdapter(Protocol):
    """Thin extension point for the organizer's actual submission transport.

    No concrete organizer adapter exists yet -- the source documentation
    does not specify one. Implement this once it is documented rather than
    guessing a transport now.
    """

    def export(self, predictions: Sequence[KisPrediction], *, query_id: str) -> Any: ...


def validate_kis_predictions(predictions: Sequence[Any]) -> list[KisPrediction]:
    """Validate and normalize ranked KIS candidates into the export contract.

    Input items must expose ``.video_id`` and ``.frame_id`` (e.g.
    ``mvr.retrieval.kis.KisCandidate``). Raises ``InvalidKisPredictionError``
    on any contract violation rather than silently dropping or coercing bad
    data.

    Exact duplicate ``(video_id, frame_id)`` pairs are removed first,
    keeping the first (highest-ranked) occurrence and preserving input
    order; ranks are then reassigned 1..N so they stay contiguous. The
    maximum-100 rule is enforced on the deduplicated result, since exact
    duplicates do not consume additional prediction budget.
    """
    validated: list[KisPrediction] = []
    seen: set[tuple[str, int]] = set()
    for position, item in enumerate(predictions):
        video_id = getattr(item, "video_id", None)
        frame_id = getattr(item, "frame_id", None)
        if not isinstance(video_id, str) or not video_id.strip():
            raise InvalidKisPredictionError(
                f"prediction at position {position} has a blank/invalid video_id: {video_id!r}"
            )
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise InvalidKisPredictionError(
                f"prediction at position {position} has a non-integer frame_id: {frame_id!r}"
            )
        if frame_id < 0:
            raise InvalidKisPredictionError(
                f"prediction at position {position} has a negative frame_id: {frame_id}"
            )
        key = (video_id, frame_id)
        if key in seen:
            continue
        seen.add(key)
        validated.append(
            KisPrediction(rank=len(validated) + 1, video_id=video_id, frame_id=frame_id)
        )

    if len(validated) > MAX_PREDICTIONS_PER_QUERY:
        raise InvalidKisPredictionError(
            f"at most {MAX_PREDICTIONS_PER_QUERY} unique predictions are allowed per "
            f"query, got {len(validated)}"
        )
    return validated


def export_local_verification_csv(
    predictions: Sequence[KisPrediction], output_path: Path | str, *, query_id: str
) -> Path:
    """Write predictions as a LOCAL VERIFICATION EXPORT (CSV).

    This is explicitly not the official organizer submission file format --
    adapt it only once that format is documented.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["format", "query_id", "rank", "video_id", "frame_id"])
        for prediction in predictions:
            writer.writerow(
                [LOCAL_EXPORT_FORMAT_LABEL, query_id, prediction.rank, prediction.video_id, prediction.frame_id]
            )
    return output_path


def export_local_verification_json(
    predictions: Sequence[KisPrediction], output_path: Path | str, *, query_id: str
) -> Path:
    """Write predictions as a LOCAL VERIFICATION EXPORT (JSON).

    Same caveat as the CSV export: not the official organizer format.
    """
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
            {"rank": p.rank, "video_id": p.video_id, "frame_id": p.frame_id}
            for p in predictions
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_local_verification_json(input_path: Path | str) -> list[KisPrediction]:
    """Round-trip reader for ``export_local_verification_json``'s output."""
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if payload.get("format") != LOCAL_EXPORT_FORMAT_LABEL:
        raise InvalidKisPredictionError(
            f"unexpected local export format: {payload.get('format')!r}"
        )
    return [
        KisPrediction(rank=row["rank"], video_id=row["video_id"], frame_id=row["frame_id"])
        for row in payload["predictions"]
    ]

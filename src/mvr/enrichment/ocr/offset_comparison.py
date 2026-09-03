"""Compare two OCR runs over the same keyframes.

The temporal sampler reads each keyframe at several offsets, so OCR's cost is a
direct multiple of how many offsets it uses. Dropping the neighbours is the
largest lever available on a run measured in days -- and the only one that
spends recall to get it, because the neighbours exist to catch text that motion
blur smears in the anchor frame.

How much recall is a question about this dataset, not about the code, so it is
answered by running both samplings over one video and reading the difference.
"""
from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _normalise(text: str) -> str:
    """Two readings of one sign are one finding.

    Case and surrounding space differ between runs without the text differing,
    and Vietnamese diacritics have more than one valid encoding, so the
    comparison would otherwise count identical text as lost.
    """
    return unicodedata.normalize("NFC", text).strip().casefold()


def _keyframe_files(root: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("_") or path.name == "manifest.json":
            continue
        yield f"{path.parent.name}/{path.stem}", path


def _texts_of(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        _normalise(str(entry.get("text", "")))
        for entry in payload.get("texts", [])
        if str(entry.get("text", "")).strip()
    }


def _originals_of(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        _normalise(str(entry["text"])): str(entry["text"]).strip()
        for entry in payload.get("texts", [])
        if str(entry.get("text", "")).strip()
    }


def compare_offset_runs(baseline: Path, candidate: Path) -> dict[str, Any]:
    """What the candidate sampling found, against the baseline's findings.

    ``recall`` is the share of the baseline's distinct text the candidate also
    found, and is ``None`` when the baseline found none -- a video with no
    scene text says nothing about the trade either way.
    """
    baseline, candidate = Path(baseline), Path(candidate)

    baseline_texts: set[str] = set()
    originals: dict[str, str] = {}
    baseline_with_text = 0
    baseline_by_keyframe: dict[str, set[str]] = {}
    for identity, path in _keyframe_files(baseline):
        found = _texts_of(path)
        baseline_by_keyframe[identity] = found
        baseline_texts |= found
        originals.update(_originals_of(path))
        if found:
            baseline_with_text += 1

    candidate_texts: set[str] = set()
    candidate_originals: dict[str, str] = {}
    candidate_with_text = 0
    lost_everything: list[str] = []
    for identity, path in _keyframe_files(candidate):
        found = _texts_of(path)
        candidate_texts |= found
        candidate_originals.update(_originals_of(path))
        if found:
            candidate_with_text += 1
        elif baseline_by_keyframe.get(identity):
            lost_everything.append(identity)

    missing = baseline_texts - candidate_texts
    extra = candidate_texts - baseline_texts

    return {
        "baseline_keyframes": len(baseline_by_keyframe),
        "baseline_keyframes_with_text": baseline_with_text,
        "candidate_keyframes_with_text": candidate_with_text,
        "baseline_text_count": len(baseline_texts),
        "candidate_text_count": len(candidate_texts),
        "recall": round(len(baseline_texts & candidate_texts) / len(baseline_texts), 4)
        if baseline_texts
        else None,
        "missing_texts": sorted(originals[key] for key in missing),
        "extra_texts": sorted(candidate_originals[key] for key in extra),
        "keyframes_that_lost_all_text": sorted(lost_everything),
    }

"""Read a validated enrichment version into a per-keyframe vector catalog.

This is the index-building entry point. It deliberately does not use
``extraction_status.parquet``: that schema carries no ``video_id`` and no
pipeline writes it, whereas a version already publishes everything needed.
Each collection's ``_SUCCESS.json`` proves all six features completed, and
each provider's ``feature_mapping.csv`` carries ``video_id`` alongside
``keyframe_id`` plus the row index into that provider's ``.npy``.

Identity here is always the ``(video_id, keyframe_id)`` pair.  ``keyframe_id``
alone is not globally unique: the organizer's ids are per-video ordinals, so
177,321 of their keyframes carry only 632 distinct id strings.
"""
from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mvr.operations.enrichment_version import (
    EnrichmentVersionError,
    validate_enrichment_version,
)


REQUIRED_MAPPING_COLUMNS = ("keyframe_id", "video_id", "feature_file", "feature_index")
_BULLET = chr(10) + "  - "


@dataclass(frozen=True)
class CatalogEntry:
    """One keyframe with the row index of its vector in every provider."""

    video_id: str
    keyframe_id: str
    collection_id: str
    shot_id: int | None
    frame_id: int | None
    timestamp: float | None
    image_path: str
    vectors: Mapping[str, tuple[str, int]]

    @property
    def identity(self) -> tuple[str, str]:
        return self.video_id, self.keyframe_id


def _optional_int(value: str | None) -> int | None:
    try:
        return int((value or "").strip())
    except ValueError:
        return None


def _optional_float(value: str | None) -> float | None:
    try:
        return float((value or "").strip())
    except ValueError:
        return None


def _reject(problems: Sequence[str]) -> None:
    raise EnrichmentVersionError(
        "enrichment catalog rejected the version:" + _BULLET + _BULLET.join(problems)
    )


def _read_mapping(
    path: Path, provider: str, collection: str
) -> dict[tuple[str, str], dict[str, str]]:
    problems: list[str] = []
    rows: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [
            column
            for column in REQUIRED_MAPPING_COLUMNS
            if column not in (reader.fieldnames or ())
        ]
        if missing:
            raise EnrichmentVersionError(
                f"{collection}/{provider}: feature_mapping.csv is missing "
                f"columns {missing}"
            )
        for line_number, row in enumerate(reader, start=2):
            video_id = (row.get("video_id") or "").strip()
            keyframe_id = (row.get("keyframe_id") or "").strip()
            if not video_id or not keyframe_id:
                problems.append(
                    f"{collection}/{provider}: row {line_number} has an empty identity"
                )
                continue
            index = _optional_int(row.get("feature_index"))
            if index is None or index < 0:
                raw = row.get("feature_index")
                problems.append(
                    f"{collection}/{provider}: row {line_number} has an unusable "
                    f"feature_index {raw!r}"
                )
                continue
            identity = (video_id, keyframe_id)
            if identity in rows:
                problems.append(
                    f"{collection}/{provider}: duplicate identity "
                    f"{video_id}/{keyframe_id}"
                )
                continue
            row["feature_index"] = str(index)
            rows[identity] = row
    if problems:
        _reject(problems)
    return rows


def read_enrichment_catalog(
    *,
    client: Any,
    version: str,
    work_root: Path,
    collections: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return every keyframe of a validated version with its provider vectors.

    Fail-closed.  A keyframe that any declared provider did not embed, a vector
    file the mapping names but nobody published, or an unusable row index all
    stop the read rather than producing an index that silently omits or
    misaligns entries.
    """
    work_root = Path(work_root)
    validation = validate_enrichment_version(
        client=client, version=version, work_root=work_root
    )
    prefix = validation["prefix"]
    batch = validation["batch"]
    selected = list(collections or validation["collections"])
    unknown = [name for name in selected if name not in validation["collections"]]
    if unknown:
        raise EnrichmentVersionError(
            f"version {version} does not publish collections {unknown}"
        )

    listing = {
        str(obj["Key"]): int(obj.get("Size", -1))
        for obj in client.list_objects(f"{prefix}/")
    }
    providers = tuple(validation["ready"].get("providers") or ())
    if not providers:
        raise EnrichmentVersionError("version _READY marker declares no providers")

    entries: list[CatalogEntry] = []
    problems: list[str] = []
    for collection in selected:
        root = f"{prefix}/batch={batch}/collection={collection}"
        by_provider: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
        for provider in providers:
            local = work_root / "mappings" / collection / f"{provider}.csv"
            local.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(
                f"{root}/embeddings/{provider}/feature_mapping.csv", local
            )
            by_provider[provider] = _read_mapping(local, provider, collection)

        reference = providers[0]
        identities = set(by_provider[reference])
        for provider in providers[1:]:
            for video_id, keyframe_id in sorted(identities - set(by_provider[provider])):
                problems.append(
                    f"{collection}: {video_id}/{keyframe_id} is absent from {provider}"
                )
            for video_id, keyframe_id in sorted(set(by_provider[provider]) - identities):
                problems.append(
                    f"{collection}: {video_id}/{keyframe_id} is present in {provider} "
                    f"but absent from {reference}"
                )

        for identity in sorted(identities):
            reference_row = by_provider[reference][identity]
            vectors: dict[str, tuple[str, int]] = {}
            for provider in providers:
                row = by_provider[provider].get(identity)
                if row is None:
                    continue
                feature_file = (row.get("feature_file") or "").strip()
                vector_key = f"{root}/embeddings/{provider}/{feature_file}"
                if listing.get(vector_key, 0) <= 0:
                    problems.append(
                        f"{collection}/{provider}: mapping declares {feature_file} "
                        "but no such non-empty object was published"
                    )
                    continue
                vectors[provider] = (feature_file, int(row["feature_index"]))
            if len(vectors) != len(providers):
                continue
            video_id, keyframe_id = identity
            entries.append(
                CatalogEntry(
                    video_id=video_id,
                    keyframe_id=keyframe_id,
                    collection_id=collection,
                    shot_id=_optional_int(reference_row.get("shot_id")),
                    frame_id=_optional_int(reference_row.get("frame_id")),
                    timestamp=_optional_float(reference_row.get("timestamp")),
                    image_path=(reference_row.get("image_path") or "").strip(),
                    vectors=vectors,
                )
            )

    if problems:
        _reject(problems)
    if not entries:
        raise EnrichmentVersionError(
            f"version {version} yielded no catalog entries for {selected}"
        )

    return {
        "version": version,
        "prefix": prefix,
        "batch": batch,
        "collections": selected,
        "providers": list(providers),
        "entries": entries,
    }

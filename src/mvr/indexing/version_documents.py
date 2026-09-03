"""Build lexical search documents from an immutable enrichment version.

The version's ``metadata/{ocr,asr,objects}/part-000.parquet`` files all carry
``video_id`` beside ``keyframe_id``, so every join here is on the
``(video_id, keyframe_id)`` pair.  Joining on ``keyframe_id`` alone would merge
different videos' text under one document, exactly the failure the vector side
already guards against.

Each document also carries ``document_id``, the pair-derived identifier that
Elasticsearch should use as ``_id``.
"""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mvr.operations.enrichment_catalog import read_enrichment_catalog

from .milvus_schema import make_entity_id

LOGGER = logging.getLogger(__name__)

TEXT_FEATURES = ("ocr", "asr")


def _require_polars():
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "polars is required to read version metadata; "
            "install the project's operations extra"
        ) from exc
    return pl


def _read_feature_rows(
    *, r2_client: Any, root: str, feature: str, work_root: Path
) -> list[dict[str, Any]]:
    pl = _require_polars()
    local = work_root / "metadata" / f"{feature}.parquet"
    local.parent.mkdir(parents=True, exist_ok=True)
    r2_client.download_file(f"{root}/metadata/{feature}/part-000.parquet", local)
    frame = pl.read_parquet(local)
    if frame.height == 0:
        return []
    return frame.to_dicts()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("video_id", "")).strip(),
        str(row.get("keyframe_id", "")).strip(),
    )


def build_documents_from_version(
    *,
    r2_client: Any,
    version: str,
    work_root: Path,
    collections: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Return one lexical document per keyframe published by ``version``.

    A keyframe with no OCR, ASR or object rows still yields a document with
    empty text: dropping it would silently shrink the lexical index relative to
    the vector index built from the same version.
    """
    work_root = Path(work_root)
    catalog = read_enrichment_catalog(
        client=r2_client,
        version=version,
        work_root=work_root,
        collections=collections,
    )
    prefix = catalog["prefix"]
    batch = catalog["batch"]

    texts: dict[str, dict[tuple[str, str], list[str]]] = {
        feature: {} for feature in TEXT_FEATURES
    }
    labels: dict[tuple[str, str], Counter] = {}
    for collection in catalog["collections"]:
        root = f"{prefix}/batch={batch}/collection={collection}"
        collection_root = work_root / "text" / collection
        for feature in TEXT_FEATURES:
            for row in _read_feature_rows(
                r2_client=r2_client,
                root=root,
                feature=feature,
                work_root=collection_root,
            ):
                text = str(row.get("text") or "").strip()
                if text:
                    texts[feature].setdefault(_identity(row), []).append(text)
        for row in _read_feature_rows(
            r2_client=r2_client,
            root=root,
            feature="objects",
            work_root=collection_root,
        ):
            label = str(row.get("label") or "").strip()
            if label:
                labels.setdefault(_identity(row), Counter())[label] += 1

    documents: list[dict[str, Any]] = []
    for entry in catalog["entries"]:
        identity = entry.identity
        documents.append(
            {
                "document_id": make_entity_id(entry.video_id, entry.keyframe_id),
                "frame_id": entry.keyframe_id,
                "video_id": entry.video_id,
                "collection_id": entry.collection_id,
                "ts": entry.timestamp if entry.timestamp is not None else 0.0,
                "caption": " ".join(texts["asr"].get(identity, [])),
                "ocr_text": " ".join(texts["ocr"].get(identity, [])),
                "objects": dict(labels.get(identity, Counter())),
            }
        )

    LOGGER.info(
        "built %d lexical documents from version %s", len(documents), version
    )
    return documents

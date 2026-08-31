"""Milvus semantic index manifest — persist and stale-detect index state.

Manifest is written as JSON alongside the collection config.
If any canonical checksum or dimension changes, the manifest is STALE.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "1.0"
STATE_FRESH = "FRESH"
STATE_STALE = "STALE"
STATE_UNKNOWN = "UNKNOWN"

DEFAULT_MANIFEST_PATH = Path("data/milvus_index_manifest.json")


@dataclass
class ProviderManifestEntry:
    provider_id: str
    dimension: int
    metric: str
    normalized: bool
    checksum: str
    vector_file: str = ""
    mapping_file: str = ""


@dataclass
class SemanticIndexManifest:
    schema_version: str = MANIFEST_SCHEMA_VERSION
    collection_name: str = "aic26_keyframes"
    entity_count: int = 0
    keyframe_mapping_checksum: str = ""
    providers: dict[str, ProviderManifestEntry] = field(default_factory=dict)
    index_type: str = "HNSW"
    metric: str = "IP"
    index_params: dict[str, Any] = field(default_factory=dict)
    build_timestamp: str = ""
    update_timestamp: str = ""
    dataset_version: str = ""
    code_version: str = ""
    state: str = STATE_UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Convert ProviderManifestEntry sub-dicts
        d["providers"] = {
            pid: asdict(entry) for pid, entry in self.providers.items()
        }
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticIndexManifest":
        providers = {
            pid: ProviderManifestEntry(**entry)
            for pid, entry in data.get("providers", {}).items()
        }
        obj = cls(**{
            k: v for k, v in data.items() if k != "providers"
        })
        obj.providers = providers
        return obj


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_manifest(manifest: SemanticIndexManifest, path: Path) -> None:
    """Write manifest to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    LOGGER.info("Wrote semantic index manifest: %s", path)


def load_manifest(path: Path) -> SemanticIndexManifest | None:
    """Load manifest from JSON file. Returns None if file missing."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return SemanticIndexManifest.from_dict(data)
    except Exception as exc:
        LOGGER.warning("Could not load manifest from %s: %s", path, exc)
        return None


def check_staleness(
    existing: SemanticIndexManifest,
    *,
    current_keyframe_checksum: str,
    current_provider_checksums: dict[str, str],
    current_dimensions: dict[str, int],
) -> tuple[str, list[str]]:
    """Compare existing manifest against current corpus state.

    Returns (state, reasons) where state is FRESH or STALE.
    """
    reasons: list[str] = []

    if existing.keyframe_mapping_checksum != current_keyframe_checksum:
        reasons.append(
            f"keyframe mapping checksum changed: "
            f"{existing.keyframe_mapping_checksum!r} → {current_keyframe_checksum!r}"
        )

    for pid, current_checksum in current_provider_checksums.items():
        entry = existing.providers.get(pid)
        if entry is None:
            reasons.append(f"provider {pid!r} missing from manifest")
            continue
        if entry.checksum != current_checksum:
            reasons.append(
                f"provider {pid!r} checksum changed: "
                f"{entry.checksum!r} → {current_checksum!r}"
            )

    for pid, current_dim in current_dimensions.items():
        entry = existing.providers.get(pid)
        if entry and entry.dimension != current_dim:
            reasons.append(
                f"provider {pid!r} dimension changed: "
                f"{entry.dimension} → {current_dim}"
            )

    state = STATE_STALE if reasons else STATE_FRESH
    return state, reasons


def build_manifest(
    *,
    collection_name: str,
    entity_count: int,
    keyframe_mapping_checksum: str,
    providers: dict[str, ProviderManifestEntry],
    index_type: str,
    metric: str,
    index_params: dict[str, Any],
    dataset_version: str = "",
    code_version: str = "",
    state: str = STATE_FRESH,
) -> SemanticIndexManifest:
    """Construct a fresh SemanticIndexManifest."""
    now = _now_utc()
    return SemanticIndexManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        collection_name=collection_name,
        entity_count=entity_count,
        keyframe_mapping_checksum=keyframe_mapping_checksum,
        providers=providers,
        index_type=index_type,
        metric=metric,
        index_params=index_params,
        build_timestamp=now,
        update_timestamp=now,
        dataset_version=dataset_version,
        code_version=code_version,
        state=state,
    )

"""Provider-neutral contracts for offline embedding artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


MANIFEST_FILENAME = "manifest.json"
SUPPORTED_DTYPES = frozenset({"float16", "float32", "float64"})
_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "model_name",
        "checkpoint_or_pretrained",
        "model_revision",
        "embedding_dimension",
        "dtype",
        "normalized",
        "mapping_path",
        "mapping_sha256",
        "keyframe_count",
        "created_at",
    }
)
_REQUIRED_MAPPING_FIELDS = frozenset(
    {"keyframe_id", "video_id", "feature_file", "feature_index"}
)


class EmbeddingArtifactError(ValueError):
    """Raised when an embedding artifact violates the shared contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str | None, asset_name: str) -> str:
    """Fail closed when a local model asset is absent, unlocked, or altered."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{asset_name} not found: {path}")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise EmbeddingArtifactError(f"{asset_name} has no locked SHA256")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise EmbeddingArtifactError(
            f"{asset_name} SHA256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return actual_sha256


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbeddingArtifactError(f"{field} must be a non-empty string")
    return value


def _safe_relative_path(root: Path, value: str, field: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        raise EmbeddingArtifactError(f"{field} must be relative to the provider root")
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise EmbeddingArtifactError(f"{field} escapes the provider root: {value}") from exc
    return resolved


@dataclass(frozen=True)
class EmbeddingManifest:
    schema_version: str
    model_id: str
    model_name: str
    checkpoint_or_pretrained: str
    model_revision: str | None
    embedding_dimension: int
    dtype: str
    normalized: bool
    mapping_path: str
    mapping_sha256: str
    keyframe_count: int
    created_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbeddingManifest":
        missing = sorted(_REQUIRED_MANIFEST_FIELDS - payload.keys())
        if missing:
            raise EmbeddingArtifactError(
                f"manifest is missing required fields: {', '.join(missing)}"
            )

        model_id = _require_string(payload["model_id"], "model_id")
        if not _MODEL_ID_PATTERN.fullmatch(model_id):
            raise EmbeddingArtifactError(
                "model_id must contain only lowercase letters, digits, '_' or '-'"
            )

        dimension = payload["embedding_dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise EmbeddingArtifactError("embedding_dimension must be a positive integer")

        dtype = _require_string(payload["dtype"], "dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise EmbeddingArtifactError(f"unsupported embedding dtype: {dtype}")

        normalized = payload["normalized"]
        if not isinstance(normalized, bool):
            raise EmbeddingArtifactError("normalized must be a boolean")

        count = payload["keyframe_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise EmbeddingArtifactError("keyframe_count must be a non-negative integer")

        revision = payload["model_revision"]
        if revision is not None:
            revision = _require_string(revision, "model_revision")

        created_at = _require_string(payload["created_at"], "created_at")
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise EmbeddingArtifactError("created_at must be an ISO-8601 timestamp") from exc

        checksum = _require_string(payload["mapping_sha256"], "mapping_sha256").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise EmbeddingArtifactError("mapping_sha256 must be a 64-character hex digest")

        return cls(
            schema_version=_require_string(payload["schema_version"], "schema_version"),
            model_id=model_id,
            model_name=_require_string(payload["model_name"], "model_name"),
            checkpoint_or_pretrained=_require_string(
                payload["checkpoint_or_pretrained"], "checkpoint_or_pretrained"
            ),
            model_revision=revision,
            embedding_dimension=dimension,
            dtype=dtype,
            normalized=normalized,
            mapping_path=_require_string(payload["mapping_path"], "mapping_path"),
            mapping_sha256=checksum,
            keyframe_count=count,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EmbeddingMappingRecord:
    keyframe_id: str
    video_id: str
    feature_file: str
    feature_index: int
    frame_id: str | None = None
    shot_id: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class EmbeddingProviderArtifact:
    root: Path
    manifest: EmbeddingManifest
    mapping_path: Path

    def resolve_vector_path(self, feature_file: str) -> Path:
        return _safe_relative_path(self.root, feature_file, "feature_file")


class EmbeddingProviderRegistry:
    """Deterministic lookup for discovered provider artifacts."""

    def __init__(self, providers: Iterable[EmbeddingProviderArtifact]) -> None:
        by_id: dict[str, EmbeddingProviderArtifact] = {}
        for provider in providers:
            model_id = provider.manifest.model_id
            if model_id in by_id:
                raise EmbeddingArtifactError(f"duplicate model_id: {model_id}")
            by_id[model_id] = provider
        self._providers = tuple(by_id[key] for key in sorted(by_id))
        self._by_id = by_id

    def __iter__(self):
        return iter(self._providers)

    def __len__(self) -> int:
        return len(self._providers)

    def get(self, model_id: str) -> EmbeddingProviderArtifact:
        try:
            return self._by_id[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown embedding provider: {model_id}") from exc

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(provider.manifest.model_id for provider in self._providers)


def load_mapping(artifact: EmbeddingProviderArtifact) -> list[EmbeddingMappingRecord]:
    try:
        with artifact.mapping_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_MAPPING_FIELDS - fields)
            if missing:
                raise EmbeddingArtifactError(
                    f"mapping is missing required columns: {', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise EmbeddingArtifactError(f"cannot read mapping: {artifact.mapping_path}") from exc

    records: list[EmbeddingMappingRecord] = []
    # See offline._validate_inputs: identity is (video_id, keyframe_id).
    seen_ids: set[tuple[str, str]] = set()
    for line_number, row in enumerate(rows, start=2):
        keyframe_id = (row.get("keyframe_id") or "").strip()
        video_id = (row.get("video_id") or "").strip()
        feature_file = (row.get("feature_file") or "").strip()
        if not keyframe_id or not video_id or not feature_file:
            raise EmbeddingArtifactError(f"mapping row {line_number} has empty identity/locator")
        if (video_id, keyframe_id) in seen_ids:
            raise EmbeddingArtifactError(
                f"duplicate keyframe in mapping: {video_id}/{keyframe_id}"
            )
        seen_ids.add((video_id, keyframe_id))
        try:
            feature_index = int(row["feature_index"])
        except (TypeError, ValueError) as exc:
            raise EmbeddingArtifactError(
                f"mapping row {line_number} has invalid feature_index"
            ) from exc
        if feature_index < 0:
            raise EmbeddingArtifactError(
                f"mapping row {line_number} has negative feature_index"
            )
        artifact.resolve_vector_path(feature_file)
        records.append(
            EmbeddingMappingRecord(
                keyframe_id=keyframe_id,
                video_id=video_id,
                feature_file=feature_file,
                feature_index=feature_index,
                frame_id=(row.get("frame_id") or row.get("frame_idx") or None),
                shot_id=(row.get("shot_id") or None),
                timestamp=(row.get("timestamp") or None),
            )
        )
    if len(records) != artifact.manifest.keyframe_count:
        raise EmbeddingArtifactError(
            "mapping row count does not match manifest keyframe_count: "
            f"{len(records)} != {artifact.manifest.keyframe_count}"
        )
    return records


def load_provider_artifact(
    provider_root: Path,
    *,
    verify_checksum: bool = False,
    verify_mapping: bool = False,
) -> EmbeddingProviderArtifact:
    provider_root = Path(provider_root)
    manifest_path = provider_root / MANIFEST_FILENAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbeddingArtifactError(f"invalid provider manifest: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EmbeddingArtifactError(f"provider manifest must contain an object: {manifest_path}")
    manifest = EmbeddingManifest.from_dict(payload)
    if provider_root.name != manifest.model_id:
        raise EmbeddingArtifactError(
            f"provider directory '{provider_root.name}' does not match model_id "
            f"'{manifest.model_id}'"
        )
    mapping_path = _safe_relative_path(provider_root, manifest.mapping_path, "mapping_path")
    if not mapping_path.is_file():
        raise EmbeddingArtifactError(f"mapping file does not exist: {mapping_path}")
    if verify_checksum:
        actual = sha256_file(mapping_path)
        if actual != manifest.mapping_sha256:
            raise EmbeddingArtifactError(
                f"mapping checksum mismatch for {manifest.model_id}: "
                f"expected {manifest.mapping_sha256}, got {actual}"
            )
    artifact = EmbeddingProviderArtifact(provider_root.resolve(), manifest, mapping_path)
    if verify_mapping:
        load_mapping(artifact)
    return artifact


def discover_embedding_providers(
    embeddings_root: Path,
    *,
    verify_checksum: bool = False,
    verify_mapping: bool = False,
) -> EmbeddingProviderRegistry:
    root = Path(embeddings_root)
    if not root.exists():
        return EmbeddingProviderRegistry(())
    if not root.is_dir():
        raise EmbeddingArtifactError(f"embedding provider root is not a directory: {root}")

    providers = []
    for child in sorted(root.iterdir(), key=lambda path: path.name):
        if not child.is_dir() or not (child / MANIFEST_FILENAME).is_file():
            continue
        try:
            providers.append(
                load_provider_artifact(
                    child,
                    verify_checksum=verify_checksum,
                    verify_mapping=verify_mapping,
                )
            )
        except EmbeddingArtifactError as exc:
            raise EmbeddingArtifactError(f"invalid provider '{child.name}': {exc}") from exc
    return EmbeddingProviderRegistry(providers)


def write_manifest(path: Path, manifest: EmbeddingManifest) -> None:
    """Atomically write a provider manifest without overwriting vector data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

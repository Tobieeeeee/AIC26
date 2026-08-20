"""Runtime-readable, no-download contracts for embedding providers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import EmbeddingArtifactError, EmbeddingManifest, SUPPORTED_DTYPES


_MODEL_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_METRICS = frozenset({"inner_product", "cosine"})
_STATUSES = frozenset({"ready", "blocked"})


class ProviderSpecError(ValueError):
    """Raised when a locked provider specification is incomplete or unsafe."""


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ProviderSpecError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ProviderSpec:
    model_id: str
    status: str
    model_name: str
    checkpoint_or_pretrained: str
    model_revision: str | None
    checkpoint_sha256: str | None
    tokenizer_sha256: str | None
    source_repository: str
    source_revision: str
    official_sources: tuple[str, ...]
    image_encoder_entry_point: str
    text_encoder_entry_point: str
    image_preprocessing: dict[str, Any]
    tokenizer: dict[str, Any]
    embedding_dimension: int
    dtype: str
    normalized: bool
    similarity_metric: str
    license: str
    access_requirements: str
    dependencies: tuple[str, ...]
    reproducibility_notes: str
    blocker_reason: str | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProviderSpec":
        model_id = _required_text(payload, "model_id")
        if not _MODEL_ID_PATTERN.fullmatch(model_id):
            raise ProviderSpecError(f"invalid model_id: {model_id}")

        status = _required_text(payload, "status")
        if status not in _STATUSES:
            raise ProviderSpecError(f"unsupported provider status: {status}")

        dimension = payload.get("embedding_dimension")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ProviderSpecError("embedding_dimension must be a positive integer")
        dtype = _required_text(payload, "dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise ProviderSpecError(f"unsupported dtype: {dtype}")
        normalized = payload.get("normalized")
        if normalized is not True:
            raise ProviderSpecError("locked retrieval providers must emit normalized vectors")
        metric = _required_text(payload, "similarity_metric")
        if metric not in _METRICS:
            raise ProviderSpecError(f"unsupported similarity metric: {metric}")

        source_revision = _required_text(payload, "source_revision")
        if not _COMMIT_PATTERN.fullmatch(source_revision):
            raise ProviderSpecError("source_revision must be a full 40-character commit")
        model_revision = payload.get("model_revision")
        if model_revision is not None and not _COMMIT_PATTERN.fullmatch(model_revision):
            raise ProviderSpecError("model_revision must be null or a full 40-character commit")

        checksum = payload.get("checkpoint_sha256")
        if checksum is not None and not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ProviderSpecError("checkpoint_sha256 must be null or a SHA256 hex digest")
        if model_id in {"beit3", "perception_encoder"} and checksum is None:
            raise ProviderSpecError(f"{model_id} requires a locked checkpoint_sha256")
        tokenizer_checksum = payload.get("tokenizer_sha256")
        if tokenizer_checksum is not None and not re.fullmatch(
            r"[0-9a-f]{64}", tokenizer_checksum
        ):
            raise ProviderSpecError("tokenizer_sha256 must be null or a SHA256 hex digest")
        if model_id == "beit3" and tokenizer_checksum is None:
            raise ProviderSpecError("BEiT-3 requires a locked tokenizer_sha256")

        sources = payload.get("official_sources")
        dependencies = payload.get("dependencies")
        preprocessing = payload.get("image_preprocessing")
        tokenizer = payload.get("tokenizer")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(item, str) and item.startswith("https://") for item in sources
        ):
            raise ProviderSpecError("official_sources must contain HTTPS URLs")
        if not isinstance(dependencies, list) or not dependencies or not all(
            isinstance(item, str) and item.strip() for item in dependencies
        ):
            raise ProviderSpecError("dependencies must be a non-empty string list")
        if not isinstance(preprocessing, dict) or not preprocessing:
            raise ProviderSpecError("image_preprocessing must be a non-empty object")
        if not isinstance(tokenizer, dict) or not tokenizer:
            raise ProviderSpecError("tokenizer must be a non-empty object")

        notes = _required_text(payload, "reproducibility_notes")
        if model_revision is None and "immutable" not in notes.lower():
            raise ProviderSpecError(
                "a null model_revision must document the missing immutable revision"
            )
        blocker = payload.get("blocker_reason")
        if status == "blocked" and (not isinstance(blocker, str) or not blocker.strip()):
            raise ProviderSpecError("blocked providers require blocker_reason")
        if status == "ready" and blocker is not None:
            raise ProviderSpecError("ready providers must not define blocker_reason")

        return cls(
            model_id=model_id,
            status=status,
            model_name=_required_text(payload, "model_name"),
            checkpoint_or_pretrained=_required_text(payload, "checkpoint_or_pretrained"),
            model_revision=model_revision,
            checkpoint_sha256=checksum,
            tokenizer_sha256=tokenizer_checksum,
            source_repository=_required_text(payload, "source_repository"),
            source_revision=source_revision,
            official_sources=tuple(sources),
            image_encoder_entry_point=_required_text(payload, "image_encoder_entry_point"),
            text_encoder_entry_point=_required_text(payload, "text_encoder_entry_point"),
            image_preprocessing=dict(preprocessing),
            tokenizer=dict(tokenizer),
            embedding_dimension=dimension,
            dtype=dtype,
            normalized=normalized,
            similarity_metric=metric,
            license=_required_text(payload, "license"),
            access_requirements=_required_text(payload, "access_requirements"),
            dependencies=tuple(dependencies),
            reproducibility_notes=notes,
            blocker_reason=blocker,
        )

    def validate_artifact_compatibility(self) -> None:
        """Verify that locked fields satisfy the Task 04 manifest schema."""
        try:
            EmbeddingManifest.from_dict(
                {
                    "schema_version": "1.0",
                    "model_id": self.model_id,
                    "model_name": self.model_name,
                    "checkpoint_or_pretrained": self.checkpoint_or_pretrained,
                    "model_revision": self.model_revision or self.source_revision,
                    "embedding_dimension": self.embedding_dimension,
                    "dtype": self.dtype,
                    "normalized": self.normalized,
                    "mapping_path": "feature_mapping.csv",
                    "mapping_sha256": "0" * 64,
                    "keyframe_count": 0,
                    "created_at": "1970-01-01T00:00:00+00:00",
                }
            )
        except EmbeddingArtifactError as exc:
            raise ProviderSpecError(
                f"provider {self.model_id} is incompatible with artifact schema: {exc}"
            ) from exc


@dataclass(frozen=True)
class ProviderSpecRegistry:
    schema_version: str
    providers: tuple[ProviderSpec, ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(provider.model_id for provider in self.providers)

    def get(self, model_id: str, *, require_ready: bool = True) -> ProviderSpec:
        for provider in self.providers:
            if provider.model_id == model_id:
                if require_ready and provider.status != "ready":
                    raise ProviderSpecError(
                        f"provider {model_id} is blocked: {provider.blocker_reason}"
                    )
                return provider
        raise KeyError(f"unknown embedding provider: {model_id}")


def load_provider_specs(path: Path) -> ProviderSpecRegistry:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderSpecError(f"cannot load provider specs from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderSpecError("provider spec file must contain an object")
    schema_version = _required_text(payload, "schema_version")
    rows = payload.get("providers")
    if not isinstance(rows, list):
        raise ProviderSpecError("providers must be a list")
    providers = tuple(ProviderSpec.from_dict(row) for row in rows)
    ids = [provider.model_id for provider in providers]
    if len(ids) != len(set(ids)):
        raise ProviderSpecError("provider model_id values must be unique")
    registry = ProviderSpecRegistry(
        schema_version=schema_version,
        providers=tuple(sorted(providers, key=lambda provider: provider.model_id)),
    )
    for provider in registry.providers:
        provider.validate_artifact_compatibility()
    return registry

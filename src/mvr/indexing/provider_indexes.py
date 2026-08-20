"""Provider-isolated semantic index construction for Task05C."""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from mvr.enrichment.embeddings.artifacts import (
    EmbeddingArtifactError,
    load_mapping,
    load_provider_artifact,
    sha256_file,
)
from mvr.enrichment.embeddings.provider_specs import ProviderSpec, load_provider_specs
from mvr.indexing.validation import validate_task05a_document


class ProviderIndexError(ValueError):
    """Raised when a provider cannot be routed into a safe index contract."""


@dataclass(frozen=True)
class ProviderIndexContract:
    model_id: str
    collection_name: str
    dimension: int
    metric: str
    normalized: bool
    backend: str


@dataclass(frozen=True)
class ProviderIndexBuildResult:
    manifest_path: Path
    manifest: dict[str, Any]
    reused: bool = False


class ProviderIndexBackend(Protocol):
    name: str

    def prepare(self, contract: ProviderIndexContract, policy: str) -> int: ...

    def insert_batch(
        self,
        contract: ProviderIndexContract,
        vectors: np.ndarray,
        metadata: Sequence[Mapping[str, Any]],
    ) -> Sequence[str]: ...

    def finalize(self, contract: ProviderIndexContract) -> int: ...


class FaissProviderBackend:
    """Persistent exact FAISS backend with provider-namespaced directories."""

    name = "faiss_index_flat_ip"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict[str, Any]] = []
        self._policy: str | None = None

    @staticmethod
    def _faiss():
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError('FAISS is missing; install with ".[semantic]"') from exc
        return faiss

    def _target(self, contract: ProviderIndexContract) -> Path:
        target = (self.root / contract.collection_name).resolve()
        try:
            target.relative_to(self.root.resolve())
        except ValueError as exc:
            raise ProviderIndexError("collection path escapes backend root") from exc
        return target

    def prepare(self, contract: ProviderIndexContract, policy: str) -> int:
        target = self._target(contract)
        self._vectors = []
        self._metadata = []
        self._policy = policy
        if target.exists() and policy == "create":
            raise FileExistsError(f"FAISS collection already exists: {target}")
        if policy == "append" and target.exists():
            values = np.load(target / "vectors.npy", allow_pickle=False)
            metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
            if values.ndim != 2 or values.shape[1] != contract.dimension:
                raise ProviderIndexError("existing FAISS collection dimension mismatch")
            self._vectors.extend(np.asarray(row, dtype=np.float32) for row in values)
            self._metadata.extend(dict(row) for row in metadata)
        return len(self._metadata)

    def insert_batch(
        self,
        contract: ProviderIndexContract,
        vectors: np.ndarray,
        metadata: Sequence[Mapping[str, Any]],
    ) -> Sequence[str]:
        if vectors.shape != (len(metadata), contract.dimension):
            raise ProviderIndexError("backend batch dimension mismatch")
        existing_ids = {row["keyframe_id"] for row in self._metadata}
        duplicates = [str(row["keyframe_id"]) for row in metadata if row["keyframe_id"] in existing_ids]
        if duplicates:
            return duplicates
        self._vectors.extend(np.asarray(row, dtype=np.float32) for row in vectors)
        self._metadata.extend(dict(row) for row in metadata)
        return []

    def finalize(self, contract: ProviderIndexContract) -> int:
        if not self._vectors:
            raise ProviderIndexError("cannot create an empty provider index")
        values = np.ascontiguousarray(self._vectors, dtype=np.float32)
        faiss = self._faiss()
        if contract.metric not in {"inner_product", "cosine"}:
            raise ProviderIndexError(f"unsupported FAISS metric: {contract.metric}")
        index = faiss.IndexFlatIP(contract.dimension)
        index.add(values)
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._target(contract)
        staging = self.root / f".{contract.collection_name}.{uuid.uuid4().hex}.tmp"
        staging.mkdir()
        try:
            np.save(staging / "vectors.npy", values, allow_pickle=False)
            (staging / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            faiss.write_index(index, str(staging / "keyframes.faiss"))
            if target.exists():
                if self._policy not in {"rebuild", "append"}:
                    raise FileExistsError(f"FAISS collection already exists: {target}")
                shutil.rmtree(target)
            staging.replace(target)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return len(self._metadata)


def provider_collection_name(environment: str, dataset: str, model_id: str) -> str:
    def clean(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if not normalized:
            raise ProviderIndexError("index namespace components must be non-empty")
        return normalized

    return f"{clean(environment)}__{clean(dataset)}__{clean(model_id)}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_documents(processed_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Processed documents, keyed on (video_id, keyframe_id).

    On `keyframe_id` alone this raised on the second video of the
    organizer's corpus: 166,550 keyframes carry 632 distinct ids, so "034"
    is in 842 videos and every one after the first looked like a duplicate.
    The documents are already stored as
    `documents/<video_id>/<keyframe_id>.json` -- the layout says the key.
    """
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted((processed_root / "documents").rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ProviderIndexError(f"processed document must be an object: {path}")
        validate_task05a_document(payload)
        identity = (str(payload["video_id"]), str(payload["keyframe_id"]))
        if identity in documents:
            raise ProviderIndexError(
                f"duplicate processed keyframe: {identity[0]}/{identity[1]}"
            )
        documents[identity] = payload
    if not documents:
        raise ProviderIndexError("no processed documents found")
    return documents


def _cross_check_source_manifest(
    processed_manifest: Mapping[str, Any], provider_id: str, provider_root: Path
) -> str:
    expected = (
        processed_manifest.get("source_branch_manifests", {})
        .get("embeddings", {})
        .get(provider_id, {})
        .get("sha256")
    )
    actual = sha256_file(provider_root / "manifest.json")
    if not isinstance(expected, str) or expected != actual:
        raise ProviderIndexError(
            f"source embedding manifest hash mismatch for {provider_id}: "
            f"processed={expected!r}, actual={actual}"
        )
    return actual


def _contract(
    spec: ProviderSpec,
    *,
    dimension: int,
    normalized: bool,
    environment: str,
    dataset: str,
    backend: ProviderIndexBackend,
) -> ProviderIndexContract:
    if dimension != spec.embedding_dimension:
        raise ProviderIndexError(
            f"{spec.model_id} artifact dimension {dimension} != locked {spec.embedding_dimension}"
        )
    if normalized != spec.normalized:
        raise ProviderIndexError(f"{spec.model_id} normalized state differs from locked spec")
    return ProviderIndexContract(
        model_id=spec.model_id,
        collection_name=provider_collection_name(environment, dataset, spec.model_id),
        dimension=dimension,
        metric=spec.similarity_metric,
        normalized=normalized,
        backend=backend.name,
    )


def _assert_provider_route(contract: ProviderIndexContract, provider_id: str) -> None:
    expected_suffix = f"__{provider_id}"
    if contract.model_id != provider_id or not contract.collection_name.endswith(
        expected_suffix
    ):
        raise ProviderIndexError(
            f"cross-provider collection routing rejected: {provider_id} -> "
            f"{contract.collection_name}"
        )


def build_provider_index(
    *,
    provider_root: Path,
    processed_root: Path,
    output_root: Path,
    provider_specs_path: Path,
    backend: ProviderIndexBackend,
    environment: str,
    dataset: str,
    policy: str = "create",
    batch_size: int = 256,
    collection_name: str | None = None,
) -> ProviderIndexBuildResult:
    """Build exactly one provider index after identity and source gates pass."""

    if policy not in {"create", "rebuild", "append"}:
        raise ProviderIndexError("policy must be create, rebuild, or append")
    if batch_size <= 0:
        raise ProviderIndexError("batch_size must be positive")
    provider_root = Path(provider_root)
    processed_root = Path(processed_root)
    output_root = Path(output_root)
    try:
        artifact = load_provider_artifact(
            provider_root, verify_checksum=True, verify_mapping=True
        )
        records = load_mapping(artifact)
    except EmbeddingArtifactError as exc:
        raise ProviderIndexError(str(exc)) from exc
    provider_id = artifact.manifest.model_id
    spec = load_provider_specs(provider_specs_path).get(provider_id)
    contract = _contract(
        spec,
        dimension=artifact.manifest.embedding_dimension,
        normalized=artifact.manifest.normalized,
        environment=environment,
        dataset=dataset,
        backend=backend,
    )
    if collection_name is not None:
        contract = ProviderIndexContract(
            model_id=contract.model_id,
            collection_name=collection_name,
            dimension=contract.dimension,
            metric=contract.metric,
            normalized=contract.normalized,
            backend=contract.backend,
        )
    _assert_provider_route(contract, provider_id)

    processed_manifest_path = processed_root / "manifest.json"
    processed_manifest = json.loads(processed_manifest_path.read_text(encoding="utf-8"))
    source_manifest_hash = _cross_check_source_manifest(
        processed_manifest, provider_id, provider_root
    )
    documents = _load_documents(processed_root)
    # Same identity as the documents, or the two sides cannot be compared:
    # on keyframe_id alone both collapse to 632 entries and the mismatch
    # check below would compare two wrong sets and find them equal.
    record_by_id = {(record.video_id, record.keyframe_id): record for record in records}
    if len(record_by_id) != len(records):
        raise ProviderIndexError(f"duplicate provider mapping IDs for {provider_id}")
    missing = sorted(set(documents) - set(record_by_id))
    extra = sorted(set(record_by_id) - set(documents))
    if missing or extra:
        raise ProviderIndexError(
            f"provider/canonical identity mismatch: missing={missing}, extra={extra}"
        )

    manifest_path = output_root / provider_id / "index_manifest.json"
    if policy == "create" and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("model_id") == provider_id
            and existing.get("source_embedding_manifest_sha256") == source_manifest_hash
            and existing.get("collection_name") == contract.collection_name
        ):
            return ProviderIndexBuildResult(manifest_path, existing, reused=True)
        raise FileExistsError(f"provider index already exists: {manifest_path.parent}")

    vectors: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for identity in sorted(documents):
        video_id, keyframe_id = identity
        document = documents[identity]
        flags = document["indexability"].get("embeddings", {})
        if flags.get(provider_id) is not True:
            continue
        locator = document.get("embeddings", {}).get(provider_id)
        record = record_by_id[identity]
        if not isinstance(locator, Mapping) or (
            locator.get("feature_file") != record.feature_file
            or int(locator.get("row_idx", -1)) != record.feature_index
        ):
            raise ProviderIndexError(
                f"processed locator mismatch for {provider_id}/{video_id}/{keyframe_id}"
            )
        matrix = np.load(
            artifact.resolve_vector_path(record.feature_file),
            mmap_mode="r",
            allow_pickle=False,
        )
        if (
            matrix.ndim != 2
            or str(matrix.dtype) != artifact.manifest.dtype
            or matrix.shape[1] != contract.dimension
            or record.feature_index >= matrix.shape[0]
        ):
            raise ProviderIndexError(f"invalid vector matrix/locator for {provider_id}/{keyframe_id}")
        vector = np.asarray(matrix[record.feature_index], dtype=np.float32)
        if not np.isfinite(vector).all():
            raise ProviderIndexError(f"non-finite vector for {provider_id}/{keyframe_id}")
        if contract.normalized and not np.isclose(
            float(np.linalg.norm(vector)), 1.0, atol=1e-3, rtol=1e-3
        ):
            raise ProviderIndexError(f"non-normalized vector for {provider_id}/{keyframe_id}")
        vectors.append(vector)
        metadata.append(
            {
                "keyframe_id": keyframe_id,
                "video_id": document["video_id"],
                "frame_id": document["frame_idx"],
                "shot_id": document["shot_id"],
                "timestamp": document["timestamp"],
            }
        )
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if values.shape != (len(vectors), contract.dimension):
        raise ProviderIndexError("provider vector batch has an invalid shape")

    _assert_provider_route(contract, provider_id)
    initial_count = backend.prepare(contract, policy)
    failed_ids: list[str] = []
    for start in range(0, len(metadata), batch_size):
        failed_ids.extend(
            backend.insert_batch(
                contract,
                values[start : start + batch_size],
                metadata[start : start + batch_size],
            )
        )
    backend_count = backend.finalize(contract)
    failed_ids = sorted(set(failed_ids))
    inserted_count = len(metadata) - len(failed_ids)
    if backend_count != initial_count + inserted_count:
        raise ProviderIndexError(
            f"backend count mismatch: {backend_count} != {initial_count + inserted_count}"
        )
    manifest = {
        "schema_version": "1.0",
        "model_id": provider_id,
        "source_embedding_manifest_path": str((provider_root / "manifest.json").resolve()),
        "source_embedding_manifest_sha256": source_manifest_hash,
        "backend": backend.name,
        "collection_name": contract.collection_name,
        "dimension": contract.dimension,
        "metric": contract.metric,
        "normalized": contract.normalized,
        "policy": policy,
        "vectors_expected": len(documents),
        "indexed_count": backend_count,
        "inserted_count": inserted_count,
        "failed_count": len(failed_ids),
        "failed_ids": failed_ids,
        "missing_ids": 0,
        "extra_ids": 0,
        "duplicate_ids": 0,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _atomic_json(manifest_path, manifest)
    return ProviderIndexBuildResult(manifest_path, manifest)

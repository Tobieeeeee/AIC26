"""Approval-gated, resumable full-corpus multi-embedding orchestration."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mvr.enrichment.embeddings.artifacts import load_mapping, load_provider_artifact


PROVIDERS = ("beit3", "metaclip2", "perception_encoder")
STAGES = (
    "canonical_mapping",
    "extract_beit3",
    "audit_beit3",
    "extract_metaclip2",
    "audit_metaclip2",
    "extract_perception_encoder",
    "audit_perception_encoder",
    "task05a_validation",
    "provider_indexes",
    "online_smoke",
    "benchmark",
)


class FullCorpusError(RuntimeError):
    """Raised when an operational safety or identity gate fails."""


@dataclass(frozen=True)
class FullCorpusConfig:
    config_path: Path
    repository_root: Path
    dataset_root: Path
    mapping_archive: Path
    keyframes_archive: Path
    canonical_mapping: Path
    keyframe_root: Path
    output_root: Path
    dataset_prefix: str
    providers: tuple[str, ...]
    commands: Mapping[str, tuple[str, ...]]
    resource_estimates: Mapping[str, Any]
    artifact_policy: str
    index_policy: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_full_corpus_config(path: Path, repository_root: Path) -> FullCorpusConfig:
    path = Path(path).resolve()
    repository_root = Path(repository_root).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_root = _resolve(repository_root, payload["dataset_root"])
    output_root = _resolve(repository_root, payload["output_root"])
    try:
        output_root.relative_to(repository_root)
    except ValueError:
        pass
    else:
        raise FullCorpusError("full-corpus output_root must be outside the Git repository")
    providers = tuple(payload["providers"])
    if providers != PROVIDERS:
        raise FullCorpusError(f"providers must be in audited order: {PROVIDERS}")
    artifact_policy = payload.get("artifact_policy", "refuse_overwrite")
    index_policy = payload.get("index_policy", "create")
    if artifact_policy not in {"refuse_overwrite", "resume"}:
        raise FullCorpusError("artifact_policy must be refuse_overwrite or resume")
    if index_policy not in {"create", "rebuild"}:
        raise FullCorpusError("index_policy must be create or rebuild")
    commands = {
        stage: tuple(str(part) for part in payload["commands"][stage])
        for stage in STAGES
    }
    if any(not command for command in commands.values()):
        raise FullCorpusError("every full-corpus stage requires an explicit command")
    return FullCorpusConfig(
        config_path=path,
        repository_root=repository_root,
        dataset_root=dataset_root,
        mapping_archive=_resolve(dataset_root, payload["mapping_archive"]),
        keyframes_archive=_resolve(dataset_root, payload["keyframes_archive"]),
        canonical_mapping=_resolve(dataset_root, payload["canonical_mapping"]),
        keyframe_root=_resolve(dataset_root, payload["keyframe_root"]),
        output_root=output_root,
        dataset_prefix=str(payload["dataset_prefix"]),
        providers=providers,
        commands=commands,
        resource_estimates=dict(payload["resource_estimates"]),
        artifact_policy=artifact_policy,
        index_policy=index_policy,
    )


def audit_source_archives(config: FullCorpusConfig) -> dict[str, Any]:
    """Resolve full-corpus identity counts from ZIP central directories only."""

    if not config.mapping_archive.is_file() or not config.keyframes_archive.is_file():
        raise FullCorpusError("mapping/keyframe archives must exist for dry-run")
    keyframes: set[tuple[str, int]] = set()
    with zipfile.ZipFile(config.keyframes_archive) as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.suffix.lower() != ".jpg" or len(path.parts) < 2:
                continue
            video_id = path.parts[-2]
            if not video_id.startswith(config.dataset_prefix):
                continue
            try:
                keyframes.add((video_id, int(path.stem)))
            except ValueError:
                continue
    mappings: list[tuple[str, int]] = []
    keyframe_videos = {video_id for video_id, _ in keyframes}
    with zipfile.ZipFile(config.mapping_archive) as archive:
        for name in sorted(archive.namelist()):
            path = Path(name)
            if path.suffix.lower() != ".csv":
                continue
            video_id = path.stem
            if (
                not video_id.startswith(config.dataset_prefix)
                or video_id not in keyframe_videos
            ):
                continue
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                for row in reader:
                    mappings.append((video_id, int(str(row["n"]).strip())))
    mapping_set = set(mappings)
    duplicates = len(mappings) - len(mapping_set)
    missing_keyframes = sorted(mapping_set - keyframes)
    extra_keyframes = sorted(keyframes - mapping_set)
    if duplicates or missing_keyframes or extra_keyframes:
        raise FullCorpusError(
            "source archive identity mismatch: "
            f"duplicates={duplicates}, missing_keyframes={len(missing_keyframes)}, "
            f"extra_keyframes={len(extra_keyframes)}"
        )
    canonical_ids = {
        f"{video_id}__{number:06d}" for video_id, number in mapping_set
    }
    if len(canonical_ids) != len(mapping_set):
        raise FullCorpusError("derived canonical keyframe_id values are not unique")
    return {
        "expected_keyframes": len(mapping_set),
        "unique_keyframe_ids": len(canonical_ids),
        "video_count": len({video_id for video_id, _ in mapping_set}),
        "duplicates": duplicates,
        "missing": len(missing_keyframes),
        "extra": len(extra_keyframes),
        "mapping_archive_sha256": _sha256(config.mapping_archive),
        "keyframes_archive_sha256": _sha256(config.keyframes_archive),
    }


def dry_run_plan(config: FullCorpusConfig) -> dict[str, Any]:
    source_audit = audit_source_archives(config)
    unresolved_stages = [
        stage
        for stage, command in config.commands.items()
        if any("<" in part or ">" in part for part in command)
    ]
    blockers = [f"unresolved command: {stage}" for stage in unresolved_stages]
    if not config.keyframe_root.is_dir():
        blockers.append(f"extracted keyframe root missing: {config.keyframe_root}")
    if not config.canonical_mapping.is_file():
        blockers.append(f"canonical mapping target not built: {config.canonical_mapping}")
    return {
        "mode": "dry_run",
        "config": str(config.config_path),
        "dataset_root": str(config.dataset_root),
        "canonical_mapping_target": str(config.canonical_mapping),
        "keyframe_root": str(config.keyframe_root),
        "output_root": str(config.output_root),
        "providers": list(config.providers),
        "source_audit": source_audit,
        "resource_estimates": dict(config.resource_estimates),
        "artifact_policy": config.artifact_policy,
        "index_policy": config.index_policy,
        "stages": [
            {"order": index, "stage": stage, "command": list(config.commands[stage])}
            for index, stage in enumerate(STAGES, start=1)
        ],
        "artifacts": {
            "embeddings": str(config.output_root / "features" / "embeddings"),
            "task05a": str(config.output_root / "processed"),
            "indexes": str(config.output_root / "indexes"),
            "status": str(config.output_root / "full_run_status.json"),
            "failed_ids": str(config.output_root / "failed_ids"),
        },
        "affected_indexes": [
            f"<environment>__<dataset>__{provider}" for provider in config.providers
        ],
        "approval_required": (
            "Explicit owner approval of this exact plan is required before any "
            "non-dry-run extraction or index rebuild."
        ),
        "execution_ready": not blockers,
        "blockers": blockers,
    }


def audit_provider_identity(
    provider_root: Path, canonical_mapping: Path
) -> dict[str, Any]:
    """Audit one completed provider before the next extraction may start."""

    with Path(canonical_mapping).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    canonical_ids = [str(row.get("keyframe_id", "")).strip() for row in rows]
    if any(not keyframe_id for keyframe_id in canonical_ids):
        raise FullCorpusError("canonical mapping contains blank keyframe_id")
    canonical_set = set(canonical_ids)
    canonical_duplicates = len(canonical_ids) - len(canonical_set)
    if canonical_duplicates:
        raise FullCorpusError("canonical mapping contains duplicate keyframe_id")
    artifact = load_provider_artifact(
        Path(provider_root), verify_checksum=True, verify_mapping=True
    )
    records = load_mapping(artifact)
    record_ids = [record.keyframe_id for record in records]
    record_set = set(record_ids)
    duplicates = len(record_ids) - len(record_set)
    missing_ids = sorted(canonical_set - record_set)
    extra_ids = sorted(record_set - canonical_set)
    invalid_ids = []
    for record in records:
        try:
            matrix = np.load(
                artifact.resolve_vector_path(record.feature_file),
                mmap_mode="r",
                allow_pickle=False,
            )
            vector = matrix[record.feature_index]
        except (OSError, ValueError, IndexError):
            invalid_ids.append(record.keyframe_id)
            continue
        if (
            matrix.ndim != 2
            or str(matrix.dtype) != artifact.manifest.dtype
            or vector.shape != (artifact.manifest.embedding_dimension,)
            or not np.isfinite(vector).all()
            or (
                artifact.manifest.normalized
                and not np.isclose(
                    float(np.linalg.norm(vector)), 1.0, atol=1e-3, rtol=1e-3
                )
            )
        ):
            invalid_ids.append(record.keyframe_id)
    return {
        "status": (
            "pass"
            if not (missing_ids or extra_ids or duplicates or invalid_ids)
            else "fail"
        ),
        "model_id": artifact.manifest.model_id,
        "expected": len(canonical_set),
        "valid": len(canonical_set) - len(missing_ids) - len(set(invalid_ids)),
        "missing": len(missing_ids),
        "extra": len(extra_ids),
        "duplicates": duplicates,
        "invalid": len(set(invalid_ids)),
        "dimension": artifact.manifest.embedding_dimension,
        "dtype": artifact.manifest.dtype,
        "normalized": artifact.manifest.normalized,
        "mapping_sha256": artifact.manifest.mapping_sha256,
        "manifest_sha256": _sha256(Path(provider_root) / "manifest.json"),
        "failed_ids": {
            "missing": missing_ids,
            "extra": extra_ids,
            "invalid": sorted(set(invalid_ids)),
        },
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_full_corpus(
    config: FullCorpusConfig,
    *,
    approved: bool,
    execute_stage: Callable[[str, Sequence[str]], Mapping[str, Any]],
    state_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one audited stage at a time; intended for explicit operator use."""

    if not approved:
        raise FullCorpusError("expensive full-corpus execution requires explicit approval")
    if config.index_policy == "rebuild":
        raise FullCorpusError(
            "index_policy=rebuild requires a separate destructive-target approval"
        )
    plan = dry_run_plan(config)
    state_path = Path(state_path or config.output_root / "full_run_status.json")
    config_hash = _sha256(config.config_path)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_hash:
            raise FullCorpusError("resume state config hash mismatch")
    else:
        state = {
            "schema_version": "1.0",
            "config_sha256": config_hash,
            "source_audit": plan["source_audit"],
            "completed_stages": [],
            "stage_results": {},
        }
    completed = set(state["completed_stages"])
    for stage in STAGES:
        if stage in completed:
            continue
        result = dict(execute_stage(stage, config.commands[stage]))
        if result.get("status") != "pass":
            state["failed_stage"] = stage
            state["stage_results"][stage] = result
            _atomic_json(state_path, state)
            raise FullCorpusError(f"full-corpus stage failed: {stage}")
        if stage.startswith("audit_"):
            expected = plan["source_audit"]["expected_keyframes"]
            required_counts = {
                "valid": expected,
                "missing": 0,
                "extra": 0,
                "duplicates": 0,
                "invalid": 0,
            }
            if any(result.get(key) != value for key, value in required_counts.items()):
                state["failed_stage"] = stage
                state["stage_results"][stage] = result
                _atomic_json(state_path, state)
                raise FullCorpusError(f"provider identity audit failed: {stage}")
        state["stage_results"][stage] = result
        state["completed_stages"].append(stage)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(state_path, state)
        completed.add(stage)
    state["status"] = "complete"
    _atomic_json(state_path, state)
    return state

"""Static architecture and repository-hygiene release audit."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from mvr.enrichment.embeddings.provider_specs import load_provider_specs


BANNED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".npy", ".npz",
    ".faiss", ".index", ".db", ".sqlite", ".mp4", ".avi", ".mov", ".mkv",
    ".log", ".zip",
}
MAX_TRACKED_BYTES = 10 * 1024 * 1024


class ReleaseGateError(RuntimeError):
    """Raised when the reproducible release audit cannot run."""


def _git(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise ReleaseGateError(completed.stderr.strip() or "git command failed")
    return [line for line in completed.stdout.splitlines() if line]


def audit_multi_embedding_release(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    specs = load_provider_specs(root / "configs" / "embedding_providers.json")
    provider_matrix = {
        spec.model_id: {
            "revision": spec.model_revision,
            "source_revision": spec.source_revision,
            "dimension": spec.embedding_dimension,
            "dtype": spec.dtype,
            "normalized": spec.normalized,
            "metric": spec.similarity_metric,
        }
        for spec in specs.providers
    }
    task05a = (root / "src/mvr/indexing/task05a.py").read_text(encoding="utf-8")
    query = (root / "src/mvr/enrichment/embeddings/query_encoders.py").read_text(encoding="utf-8")
    indexes = (root / "src/mvr/indexing/provider_indexes.py").read_text(encoding="utf-8")
    rrf = (root / "src/mvr/retrieval/semantic_rrf.py").read_text(encoding="utf-8")
    lexical = (root / "src/mvr/indexing/lexical_index.py").read_text(encoding="utf-8")
    real_fasttest_config = (
        root / "configs/multi_embedding_real_fasttest.json"
    ).read_text(encoding="utf-8")
    replay_verifier = (
        root / "scripts/verification/run_real_multi_embedding_retrieval.py"
    ).read_text(encoding="utf-8")
    architecture_checks = {
        "provider_artifact_layout": "features/embeddings/<model_id>" in (
            root / "docs/MULTI_EMBEDDING_ARTIFACTS.md"
        ).read_text(encoding="utf-8"),
        "three_provider_dimensions": {
            key: value["dimension"] for key, value in provider_matrix.items()
        } == {"beit3": 768, "metaclip2": 512, "perception_encoder": 1024},
        "aggregate_vector_any": 'indexability["vector"] = any(indexability["embeddings"].values())' in task05a,
        "provider_specific_indexability": 'flags.get(provider_id) is not True' in indexes,
        "query_cross_model_guard": "cannot search" in query and "dimension mismatch" in query,
        "index_cross_model_guard": "cross-provider collection routing rejected" in indexes,
        "rrf_uses_rank_formula": "1.0 / (rrf_k + rank)" in rrf,
        "rrf_does_not_read_raw_score": ".score" not in rrf,
        "semantic_rrf_separate_from_cross_modal_hybrid": "semantic_rrf.py" not in (
            root / "src/mvr/retrieval/hybrid.py"
        ).read_text(encoding="utf-8"),
        "task05b_fields_unchanged": all(
            field in lexical for field in ('"ocr_text"', '"asr_text"')
        ),
        "real_fasttest_config_portable": re.search(
            r'(?i)[a-z]:[\\/]', real_fasttest_config
        ) is None,
        "beit3_assets_checksum_locked": bool(
            specs.get("beit3").checkpoint_sha256
            and specs.get("beit3").tokenizer_sha256
        ),
        "recorded_query_evidence_verified": all(
            marker in replay_verifier
            for marker in (
                'manifest.get("vector_sha256")',
                'manifest.get("checkpoint_sha256")',
                'manifest.get("tokenizer_sha256")',
                'source_embedding_manifest_sha256',
            )
        ),
    }

    tracked = _git(root, "ls-files")
    staged = _git(root, "diff", "--cached", "--name-only")
    banned_tracked = []
    oversized_tracked = []
    for relative in tracked:
        path = root / relative
        if path.suffix.lower() in BANNED_SUFFIXES:
            banned_tracked.append(relative)
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            oversized_tracked.append({"path": relative, "bytes": path.stat().st_size})
    banned_staged = [
        relative for relative in staged if (root / relative).suffix.lower() in BANNED_SUFFIXES
    ]
    secret_name_pattern = re.compile(r"(^|/)(\.env($|\.)|.*credential.*|.*secret.*)", re.I)
    suspicious_names = [relative for relative in tracked if secret_name_pattern.search(relative)]
    hygiene = {
        "tracked_file_count": len(tracked),
        "staged_files": staged,
        "banned_tracked_artifacts": banned_tracked,
        "banned_staged_artifacts": banned_staged,
        "oversized_tracked_files": oversized_tracked,
        "suspicious_tracked_secret_names": suspicious_names,
    }
    code_gate_pass = all(architecture_checks.values()) and not any(
        (banned_tracked, banned_staged, oversized_tracked, suspicious_names)
    )
    operational_blockers = [
        "full-corpus extraction, provider audits, indexes, online smoke, and benchmark not run",
        "representative-ground-truth quality evaluation not available",
    ]
    return {
        "schema_version": "1.0",
        "code_structural_gate": "pass" if code_gate_pass else "fail",
        "operational_release_gate": "blocked",
        "provider_matrix": provider_matrix,
        "architecture_checks": architecture_checks,
        "repository_hygiene": hygiene,
        "real_fasttest_evidence": (
            "not evaluated by this static audit; validate with the recorded-artifact "
            "audit and opt-in provider runtime tests"
        ),
        "operational_blockers": operational_blockers,
        "quality_claim": "none; structural and latency smoke only",
    }


def serialize_release_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

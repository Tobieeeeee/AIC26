from __future__ import annotations

import json
from pathlib import Path

import mvr.evaluation.multi_embedding_release as release_module
from mvr.evaluation.multi_embedding_release import (
    audit_multi_embedding_release,
    serialize_release_report,
)


def test_release_gate_reports_architecture_hygiene_and_operational_blockers(
    monkeypatch,
) -> None:
    root = Path(__file__).resolve().parents[3]
    monkeypatch.setattr(release_module, "_git", lambda _root, *args: [])
    report = audit_multi_embedding_release(root)
    assert report["code_structural_gate"] == "pass"
    assert report["operational_release_gate"] == "blocked"
    assert all(report["architecture_checks"].values())
    assert report["provider_matrix"]["beit3"]["dimension"] == 768
    assert report["provider_matrix"]["metaclip2"]["dimension"] == 512
    assert report["provider_matrix"]["perception_encoder"]["dimension"] == 1024
    assert report["repository_hygiene"]["banned_tracked_artifacts"] == []
    assert report["repository_hygiene"]["oversized_tracked_files"] == []
    assert len(report["operational_blockers"]) == 2
    assert all("not run locally" not in item for item in report["operational_blockers"])
    assert report["real_fasttest_evidence"].startswith("not evaluated")
    assert json.loads(serialize_release_report(report))["quality_claim"].startswith("none")

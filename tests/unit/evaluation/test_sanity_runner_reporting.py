from __future__ import annotations

import json
from pathlib import Path

from mvr.evaluation.sanity_contracts import RetrievalHit, SanityQuery
from mvr.evaluation.sanity_reporting import write_run_reports
from mvr.evaluation.sanity_runner import run_sanity_suite


class FakeAdapter:
    provider_id = "fake"

    def search(self, query: str, top_k: int) -> list[RetrievalHit]:
        if not query.strip():
            raise ValueError("query must be non-empty")
        return [
            RetrievalHit(
                rank=1,
                score=1.0,
                keyframe_id="L21_V001_S0001_K01",
                video_id="L21_V001",
                frame_id=10,
                provider=self.provider_id,
            )
        ][:top_k]

    def provenance(self) -> dict:
        return {"provider_id": self.provider_id, "version": "test"}


def _manifest() -> dict:
    return {
        "run_id": "test-run",
        "timestamp": "2026-08-20T00:00:00Z",
        "git_commit": "abc",
        "git_dirty": True,
        "query_suite_path": "queries.json",
        "query_suite_sha256": "hash",
        "providers": [{"provider_id": "fake"}],
        "top_k": 5,
        "repetitions": 3,
    }


def test_runner_handles_positive_and_expected_error_queries() -> None:
    run = run_sanity_suite(
        adapter=FakeAdapter(),
        queries=[
            SanityQuery("q1", "hello"),
            SanityQuery("q_empty", "", expect_error=True),
        ],
        top_k=5,
        repetitions=3,
        manifest=_manifest(),
    )

    assert run.summary["total_queries"] == 2
    assert run.summary["failed_attempts"] == 0
    assert run.summary["provider_errors"] == 3
    assert run.summary["determinism"]["deterministic_order_count"] == 1
    assert all(result.latency_ms >= 0 for result in run.results)


def test_reporting_writes_machine_and_human_readable_outputs(tmp_path: Path) -> None:
    run = run_sanity_suite(
        adapter=FakeAdapter(),
        queries=[SanityQuery("q1", "hello")],
        top_k=5,
        repetitions=3,
        manifest=_manifest(),
    )
    output = write_run_reports(run, output_root=tmp_path)

    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "results.jsonl",
        "summary.json",
        "report.md",
    }
    lines = (output / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["ordered_keyframe_ids"] == [
        "L21_V001_S0001_K01"
    ]
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["contract_failures"] == 0
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "GT-free" in report

"""Machine-readable and Markdown output for retrieval sanity runs."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .sanity_contracts import SanityRun


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown_report(
    run: SanityRun,
    *,
    regression_diff: Mapping[str, Any] | None = None,
) -> str:
    summary = run.summary
    latency = summary["latency_ms"]
    determinism = summary["determinism"]
    lines = [
        "# Retrieval Sanity Run",
        "",
        f"- Run ID: `{run.manifest['run_id']}`",
        f"- Provider: `{run.manifest['providers'][0]['provider_id']}`",
        f"- Queries: {summary['total_queries']}",
        f"- Repetitions: {run.manifest['repetitions']}",
        f"- Successful attempts: {summary['successful_attempts']}",
        f"- Failed attempts: {summary['failed_attempts']}",
        f"- Provider errors: {summary['provider_errors']}",
        "",
        "## Contract and integrity",
        "",
        f"- Contract failures: {summary['contract_failures']}",
        f"- Top-K failures: {summary['top_k_failures']}",
        f"- Invalid rank results: {summary['invalid_rank_results']}",
        f"- Non-finite score results: {summary['non_finite_score_results']}",
        f"- Duplicate keyframe occurrences: {summary['duplicate_keyframe_occurrences']}",
        f"- Identity conflicts: {summary['identity_conflict_occurrences']}",
        "",
        "## Latency",
        "",
        f"- Mean: {latency['mean']:.3f} ms",
        f"- P50: {latency['p50']:.3f} ms",
        f"- P95: {latency['p95']:.3f} ms",
        f"- Min: {latency['min']:.3f} ms",
        f"- Max: {latency['max']:.3f} ms",
        "",
        "## Determinism",
        "",
        f"- Comparable queries: {determinism['comparable_query_count']}",
        f"- Deterministic ordered rankings: {determinism['deterministic_order_count']}",
        f"- Changed ordered rankings: {determinism['changed_order_count']}",
        f"- Score drift cases: {determinism['score_drift_count']}",
    ]
    if regression_diff is not None:
        lines.extend(
            [
                "",
                "## Regression change detection",
                "",
                f"- Compared rankings: {regression_diff['compared_ranking_count']}",
                f"- Changed rankings: {regression_diff['changed_ranking_count']}",
                "- Interpretation: ranking change only; no better/worse conclusion.",
            ]
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This run is GT-free. It reports pipeline correctness, contract integrity, "
            "latency, determinism, and ranking-change signals only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_run_reports(
    run: SanityRun,
    *,
    output_root: Path,
    regression_diff: Mapping[str, Any] | None = None,
) -> Path:
    run_directory = Path(output_root) / run.manifest["run_id"]
    run_directory.mkdir(parents=True, exist_ok=False)
    _write_json(run_directory / "manifest.json", run.manifest)
    with (run_directory / "results.jsonl").open("w", encoding="utf-8") as handle:
        for result in run.results:
            handle.write(
                json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    _write_json(
        run_directory / "summary.json",
        {
            **run.summary,
            "determinism_results": [
                result.to_dict() for result in run.determinism
            ],
        },
    )
    if regression_diff is not None:
        _write_json(run_directory / "regression_diff.json", regression_diff)
    (run_directory / "report.md").write_text(
        render_markdown_report(run, regression_diff=regression_diff),
        encoding="utf-8",
    )
    return run_directory

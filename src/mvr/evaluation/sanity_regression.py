"""Regression snapshots that detect ranking changes without judging quality."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sanity_contracts import QueryRunResult, SanityRun


SNAPSHOT_SCHEMA_VERSION = "1.0"


def build_snapshot(run: SanityRun) -> dict[str, Any]:
    first_results: dict[tuple[str, str], QueryRunResult] = {}
    for result in run.results:
        key = (result.provider_id, result.query_id)
        if result.error is None and key not in first_results:
            first_results[key] = result

    rankings = [
        {
            "query_id": query.query_id,
            "provider_id": provider_id,
            "top_k": result.top_k_requested,
            "keyframe_ids": list(result.ordered_keyframe_ids),
        }
        for (provider_id, _query_id), result in sorted(first_results.items())
        for query in run.queries
        if query.query_id == result.query_id
    ]
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_at": run.manifest["timestamp"],
        "git_commit": run.manifest["git_commit"],
        "git_dirty": run.manifest["git_dirty"],
        "query_suite_path": run.manifest["query_suite_path"],
        "query_suite_sha256": run.manifest["query_suite_sha256"],
        "top_k": run.manifest["top_k"],
        "providers": run.manifest["providers"],
        "rankings": rankings,
    }


def save_snapshot(path: Path, snapshot: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("snapshot must be a JSON object")
    if payload.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema_version")
    rankings = payload.get("rankings")
    if not isinstance(rankings, list):
        raise ValueError("snapshot rankings must be a list")
    return dict(payload)


def _rank_map(ids: Sequence[str]) -> dict[str, int]:
    return {keyframe_id: rank for rank, keyframe_id in enumerate(ids, start=1)}


def compare_snapshots(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
) -> dict[str, Any]:
    def index_rankings(snapshot: Mapping[str, Any]) -> dict[tuple[str, str], list[str]]:
        indexed: dict[tuple[str, str], list[str]] = {}
        for row in snapshot.get("rankings", []):
            if not isinstance(row, Mapping):
                raise ValueError("snapshot ranking row must be an object")
            provider_id = row.get("provider_id")
            query_id = row.get("query_id")
            keyframe_ids = row.get("keyframe_ids")
            if (
                not isinstance(provider_id, str)
                or not isinstance(query_id, str)
                or not isinstance(keyframe_ids, list)
                or any(not isinstance(item, str) for item in keyframe_ids)
            ):
                raise ValueError("malformed snapshot ranking row")
            indexed[(provider_id, query_id)] = list(keyframe_ids)
        return indexed

    baseline_rows = index_rankings(baseline)
    current_rows = index_rankings(current)
    all_keys = sorted(set(baseline_rows) | set(current_rows))
    differences: list[dict[str, Any]] = []

    for provider_id, query_id in all_keys:
        before = baseline_rows.get((provider_id, query_id), [])
        after = current_rows.get((provider_id, query_id), [])
        before_set = set(before)
        after_set = set(after)
        intersection = before_set & after_set
        union = before_set | after_set
        before_ranks = _rank_map(before)
        after_ranks = _rank_map(after)
        shared_displacements = {
            keyframe_id: after_ranks[keyframe_id] - before_ranks[keyframe_id]
            for keyframe_id in sorted(intersection)
            if after_ranks[keyframe_id] != before_ranks[keyframe_id]
        }
        differences.append(
            {
                "provider_id": provider_id,
                "query_id": query_id,
                "exact_order_match": before == after,
                "added_keyframe_ids": sorted(after_set - before_set),
                "removed_keyframe_ids": sorted(before_set - after_set),
                "rank_displacement": shared_displacements,
                "ranking_change_statistics": {
                    "intersection_count": len(intersection),
                    "union_count": len(union),
                    "overlap_fraction": (
                        len(intersection) / max(len(before), len(after), 1)
                    ),
                    "jaccard_change_statistic": (
                        len(intersection) / len(union) if union else 1.0
                    ),
                },
            }
        )

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "baseline_query_suite_sha256": baseline.get("query_suite_sha256"),
        "current_query_suite_sha256": current.get("query_suite_sha256"),
        "compared_ranking_count": len(all_keys),
        "changed_ranking_count": sum(
            not row["exact_order_match"] for row in differences
        ),
        "all_exact_order_match": all(
            row["exact_order_match"] for row in differences
        ),
        "differences": differences,
        "interpretation": (
            "These values describe ranking change only and do not indicate "
            "improvement or degradation."
        ),
    }

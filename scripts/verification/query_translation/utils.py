"""Shared utilities for the TKIS query translation benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import MethodSummary, TranslationResult


def load_queries(path: Path) -> list[dict[str, str]]:
    """Load a query CSV file and validate required columns."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Query file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Query file is empty: {path}")

    required = {"query_id", "query_vi"}
    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Query CSV is missing columns: {sorted(missing)}"
        )

    return rows


# ── Translation cache ──────────────────────────────────────────────


def _cache_key(text: str, method: str, config: str = "") -> str:
    """Deterministic cache key from query text + method + config."""
    payload = f"{method}|{config}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_get(
    cache_dir: Path,
    text: str,
    method: str,
    config: str = "",
) -> str | None:
    """Return cached translation or None if not cached."""
    key = _cache_key(text, method, config)
    cache_file = cache_dir / f"{key}.json"

    if not cache_file.is_file():
        return None

    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return str(data["translated_text"])
    except (json.JSONDecodeError, KeyError):
        return None


def cache_put(
    cache_dir: Path,
    text: str,
    method: str,
    translated: str,
    config: str = "",
) -> None:
    """Store a translation in the local cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(text, method, config)
    cache_file = cache_dir / f"{key}.json"

    # Never cache secrets — only store text + method + result.
    payload = {
        "method": method,
        "config": config,
        "source_text": text,
        "translated_text": translated,
    }

    cache_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Output writing ─────────────────────────────────────────────────


def write_results_csv(
    results: list[TranslationResult],
    path: Path,
) -> None:
    """Write per-query results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].to_dict().keys())

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow(result.to_dict())


def write_results_json(
    results: list[TranslationResult],
    path: Path,
) -> None:
    """Write per-query results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.to_dict() for r in results]

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_summary(
    results: list[TranslationResult],
) -> list[MethodSummary]:
    """Compute per-method aggregate statistics."""
    by_method: dict[str, list[TranslationResult]] = {}

    for r in results:
        by_method.setdefault(r.method, []).append(r)

    summaries = []

    for method, group in sorted(by_method.items()):
        successes = [r for r in group if r.success]
        failures = [r for r in group if not r.success]
        latencies = [r.latency_ms for r in successes] or [0.0]

        summaries.append(
            MethodSummary(
                method=method,
                total_queries=len(group),
                success_count=len(successes),
                failure_count=len(failures),
                failure_rate=(
                    len(failures) / len(group) if group else 0.0
                ),
                mean_latency_ms=statistics.mean(latencies),
                median_latency_ms=statistics.median(latencies),
                p95_latency_ms=_percentile(latencies, 95),
            )
        )

    return summaries


def write_summary_json(
    summaries: list[MethodSummary],
    path: Path,
) -> None:
    """Write method summary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [s.to_dict() for s in summaries]

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile without numpy dependency."""
    if not data:
        return 0.0

    sorted_data = sorted(data)
    k = (pct / 100.0) * (len(sorted_data) - 1)
    lower = int(k)
    upper = min(lower + 1, len(sorted_data) - 1)
    weight = k - lower

    return sorted_data[lower] + weight * (
        sorted_data[upper] - sorted_data[lower]
    )


def now_iso() -> str:
    """Current UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()

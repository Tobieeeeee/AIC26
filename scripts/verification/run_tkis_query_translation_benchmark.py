#!/usr/bin/env python
"""TKIS Query Translation Benchmark Runner.

Translates Vietnamese queries through multiple backends and records
latency/failure metrics.  This is an experiment-only benchmark —
it does NOT connect to KIS retrieval.

Usage
-----
python scripts/verification/run_tkis_query_translation_benchmark.py \
    --input  artifacts/verification/tkis_query_translation/queries.csv \
    --output artifacts/verification/tkis_query_translation/results \
    --methods vi_raw,local_vi2en,google_translate,deepl
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Allow importing the sibling package when running as a script.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from query_translation.base import Translator
from query_translation.models import TranslationResult
from query_translation.utils import (
    cache_get,
    cache_put,
    compute_summary,
    load_queries,
    now_iso,
    write_results_csv,
    write_results_json,
    write_summary_json,
)

VALID_METHODS = {"vi_raw", "local_vi2en", "google_translate", "deepl"}

DEFAULT_INPUT = Path(
    "artifacts/verification/tkis_query_translation/queries.csv"
)
DEFAULT_OUTPUT = Path(
    "artifacts/verification/tkis_query_translation/results"
)


# ── vi_raw pass-through ────────────────────────────────────────────


class ViRawTranslator(Translator):
    """Control: returns the original Vietnamese text unchanged."""

    @property
    def name(self) -> str:
        return "vi_raw"

    def translate(self, text: str) -> str:
        return text


# ── Factory ────────────────────────────────────────────────────────


def _create_translator(
    method: str,
    *,
    local_model: str,
    device: str,
) -> Translator:
    """Create the translator for *method*, raising on failure."""
    if method == "vi_raw":
        return ViRawTranslator()

    if method == "local_vi2en":
        from query_translation.local import LocalTranslator

        return LocalTranslator(
            model_name=local_model,
            device=device,
        )

    if method == "google_translate":
        from query_translation.google import GoogleTranslator

        return GoogleTranslator()

    if method == "deepl":
        from query_translation.deepl import DeepLTranslator

        return DeepLTranslator()

    raise ValueError(f"Unknown translation method: {method}")


# ── CLI ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TKIS Query Translation Benchmark"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the query CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="vi_raw,local_vi2en",
        help=(
            "Comma-separated list of methods to run.  "
            f"Valid: {', '.join(sorted(VALID_METHODS))}"
        ),
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="Helsinki-NLP/opus-mt-vi-en",
        help="HuggingFace model identifier for local_vi2en.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for local model (cpu / cuda / auto).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of queries for quick testing.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the local translation cache.",
    )

    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    invalid = set(methods) - VALID_METHODS

    if invalid:
        print(f"ERROR: Unknown methods: {sorted(invalid)}")
        sys.exit(1)

    if not methods:
        print("ERROR: No methods specified.")
        sys.exit(1)

    # Load queries.
    print(f"[1/4] Loading queries from {args.input} ...")
    queries = load_queries(args.input)

    if args.limit is not None:
        queries = queries[: args.limit]

    print(f"       {len(queries)} queries loaded.")

    # Prepare output dir.
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = output_dir / ".cache"

    # Run each method.
    all_results: list[TranslationResult] = []

    for method in methods:
        print(f"\n[2/4] Method: {method}")

        # Try to create translator.
        translator: Translator | None = None
        init_error: str | None = None

        try:
            translator = _create_translator(
                method,
                local_model=args.local_model,
                device=args.device,
            )
            print(f"       Translator loaded.")
        except Exception as exc:
            init_error = f"{type(exc).__name__}: {exc}"
            print(f"       SKIPPED — {init_error}")

        for query in queries:
            query_id = str(query["query_id"])
            query_vi = str(query["query_vi"])

            if translator is None:
                # Record init failure for every query in this method.
                all_results.append(
                    TranslationResult(
                        query_id=query_id,
                        query_vi=query_vi,
                        method=method,
                        translated_text="",
                        latency_ms=0.0,
                        success=False,
                        error=init_error,
                        model_or_provider=method,
                        timestamp=now_iso(),
                    )
                )
                continue

            # Check cache.
            cache_config = (
                args.local_model if method == "local_vi2en" else method
            )

            if not args.no_cache:
                cached = cache_get(
                    cache_dir, query_vi, method, cache_config
                )
            else:
                cached = None

            if cached is not None:
                all_results.append(
                    TranslationResult(
                        query_id=query_id,
                        query_vi=query_vi,
                        method=method,
                        translated_text=cached,
                        latency_ms=0.0,
                        success=True,
                        model_or_provider=cache_config,
                        timestamp=now_iso(),
                    )
                )
                continue

            # Translate.
            start = time.perf_counter()

            try:
                translated = translator.translate(query_vi)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                all_results.append(
                    TranslationResult(
                        query_id=query_id,
                        query_vi=query_vi,
                        method=method,
                        translated_text=translated,
                        latency_ms=elapsed_ms,
                        success=True,
                        model_or_provider=cache_config,
                        timestamp=now_iso(),
                    )
                )

                # Cache on success.
                if not args.no_cache:
                    cache_put(
                        cache_dir,
                        query_vi,
                        method,
                        translated,
                        cache_config,
                    )

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                all_results.append(
                    TranslationResult(
                        query_id=query_id,
                        query_vi=query_vi,
                        method=method,
                        translated_text="",
                        latency_ms=elapsed_ms,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        model_or_provider=cache_config,
                        timestamp=now_iso(),
                    )
                )

        method_results = [r for r in all_results if r.method == method]
        ok = sum(1 for r in method_results if r.success)
        fail = sum(1 for r in method_results if not r.success)
        print(f"       Done: {ok} success, {fail} failures.")

    # Write outputs.
    print(f"\n[3/4] Writing results ...")

    write_results_csv(all_results, output_dir / "translation_results.csv")
    write_results_json(all_results, output_dir / "translation_results.json")

    summaries = compute_summary(all_results)
    write_summary_json(summaries, output_dir / "translation_summary.json")

    # Console report.
    print(f"\n[4/4] Summary")
    print("=" * 60)

    for s in summaries:
        print(f"\n  Method              : {s.method}")
        print(f"  Queries             : {s.total_queries}")
        print(f"  Success             : {s.success_count}")
        print(f"  Failures            : {s.failure_count}")
        print(f"  Failure rate        : {s.failure_rate:.2%}")
        print(f"  Mean latency (ms)   : {s.mean_latency_ms:.1f}")
        print(f"  Median latency (ms) : {s.median_latency_ms:.1f}")
        print(f"  P95 latency (ms)    : {s.p95_latency_ms:.1f}")

    print()
    print(f"Results CSV  : {output_dir / 'translation_results.csv'}")
    print(f"Results JSON : {output_dir / 'translation_results.json'}")
    print(f"Summary      : {output_dir / 'translation_summary.json'}")


if __name__ == "__main__":
    main()

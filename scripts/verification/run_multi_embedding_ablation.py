"""Run the seven structural semantic ablations from a machine-readable fixture."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mvr.evaluation.multi_embedding_ablation import (
    AblationConfig,
    render_ablation_markdown,
    run_ablation_suite,
)
from mvr.retrieval.provider_semantic import (
    IndependentSemanticResponse,
    ProviderSemanticResult,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run seven multi-embedding structural ablations")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = AblationConfig(
        run_id=payload["run_id"],
        queries=tuple(payload["queries"]),
        canonical_ids=frozenset(payload["canonical_ids"]),
        expected_keyframes=int(payload.get("expected_keyframes", 5)),
        rrf_k=int(payload.get("rrf_k", 60)),
        top_k=int(payload.get("top_k", 5)),
        warmup_runs=int(payload.get("warmup_runs", 1)),
        measured_runs=int(payload.get("measured_runs", 3)),
        query_set_version=payload["query_set_version"],
        ground_truth_version=payload.get("ground_truth_version"),
        representative_ground_truth=bool(payload.get("representative_ground_truth", False)),
        provider_provenance=payload.get("provider_provenance", {}),
    )
    fixture_rankings = payload["fixture_rankings"]

    def retrieve(query: str, providers, top_k: int):
        query_id = next(item["id"] for item in config.queries if item["text"] == query)
        lists = {}
        for provider_id in providers:
            lists[provider_id] = tuple(
                ProviderSemanticResult(model_id=provider_id, **row)
                for row in fixture_rankings[query_id][provider_id][:top_k]
            )
        return IndependentSemanticResponse(
            ranked_lists=lists,
            requested_providers=tuple(providers),
            successful_providers=tuple(providers),
            failed_providers=(),
            errors={},
            failure_policy="fail_closed",
            timings_ms={
                provider_id: {"encoding_ms": 0.0, "search_ms": 0.0}
                for provider_id in providers
            },
        )

    report = run_ablation_suite(config, retrieve)
    output_root = REPOSITORY_ROOT / payload["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "results.json"
    markdown_path = output_root / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_ablation_markdown(report), encoding="utf-8")
    print(json.dumps({"results": str(json_path), "report": str(markdown_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

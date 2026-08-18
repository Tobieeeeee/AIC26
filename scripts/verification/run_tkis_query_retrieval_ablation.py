"""TKIS Query-Variant Retrieval Ablation Benchmark Runner (Task 03)."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from scripts.verification.query_decomposition.qwen import QwenDecomposer, SCHEMA_VERSION as QWEN_SCHEMA_VERSION
from scripts.verification.query_retrieval_ablation.inputs import AblationQuery, load_ablation_queries
from scripts.verification.query_retrieval_ablation.metrics import compute_pairwise_comparison
from scripts.verification.query_retrieval_ablation.models import VariantSummary
from scripts.verification.query_retrieval_ablation.runner import AblationRunner
from scripts.verification.query_retrieval_ablation.variants import freeze_query_variants
from scripts.verification.query_translation.local import LocalMarianTranslator


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    audit_path = args.output / "query_source_audit.json"

    # Gate 0: Audit
    queries: list[AblationQuery] = []
    try:
        queries = load_ablation_queries(args.gt)
    except Exception as e:
        # Generate audit report for data blocker
        audit = {
            "query_source": str(args.gt),
            "query_count": 0,
            "query_language": "unknown",
            "queries_organizer_provided": False,
            "gt_source": str(args.gt),
            "gt_availability": False,
            "representative": False,
            "notes": f"Data Blocker: {e}"
        }
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Data Blocker reported in {audit_path}")
        return

    # If we have queries, proceed
    audit = {
        "query_source": str(args.gt),
        "query_count": len(queries),
        "query_language": "vi",
        "queries_organizer_provided": False,
        "gt_source": str(args.gt),
        "gt_availability": True,
        "representative": False,  # assuming internal for now
        "notes": "DEVELOPMENT-ONLY ablation"
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    # Freeze variants
    translator = LocalMarianTranslator(device=args.device)
    decomposer = QwenDecomposer(device=args.device)
    variants_path = args.output / "query_variants.json"
    
    variants = freeze_query_variants(queries, translator, decomposer, variants_path)

    # Initialize semantic service
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.retrieval.semantic import SemanticRetriever
    from mvr.ingestion.aic_dataset import AicDatasetRegistry

    faiss_index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=faiss_index.manifest["embedding_model"],
        pretrained=faiss_index.manifest["pretrained"],
        device=args.device,
    )
    semantic_service = SemanticRetriever(encoder, faiss_index)
    aic_registry = AicDatasetRegistry(args.dataset_root)

    runner = AblationRunner(semantic_service, aic_registry, top_k=100, rrf_k=60)
    
    all_records = []
    for q in queries:
        v = next((v for v in variants if v.query_id == q.query_id), None)
        if v:
            all_records.extend(runner.run_query(q, v))

    # Output results
    results_json = args.output / "ablation_results.json"
    results_csv = args.output / "ablation_results.csv"

    results_json.write_text(
        json.dumps([r.to_dict() for r in all_records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if all_records:
        with results_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].to_dict().keys())
            writer.writeheader()
            for r in all_records:
                writer.writerow(r.to_dict())

    # Calculate summaries
    from collections import defaultdict
    import statistics
    
    by_variant = defaultdict(list)
    for r in all_records:
        by_variant[r.variant].append(r)

    summaries = []
    for v_name, records in by_variant.items():
        latencies = [r.retrieval_latency_ms for r in records]
        summaries.append(
            VariantSummary(
                variant=v_name,
                total_queries=len(records),
                r_at_1=statistics.mean([r.r_at_1 for r in records]),
                r_at_5=statistics.mean([r.r_at_5 for r in records]),
                r_at_20=statistics.mean([r.r_at_20 for r in records]),
                r_at_50=statistics.mean([r.r_at_50 for r in records]),
                r_at_100=statistics.mean([r.r_at_100 for r in records]),
                final_score=statistics.mean([r.final_score for r in records]),
                mean_latency_ms=statistics.mean(latencies),
                median_latency_ms=statistics.median(latencies),
            )
        )

    summary_path = args.output / "ablation_summary.json"
    summary_path.write_text(
        json.dumps([s.to_dict() for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Pairwise
    pairs = [
        ("en_full", "vi_raw"),
        ("qwen_keyword", "en_full"),
        ("en_full_qwen_rrf", "en_full"),
        ("en_full_qwen_rrf", "qwen_keyword"),
    ]
    pairwise_results = []
    for a, b in pairs:
        comp = compute_pairwise_comparison(by_variant[a], by_variant[b], a, b)
        pairwise_results.append(comp)

    pairwise_csv = args.output / "ablation_pairwise.csv"
    if pairwise_results:
        with pairwise_csv.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=pairwise_results[0].to_dict().keys())
            writer.writeheader()
            for c in pairwise_results:
                writer.writerow(c.to_dict())

    # Markdown Report
    report_path = args.output / "ablation_report.md"
    report_lines = [
        "# TKIS Query-Variant Retrieval Ablation Report",
        f"- Git Commit: `{_get_git_commit()}`",
        f"- Query Source: `{args.gt}`",
        f"- Query Count: {len(queries)}",
        f"- Representative: {audit['representative']} (DEVELOPMENT-ONLY)",
        f"- Translation Model: {translator.name}",
        f"- Qwen Model: {decomposer.name}",
        f"- Qwen Schema Version: {QWEN_SCHEMA_VERSION}",
        f"- Semantic Model: {faiss_index.manifest.get('embedding_model', 'unknown')}",
        f"- Pretrained: {faiss_index.manifest.get('pretrained', 'unknown')}",
        f"- Embedding Dimension: {faiss_index.manifest.get('dimension', 'unknown')}",
        f"- Index Path: `{args.index}`",
        f"- Device: `{args.device}`",
        f"- Refinement: OFF",
        f"- Final top_k: 100",
        f"- RRF k: 60",
        "",
        "## Summary Table",
        "| Variant | R@1 | R@5 | R@20 | R@50 | R@100 | Final | Retrieval latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        report_lines.append(
            f"| {s.variant} | {s.r_at_1:.4f} | {s.r_at_5:.4f} | {s.r_at_20:.4f} | {s.r_at_50:.4f} | {s.r_at_100:.4f} | {s.final_score:.4f} | {s.mean_latency_ms:.2f}ms |"
        )
    
    report_lines.append("")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

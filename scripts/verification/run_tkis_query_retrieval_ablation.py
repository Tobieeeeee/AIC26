"""Run the TKIS query-variant retrieval ablation and report the result.

Answers one question: does sending the original Vietnamese to BM25, instead
of the English translation everything currently gets, retrieve better?

The five variants, all scored by the official KIS rule:

    vi_raw            Vietnamese to both branches
    en_full           English to both branches            <- today's baseline
    qwen_keyword      Qwen-decomposed keywords to both
    en_full_qwen_rrf  the two English variants fused
    vi_lex_en_dense   English to dense, Vietnamese to BM25 <- the proposal

`vi_lex_en_dense` needs a lexical branch. Without `--elasticsearch-url` it
records `lexical_branch_unavailable` instead of running, because with no
lexical branch it is byte-for-byte `en_full` and reporting it as a result
would put a duplicate row beside its own baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for candidate in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from scripts.verification.query_retrieval_ablation.inputs import (  # noqa: E402
    load_ablation_queries,
)
from scripts.verification.query_retrieval_ablation.metrics import (  # noqa: E402
    compare_variants,
    summarize_variant,
)
from scripts.verification.query_retrieval_ablation.models import (  # noqa: E402
    QueryVariant,
)
from scripts.verification.query_retrieval_ablation.runner import (  # noqa: E402
    AblationRunner,
)

VARIANTS = (
    "vi_raw",
    "en_full",
    "qwen_keyword",
    "en_full_qwen_rrf",
    "vi_lex_en_dense",
)

#: The comparison the benchmark exists to settle: baseline, then candidate.
HEADLINE_COMPARISON = ("en_full", "vi_lex_en_dense")


def load_variants(path: Path) -> dict[str, QueryVariant]:
    """Read the frozen variant manifest `variants.freeze_query_variants` writes.

    Frozen deliberately: translation and decomposition are model calls, and
    re-running them per benchmark would make two runs of the same queries
    incomparable for reasons that have nothing to do with retrieval.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    variants: dict[str, QueryVariant] = {}
    for row in payload:
        variant = QueryVariant.from_dict(row)
        if variant.query_id in variants:
            raise ValueError(f"duplicate query_id in manifest: {variant.query_id}")
        variants[variant.query_id] = variant
    return variants


def build_multimodal_service(
    semantic_service, elasticsearch_url: str, index_name: str
):
    """Wire the BM25 branch, or return None when no Elasticsearch was given."""
    if not elasticsearch_url or not index_name:
        return None

    from elasticsearch import Elasticsearch

    from mvr.retrieval.lexical_bm25 import search_task05b_lexical
    from mvr.retrieval.qna_fusion import FusedRetriever

    client = Elasticsearch(elasticsearch_url)

    def lexical_search(query, *, top_k):
        return search_task05b_lexical(client, query, index_name=index_name, top_k=top_k)

    return FusedRetriever(
        semantic_service=semantic_service, lexical_search=lexical_search
    )


def _cell(value: float | None) -> str:
    return "  --  " if value is None else f"{value:.4f}"


def render_report(summaries, comparison) -> str:
    """A fixed-width table, plus the one comparison that decides the question."""
    lines = [
        "TKIS query-variant retrieval ablation",
        "",
        f"{'variant':<18}{'R@1':>8}{'R@5':>8}{'R@20':>8}{'R@50':>8}{'R@100':>8}"
        f"{'Final':>8}{'ran':>6}{'failed':>8}",
        "-" * 80,
    ]
    for summary in summaries:
        lines.append(
            f"{summary.variant:<18}"
            f"{_cell(summary.r_at_1):>8}{_cell(summary.r_at_5):>8}"
            f"{_cell(summary.r_at_20):>8}{_cell(summary.r_at_50):>8}"
            f"{_cell(summary.r_at_100):>8}{_cell(summary.final_score):>8}"
            f"{summary.successful_queries:>6}{summary.failed_queries:>8}"
        )

    failures = [s for s in summaries if s.error_counts]
    if failures:
        lines += ["", "Failures (excluded from every mean above):"]
        for summary in failures:
            for error, count in summary.error_counts.items():
                lines.append(f"  {summary.variant:<18} {error:<30} {count}")

    lines += [
        "",
        f"{comparison.variant_b} vs {comparison.variant_a}"
        f"  (baseline = {comparison.variant_a}):",
        f"  compared     {comparison.valid_pairs}"
        f"   (excluded {comparison.excluded_pairs} where either side failed)",
        f"  better       {comparison.wins}"
        f"    same {comparison.ties}    worse {comparison.losses}",
        f"  rank moved   earlier {comparison.moves_earlier}"
        f"   later {comparison.moves_later}",
        f"  rescued      {comparison.rescues}"
        f"    regressed {comparison.regressions}",
    ]
    if comparison.valid_pairs == 0:
        lines += [
            "",
            "NOTHING WAS COMPARED. Every pair was excluded, so this run says",
            "nothing about the split either way -- read the failures above.",
        ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument(
        "--variants-manifest",
        type=Path,
        required=True,
        help="JSON written by variants.freeze_query_variants",
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path)
    parser.add_argument("--elasticsearch-url", default="")
    parser.add_argument(
        "--lexical-index",
        default="",
        help="Task05B index name; required for the vi_lex_en_dense variant",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tkis"))
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def build_semantic_service(index_dir: Path, device: str):
    """Take the encoder from the index manifest, never from a constant.

    An index built with ViT-B-32 queried by an RN50 text encoder returns
    ranked, plausible, meaningless results -- the vectors are the wrong
    shape of wrong, not an error. `run_real_data_kis_smoke.py` reads the
    manifest for this reason and so does this.
    """
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.retrieval.semantic import SemanticRetriever

    index = FaissIndex.load(index_dir)
    encoder = OpenClipEncoder(
        model_name=index.manifest.get("embedding_model", "ViT-B-32"),
        pretrained=index.manifest.get("pretrained", "laion2b_s34b_b79k"),
        device=device,
    )
    return SemanticRetriever(encoder, index)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    queries = load_ablation_queries(args.ground_truth)
    if args.limit:
        queries = queries[: args.limit]
    variants = load_variants(args.variants_manifest)

    missing = [q.query_id for q in queries if q.query_id not in variants]
    if missing:
        raise SystemExit(
            f"{len(missing)} queries have no frozen variant: {missing[:5]}. "
            "Run variants.freeze_query_variants first."
        )

    semantic_service = build_semantic_service(args.index_dir, args.device)
    multimodal_service = build_multimodal_service(
        semantic_service, args.elasticsearch_url, args.lexical_index
    )
    if multimodal_service is None:
        print(
            "No lexical branch (pass --elasticsearch-url and --lexical-index). "
            "vi_lex_en_dense will not run.",
            file=sys.stderr,
        )

    pipeline_resolver = None
    if args.mapping_csv:
        from mvr.retrieval.kis import KeyframeFrameIndexResolver

        pipeline_resolver = KeyframeFrameIndexResolver.from_csv(args.mapping_csv)

    runner = AblationRunner(
        semantic_service=semantic_service,
        aic_registry=None,
        top_k=args.top_k,
        rrf_k=args.rrf_k,
        multimodal_service=multimodal_service,
        pipeline_resolver=pipeline_resolver,
    )

    by_variant: dict[str, list] = {name: [] for name in VARIANTS}
    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {query.query_id}", file=sys.stderr)
        for record in runner.run_query(query, variants[query.query_id]):
            by_variant[record.variant].append(record)

    summaries = [summarize_variant(name, by_variant[name]) for name in VARIANTS]
    baseline, candidate = HEADLINE_COMPARISON
    comparison = compare_variants(
        baseline, candidate, by_variant[baseline], by_variant[candidate]
    )

    report = render_report(summaries, comparison)
    print(report)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "ablation_results.json").write_text(
        json.dumps(
            {
                "query_count": len(queries),
                "summaries": [s.to_dict() for s in summaries],
                "comparison": comparison.to_dict(),
                "records": [
                    record.to_dict()
                    for name in VARIANTS
                    for record in by_variant[name]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

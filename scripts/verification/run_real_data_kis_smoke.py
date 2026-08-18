import argparse
import json
import time
from typing import Any
from pathlib import Path

from mvr.retrieval.kis import search_kis

try:
    from mvr.ingestion.aic_dataset import AicDatasetRegistry
except ImportError:
    AicDatasetRegistry = None

try:
    from scripts.verification.query_translation.local import LocalTranslator
    translator_instance = LocalTranslator()
    def translate_vi_to_en(text):
        return translator_instance.translate(text)
except Exception as e:
    print(f"Translation module load error: {e}")
    translate_vi_to_en = None

try:
    from scripts.verification.query_decomposition.qwen import QwenDecomposer
    from scripts.verification.query_decomposition.utils import build_keyword_query
    qwen_decomposer = QwenDecomposer("Qwen/Qwen2.5-0.5B-Instruct")
    def decompose_query_structured(query):
        concepts, kw_q, raw = qwen_decomposer.decompose(query)
        return concepts
except Exception as e:
    print(f"Decomposition module load error: {e}")
    decompose_query_structured = None
    build_keyword_query = None

try:
    from scripts.verification.query_retrieval_ablation.fusion import reciprocal_rank_fusion
except ImportError:
    reciprocal_rank_fusion = None


def run_smoke_test(
    queries: list[dict],
    modes: list[str],
    top_k: int,
    semantic_service: Any,
    registry: AicDatasetRegistry = None,
    pipeline_resolver: Any = None,
    output_dir: Path = None,
):
    results = []

    for idx, q_dict in enumerate(queries):
        query_id = q_dict.get("query_id", f"query_{idx:03d}")
        query_vi = q_dict.get("query_vi", "")
        
        print(f"\n--- Processing Query: {query_id} ---")
        print(f"Original (VI): {query_vi}")

        # Shared pre-processing
        query_en = None
        translation_latency = 0.0
        
        keyword_query = None
        decomposition_latency = 0.0

        if "en_full" in modes or "qwen_keyword" in modes or "rrf" in modes:
            if translate_vi_to_en:
                start_t = time.perf_counter()
                try:
                    query_en = translate_vi_to_en(query_vi)
                except Exception as e:
                    print(f"Translation failed: {e}")
                    query_en = None
                translation_latency = (time.perf_counter() - start_t) * 1000.0
            else:
                print("Warning: Translation module unavailable.")

        if "qwen_keyword" in modes or "rrf" in modes:
            if decompose_query_structured and build_keyword_query and query_en:
                start_t = time.perf_counter()
                try:
                    struct = decompose_query_structured(query_en)
                    keyword_query = build_keyword_query(struct)
                except Exception as e:
                    print(f"Decomposition failed: {e}")
                    keyword_query = None
                decomposition_latency = (time.perf_counter() - start_t) * 1000.0
            else:
                print("Warning: Qwen decomposition unavailable or translation failed.")

        # Execute modes
        mode_results = {}
        c_en = []
        c_qwen = []
        
        def _search(text: str):
            if not text:
                return [], 0.0, "empty_query"
            start_s = time.perf_counter()
            try:
                candidates = search_kis(
                    text,
                    top_k=top_k,
                    semantic_service=semantic_service,
                    multimodal_service=None,
                    pipeline_resolver=pipeline_resolver,
                    aic_registry=registry
                )
                lat = (time.perf_counter() - start_s) * 1000.0
                err = "empty_mapped_candidates" if not candidates else None
                return candidates, lat, err
            except Exception as e:
                lat = (time.perf_counter() - start_s) * 1000.0
                return [], lat, f"retrieval_crash: {str(e)}"

        for mode in modes:
            cands = []
            lat = 0.0
            err = None
            
            if mode == "vi_raw":
                cands, lat, err = _search(query_vi)
                
            elif mode == "en_full":
                if not query_en:
                    err = "translation_failed"
                else:
                    cands, lat, err = _search(query_en)
                c_en = cands

            elif mode == "qwen_keyword":
                if not keyword_query:
                    err = "decomposition_failed"
                else:
                    cands, lat, err = _search(keyword_query)
                c_qwen = cands

            elif mode == "rrf":
                if not reciprocal_rank_fusion:
                    err = "rrf_module_missing"
                elif not query_en:
                    err = "rrf_missing_en_branch"
                elif not keyword_query:
                    err = "rrf_missing_keyword_branch"
                else:
                    start_f = time.perf_counter()
                    # Reusing cached candidates from en_full and qwen_keyword if they were run.
                    # Wait, if they were NOT in modes, c_en/c_qwen are empty!
                    # If rrf is requested but not the individual modes, we must run them!
                    if not c_en and query_en:
                         c_en, _, _ = _search(query_en)
                    if not c_qwen and keyword_query:
                         c_qwen, _, _ = _search(keyword_query)
                         
                    cands = reciprocal_rank_fusion([c_en, c_qwen], rrf_k=60, top_k=top_k)
                    fusion_time = (time.perf_counter() - start_f) * 1000.0
                    lat = fusion_time
                    err = "empty_mapped_candidates" if not cands else None

            else:
                err = f"unknown_mode_{mode}"
                
            top_k_list = [
                {
                    "rank": c.rank,
                    "video_id": c.video_id,
                    "frame_id": c.frame_id,
                    "score": c.score
                } for c in cands
            ]
            
            status = "PASS" if not err else "FAIL"
            print(f"  [{mode:12s}] {status:4s} | Candidates: {len(cands):<3d} | Latency: {lat:.1f}ms")
            if err:
                print(f"    Error: {err}")

            mode_results[mode] = {
                "status": status,
                "candidate_count": len(cands),
                "latency_ms": round(lat, 2),
                "error": err,
                "top_k": top_k_list
            }

        res = {
            "query_id": query_id,
            "query_vi": query_vi,
            "query_en": query_en,
            "keyword_query": keyword_query,
            "translation_latency_ms": round(translation_latency, 2),
            "decomposition_latency_ms": round(decomposition_latency, 2),
            "modes": mode_results
        }
        results.append(res)
        
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "query_results.json"
        json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        
        md_path = output_dir / "smoke_report.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write("# Real-Data No-GT Smoke Test Report\n\n")
            f.write("This is a qualitative smoke test. No R@K or Final Score conclusions are made.\n\n")
            f.write("```text\n")
            f.write(f"{'Query':<15} {'Mode':<15} {'Status':<8} {'Cands':<6} {'Latency':<10} {'Error'}\n")
            f.write("-" * 70 + "\n")
            for r in results:
                q_id = r["query_id"]
                for m_name, m_data in r["modes"].items():
                    status = m_data["status"]
                    cands = m_data["candidate_count"]
                    lat = f"{m_data['latency_ms']:.1f}ms"
                    err = m_data["error"] or ""
                    f.write(f"{q_id:<15} {m_name:<15} {status:<8} {cands:<6} {lat:<10} {err}\n")
            f.write("```\n")
        print(f"\nSaved results to {output_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Real-Data KIS Smoke Test")
    parser.add_argument("--query", type=str, help="Single Vietnamese query")
    parser.add_argument("--queries-file", type=Path, help="JSON file with queries")
    parser.add_argument("--modes", type=str, default="en_full", help="Comma-separated modes (vi_raw,en_full,qwen_keyword,rrf)")
    parser.add_argument("--top-k", type=int, default=20, help="Top-K candidates to retrieve")
    parser.add_argument("--data-root", type=Path, help="AIC dataset root path")
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/indexes/aic_l21"), help="Semantic index directory")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/verification/tkis_real_data_smoke"), help="Output directory")
    parser.add_argument("--device", type=str, help="Device for semantic model")
    parser.add_argument("--mapping-csv", type=Path, help="CSV file mapping keyframe_id to frame_idx")
    
    parser.add_argument("--query-column", type=str, default="text", help="Column containing the Vietnamese query in CSV")
    parser.add_argument("--limit", type=int, help="Limit number of queries to run")
    
    args = parser.parse_args()
    
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("No valid modes provided.")
        return
        
    queries = []
    if args.query:
        queries.append({"query_id": "cli_001", "query_vi": args.query})
    elif args.queries_file and args.queries_file.exists():
        if args.queries_file.suffix.lower() == '.csv':
            import pandas as pd
            df = pd.read_csv(args.queries_file, encoding='utf-8')
            if 'task' in df.columns:
                df = df[df['task'] == 'TKIS']
            col = args.query_column
            if col in df.columns:
                for idx, row in df.iterrows():
                    queries.append({"query_id": str(row.get('query_id', f'Q_{idx}')), "query_vi": str(row[col])})
            else:
                print(f"Column {col} not found in CSV. Available: {df.columns.tolist()}")
                return
        else:
            queries = json.loads(args.queries_file.read_text(encoding="utf-8"))
            
    if args.limit and args.limit > 0:
        queries = queries[:args.limit]
        
    if not queries:
        print("No queries provided. Use --query or --queries-file.")
        return
        
    print(f"Loading semantic index from {args.index_dir}...")
    from mvr.retrieval.semantic import SemanticRetriever
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.indexing.faiss_index import FaissIndex
    
    device = args.device or "auto"
    index = FaissIndex.load(args.index_dir)
    model_name = index.manifest.get("embedding_model", "ViT-B-32")
    pretrained = index.manifest.get("pretrained", "laion2b_s34b_b79k")
    encoder = OpenClipEncoder(model_name=model_name, pretrained=pretrained, device=device)
    semantic_service = SemanticRetriever(encoder, index)
    
    registry = None
    if args.data_root and AicDatasetRegistry:
        registry = AicDatasetRegistry(args.data_root)
        print(f"Loaded AIC registry from {args.data_root}")
        
    mapping_path = args.mapping_csv if args.mapping_csv else Path("artifacts/ingest_public_test/keyframe_mapping.csv")
    if not mapping_path.exists():
        print(f"Public keyframe mapping unavailable: {mapping_path}")
        return
        
    from mvr.retrieval.kis import KeyframeFrameIndexResolver
    pipeline_resolver = KeyframeFrameIndexResolver.from_csv(mapping_path)
    print(f"Loaded pipeline resolver from {mapping_path}")
    run_smoke_test(queries, modes, args.top_k, semantic_service, registry, pipeline_resolver, args.output_dir)

if __name__ == "__main__":
    main()

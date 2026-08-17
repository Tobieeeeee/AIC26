import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from scripts.verification.query_decomposition.models import DecompositionResult, QueryConcepts
from scripts.verification.query_decomposition.qwen import QwenDecomposer


def load_input_queries(input_path: str, source_method: str) -> List[Dict[str, str]]:
    """Load valid English queries from Task 01 output or a generic CSV."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    queries = []
    
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for row in data:
            if row.get("method") == source_method and row.get("success") is True and row.get("translated_text"):
                queries.append({
                    "query_id": row["query_id"],
                    "query_vi": row.get("query_vi", ""),
                    "query_en": row["translated_text"]
                })
    elif path.suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "query_id" in row and "query_en" in row and row["query_en"].strip():
                    queries.append({
                        "query_id": row["query_id"],
                        "query_vi": row.get("query_vi", ""),
                        "query_en": row["query_en"].strip()
                    })
    else:
        raise ValueError("Unsupported input format. Must be .json or .csv")
        
    return queries


def get_cache_key(query_en: str, model_name: str) -> str:
    """Stable cache key for the query and model combination."""
    import hashlib
    key_content = f"{query_en}|{model_name}|v1_schema|do_sample=False".encode("utf-8")
    return hashlib.md5(key_content).hexdigest()


def load_cache(cache_dir: Path) -> Dict[str, Any]:
    cache_file = cache_dir / "cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache_dir: Path, cache: Dict[str, Any]):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cache.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_benchmark(args):
    input_path = args.input
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/4] Loading queries from {input_path} (source method: {args.source_method}) ...")
    queries = load_input_queries(input_path, args.source_method)
    
    if args.limit and args.limit > 0:
        queries = queries[:args.limit]
        
    print(f"       {len(queries)} valid queries loaded.")
    
    cache_dir = output_dir / ".cache"
    cache = load_cache(cache_dir) if not args.no_cache else {}
    
    print(f"\n[2/4] Decomposing with: {args.qwen_model}")
    
    decomposer = QwenDecomposer(model_name=args.qwen_model, device=args.device)
    
    results: List[DecompositionResult] = []
    
    success_count = 0
    failure_count = 0
    json_parse_failures = 0
    
    for q in queries:
        query_id = q["query_id"]
        query_vi = q["query_vi"]
        query_en = q["query_en"]
        
        cache_key = get_cache_key(query_en, args.qwen_model)
        
        if not args.no_cache and cache_key in cache:
            cached_data = cache[cache_key]
            # Reconstruct QueryConcepts
            concepts = QueryConcepts(**cached_data["concepts"])
            
            res = DecompositionResult(
                query_id=query_id,
                query_vi=query_vi,
                query_en_full=query_en,
                qwen_model=args.qwen_model,
                concepts=concepts,
                keyword_query=cached_data["keyword_query"],
                latency_ms=cached_data["latency_ms"],
                success=cached_data["success"],
                error=cached_data["error"],
                raw_model_output=cached_data["raw_model_output"]
            )
            results.append(res)
            if res.success:
                success_count += 1
            else:
                failure_count += 1
            continue
            
        start_t = time.time()
        try:
            concepts, keyword_query, raw_output = decomposer.decompose(query_en)
            latency_ms = (time.time() - start_t) * 1000.0
            
            res = DecompositionResult(
                query_id=query_id,
                query_vi=query_vi,
                query_en_full=query_en,
                qwen_model=args.qwen_model,
                concepts=concepts,
                keyword_query=keyword_query,
                latency_ms=latency_ms,
                success=True,
                raw_model_output=raw_output
            )
            success_count += 1
        except Exception as e:
            latency_ms = (time.time() - start_t) * 1000.0
            error_msg = str(e)
            if "Failed to parse JSON" in error_msg:
                json_parse_failures += 1
                
            res = DecompositionResult(
                query_id=query_id,
                query_vi=query_vi,
                query_en_full=query_en,
                qwen_model=args.qwen_model,
                concepts=QueryConcepts(),
                keyword_query="",
                latency_ms=latency_ms,
                success=False,
                error=error_msg
            )
            failure_count += 1
            
        results.append(res)
        
        # Save to cache
        cache[cache_key] = res.to_dict()

    if not args.no_cache:
        save_cache(cache_dir, cache)
        
    print(f"       Done: {success_count} success, {failure_count} failures.")

    print("\n[3/4] Writing results ...")
    
    json_path = output_dir / "query_decomposition_results.json"
    csv_path = output_dir / "query_decomposition_results.csv"
    summary_path = output_dir / "query_decomposition_summary.json"
    
    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
    # Write CSV
    if results:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "query_id", "query_vi", "query_en_full", "qwen_model", 
                "keyword_query", "latency_ms", "success", "error", "concepts"
            ])
            writer.writeheader()
            for r in results:
                d = r.to_dict()
                d["concepts"] = json.dumps(d["concepts"], ensure_ascii=False)
                # Omit raw_model_output from CSV to keep it clean
                d.pop("raw_model_output", None)
                d.pop("timestamp", None)
                writer.writerow(d)
                
    # Calculate summary
    successful_latencies = [r.latency_ms for r in results if r.success]
    total = len(results)
    
    if successful_latencies:
        import statistics
        mean_lat = statistics.mean(successful_latencies)
        median_lat = statistics.median(successful_latencies)
        sorted_lat = sorted(successful_latencies)
        p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 0 else 0.0
    else:
        mean_lat = 0.0
        median_lat = 0.0
        p95_lat = 0.0

    summary = {
        "queries": total,
        "successes": success_count,
        "failures": failure_count,
        "failure_rate": (failure_count / total) if total > 0 else 0.0,
        "mean_latency_ms": round(mean_lat, 1),
        "median_latency_ms": round(median_lat, 1),
        "p95_latency_ms": round(p95_lat, 1),
        "json_parse_failures": json_parse_failures,
        "repair_attempts": json_parse_failures, # Simplified tracking
        "repair_successes": 0 # Tracked within QwenDecomposer if we refactored
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[4/4] Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<20}: {v}")
        
    print(f"\nResults CSV  : {csv_path.absolute()}")
    print(f"Results JSON : {json_path.absolute()}")
    print(f"Summary      : {summary_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TKIS Qwen Query Decomposition Benchmark")
    parser.add_argument("--input", required=True, help="Path to input JSON or CSV file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--source-method", default="local_vi2en", help="Method name to filter by in JSON input")
    parser.add_argument("--qwen-model", default="Qwen/Qwen1.5-1.8B", help="Hugging Face model ID")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device to run on")
    parser.add_argument("--limit", type=int, help="Limit number of queries to process")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    run_benchmark(args)

"""Minimal end-to-end TRAKE pipeline runner: one query file in, ranked
never-blank chain predictions out.

Wires the real Phase 02-08 components (event/gap contract, orchestration,
dense scoring, chain alignment, never-blank fallback, local verification
export) against a local AIC dataset root and a persisted semantic index.
Closes the gap docs/AIC_TRAKE_FINAL_REPORT.md's "API/library-only, no
mvr.cli subcommand" limitation flagged: apply_never_blank_fallback and the
export functions previously had no real caller outside
tests/integration/test_aic_trake_e2e.py.

No organizer TRAKE query file format is documented anywhere in this repo's
source material, so the input format below is this script's own -- the
same {"events": [...], "gap_phrases": [...]} shape POST /search/trake
already accepts and validates (mvr.api.schemas.TrakeSearchRequest), not
invented ad hoc for this script alone.

Example:
    python scripts/run_trake_pipeline.py \\
        --query-file query.json \\
        --index artifacts/indexes/aic_l21 \\
        --dataset-root "E:\\AIC\\AIC_DATASET_ROOT" \\
        --output-dir artifacts/verification/trake_submission

query.json:
    {
      "events": ["a car crashes into a pole at night", "police arrive"],
      "gap_phrases": ["rồi"]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to a legacy codepage (e.g. cp1252) that
    # cannot encode Vietnamese text; force UTF-8 so event/translation output
    # never crashes on non-ASCII characters regardless of the caller's
    # console configuration.
    sys.stdout.reconfigure(encoding="utf-8")


def load_query_file(path: Path) -> tuple[list[str], list[str]]:
    """Read ``{"events": [...], "gap_phrases": [...]}`` -- the same shape
    ``TrakeSearchRequest`` validates at the API layer. Raises a clear error
    rather than guessing on a malformed file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    gap_phrases = payload.get("gap_phrases", [])
    if not isinstance(events, list) or len(events) < 2 or not all(
        isinstance(e, str) and e.strip() for e in events
    ):
        raise SystemExit(f"{path}: 'events' must be a list of at least 2 non-blank strings")
    if not isinstance(gap_phrases, list) or len(gap_phrases) != len(events) - 1:
        raise SystemExit(
            f"{path}: expected {len(events) - 1} gap_phrases for {len(events)} events, "
            f"got {len(gap_phrases) if isinstance(gap_phrases, list) else gap_phrases!r}"
        )
    return events, gap_phrases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TRAKE query end-to-end")
    parser.add_argument(
        "--query-file", type=Path, required=True,
        help='JSON file: {"events": [...], "gap_phrases": [...]}',
    )
    parser.add_argument("--index", type=Path, required=True, help="Semantic FAISS index directory")
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="AIC dataset root with videos/keyframes/map_keyframes",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write the local verification export (CSV + JSON)",
    )
    parser.add_argument("--query-id", type=str, default=None, help="Defaults to the query file's stem")
    parser.add_argument(
        "--top-k", type=int, default=100,
        help=(
            "How many ranked chains to export (competition maximum: 100). "
            "Defaults to the full budget: the official score averages R@1/5/"
            "20/50/100, so a smaller value discards score that costs nothing "
            "but compute to claim."
        ),
    )
    parser.add_argument(
        "--alignment-strategy", choices=("atlas", "lucifer", "ucese"), default="atlas",
        help=(
            "Defaults to the evidence-backed production default: ATLAS tied "
            "Lucifer on every real benchmark query while running faster and "
            "without a tuning cliff -- see docs/AIC_TRAKE_BENCHMARK_RESULTS.md."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--translate-query", action="store_true",
        help=(
            "Translate each event to English before retrieval. CLIP's text "
            "tower is English-trained: measured on this index, a Vietnamese "
            "query ranked the correct frame 192/268 vs. #1/268 in English "
            "for the same target. See docs/AIC_TRAKE_FINAL_REPORT.md's "
            "'Query Translation' section for the measured before/after."
        ),
    )
    parser.add_argument(
        "--translation-model", type=str, default=None,
        help="Defaults to HelsinkiViEnTranslator's model",
    )
    parser.add_argument(
        "--allow-translation-download", action="store_true",
        help="Permit downloading the translation model (fails closed otherwise)",
    )
    parser.add_argument(
        "--official-csv", type=Path, default=None,
        help=(
            "Also write the REAL AIC26 organizer submission CSV to this exact "
            "path. No header row, comma-delimited, video_id + N frame IDs per "
            "row -- verified byte-for-byte against the organizer's own "
            "published example from https://sotuyenaic.oj.io.vn/. This is "
            "NOT the same file as --output-dir's local verification export; "
            "write both if you want to keep the debug copy too. Prefer "
            "--query-number + --official-dir below unless you need an exact "
            "custom path/filename."
        ),
    )
    parser.add_argument(
        "--query-number", type=int, default=None,
        help=(
            "This query's sequence number in the organizer's batch (the 'N' "
            "in the query-N-trake.txt file the organizer handed out). "
            "Combined with --official-dir, generates the exact "
            "query-<N>-trake.csv filename the organizer expects "
            "(mvr.submission.trake_export.official_submission_filename) -- "
            "ignored if --official-csv is also given."
        ),
    )
    parser.add_argument(
        "--official-dir", type=Path, default=None,
        help="Directory to write the --query-number-generated official CSV into",
    )
    args = parser.parse_args()

    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry
    from mvr.retrieval.semantic import SemanticRetriever
    from mvr.retrieval.trake import search_trake
    from mvr.retrieval.trake_align import AtlasAlignment, LuciferAlignment, UceseAlignment
    from mvr.retrieval.trake_events import TrakeEventQuery, TrakeQuery, translate_gap_phrase
    from mvr.submission.trake_export import (
        InvalidTrakePredictionError,
        apply_never_blank_fallback,
        export_local_trake_verification_csv,
        export_local_trake_verification_json,
        export_official_trake_submission_csv,
        official_submission_filename,
        validate_trake_prediction,
    )

    official_csv_path = args.official_csv
    if official_csv_path is None and args.query_number is not None:
        if args.official_dir is None:
            raise SystemExit("--query-number requires --official-dir")
        official_csv_path = args.official_dir / official_submission_filename(
            args.query_number, "trake"
        )

    events, gap_phrases = load_query_file(args.query_file)
    query_id = args.query_id or args.query_file.stem
    print(f"[events] {events}")
    print(f"[gap_phrases] {gap_phrases}")

    index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=index.manifest["embedding_model"],
        pretrained=index.manifest["pretrained"],
        device=args.device,
    )
    semantic_service = SemanticRetriever(encoder, index)
    aic_registry = AicDatasetRegistry(args.dataset_root)

    query_translator = None
    if args.translate_query:
        from mvr.enrichment.qna_translation import HelsinkiViEnTranslator

        translator_kwargs = {"allow_model_download": args.allow_translation_download}
        if args.translation_model:
            translator_kwargs["model_name"] = args.translation_model
        query_translator = HelsinkiViEnTranslator(**translator_kwargs)
        print("[retrieval query EN]", [query_translator.translate(e) for e in events])

    query = TrakeQuery(
        events=tuple(TrakeEventQuery(i, text) for i, text in enumerate(events, start=1)),
        gaps=tuple(translate_gap_phrase(phrase) for phrase in gap_phrases),
    )

    strategy_classes = {"atlas": AtlasAlignment, "lucifer": LuciferAlignment, "ucese": UceseAlignment}
    strategy = strategy_classes[args.alignment_strategy]()
    print(f"[alignment_strategy] {args.alignment_strategy}")

    chains = search_trake(
        query,
        alignment_strategy=strategy,
        semantic_service=semantic_service,
        aic_registry=aic_registry,
        top_k=args.top_k,
        query_translator=query_translator,
    )

    if not chains:
        print("(no candidate chains found)")
        return

    # fps is per-video, not per-chain -- cache it so a 100-chain result
    # from N videos does only N registry lookups, not 100.
    fps_cache: dict[str, float] = {}

    def video_fps(video_id: str) -> float:
        if video_id not in fps_cache:
            fps_cache[video_id] = aic_registry.resolve_frame(video_id, "1").fps
        return fps_cache[video_id]

    rows = []
    for chain in chains:
        try:
            row = apply_never_blank_fallback(chain, query.gaps, video_fps=video_fps(chain.video_id))
            validate_trake_prediction(row, k=len(query.events))
        except InvalidTrakePredictionError as exc:
            print(f"[skip] {chain.video_id} {chain.frames}: {exc}")
            continue
        rows.append(row)

    if not rows:
        print("(every candidate chain failed the never-blank fallback / validation)")
        return

    csv_path = export_local_trake_verification_csv(
        rows, args.output_dir / f"{query_id}.csv", query_id=query_id
    )
    json_path = export_local_trake_verification_json(
        rows, args.output_dir / f"{query_id}.json", query_id=query_id
    )
    print(f"wrote {len(rows)} ranked chains to {csv_path} and {json_path}")
    for rank, row in enumerate(rows[:5], start=1):
        print(f"  #{rank} {row.video_id} {row.frames} interpolated={row.is_interpolated}")
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more")

    if official_csv_path is not None:
        try:
            official_path = export_official_trake_submission_csv(rows, official_csv_path)
        except InvalidTrakePredictionError as exc:
            raise SystemExit(f"cannot write the official submission CSV: {exc}") from exc
        print(f"wrote the official AIC26 submission CSV to {official_path}")
        print(
            "remember: put this file inside a 'submission/' folder and zip that "
            "folder (not the loose CSV) before uploading to "
            "https://sotuyenaic.oj.io.vn/"
        )


if __name__ == "__main__":
    main()

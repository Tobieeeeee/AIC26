"""Minimal end-to-end Q&A pipeline runner: one organizer query file in,
ranked answered predictions out.

Wires the real Task 02-05 components (structured context, LLM answer
reasoner, orchestration) against a local AIC dataset root and a persisted
semantic index. No Elasticsearch, no live COCO-detector fallback -- matches
the default scope documented in docs/AIC_QNA_IMPLEMENTATION_NOTES.md.

Example:
    python scripts/run_qna_pipeline.py \\
        --query-file "D:\\AIC_query\\query-p1-groupA\\query-p1-15-qa.txt" \\
        --index "artifacts\\indexes\\qna_l21_l22" \\
        --dataset-root "D:\\AIC26_test_root" \\
        --allow-model-download
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    # Windows consoles often default to a legacy codepage (e.g. cp1252) that
    # cannot encode Vietnamese text; force UTF-8 so query/answer output
    # never crashes on non-ASCII characters regardless of the caller's
    # console configuration.
    sys.stdout.reconfigure(encoding="utf-8")


def split_event_and_question(text: str) -> tuple[str, str]:
    """Split one organizer Q&A query file into (event_description, question).

    Organizer files are a single text blob ending in a "Hỏi ...?" question
    sentence -- there is no separate field for it. Splits at the first
    sentence containing "Hỏi" or ending in "?"; if no such sentence exists,
    returns the full text for both halves rather than guessing which part
    is the question.
    """
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    question_idx = next(
        (i for i, s in enumerate(sentences) if "Hỏi" in s or s.strip().endswith("?")),
        None,
    )
    if question_idx is None:
        return text.strip(), text.strip()
    question = " ".join(sentences[question_idx:]).strip()
    event_description = " ".join(sentences[:question_idx]).strip() or text.strip()
    return event_description, question


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Q&A query end-to-end")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True, help="Semantic FAISS index directory")
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="AIC dataset root with videos/keyframes/map_keyframes/objects",
    )
    parser.add_argument(
        "--top-k", type=int, default=100,
        help=(
            "How many ranked answers to produce (competition maximum: 100). "
            "Defaults to the full budget because the final score averages "
            "R@1/5/20/50/100 -- three of those five cutoffs are unreachable "
            "below 20 answers, so a smaller value discards score that costs "
            "nothing but compute to claim. Lower it for quick local smoke "
            "tests, where OCR over 100 frames is the slow part."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--qna-model", type=str, default=None, help="Defaults to HuggingFaceCausalLmBackend's default model")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR context (faster, less accurate)")
    parser.add_argument(
        "--min-object-confidence", type=float, default=0.0,
        help=(
            "Drop object detections below this confidence. Default 0.0 (no "
            "filtering): the threshold policy belongs to whoever produces the "
            "detections, not to Q&A. Pass one when reading raw organizer "
            "objects JSON, which is unfiltered (~100 detections/frame)."
        ),
    )
    parser.add_argument(
        "--translate-query", action="store_true",
        help=(
            "Translate the event description to English before retrieval. "
            "CLIP's text tower is English-trained: on the real organizer "
            "index the same target video ranked nowhere for a Vietnamese "
            "query but #112/166550 for an English one. The question itself "
            "is never translated."
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
        "--temporal-radius", type=int, default=0,
        help=(
            "Also predict each hit's neighbouring keyframes, up to this many "
            "positions either side. Default 0 (retrieval's ranking untouched). "
            "Targets the measured failure mode: on p1-15 the correct video "
            "ranked #1 and #2 while the answer sat four keyframes further on, "
            "which scores exactly zero. Neighbours are resolved through the "
            "same frame map as the hits, so non-existent ones are dropped."
        ),
    )
    parser.add_argument(
        "--neighbour-cost", type=float, default=1.0,
        help=(
            "How the answer budget splits between more videos and more frames "
            "of the same video. High (e.g. 50) submits every retrieval hit "
            "before any neighbour; low (e.g. 0.1) exhausts the top hit's "
            "neighbourhood first; 1.0 treats one rank down and one keyframe "
            "aside as equally speculative."
        ),
    )
    parser.add_argument(
        "--records", type=Path, default=None,
        help=(
            "Directory of Task05A documents (<root>/<video_id>/<keyframe_id>.json). "
            "When a candidate's document exists it supplies OCR, ASR and object "
            "counts directly, skipping the live OCR pass entirely -- that pass "
            "costs ~6.7s per candidate on CPU, and it reaches no speech at all. "
            "Candidates with no document fall back to the live path, so partial "
            "coverage is fine."
        ),
    )
    parser.add_argument(
        "--export-prompts", type=Path, default=None,
        help=(
            "Collect prompts to this JSON file instead of generating answers "
            "locally. No LLM is loaded. Feed the file to a GPU host, then "
            "merge the results with scripts/import_qna_answers.py."
        ),
    )
    args = parser.parse_args()

    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.enrichment.ocr.easyocr_engine import EasyOcrEngine
    from mvr.enrichment.qna_answer import AnswerReasoner, HuggingFaceCausalLmBackend
    from mvr.enrichment.qna_batch import CollectingBackend, write_prompt_batch
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry
    from mvr.retrieval.qna import KeyframeNeighbourIndex, search_qna
    from mvr.retrieval.semantic import SemanticRetriever

    text = args.query_file.read_text(encoding="utf-8")
    event_description, question = split_event_and_question(text)
    print(f"[event_description] {event_description}")
    print(f"[question] {question}")

    index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=index.manifest["embedding_model"],
        pretrained=index.manifest["pretrained"],
        device=args.device,
    )
    semantic_service = SemanticRetriever(encoder, index)

    aic_registry = AicDatasetRegistry(args.dataset_root)
    objects_root = args.dataset_root / "objects"

    def objects_path_resolver(video_id: str, keyframe_id: str) -> Path | None:
        path = objects_root / video_id / f"{keyframe_id}.json"
        return path if path.is_file() else None

    # Temporally-expanded neighbours are not retrieval hits, so the index
    # supplied no image for them. Without this the neighbour reaches the
    # reasoner with no OCR and no objects -- for Q&A, almost no signal.
    #
    # The index's own metadata is the exact answer: it already stores an
    # image path for every keyframe it holds, which is precisely the set
    # neighbours are drawn from. Consulting it beats the registry, whose
    # root may be a different (smaller) normalized tree than the one the
    # index was built from -- that mismatch silently starved every
    # neighbour of its image. The registry stays as a fallback.
    image_paths = {
        (str(entry["video_id"]), str(entry["frame_id"])): entry["image_path"]
        for entry in index.metadata
        if entry.get("image_path")
    }

    def keyframe_path_resolver(video_id: str, keyframe_id: str) -> Path | str | None:
        known = image_paths.get((video_id, keyframe_id))
        if known is not None:
            return known
        return aic_registry.keyframe_path(video_id, keyframe_id)

    # Expand to neighbours by position in time rather than by arithmetic on
    # the id string. Same result where the ids happen to be one running
    # sequence per video, but it also refuses to walk off the end of a video
    # -- and it is the only thing that works at all for shot-scoped ids like
    # "<video>_S0007_K01", which this repo's own ingestion produces.
    neighbour_index = KeyframeNeighbourIndex.from_records(index.metadata)

    record_resolver = None
    if args.records is not None:
        records_root = args.records

        def record_resolver(video_id: str, keyframe_id: str) -> Path | None:  # noqa: F811
            path = records_root / video_id / f"{keyframe_id}.json"
            return path if path.is_file() else None

    ocr_engine = None
    image_loader = None
    if not args.no_ocr:
        from PIL import Image

        ocr_engine = EasyOcrEngine()
        image_loader = lambda path: Image.open(path).convert("RGB")  # noqa: E731

    query_translator = None
    if args.translate_query:
        from mvr.enrichment.qna_translation import HelsinkiViEnTranslator

        translator_kwargs = {"allow_model_download": args.allow_translation_download}
        if args.translation_model:
            translator_kwargs["model_name"] = args.translation_model
        query_translator = HelsinkiViEnTranslator(**translator_kwargs)
        print(f"[retrieval query EN] {query_translator.translate(event_description)}")

    if args.export_prompts is not None:
        collector = CollectingBackend()
        answer_reasoner = AnswerReasoner(collector)
    else:
        collector = None
        backend_kwargs = {"allow_model_download": args.allow_model_download, "device": args.device}
        if args.qna_model:
            backend_kwargs["model_name"] = args.qna_model
        answer_reasoner = AnswerReasoner(HuggingFaceCausalLmBackend(**backend_kwargs))

    results = search_qna(
        event_description,
        question,
        top_k=args.top_k,
        semantic_service=semantic_service,
        multimodal_service=None,
        aic_registry=aic_registry,
        answer_reasoner=answer_reasoner,
        ocr_engine=ocr_engine,
        image_loader=image_loader,
        objects_path_resolver=objects_path_resolver,
        keyframe_path_resolver=keyframe_path_resolver,
        record_resolver=record_resolver,
        min_object_confidence=args.min_object_confidence,
        query_translator=query_translator,
        temporal_radius=args.temporal_radius,
        neighbour_cost=args.neighbour_cost,
        neighbour_index=neighbour_index,
    )

    if not results:
        print("(no candidates resolved to an original-video frame)")
        return

    if collector is not None:
        out = write_prompt_batch(
            args.export_prompts,
            query_id=args.query_file.stem,
            event_description=event_description,
            question=question,
            candidates=results,
            prompts=collector.prompts,
        )
        print(f"wrote {len(results)} prompts to {out}")
        print("next: generate answers on GPU, then run scripts/import_qna_answers.py")
        return

    for r in results:
        origin = (
            "" if r.expanded_from is None
            else f" [{r.temporal_offset:+d} from {r.expanded_from}]"
        )
        print(
            f"#{r.rank} {r.video_id} frame={r.frame_id} score={r.score:.4f} "
            f"answer={r.answer!r} source={r.answer_source}{origin}"
        )


if __name__ == "__main__":
    main()

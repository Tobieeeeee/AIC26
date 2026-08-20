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
    parser.add_argument("--top-k", type=int, default=5)
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
    from mvr.retrieval.qna import search_qna
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
        min_object_confidence=args.min_object_confidence,
        query_translator=query_translator,
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
        print(
            f"#{r.rank} {r.video_id} frame={r.frame_id} score={r.score:.4f} "
            f"answer={r.answer!r} source={r.answer_source}"
        )


if __name__ == "__main__":
    main()

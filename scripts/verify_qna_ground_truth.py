"""Human verification aid for building Q&A ground truth.

Turning candidate videos into real ground truth needs a person to watch the
footage -- this script removes everything around that: it shows the query,
the candidate video's metadata, a YouTube link seeked to the retrieved
moment, and the top retrieved keyframe images, then records the verdict.

It never decides correctness itself. Nothing here infers an answer, and a
candidate is only ever written to the ground-truth file with the frame
interval and answer a human supplied.

Ground truth is written in the shape ``mvr.evaluation.aic_qna.QnaGroundTruth``
consumes: ``video_id``, inclusive ``start_frame``/``end_frame``, ``answer``.

Example (review one query):
    python scripts/verify_qna_ground_truth.py \\
        --query-file "D:\\AIC_query\\query-p1-groupA\\query-p1-19-qa.txt" \\
        --index "artifacts\\indexes\\qna_organizer_all" \\
        --dataset-root "D:\\AIC26_data" \\
        --translate --top-k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def youtube_url_at(watch_url: str, seconds: float) -> str:
    """Build a watch URL seeked to a timestamp, for jumping straight there."""
    if not watch_url:
        return ""
    separator = "&" if "?" in watch_url else "?"
    return f"{watch_url}{separator}t={int(seconds)}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Review Q&A candidates and record ground truth")
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="Raw organizer root (holds media-info-aic25-b1/, Keyframes_L*/, ...)",
    )
    parser.add_argument(
        "--registry-root", type=Path, default=None,
        help=(
            "Root in AicDatasetRegistry's normalized layout "
            "(<root>/map_keyframes/<video_id>.csv). Needed to resolve the "
            "original frame_idx, which is what a submission carries. "
            "Defaults to --dataset-root, whose raw organizer layout uses "
            "map-keyframes-aic25-b1/map-keyframes/ and will NOT resolve."
        ),
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--allow-translation-download", action="store_true")
    parser.add_argument(
        "--ground-truth", type=Path, default=Path("artifacts/qna/ground_truth.json"),
        help="JSON file the verdict is appended to",
    )
    parser.add_argument(
        "--open-images", action="store_true",
        help="Open the top keyframe images in the default viewer",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex
    from mvr.enrichment.qna_translation import (
        HelsinkiViEnTranslator,
        translate_for_retrieval,
    )
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry, MissingFrameMappingError
    from mvr.retrieval.semantic import SemanticRetriever

    sys.path.insert(0, str(Path(__file__).parent))
    from run_qna_pipeline import split_event_and_question

    text = args.query_file.read_text(encoding="utf-8")
    event, question = split_event_and_question(text)

    print("=" * 78)
    print(f"QUERY : {args.query_file.stem}")
    print(f"MO TA : {event}")
    print(f"HOI   : {question}")
    print("=" * 78)

    index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=index.manifest["embedding_model"],
        pretrained=index.manifest["pretrained"],
        device=args.device,
    )

    translator = None
    if args.translate:
        translator = HelsinkiViEnTranslator(
            allow_model_download=args.allow_translation_download
        )
    retrieval_query = translate_for_retrieval(event, translator)
    if translator is not None:
        print(f"\n[query EN] {retrieval_query}")

    results = SemanticRetriever(encoder, index).search(retrieval_query, top_k=args.top_k)

    media = MediaInfoSearchIndex.from_directory(args.dataset_root / "media-info-aic25-b1" / "media-info")
    registry = AicDatasetRegistry(args.registry_root or args.dataset_root)

    print(f"\nTOP {len(results)} UNG VIEN:\n")
    unresolved = 0
    for hit in results:
        info = media.get(hit.video_id)
        title = (info.title.strip()[:64] if info else "(khong co media_info)")
        try:
            frame_idx = registry.resolve_frame(hit.video_id, hit.frame_id).frame_idx
        except MissingFrameMappingError:
            frame_idx = None
            unresolved += 1
        print(f"  #{hit.rank}  {hit.video_id}  keyframe={hit.frame_id}  "
              f"frame_idx={frame_idx}  t={hit.timestamp:.1f}s  score={hit.score:.4f}")
        print(f"      {title}")
        if info and info.watch_url:
            print(f"      {youtube_url_at(info.watch_url, hit.timestamp)}")
        print(f"      {hit.image_path}")
        if args.open_images:
            import webbrowser

            webbrowser.open(Path(hit.image_path).as_uri())
        print()

    if unresolved:
        registry_root = args.registry_root or args.dataset_root
        print("!" * 78)
        print(f"CANH BAO: khong resolve duoc frame_idx cho {unresolved}/{len(results)} ung vien.")
        print(f"  Registry dang doc: {registry_root / 'map_keyframes'}")
        print("  frame_idx la truong BAT BUOC khi nop bai. Neu tat ca deu None,")
        print("  nhieu kha nang --registry-root dang tro vao thu muc goc cua BTC")
        print("  (dung 'map-keyframes-aic25-b1/map-keyframes/') thay vi layout")
        print("  chuan hoa ma AicDatasetRegistry can ('map_keyframes/').")
        print("!" * 78)
        print()

    print("-" * 78)
    print("Xem video/anh o tren. Neu tim duoc doan chua dap an, nhap thong tin;")
    print("neu khong chac, go Enter de bo qua (khong ghi gi ca).")
    print("-" * 78)

    video_id = input("video_id dung (Enter = bo qua): ").strip()
    if not video_id:
        print("bo qua, khong ghi ground truth.")
        return
    start_frame = input("start_frame (frame dau cua doan dap an): ").strip()
    end_frame = input("end_frame   (frame cuoi, bao gom): ").strip()
    answer = input("dap an dung: ").strip()

    if not (start_frame.isdigit() and end_frame.isdigit() and answer):
        raise SystemExit("thieu thong tin, khong ghi ground truth.")
    if int(end_frame) < int(start_frame):
        raise SystemExit("end_frame nho hon start_frame.")

    # Validate against the real data before writing. Both checks exist
    # because both mistakes actually happened during the first verification
    # pass, and each silently makes the query score 0 forever.
    # Deliberate private access: this reuses the registry's already-tested
    # CSV loading (including its keyframe-ref normalization) rather than
    # re-reading map_keyframes here, and adding a public accessor would mean
    # editing shared ingestion code that is outside this work's scope.
    frame_map = registry._load_frame_map(video_id)
    if not frame_map:
        raise SystemExit(
            f"video_id {video_id!r} khong ton tai (khong tim thay map_keyframes CSV).\n"
            f"  Kiem tra lai chinh ta -- vi du 'L030_V072' vs 'L30_V072'."
        )
    frame_indices = [m.frame_idx for m in frame_map.values()]
    inside = [f for f in frame_indices if int(start_frame) <= f <= int(end_frame)]
    if not inside:
        print()
        print("!" * 78)
        print(f"CANH BAO: khong co keyframe nao cua {video_id} nam trong "
              f"[{start_frame},{end_frame}].")
        print("  He thong chi xuat ra duoc frame_idx cua cac keyframe co san,")
        print("  nen khoang nay se KHONG BAO GIO cham duoc diem.")
        print("  Nen lay frame_idx cua keyframe dau va cuoi ma dap an con thay duoc.")
        print("!" * 78)
        if input("Van muon ghi? (go 'yes' de ghi): ").strip().lower() != "yes":
            raise SystemExit("da huy, khong ghi ground truth.")
    else:
        print(f"  (kiem tra: {len(inside)} keyframe nam trong khoang -- cham diem duoc)")

    entry = {
        "video_id": video_id,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "answer": answer,
        "question": question,
        "verified_by": "human",
    }

    path = args.ground_truth
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload[args.query_file.stem] = entry
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nda ghi ground truth cho {args.query_file.stem} vao {path}")
    print(f"tong so query da co ground truth: {len(payload)}")


if __name__ == "__main__":
    main()

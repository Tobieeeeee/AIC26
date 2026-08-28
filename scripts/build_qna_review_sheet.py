"""Build one HTML sheet for reviewing every Q&A query's candidates at once.

Verifying ground truth is a *visual* task: a person has to look at frames
and decide which video holds the answer. Doing that through terminal output,
one query at a time, is far slower than scanning a page of thumbnails -- so
this renders all queries and their candidates into a single local page with
images, timestamps, original frame indices, and YouTube links seeked to the
retrieved moment.

It decides nothing. No answer is inferred and no ground truth is written;
the page is a lookup aid, and the verdict is recorded by a human through
``scripts/verify_qna_ground_truth.py``.

Candidates come from two sources, both shown:

- the top-K semantic retrieval hits;
- any video named in ``--expected`` (from the media_info keyword search),
  force-included even when it ranks far outside the top-K -- for the
  Nguyen Trung Truc query the likely source video sits at rank 153, so
  retrieval alone would never surface it for review.

Images are referenced by local path rather than embedded: 180 keyframes
would make a ~30MB page.

Example:
    python scripts/build_qna_review_sheet.py \\
        --query-dir "D:\\AIC_query" \\
        --index "artifacts\\indexes\\qna_organizer_all" \\
        --dataset-root "<your-dataset-root>" \\
        --registry-root "D:\\AIC26_test_root" \\
        --expected artifacts/qna/expected_videos.json \\
        --translate --allow-translation-download \\
        --output artifacts/qna/review.html
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PAGE_CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 24px;
       background: #f6f7f9; color: #14161a; }
@media (prefers-color-scheme: dark) {
  body { background: #14161a; color: #e8eaed; }
  .query, .card { background: #1e2126 !important; border-color: #2c3037 !important; }
  .meta { color: #9aa0a6 !important; }
}
h1 { font-size: 20px; margin: 0 0 20px; }
.query { background: #fff; border: 1px solid #e3e5e8; border-radius: 10px;
         padding: 18px; margin-bottom: 28px; }
.qid { font-weight: 700; font-size: 15px; }
.desc { margin: 8px 0; line-height: 1.5; }
.question { font-weight: 600; color: #b3261e; margin: 8px 0 4px; }
@media (prefers-color-scheme: dark) { .question { color: #f2b8b5; } }
.en { font-size: 13px; font-style: italic; opacity: .75; margin-bottom: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(215px, 1fr));
        gap: 12px; }
.card { border: 1px solid #e3e5e8; border-radius: 8px; overflow: hidden;
        background: #fff; }
.card.expected { border-color: #f9ab00; border-width: 2px; }
.card img { width: 100%; height: 122px; object-fit: cover; display: block;
            background: #ddd; }
.card .body { padding: 8px 10px; }
.vid { font-weight: 600; font-size: 13px; }
.meta { font-size: 11.5px; color: #5f6368; line-height: 1.5; }
.badge { display: inline-block; background: #f9ab00; color: #000; font-size: 10px;
         font-weight: 700; padding: 1px 6px; border-radius: 3px; margin-left: 4px; }
a { color: #1a73e8; }
"""


def render_card(item: dict) -> str:
    image_uri = Path(item["image_path"]).as_uri() if item["image_path"] else ""
    link = (
        f'<a href="{html.escape(item["watch_url"])}" target="_blank">xem tu {item["timestamp"]:.0f}s</a>'
        if item["watch_url"]
        else ""
    )
    badge = '<span class="badge">UNG VIEN</span>' if item["expected"] else ""
    rank = f'#{item["rank"]}' if item["rank"] is not None else "ngoai top-K"
    return f"""
    <div class="card{' expected' if item['expected'] else ''}">
      <img src="{html.escape(image_uri)}" loading="lazy" alt="">
      <div class="body">
        <div class="vid">{html.escape(item['video_id'])}{badge}</div>
        <div class="meta">
          {rank} &middot; score {item['score']:.4f}<br>
          keyframe {html.escape(str(item['keyframe_id']))} &middot;
          frame_idx <b>{item['frame_idx']}</b><br>
          {html.escape(item['title'][:52])}<br>
          {link}
        </div>
      </div>
    </div>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a Q&A candidate review sheet")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--expected", type=Path, default=None)
    parser.add_argument("--pattern", default="*-qa.txt")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--expected-frames", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--translate", action="store_true")
    parser.add_argument("--allow-translation-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/qna/review.html"))
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

    index = FaissIndex.load(args.index)
    encoder = OpenClipEncoder(
        model_name=index.manifest["embedding_model"],
        pretrained=index.manifest["pretrained"],
        device=args.device,
    )
    retriever = SemanticRetriever(encoder, index)
    media = MediaInfoSearchIndex.from_directory(
        args.dataset_root / "media-info-aic25-b1" / "media-info"
    )
    registry = AicDatasetRegistry(args.registry_root or args.dataset_root)

    translator = None
    if args.translate:
        translator = HelsinkiViEnTranslator(
            allow_model_download=args.allow_translation_download
        )

    expected = {}
    if args.expected and args.expected.is_file():
        expected = {
            k: v for k, v in json.loads(args.expected.read_text(encoding="utf-8")).items()
            if not k.startswith("_")
        }

    rows_by_video: dict[str, list[int]] = {}
    for row, meta in enumerate(index.metadata):
        rows_by_video.setdefault(meta["video_id"], []).append(row)

    def describe(video_id: str, meta: dict, rank, score: float, is_expected: bool) -> dict:
        info = media.get(video_id)
        try:
            frame_idx = registry.resolve_frame(video_id, str(meta["frame_id"])).frame_idx
        except MissingFrameMappingError:
            frame_idx = None
        return {
            "video_id": video_id,
            "keyframe_id": meta["frame_id"],
            "frame_idx": frame_idx,
            "timestamp": float(meta.get("timestamp") or 0.0),
            "score": score,
            "rank": rank,
            "image_path": meta.get("image_path", ""),
            "title": info.title if info else "",
            "watch_url": (
                f"{info.watch_url}{'&' if '?' in info.watch_url else '?'}t={int(meta.get('timestamp') or 0)}s"
                if info and info.watch_url else ""
            ),
            "expected": is_expected,
        }

    sections = []
    for query_file in sorted(args.query_dir.glob(f"**/{args.pattern}")):
        query_id = query_file.stem
        event, question = split_event_and_question(query_file.read_text(encoding="utf-8"))
        retrieval_query = translate_for_retrieval(event, translator)

        cards = []
        seen_rows: set[str] = set()
        for hit in retriever.search(retrieval_query, top_k=args.top_k):
            key = f"{hit.video_id}/{hit.frame_id}"
            seen_rows.add(key)
            cards.append(
                describe(
                    hit.video_id,
                    {"frame_id": hit.frame_id, "timestamp": hit.timestamp,
                     "image_path": hit.image_path},
                    hit.rank, hit.score, hit.video_id == expected.get(query_id),
                )
            )

        # Force-include the media_info candidate even when it ranks far out.
        target = expected.get(query_id)
        for row in rows_by_video.get(target, [])[: args.expected_frames]:
            meta = index.metadata[row]
            key = f"{meta['video_id']}/{meta['frame_id']}"
            if key in seen_rows:
                continue
            cards.append(describe(target, meta, None, 0.0, True))

        print(f"{query_id:20} {len(cards)} the")
        sections.append(f"""
  <div class="query">
    <div class="qid">{html.escape(query_id)}</div>
    <div class="desc">{html.escape(event)}</div>
    <div class="question">HOI: {html.escape(question)}</div>
    <div class="en">retrieval query: {html.escape(retrieval_query)}</div>
    <div class="grid">{''.join(render_card(c) for c in cards)}</div>
  </div>""")

    page = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8">
<title>Q&amp;A candidate review</title><style>{PAGE_CSS}</style></head>
<body>
<h1>Q&amp;A candidate review &middot; {len(sections)} query
    &middot; vien vang = ung vien tu media_info</h1>
{''.join(sections)}
</body></html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"\nwrote {args.output.resolve()}")
    print("mo bang trinh duyet, roi ghi ket qua bang scripts/verify_qna_ground_truth.py")


if __name__ == "__main__":
    main()

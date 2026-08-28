"""Locate the likely source video of each Q&A query via organizer metadata.

Answers the question that gates everything else: **is the video holding the
answer even present in our semantic index?** If it is not, retrieval cannot
surface it and no amount of answer-model quality helps -- so this must be
checked before tuning prompts or models.

Uses ``media_info`` (title/description/keywords), the only per-video signal
covering all 873 organizer videos. Matching is a deliberately simple
diacritic-insensitive substring search: this is a lookup aid for a human,
not a retrieval component.

Example:
    python scripts/find_qna_source_videos.py \\
        --query-dir "D:\\AIC_query" \\
        --media-info "<your-dataset-root>\\media-info-aic25-b1\\media-info" \\
        --indexed-dir "D:\\AIC26_test_root\\keyframes"
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Words too common in Vietnamese news queries to discriminate between videos.
STOPWORDS = {
    "doan", "video", "clip", "trong", "co", "cua", "mot", "nay", "la", "gi",
    "hoi", "cho", "biet", "ten", "duoc", "va", "voi", "tai", "tren", "hinh",
    "anh", "canh", "nguoi", "cac", "nhung", "khi", "den", "tu", "ve", "ra",
    "thi", "se", "da", "dang", "bang", "sau", "truoc", "hay", "phia", "mau",
}
MIN_TERM_WORDS = 2  # single words are too noisy; prefer phrases
MAX_TERM_WORDS = 4


def extract_search_terms(text: str, *, limit: int = 12) -> list[str]:
    """Pull candidate n-gram phrases out of a query.

    Proper nouns and multi-word phrases discriminate; bare common words do
    not. Capitalised runs (place names, organisations) are taken first since
    they are the strongest signal in this corpus.
    """
    from mvr.enrichment.qna_media_info import normalize_text

    terms: list[str] = []

    # Capitalised runs, e.g. "Nguyễn Trung Trực", "FANA", "Khánh Hòa".
    for match in re.findall(r"\b[A-ZĐÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZĐÀ-Ỹ][\wÀ-ỹ]*)*", text):
        normalized = normalize_text(match)
        words = [w for w in normalized.split() if w not in STOPWORDS]
        if not words:
            continue
        phrase = " ".join(words)
        if len(phrase) >= 4 and phrase not in terms:
            terms.append(phrase)

    # Content-word n-grams as a fallback signal.
    words = [w for w in normalize_text(text).split() if w not in STOPWORDS and len(w) > 2]
    for size in range(MIN_TERM_WORDS, MAX_TERM_WORDS + 1):
        for i in range(len(words) - size + 1):
            phrase = " ".join(words[i : i + size])
            if phrase not in terms:
                terms.append(phrase)

    return terms[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Find likely source videos for Q&A queries")
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--media-info", type=Path, required=True)
    parser.add_argument(
        "--indexed-dir", type=Path, default=None,
        help="Directory of indexed keyframe folders, to flag which hits are reachable",
    )
    parser.add_argument("--pattern", default="*-qa.txt")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    from mvr.enrichment.qna_media_info import MediaInfoSearchIndex

    index = MediaInfoSearchIndex.from_directory(args.media_info)
    print(f"media_info: {len(index)} videos")

    indexed: set[str] = set()
    if args.indexed_dir is not None and args.indexed_dir.is_dir():
        indexed = {p.name for p in args.indexed_dir.iterdir() if p.is_dir()}
        print(f"indexed   : {len(indexed)} videos")
    print()

    query_files = sorted(args.query_dir.glob(f"**/{args.pattern}"))
    reachable = 0
    for query_file in query_files:
        text = query_file.read_text(encoding="utf-8")
        terms = extract_search_terms(text)
        hits = index.search(terms, limit=args.top)

        print(f"=== {query_file.stem}")
        if not hits:
            print("    (khong tim thay video nao khop)")
            print()
            continue
        top_in_index = False
        for hit in hits:
            mark = "IN INDEX " if hit.video_id in indexed else "not indexed"
            if hit.video_id in indexed:
                top_in_index = True
            title = hit.info.title.strip()[:70]
            print(f"    {hit.video_id:12} [{mark}] {hit.matched_terms} terms | {title}")
        reachable += 1 if top_in_index else 0
        print()

    if indexed:
        print(f"tong ket: {reachable}/{len(query_files)} query co video ung vien nam trong index")


if __name__ == "__main__":
    main()

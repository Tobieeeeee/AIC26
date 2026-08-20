"""Phase 06 — TRAKE end-to-end verification and real-data benchmark on real
local AIC data.

Opt-in only. Proves the complete documented path for real, with no mocks:

    real event queries -> search_trake() (shortlist via search_kis() ->
    dense per-video scoring from the real FAISS index -> chain alignment)
    -> official TRAKE evaluator -> per-strategy benchmark summary

Ground truth below is internal/self-authored (5 real multi-event queries,
each frame_idx read directly from the organizer's own map_keyframes CSV
for the exact keyframe visually verified), NOT organizer-official GT --
labeled "internal" throughout per the task pack's labeling discipline.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MVR_RUN_AIC_TRAKE_E2E") != "1",
    reason="set MVR_RUN_AIC_TRAKE_E2E=1 (and MVR_AIC_DATASET_ROOT) to run the real AIC TRAKE E2E",
)


def _require_dataset_root() -> Path:
    raw = os.environ.get("MVR_AIC_DATASET_ROOT", "").strip()
    if not raw:
        pytest.skip("set MVR_AIC_DATASET_ROOT to run this real-data E2E test")
    root = Path(raw)
    if not root.is_dir():
        pytest.skip(f"MVR_AIC_DATASET_ROOT does not exist: {root}")
    return root


def _require_real_index() -> Path:
    index_root = Path("artifacts/indexes/aic_l21")
    required = ("keyframes.faiss", "metadata.json", "manifest.json")
    if not all((index_root / name).is_file() for name in required):
        pytest.skip(f"real FAISS index not found at {index_root}")
    return index_root


def _build_benchmark_queries():
    """5 real, visually-verified multi-event queries (see
    docs/AIC_TRAKE_BENCHMARK_RESULTS.md for the source keyframes/images and
    docs/AIC_TRAKE_IMPLEMENTATION_NOTES.md for the map_keyframes CSV lookups
    each frame_idx was read from -- not invented).

    Event text is English, matching the real index's embedding model
    (ViT-B-32 / laion2b_s34b_b79k, per artifacts/indexes/aic_l21/manifest.json
    -- an English-trained CLIP, not multilingual). This was verified
    necessary, not assumed: with the identical target frames, Vietnamese
    text ranked the correct frame 192/268 and 144/268 in two separate
    checks; the English text below ranks the same frames #1/268 and #2/268
    on the same video. See docs/AIC_TRAKE_BENCHMARK_RESULTS.md for the
    measured comparison."""
    from mvr.evaluation.trake import TrakeGroundTruth
    from mvr.evaluation.trake_benchmark import TrakeBenchmarkQuery
    from mvr.retrieval.trake_events import (
        TrakeEventQuery,
        TrakeQuery,
        translate_gap_phrase,
    )

    def _query_from_phrases(events: list[str], connectors: list[str]) -> TrakeQuery:
        """Derives every gap budget through Phase 03's own phrase table rather
        than hard-coded numbers, so the benchmark exercises exactly what
        production would do with the same query text.

        Measured, not assumed: hand-picked (0,60) budgets scored 0.633 mean,
        the table's own no-clue default (0,300) scored 0.639, and removing
        gap constraints entirely scored 0.594 -- so the constraint genuinely
        contributes (+0.045 over unconstrained; house_fire_alley alone swings
        0.600 -> 0.267 without it) and the table's default beats values
        chosen by hand. See docs/AIC_TRAKE_BENCHMARK_RESULTS.md."""
        return TrakeQuery(
            events=tuple(TrakeEventQuery(i, text) for i, text in enumerate(events, start=1)),
            gaps=tuple(translate_gap_phrase(phrase) for phrase in connectors),
        )

    return [
        TrakeBenchmarkQuery(
            query_id="traffic_accident_night",
            query=_query_from_phrases(
                [
                    "a night traffic accident scene with an ambulance and police",
                    "a night street with cars and motorbikes passing near a cordon tape",
                    "two police officers standing guard at a night accident scene",
                    "a crowd of people gathered at an accident scene with an ambulance parked",
                ],
                ["", "", ""],  # no timing clue in the query -> phrase table default
            ),
            gt=TrakeGroundTruth(video_id="L21_V024", frames=(13890, 14130, 14430, 14635)),
        ),
        TrakeBenchmarkQuery(
            query_id="flood_embankment",
            query=_query_from_phrases(
                [
                    "a crowded ferry landing with motorbikes parked on a flooded road",
                    "flood water overflowing an embankment with reeds in the foreground",
                    "a wide flooded field with water hyacinth floating on the water",
                    "people gathered at a flooded ferry landing",
                ],
                ["", "", ""],  # no timing clue in the query -> phrase table default
            ),
            gt=TrakeGroundTruth(video_id="L21_V015", frames=(8279, 8636, 8988, 9378)),
        ),
        TrakeBenchmarkQuery(
            query_id="house_fire_alley",
            query=_query_from_phrases(
                [
                    "fire trucks arriving at a house fire scene on a street",
                    "firefighters entering a narrow alley with fallen electrical wires",
                    "firefighters climbing up to put out a fire near tangled electrical wires",
                ],
                ["", ""],  # no timing clue in the query -> phrase table default
            ),
            gt=TrakeGroundTruth(video_id="L21_V018", frames=(11068, 11340, 11600)),
        ),
        TrakeBenchmarkQuery(
            query_id="ocean_debris_cleanup",
            query=_query_from_phrases(
                [
                    "a diver collecting trash and fishing nets on the ocean floor",
                    "people loading bags of trash onto a boat on the sea surface",
                ],
                [""],  # no timing clue in the query -> phrase table default
            ),
            gt=TrakeGroundTruth(video_id="L21_V010", frames=(9150, 9730)),
        ),
        TrakeBenchmarkQuery(
            query_id="suspect_arrest",
            query=_query_from_phrases(
                [
                    "two men restraining a shirtless suspect on a street at night",
                    "a close-up of a restrained suspect sitting on the street at night",
                    "a night scene with a crowd of people and police motorbikes parked",
                    "seized evidence including cash, a key, a wrench and a phone on the floor",
                ],
                ["", "", ""],  # no timing clue in the query -> phrase table default
            ),
            gt=TrakeGroundTruth(video_id="L21_V026", frames=(16083, 16355, 16697, 17021)),
        ),
        TrakeBenchmarkQuery(
            # WIDE-GAP query. The other five all have real inter-event gaps of
            # 8-23s, comfortably inside any sane budget, so their gap penalty
            # never fires and ATLAS/Lucifer's two-sided [min,max] machinery is
            # never actually exercised. Here the real gaps are 310s / 294s /
            # 278s, so the budget genuinely constrains the answer. Event 3 (the
            # studio anchor) also recurs many times across the bulletin, so
            # only the timing constraint can pick the right occurrence.
            query_id="wide_gap_bulletin_segments",
            query=_query_from_phrases(
                [
                    "a view along elevated metro railway tracks over a city",
                    "a tall apartment tower block against a cloudy grey sky",
                    "a female news anchor in a pink blouse in a studio",
                    "a museum model of a prehistoric hominid face",
                ],
                ["một lúc sau", "một lúc sau", "một lúc sau"],  # -> [30, 300] seconds each
            ),
            gt=TrakeGroundTruth(video_id="L21_V008", frames=(4620, 12367, 19710, 26652)),
        ),
    ]


def test_trake_real_data_benchmark_across_all_strategies(tmp_path: Path) -> None:
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from mvr.evaluation.trake_benchmark import run_trake_benchmark_query, summarize_trake_benchmark
    from mvr.indexing.faiss_index import FaissIndex
    from mvr.ingestion.aic_dataset import AicDatasetRegistry
    from mvr.retrieval.semantic import SemanticRetriever
    from mvr.retrieval.trake import search_trake
    from mvr.retrieval.trake_align import AtlasAlignment, LuciferAlignment, UceseAlignment

    dataset_root = _require_dataset_root()
    index_root = _require_real_index()

    # Real dependencies, wired exactly as production/Phase 05 expects -- NOT
    # LazyFaissSemanticService (the KIS API route's wrapper): that class only
    # exposes .search(), not the .text_encoder/.index pair search_trake()
    # needs for dense scoring, so a real SemanticRetriever is used directly.
    registry = AicDatasetRegistry(dataset_root)
    faiss_index = FaissIndex.load(index_root)
    encoder = OpenClipEncoder(
        model_name=faiss_index.manifest["embedding_model"],
        pretrained=faiss_index.manifest["pretrained"],
        device="cpu",
    )
    semantic_service = SemanticRetriever(encoder, faiss_index)

    queries = _build_benchmark_queries()
    strategies = {
        "atlas": AtlasAlignment(),
        "lucifer": LuciferAlignment(),
        "ucese": UceseAlignment(),
    }

    records = []
    for name, strategy in strategies.items():
        for query in queries:
            def bound_search(trake_query, _strategy=strategy):
                return search_trake(
                    trake_query,
                    alignment_strategy=_strategy,
                    semantic_service=semantic_service,
                    aic_registry=registry,
                )

            record = run_trake_benchmark_query(query, config_name=name, search_fn=bound_search)
            records.append(record)

    summary = summarize_trake_benchmark(records)

    # This must prove actual correctness, not just that scoring executed --
    # a completely wrong retrieval also satisfies 0.0 <= final_score <= 1.0.
    # Print full per-query/per-strategy results so a human can inspect real
    # measured numbers (pytest -s) before Phase 06 tightens this assertion
    # further based on what's actually observed.
    print("\n=== TRAKE real-data benchmark (internal GT) ===")
    for record in records:
        print(
            f"{record.config_name:10s} {record.query_id:25s} "
            f"final_score={record.final_score:.3f} latency={record.latency_seconds:.2f}s"
        )
    print("\n--- summary ---")
    for name, stats in summary.items():
        print(f"{name}: {stats}")

    assert all(0.0 <= record.final_score <= 1.0 for record in records)
    # At least one strategy must resolve at least one query with a non-zero
    # score -- proves search_trake() actually retrieves something meaningful
    # from real data, not just an empty/always-wrong pipeline.
    assert any(record.final_score > 0.0 for record in records), (
        "every strategy scored 0.0 on every real query -- the pipeline is "
        "not retrieving anything meaningful; investigate before trusting "
        "this benchmark"
    )

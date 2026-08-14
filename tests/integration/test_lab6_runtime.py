"""Opt-in real-runtime smoke test for the reused Lab 6 pipeline."""
from __future__ import annotations

import csv
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("MVR_RUN_PHASE04_RUNTIME_TESTS") != "1",
    reason="set MVR_RUN_PHASE04_RUNTIME_TESTS=1 to run real Lab 6 runtimes",
)


def _hit_payload(hit, image_paths: dict[str, str]) -> dict:
    source = hit["_source"]
    frame_id = source["frame_id"]
    return {
        "frame_id": frame_id,
        "video_id": source["video_id"],
        "timestamp": float(source["ts"]),
        "score": float(hit.get("_score") or 0.0),
        "image_path": image_paths[frame_id],
        "ocr_text": source.get("ocr_text", ""),
        "objects": source.get("objects", {}),
    }


def test_real_lab6_multimodal_retrieval_runtime() -> None:
    """Catch broken real model, ES mapping/index, retrieval, or fusion boundaries."""
    from elasticsearch import Elasticsearch
    from PIL import Image

    from mvr.enrichment.objects.yolo import YoloObjectDetector
    from mvr.enrichment.ocr.easyocr_engine import (
        EasyOcrEngine,
        extract_easyocr_text,
    )
    from mvr.enrichment.ocr.paddleocr_engine import (
        PaddleOcrEngine,
        extract_paddle_lines,
        paddle_predict,
    )
    from mvr.indexing.documents import build_search_document
    from mvr.indexing.elasticsearch_index import ElasticsearchIndex
    from mvr.retrieval.hybrid import rrf_fuse, weighted_fuse
    from mvr.retrieval.lexical import search_lexical
    from mvr.retrieval.object_search import search_objects
    from mvr.retrieval.semantic import (
        ElasticsearchSemanticRetriever,
        create_lab6_openclip_encoder,
    )

    keyframe_root = Path(
        os.environ.get(
            "MVR_PHASE04_KEYFRAMES",
            "artifacts/keyframes/phase02-smoke",
        )
    ).resolve()
    mapping_path = keyframe_root / "keyframe_mapping.csv"
    assert mapping_path.is_file(), f"missing smoke mapping: {mapping_path}"
    with mapping_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5

    image_paths = {
        row["keyframe_id"]: str((keyframe_root / row["image_path"]).resolve())
        for row in rows
    }
    assert all(Path(path).is_file() for path in image_paths.values())
    images = [Image.open(image_paths[row["keyframe_id"]]).convert("RGB") for row in rows]

    easyocr = EasyOcrEngine(gpu=False)
    easy_results = []
    for row, image in zip(rows, images):
        detections = easyocr.reader.readtext(np.asarray(image))
        easy_results.append(
            {
                "frame_id": row["keyframe_id"],
                "text": extract_easyocr_text(detections),
                "detections": [
                    {"text": str(text), "confidence": float(confidence)}
                    for _, text, confidence in detections
                ],
            }
        )

    paddleocr = PaddleOcrEngine()
    paddle_results = []
    for row, image in zip(rows, images):
        lines = extract_paddle_lines(paddle_predict(paddleocr.reader, image), 0.3)
        paddle_results.append(
            {
                "frame_id": row["keyframe_id"],
                "text": " ".join(text for text, _ in lines),
                "detections": [
                    {"text": text, "confidence": confidence}
                    for text, confidence in lines
                ],
            }
        )

    yolo = YoloObjectDetector()
    object_results = [
        {
            "frame_id": row["keyframe_id"],
            "objects": yolo.detect(image),
        }
        for row, image in zip(rows, images)
    ]

    encoder = create_lab6_openclip_encoder(device="cpu")
    vectors = encoder.encode_images(list(image_paths.values()))
    assert encoder.model_name == "RN50"
    assert encoder.pretrained == "openai"
    assert vectors.shape == (len(rows), 1024)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        np.ones(len(rows)),
        atol=1e-4,
        rtol=1e-4,
    )

    documents = [
        build_search_document(
            row,
            caption="",
            ocr_text=paddle["text"],
            objects=objects["objects"],
            clip_vector=vector,
        )
        for row, paddle, objects, vector in zip(
            rows,
            paddle_results,
            object_results,
            vectors,
        )
    ]

    client = Elasticsearch(
        "http://localhost:9200",
        request_timeout=120,
        max_retries=3,
        retry_on_timeout=True,
    )
    assert client.ping(), "Elasticsearch is not reachable at localhost:9200"
    server_version = client.info()["version"]["number"]
    assert server_version == "8.11.4"

    index = ElasticsearchIndex(client)
    index.recreate(embedding_dimension=1024)
    assert index.index_documents(documents) == len(rows)
    mapping = client.indices.get_mapping(index="keyframes")
    assert mapping["keyframes"]["mappings"]["properties"]["clip_vector"]["dims"] == 1024

    lexical_query = "00"
    lexical = search_lexical(client, lexical_query, top_k=5)
    lexical_hits = list(lexical["hits"]["hits"])
    assert lexical_hits, "PaddleOCR text was indexed but BM25 returned no hit"

    detected = next(
        (
            (name, count)
            for result in object_results
            for name, count in result["objects"].items()
            if count > 0
        ),
        None,
    )
    object_query = f"{detected[0]}:1" if detected else "person:1"
    object_response = search_objects(client, object_query, top_k=5)
    object_hits = list(object_response["hits"]["hits"])
    if detected:
        assert object_hits

    semantic_query = "color test pattern"
    semantic = ElasticsearchSemanticRetriever(encoder, client).search(
        semantic_query,
        top_k=5,
    )
    semantic_hits = list(semantic["hits"]["hits"])
    assert len(semantic_hits) == len(rows)

    lexical_ids = [hit["_source"]["frame_id"] for hit in lexical_hits]
    semantic_ids = [hit["_source"]["frame_id"] for hit in semantic_hits]
    rrf = rrf_fuse([lexical_ids, semantic_ids], rrf_k=60, top_k=5)
    weighted = weighted_fuse(
        {hit["_source"]["frame_id"]: float(hit["_score"]) for hit in lexical_hits},
        {hit["_source"]["frame_id"]: float(hit["_score"]) for hit in semantic_hits},
        alpha=0.25,
        top_k=5,
    )
    assert rrf
    assert weighted

    sources = {
        hit["_source"]["frame_id"]: hit
        for hit in [*lexical_hits, *object_hits, *semantic_hits]
    }

    def fused_payload(results: list[tuple[str, float]]) -> list[dict]:
        payloads = []
        for frame_id, score in results:
            payload = _hit_payload(sources[frame_id], image_paths)
            payload["score"] = float(score)
            payloads.append(payload)
        return payloads

    report = {
        "corpus": {
            "keyframes": len(rows),
            "mapping": str(mapping_path),
        },
        "versions": {
            "easyocr": version("easyocr"),
            "paddleocr": version("paddleocr"),
            "paddlepaddle": version("paddlepaddle"),
            "ultralytics": version("ultralytics"),
            "open_clip_torch": version("open-clip-torch"),
            "elasticsearch_client": version("elasticsearch"),
            "elasticsearch_server": server_version,
        },
        "easyocr": {"threshold": 0.3, "results": easy_results},
        "paddleocr": {"threshold": 0.3, "results": paddle_results},
        "yolo": {
            "model": "yolov8n.pt",
            "confidence": 0.4,
            "results": object_results,
        },
        "rn50": {
            "model": encoder.model_name,
            "pretrained": encoder.pretrained,
            "shape": list(vectors.shape),
            "dtype": str(vectors.dtype),
            "norms": np.linalg.norm(vectors, axis=1).tolist(),
        },
        "elasticsearch": {"index": "keyframes", "documents": len(rows)},
        "bm25": {
            "query": lexical_query,
            "results": [_hit_payload(hit, image_paths) for hit in lexical_hits],
        },
        "object": {
            "query": object_query,
            "results": [_hit_payload(hit, image_paths) for hit in object_hits],
        },
        "semantic_knn": {
            "query": semantic_query,
            "results": [_hit_payload(hit, image_paths) for hit in semantic_hits],
        },
        "rrf": {"rrf_k": 60, "results": fused_payload(rrf)},
        "weighted": {"alpha": 0.25, "results": fused_payload(weighted)},
    }
    print("PHASE04_RUNTIME_REPORT=" + json.dumps(report, ensure_ascii=False))

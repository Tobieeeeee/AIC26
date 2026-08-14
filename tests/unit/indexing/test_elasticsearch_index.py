def test_mapping_preserves_lab6_analyzer_fields_and_cosine_vector() -> None:
    from mvr.indexing.elasticsearch_index import SETTINGS, build_mappings

    assert SETTINGS == {
        "analysis": {
            "analyzer": {
                "vi_folded": {
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding"],
                }
            }
        }
    }
    assert build_mappings(1024) == {
        "properties": {
            "frame_id": {"type": "keyword"},
            "video_id": {"type": "keyword"},
            "ts": {"type": "float"},
            "caption": {"type": "text", "analyzer": "vi_folded"},
            "ocr_text": {"type": "text", "analyzer": "vi_folded"},
            "objects": {
                "properties": {
                    "person": {"type": "integer"},
                    "car": {"type": "integer"},
                }
            },
            "clip_vector": {
                "type": "dense_vector",
                "dims": 1024,
                "index": True,
                "similarity": "cosine",
            },
        }
    }


def test_index_recreates_bulk_inserts_and_refreshes_lab6_documents() -> None:
    from mvr.indexing.documents import SearchDocument
    from mvr.indexing.elasticsearch_index import ElasticsearchIndex

    calls = []

    class Indices:
        def delete(self, **kwargs):
            calls.append(("delete", kwargs))

        def create(self, **kwargs):
            calls.append(("create", kwargs))

        def refresh(self, **kwargs):
            calls.append(("refresh", kwargs))

    class Client:
        indices = Indices()

        def count(self, **kwargs):
            calls.append(("count", kwargs))
            return {"count": 1}

    def bulk_fn(client, actions, **kwargs):
        calls.append(("bulk", client, list(actions), kwargs))

    document = SearchDocument(
        frame_id="frame-a",
        video_id="video-a",
        ts=1.25,
        caption="caption",
        ocr_text="ocr",
        objects={"dog": 1},
        clip_vector=[1.0, 0.0],
    )
    index = ElasticsearchIndex(Client())

    index.recreate(embedding_dimension=2)
    assert index.index_documents([document], bulk_fn=bulk_fn) == 1

    assert calls[0] == ("delete", {"index": "keyframes", "ignore_unavailable": True})
    assert calls[1][0] == "create"
    assert calls[1][1]["index"] == "keyframes"
    assert calls[1][1]["mappings"]["properties"]["clip_vector"]["dims"] == 2
    assert calls[2][0] == "bulk"
    assert calls[2][2] == [
        {"_index": "keyframes", "_id": "frame-a", **document.to_source()}
    ]
    assert calls[2][3] == {"chunk_size": 50, "request_timeout": 120}
    assert calls[3:] == [
        ("refresh", {"index": "keyframes"}),
        ("count", {"index": "keyframes"}),
    ]

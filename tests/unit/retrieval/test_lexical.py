import pytest


def test_lexical_request_preserves_ocr_boost_caption_fallback_and_filters() -> None:
    from mvr.retrieval.lexical import build_lexical_request

    assert build_lexical_request("giá vàng", top_k=5, person_min=2, car_min=1) == {
        "size": 5,
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "giá vàng",
                            "fields": ["ocr_text^2", "caption"],
                        }
                    }
                ],
                "filter": [
                    {"range": {"objects.person": {"gte": 2}}},
                    {"range": {"objects.car": {"gte": 1}}},
                ],
            }
        },
        "highlight": {"fields": {"ocr_text": {}}},
    }


def test_empty_lexical_query_uses_match_all_and_validates_inputs() -> None:
    from mvr.retrieval.lexical import build_lexical_request

    assert build_lexical_request("", top_k=1)["query"]["bool"]["must"] == [
        {"match_all": {}}
    ]
    with pytest.raises(ValueError, match="top_k must be positive"):
        build_lexical_request("query", top_k=0)
    with pytest.raises(ValueError, match="minimum counts"):
        build_lexical_request("query", person_min=-1)


def test_lexical_search_sends_request_to_selected_index() -> None:
    from mvr.retrieval.lexical import search_lexical

    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {"hits": {"hits": []}}

    response = search_lexical(Client(), "giá vàng", top_k=3, index_name="frames")

    assert response == {"hits": {"hits": []}}
    assert calls[0]["index"] == "frames"
    assert calls[0]["size"] == 3

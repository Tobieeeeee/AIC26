import pytest


def test_object_query_parses_coco_classes_and_builds_range_filters() -> None:
    from mvr.retrieval.object_search import build_object_filters, parse_object_conditions

    conditions = parse_object_conditions("person:2, traffic light:0,car:1")

    assert conditions == [("person", 2), ("traffic light", 0), ("car", 1)]
    assert build_object_filters(conditions) == [
        {"range": {"objects.person": {"gte": 2}}},
        {"range": {"objects.traffic light": {"gte": 0}}},
        {"range": {"objects.car": {"gte": 1}}},
    ]


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("", "at least one"),
        ("person", "expected COCO-class"),
        (":1", "class name"),
        ("dragon:1", "Unknown COCO"),
        ("person:-1", "non-negative integer"),
        ("person:1,person:2", "Duplicate COCO"),
    ],
)
def test_object_query_rejects_malformed_conditions(query: str, message: str) -> None:
    from mvr.retrieval.object_search import parse_object_conditions

    with pytest.raises(ValueError, match=message):
        parse_object_conditions(query)


def test_object_search_executes_bool_filter_request() -> None:
    from mvr.retrieval.object_search import search_objects

    calls = []

    class Client:
        def search(self, **kwargs):
            calls.append(kwargs)
            return {"hits": {"hits": []}}

    response = search_objects(Client(), "person:2,car:1", top_k=4)

    assert response == {"hits": {"hits": []}}
    assert calls == [
        {
            "index": "keyframes",
            "size": 4,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"objects.person": {"gte": 2}}},
                        {"range": {"objects.car": {"gte": 1}}},
                    ]
                }
            },
        }
    ]

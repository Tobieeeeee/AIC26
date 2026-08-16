import pytest

from mvr.retrieval.hybrid import min_max_normalize, rrf_fuse, weighted_fuse


def test_rrf_rewards_items_seen_in_both_lists():
    result = rrf_fuse([["a", "b"], ["b", "c"]], rrf_k=60, top_k=3)
    assert result[0][0] == "b"


def test_equal_scores_normalize_to_one():
    assert min_max_normalize({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}


def test_weighted_alpha_bounds():
    with pytest.raises(ValueError):
        weighted_fuse({}, {}, alpha=1.1)


def test_weighted_fuse_combines_modalities():
    result = weighted_fuse(
        {"ocr-hit": 10.0, "shared": 8.0},
        {"semantic-hit": 1.0, "shared": 0.9},
        alpha=0.5,
        top_k=3,
    )
    assert {item_id for item_id, _ in result} == {"ocr-hit", "semantic-hit", "shared"}


def test_rrf_preserves_one_based_scores_and_deterministic_ties():
    assert rrf_fuse([["b"], ["a"]], rrf_k=60, top_k=2) == [
        ("a", 1 / 61),
        ("b", 1 / 61),
    ]


def test_weighted_endpoint_ties_prefer_candidates_from_active_branch():
    assert [
        item_id
        for item_id, _ in weighted_fuse(
            {"a": 10.0},
            {"b": 1.0, "c": 2.0},
            alpha=0.0,
            top_k=3,
        )
    ] == ["c", "b", "a"]

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mvr.evaluation.sanity_adapters import BM25Adapter, ProviderUnavailableError


class _Indices:
    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self, *, index: str) -> bool:
        assert index == "lexical"
        return self._exists


class _Client:
    def __init__(self, *, ping: bool = True, exists: bool = True) -> None:
        self._ping = ping
        self.indices = _Indices(exists)

    def ping(self) -> bool:
        return self._ping

    def info(self) -> dict:
        return {"version": {"number": "8.11.4"}}

    def count(self, *, index: str) -> dict:
        return {"count": 5}


def test_bm25_adapter_reuses_task05b_search_and_normalizes_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    @dataclass(frozen=True)
    class Hit:
        rank: int = 1
        score: float = 2.0
        keyframe_id: str = "kf-a"
        video_id: str = "video-a"
        frame_idx: int = 42
        shot_id: int = 3
        timestamp: float = 1.4

    def fake_search(client, query, *, index_name, top_k):
        calls.append((client, query, index_name, top_k))
        return [Hit()]

    monkeypatch.setattr(
        "mvr.evaluation.sanity_adapters.search_task05b_lexical",
        fake_search,
    )
    client = _Client()
    adapter = BM25Adapter(client, index_name="lexical")

    hits = adapter.search("query", 10)

    assert calls == [(client, "query", "lexical", 10)]
    assert hits[0].keyframe_id == "kf-a"
    assert hits[0].frame_id == 42
    assert hits[0].provider == "bm25-task05b"


def test_bm25_adapter_reports_unavailable_index() -> None:
    adapter = BM25Adapter(_Client(exists=False), index_name="lexical")
    with pytest.raises(ProviderUnavailableError, match="does not exist"):
        adapter.ensure_available()

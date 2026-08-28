"""Tests for the R2 sync orchestration library."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from mvr.operations.cache import BoundedCache
from mvr.operations.r2_sync import R2Sync, SyncStats


def _make_obj(key, size=10):
    return {"Key": key, "Size": size}


def _writing_download(content: bytes | str = b"data"):
    """Return a callable that mimics client.download_file(key, local_path, expected_sha256=...)."""

    def _download(key, local_path, **kwargs):
        Path(local_path).write_bytes(
            content.encode() if isinstance(content, str) else content
        )

    return _download


class TestR2SyncVideos:
    def test_pull_videos_downloads_each(self, tmp_path):
        client = MagicMock()
        client.download_file = _writing_download(b"video")
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        stats = sync.pull_videos("batch1", "L21", ["L21_V001", "L21_V029"])
        assert stats.downloaded == 2
        assert stats.errors == 0
        assert (tmp_path / "data/raw/batch1/videos/L21/L21_V001.mp4").is_file()
        assert (tmp_path / "data/raw/batch1/videos/L21/L21_V029.mp4").is_file()

    def test_pull_videos_skips_existing(self, tmp_path):
        client = MagicMock()
        client.download_file = _writing_download(b"video")
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)
        existing = tmp_path / "data/raw/batch1/videos/L21/L21_V001.mp4"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("already here")

        stats = sync.pull_videos("batch1", "L21", ["L21_V001", "L21_V029"])
        assert stats.downloaded == 1
        assert stats.skipped_up_to_date == 1


class TestR2SyncPrefix:
    def test_pull_prefix(self, tmp_path):
        client = MagicMock()
        client.list_objects.return_value = [
            _make_obj("data/corpus/keyframes/L21_V001/001.jpg", 100),
            _make_obj("data/corpus/keyframes/L21_V001/002.jpg", 200),
        ]
        client.download_file = _writing_download(b"img")
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        stats = sync.pull_prefix("data/corpus/keyframes")
        assert stats.downloaded == 2
        assert stats.bytes_transferred == 300
        assert (tmp_path / "data/corpus/keyframes/L21_V001/001.jpg").is_file()

    def test_pull_manifest(self, tmp_path):
        client = MagicMock()
        client.download_file = _writing_download(b"{}")
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        path = sync.pull_dataset_manifest()
        assert path is not None
        assert path == (tmp_path / "data/corpus/dataset_manifest.json")


class TestR2SyncPush:
    def test_push_staging(self, tmp_path):
        client = MagicMock()
        client.upload_file.return_value = {"key": "k", "size": 5, "sha256": "abc"}
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        staging = tmp_path / "staging"
        staging.mkdir()
        (staging / "a.json").write_text("aaa")
        (staging / "b.json").write_text("bbbb")

        stats = sync.push_staging("ocr", "run-001", [staging / "a.json", staging / "b.json"])
        assert stats.uploaded == 2
        assert stats.bytes_transferred == 7
        assert client.upload_file.call_count == 2

    def test_push_staging_skips_missing(self, tmp_path):
        client = MagicMock()
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        stats = sync.push_staging("ocr", "run-001", [tmp_path / "missing.json"])
        assert stats.uploaded == 0
        assert stats.errors == 0

    def test_push_corpus_metadata(self, tmp_path):
        client = MagicMock()
        client.upload_file.return_value = {"key": "k", "size": 10, "sha256": "x"}
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        pq = tmp_path / "ocr.parquet"
        pq.write_text("parquetdata")
        stats = sync.push_corpus_metadata("ocr", "1", "L21", pq)
        assert stats.uploaded == 1
        call_key = client.upload_file.call_args[0][1]
        assert call_key == "data/corpus/metadata/ocr/batch=1/collection=L21/part-000.parquet"

    def test_push_corpus_embeddings(self, tmp_path):
        client = MagicMock()
        client.upload_file.return_value = {"key": "k", "size": 5, "sha256": "x"}
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)

        emb = tmp_path / "vectors.npy"
        emb.write_text("npy")
        stats = sync.push_corpus_embeddings("beit3", emb)
        assert stats.uploaded == 1
        call_key = client.upload_file.call_args[0][1]
        assert call_key == "data/corpus/features/embeddings/beit3/vectors.npy"


class TestSyncStats:
    def test_to_dict(self):
        stats = SyncStats(downloaded=3, skipped_up_to_date=1, uploaded=2, errors=1, bytes_transferred=100)
        d = stats.to_dict()
        assert d["downloaded"] == 3
        assert d["skipped_up_to_date"] == 1
        assert d["bytes_transferred"] == 100
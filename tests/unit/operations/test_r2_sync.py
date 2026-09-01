"""Tests for the R2 sync orchestration library."""
from __future__ import annotations

import json
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

    def test_parallel_exact_pull_is_resumable_without_cache_copy(self, tmp_path):
        client = MagicMock()
        client.download_file = MagicMock(side_effect=_writing_download(b"jpeg"))
        sync = R2Sync(client=client, local_root=tmp_path)
        existing = tmp_path / "keyframes/existing.jpg"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"already")

        stats = sync.pull_objects_parallel(
            [
                ("data/corpus/keyframes/new.jpg", tmp_path / "keyframes/new.jpg"),
                ("data/corpus/keyframes/existing.jpg", existing),
            ],
            workers=2,
        )

        assert stats.downloaded == 1
        assert stats.skipped_up_to_date == 1
        assert (tmp_path / "keyframes/new.jpg").read_bytes() == b"jpeg"
        assert client.download_file.call_count == 1


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

    def test_parallel_push_skips_remote_objects_with_same_size(self, tmp_path):
        client = MagicMock()
        client.list_objects.return_value = [
            _make_obj("data/staging/enrichment/run/a.json", 3)
        ]
        client.upload_file.return_value = {"key": "k", "size": 4, "sha256": "x"}
        sync = R2Sync(client=client, local_root=tmp_path)
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        first.write_text("aaa")
        second.write_text("bbbb")

        stats = sync.push_files_parallel(
            [
                (first, "data/staging/enrichment/run/a.json"),
                (second, "data/staging/enrichment/run/b.json"),
            ],
            workers=2,
            existing_prefix="data/staging/enrichment/run/",
        )

        assert stats.uploaded == 1
        assert stats.skipped_up_to_date == 1
        assert stats.bytes_transferred == 4
        assert client.upload_file.call_args.args[1].endswith("/b.json")

    def test_push_ingestion_run_preserves_tree_and_writes_success_last(self, tmp_path):
        client = MagicMock()
        client.upload_file.return_value = {"key": "k", "size": 5, "sha256": "x"}
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)
        output = tmp_path / "ingest"
        (output / "keyframes" / "L21_V001").mkdir(parents=True)
        (output / "selector_cache" / "L21_V001" / "candidates").mkdir(
            parents=True
        )
        (output / "selector_embeddings" / "metaclip2").mkdir(parents=True)
        (output / "keyframe_mapping.csv").write_text("keyframe_id\nkf-1\n")
        (output / "keyframes" / "L21_V001" / "kf-1.jpg").write_bytes(b"jpeg")
        (output / "selector_cache" / "L21_V001" / "candidate_embeddings.npy").write_bytes(
            b"candidate-vectors"
        )
        (output / "selector_cache" / "L21_V001" / "candidates" / "c-1.jpg").write_bytes(
            b"candidate-jpeg"
        )
        (output / "selector_embeddings" / "metaclip2" / "L21_V001.npy").write_bytes(
            b"selected-vector"
        )

        stats = sync.push_ingestion_run(
            output,
            run_id="run-001",
            batch_id="1",
            collection_id="L21",
            video_id="L21_V001",
        )

        assert stats.uploaded == 4
        assert stats.errors == 0
        keys = [call.args[1] for call in client.upload_file.call_args_list]
        assert keys == [
            "data/staging/ingestion/run-001/batch=1/collection=L21/L21_V001/"
            "keyframe_mapping.csv",
            "data/staging/ingestion/run-001/batch=1/collection=L21/L21_V001/"
            "keyframes/L21_V001/kf-1.jpg",
            "data/staging/ingestion/run-001/batch=1/collection=L21/L21_V001/"
            "selector_embeddings/metaclip2/L21_V001.npy",
            "data/staging/ingestion/run-001/batch=1/collection=L21/L21_V001/"
            "_SUCCESS.json",
        ]
        assert all("/selector_cache/" not in key for key in keys)
        marker = json.loads((output / "_SUCCESS.json").read_text(encoding="utf-8"))
        assert marker["artifact_count"] == 3
        assert (output / "selector_cache/L21_V001/candidates/c-1.jpg").is_file()

    def test_push_ingestion_run_omits_success_marker_after_partial_failure(self, tmp_path):
        client = MagicMock()
        client.upload_file.side_effect = [
            {"key": "k", "size": 5, "sha256": "x"},
            RuntimeError("upload failed"),
        ]
        cache = BoundedCache(str(tmp_path / "cache"), max_size_gb=1)
        sync = R2Sync(client=client, local_root=tmp_path, cache=cache)
        output = tmp_path / "ingest"
        output.mkdir()
        (output / "a.json").write_text("{}")
        (output / "b.json").write_text("{}")

        stats = sync.push_ingestion_run(
            output,
            run_id="run-001",
            batch_id="1",
            collection_id="L21",
            video_id="L21_V001",
        )

        assert stats.uploaded == 1
        assert stats.errors == 1
        assert not (output / "_SUCCESS.json").exists()
        assert client.upload_file.call_count == 2

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

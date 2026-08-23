"""Tests for R2 client, cache, and layout modules."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mvr.operations.r2_client import (
    R2ChecksumError,
    R2Client,
    R2ConfigError,
    _sha256_file,
)
from mvr.operations.r2_layout import (
    CORPUS_PREFIX,
    RAW_PREFIX,
    STAGING_PREFIX,
    corpus_embedding_key,
    corpus_key,
    corpus_metadata_key,
    raw_video_key,
    staging_key,
)


# --------------------------------------------------------------------------
# R2 layout / object-key tests
# --------------------------------------------------------------------------

class TestR2Layout:
    def test_raw_video_key(self):
        key = raw_video_key("batch1", "L21", "L21_V010")
        assert key == "data/raw/batch1/videos/L21/L21_V010.mp4"

    def test_staging_key(self):
        key = staging_key("ocr", "run-001", "L21_V010", "kf001.json")
        assert key == "data/staging/ocr/run-001/L21_V010/kf001.json"

    def test_corpus_key(self):
        key = corpus_key("ingest", "keyframe_mapping.csv")
        assert key == "data/corpus/ingest/keyframe_mapping.csv"

    def test_corpus_metadata_key_partitioned(self):
        key = corpus_metadata_key("ocr", "1", "L21")
        assert key == "data/corpus/metadata/ocr/batch=1/collection=L21/part-000.parquet"

    def test_corpus_embedding_key(self):
        key = corpus_embedding_key("beit3", "vectors.npy")
        assert key == "data/corpus/features/embeddings/beit3/vectors.npy"

    def test_prefix_constants(self):
        assert RAW_PREFIX == "data/raw"
        assert STAGING_PREFIX == "data/staging"
        assert CORPUS_PREFIX == "data/corpus"


# --------------------------------------------------------------------------
# R2 client tests (mocked)
# --------------------------------------------------------------------------

class TestR2Client:
    def test_missing_env_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(R2ConfigError, match="R2_BUCKET"):
                R2Client()

    def test_explicit_config_no_env(self):
        client = R2Client(
            bucket="test-bucket",
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        assert client.bucket == "test-bucket"

    def test_sha256_file(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        sha = _sha256_file(test_file)
        assert len(sha) == 64  # SHA-256 hex digest
        assert sha == _sha256_file(test_file)  # deterministic

    def test_download_checksum_mismatch(self, tmp_path):
        """A download with wrong checksum must raise R2ChecksumError."""
        client = R2Client(
            bucket="test",
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
            max_retries=1,
        )
        # Mock boto3 client
        mock_s3 = MagicMock()
        def fake_download(bucket, key, path):
            Path(path).write_text("fake content")
        mock_s3.download_file = fake_download
        client._client = mock_s3

        with pytest.raises(R2ChecksumError, match="SHA256 mismatch"):
            client.download_file(
                "test/key",
                tmp_path / "output.bin",
                expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
            )
        # Temp file should be cleaned up
        assert not list(tmp_path.glob(".*tmp*"))

    def test_download_atomic_success(self, tmp_path):
        """Successful download creates the target file atomically."""
        client = R2Client(
            bucket="test",
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
            max_retries=1,
        )
        mock_s3 = MagicMock()
        def fake_download(bucket, key, path):
            Path(path).write_text("real content")
        mock_s3.download_file = fake_download
        client._client = mock_s3

        result = client.download_file("test/key", tmp_path / "output.bin")
        assert result.is_file()
        assert result.read_text() == "real content"

    def test_upload_missing_file(self, tmp_path):
        client = R2Client(
            bucket="test",
            endpoint_url="https://test.r2.cloudflarestorage.com",
            access_key_id="key",
            secret_access_key="secret",
        )
        with pytest.raises(FileNotFoundError):
            client.upload_file(tmp_path / "nonexistent.bin", "some/key")


# --------------------------------------------------------------------------
# Cache tests
# --------------------------------------------------------------------------

from mvr.operations.cache import BoundedCache, _object_key_to_local_path


class TestCachePathMapping:
    def test_deterministic_mapping(self, tmp_path):
        path1 = _object_key_to_local_path(tmp_path, "data/raw/batch1/video.mp4")
        path2 = _object_key_to_local_path(tmp_path, "data/raw/batch1/video.mp4")
        assert path1 == path2

    def test_different_keys_different_paths(self, tmp_path):
        p1 = _object_key_to_local_path(tmp_path, "a/b.txt")
        p2 = _object_key_to_local_path(tmp_path, "a/c.txt")
        assert p1 != p2

    def test_empty_key_raises(self, tmp_path):
        with pytest.raises(ValueError):
            _object_key_to_local_path(tmp_path, "")


class TestBoundedCache:
    def test_cache_miss(self, tmp_path):
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1.0)
        assert cache.get("nonexistent/key") is None
        assert cache.stats.misses == 1

    def test_cache_hit_after_put(self, tmp_path):
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1.0)

        def download(path):
            path.write_text("cached content")

        result = cache.put("test/key.txt", download)
        assert result.is_file()
        assert result.read_text() == "cached content"
        assert cache.stats.downloads == 1

        # Second call should be a hit
        hit = cache.get("test/key.txt")
        assert hit is not None
        assert hit == result
        assert cache.stats.hits == 1

    def test_same_object_not_redownloaded(self, tmp_path):
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1.0)
        call_count = 0

        def download(path):
            nonlocal call_count
            call_count += 1
            path.write_text("content")

        cache.put("test/key.txt", download)
        cache.put("test/key.txt", download)
        assert call_count == 1
        assert cache.stats.downloads == 1

    def test_partial_download_rejected(self, tmp_path):
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1.0)

        def bad_download(path):
            # Write empty file (simulates partial/failed download)
            path.write_bytes(b"")

        with pytest.raises(RuntimeError, match="empty"):
            cache.put("test/key.txt", bad_download)

    def test_size_bound_eviction(self, tmp_path):
        # 1 KB max cache
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1e-6)

        def download_big(path):
            path.write_bytes(b"x" * 2000)

        cache.put("big1.bin", download_big)
        # This should trigger eviction
        cache.put("big2.bin", download_big)
        assert cache.stats.evictions > 0

    def test_clear(self, tmp_path):
        cache = BoundedCache(cache_dir=tmp_path / "cache", max_size_gb=1.0)

        def download(path):
            path.write_text("content")

        cache.put("test/a.txt", download)
        cache.put("test/b.txt", download)
        removed = cache.clear()
        assert removed == 2
        assert not cache.contains("test/a.txt")

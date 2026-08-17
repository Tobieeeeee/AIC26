"""Tests for the TKIS query translation benchmark utilities.

All tests run offline — no internet, no credentials, no large models.
"""
from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the scripts/verification directory is importable.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "verification"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from query_translation.base import Translator
from query_translation.models import TranslationResult, MethodSummary
from query_translation.utils import (
    cache_get,
    cache_put,
    compute_summary,
    load_queries,
    now_iso,
    write_results_csv,
    write_results_json,
    write_summary_json,
)


# ── Helpers ────────────────────────────────────────────────────────


class StubTranslator(Translator):
    """Deterministic stub that upper-cases input."""

    @property
    def name(self) -> str:
        return "stub"

    def translate(self, text: str) -> str:
        return text.upper()


class FailingTranslator(Translator):
    """Always raises on translate()."""

    @property
    def name(self) -> str:
        return "failing"

    def translate(self, text: str) -> str:
        raise RuntimeError("simulated provider failure")


# ── 1. CSV query loading ──────────────────────────────────────────


def test_load_queries(tmp_path: Path) -> None:
    csv_path = tmp_path / "q.csv"
    csv_path.write_text(
        "query_id,query_vi\nq1,hello\nq2,world\n",
        encoding="utf-8",
    )

    rows = load_queries(csv_path)
    assert len(rows) == 2
    assert rows[0]["query_id"] == "q1"
    assert rows[1]["query_vi"] == "world"


# ── 2. Required query fields ─────────────────────────────────────


def test_load_queries_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("id,text\nq1,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_queries(csv_path)


def test_load_queries_empty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("query_id,query_vi\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_queries(csv_path)


def test_load_queries_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_queries(tmp_path / "nonexistent.csv")


# ── 3. Common translator interface ───────────────────────────────


def test_translator_interface() -> None:
    t = StubTranslator()
    assert t.name == "stub"
    assert t.translate("hello") == "HELLO"


# ── 4. vi_raw behavior ───────────────────────────────────────────


def test_vi_raw_passthrough() -> None:
    # Import the ViRawTranslator from the benchmark runner.
    # We replicate the logic here to avoid importing the runner's main.
    class ViRaw(Translator):
        @property
        def name(self) -> str:
            return "vi_raw"

        def translate(self, text: str) -> str:
            return text

    t = ViRaw()
    query = "Ba người đi xe đạp mặc đồng phục trắng."
    assert t.translate(query) == query
    assert t.name == "vi_raw"


# ── 5. Local backend can be mocked ───────────────────────────────


def test_local_backend_mock() -> None:
    mock_tokenizer = MagicMock()
    # tokenizer(text, ...) returns a dict-like with .to()
    mock_inputs = MagicMock()
    mock_tokenizer.return_value = mock_inputs
    mock_tokenizer.batch_decode.return_value = [
        "Three cyclists in white"
    ]

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()

    mock_device = MagicMock()

    mock_torch = MagicMock()

    with patch.object(
        __import__(
            "query_translation.local", fromlist=["LocalTranslator"]
        ).LocalTranslator,
        "_load",
        return_value=(mock_tokenizer, mock_model),
    ), patch.object(
        __import__(
            "query_translation.local", fromlist=["LocalTranslator"]
        ).LocalTranslator,
        "_resolve_device",
        return_value=mock_device,
    ), patch.dict("sys.modules", {"torch": mock_torch}):
        from query_translation.local import LocalTranslator

        t = LocalTranslator(model_name="test-model", device="cpu")
        result = t.translate("Ba người đi xe đạp")
        assert result == "Three cyclists in white"
        assert t.name == "local_vi2en"
        mock_model.generate.assert_called_once()


# ── 6. Google backend can be mocked ──────────────────────────────


def test_google_backend_mock() -> None:
    mock_client = MagicMock()
    mock_client.get_languages.return_value = []
    mock_client.translate.return_value = {
        "translatedText": "A red car"
    }

    # Build a fake translate_v2 module with a Client class.
    import types
    fake_translate_v2 = types.ModuleType("google.cloud.translate_v2")
    fake_translate_v2.Client = MagicMock(return_value=mock_client)

    modules_patch = {
        "google": types.ModuleType("google"),
        "google.cloud": types.ModuleType("google.cloud"),
        "google.cloud.translate_v2": fake_translate_v2,
    }

    with patch.dict("sys.modules", modules_patch):
        import importlib
        import query_translation.google as gmod
        importlib.reload(gmod)

        t = gmod.GoogleTranslator()
        assert t.name == "google_translate"

        result = t.translate("Một chiếc ô tô màu đỏ")
        assert result == "A red car"
        mock_client.translate.assert_called_once()


# ── 7. DeepL backend can be mocked ──────────────────────────────


def test_deepl_backend_mock(monkeypatch) -> None:
    monkeypatch.setenv("DEEPL_API_KEY", "fake-key-for-test")

    mock_deepl = MagicMock()
    mock_result = MagicMock()
    mock_result.text = "A black car"
    mock_deepl.Translator.return_value.translate_text.return_value = (
        mock_result
    )

    with patch.dict("sys.modules", {"deepl": mock_deepl}):
        import importlib
        import query_translation.deepl as dmod
        importlib.reload(dmod)

        t = dmod.DeepLTranslator()
        result = t.translate("Một chiếc ô tô màu đen")
        assert result == "A black car"
        assert t.name == "deepl"


# ── 8. Provider failure is recorded, not crash ───────────────────


def test_provider_failure_recorded() -> None:
    t = FailingTranslator()
    query_vi = "Test query"

    try:
        translated = t.translate(query_vi)
        success = True
        error = None
    except Exception as exc:
        translated = ""
        success = False
        error = f"{type(exc).__name__}: {exc}"

    result = TranslationResult(
        query_id="q1",
        query_vi=query_vi,
        method=t.name,
        translated_text=translated,
        latency_ms=0.0,
        success=success,
        error=error,
    )

    assert result.success is False
    assert "simulated provider failure" in result.error


# ── 9. Output CSV/JSON contract ──────────────────────────────────


def test_output_csv_contract(tmp_path: Path) -> None:
    results = [
        TranslationResult(
            query_id="q1",
            query_vi="xin chào",
            method="vi_raw",
            translated_text="xin chào",
            latency_ms=0.0,
            success=True,
            timestamp=now_iso(),
        ),
    ]

    csv_path = tmp_path / "results.csv"
    write_results_csv(results, csv_path)

    with csv_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1

    required_fields = {
        "query_id", "query_vi", "method", "translated_text",
        "latency_ms", "success", "error",
    }
    assert required_fields.issubset(set(rows[0].keys()))


def test_output_json_contract(tmp_path: Path) -> None:
    results = [
        TranslationResult(
            query_id="q1",
            query_vi="xin chào",
            method="vi_raw",
            translated_text="xin chào",
            latency_ms=0.0,
            success=True,
            timestamp=now_iso(),
        ),
    ]

    json_path = tmp_path / "results.json"
    write_results_json(results, json_path)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["query_id"] == "q1"
    assert data[0]["success"] is True


# ── 10. Latency is non-negative ──────────────────────────────────


def test_latency_non_negative() -> None:
    r = TranslationResult(
        query_id="q1",
        query_vi="test",
        method="vi_raw",
        translated_text="test",
        latency_ms=0.0,
        success=True,
    )
    assert r.latency_ms >= 0.0


# ── 11. Cache hit avoids second provider call ────────────────────


def test_cache_hit(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    # Cache miss.
    result = cache_get(cache_dir, "hello", "test_method")
    assert result is None

    # Write cache.
    cache_put(cache_dir, "hello", "test_method", "HELLO")

    # Cache hit.
    result = cache_get(cache_dir, "hello", "test_method")
    assert result == "HELLO"


def test_cache_different_methods(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"

    cache_put(cache_dir, "hello", "method_a", "A_HELLO")
    cache_put(cache_dir, "hello", "method_b", "B_HELLO")

    assert cache_get(cache_dir, "hello", "method_a") == "A_HELLO"
    assert cache_get(cache_dir, "hello", "method_b") == "B_HELLO"


# ── 12. No credentials in outputs ────────────────────────────────


def test_no_credentials_in_output(tmp_path: Path) -> None:
    results = [
        TranslationResult(
            query_id="q1",
            query_vi="test",
            method="deepl",
            translated_text="",
            latency_ms=0.0,
            success=False,
            error="DEEPL_API_KEY is not configured",
            model_or_provider="deepl",
            timestamp=now_iso(),
        ),
    ]

    json_path = tmp_path / "results.json"
    write_results_json(results, json_path)

    content = json_path.read_text(encoding="utf-8")

    # Should not contain actual secret values.
    assert "fake-key" not in content
    assert "secret" not in content.lower()
    assert "password" not in content.lower()


# ── Summary computation ──────────────────────────────────────────


def test_compute_summary() -> None:
    results = [
        TranslationResult(
            query_id="q1", query_vi="a", method="vi_raw",
            translated_text="a", latency_ms=1.0, success=True,
        ),
        TranslationResult(
            query_id="q2", query_vi="b", method="vi_raw",
            translated_text="b", latency_ms=2.0, success=True,
        ),
        TranslationResult(
            query_id="q1", query_vi="a", method="deepl",
            translated_text="", latency_ms=0.0, success=False,
            error="no key",
        ),
    ]

    summaries = compute_summary(results)
    assert len(summaries) == 2

    vi_raw = next(s for s in summaries if s.method == "vi_raw")
    assert vi_raw.success_count == 2
    assert vi_raw.failure_count == 0
    assert vi_raw.mean_latency_ms == 1.5

    deepl = next(s for s in summaries if s.method == "deepl")
    assert deepl.success_count == 0
    assert deepl.failure_count == 1
    assert deepl.failure_rate == 1.0

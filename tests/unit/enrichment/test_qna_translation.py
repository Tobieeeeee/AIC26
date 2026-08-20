import pytest


class _FakeTranslator:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    def translate(self, text: str) -> str:
        self.calls.append(text)
        return self._mapping.get(text, f"EN({text})")


def test_passthrough_translator_returns_input_unchanged() -> None:
    from mvr.enrichment.qna_translation import PassthroughTranslator

    translator = PassthroughTranslator()

    assert translator.translate("một đoạn mô tả") == "một đoạn mô tả"


def test_translate_for_retrieval_uses_the_translator() -> None:
    from mvr.enrichment.qna_translation import translate_for_retrieval

    translator = _FakeTranslator({"đình thần": "communal temple"})

    assert translate_for_retrieval("đình thần", translator) == "communal temple"


def test_translate_for_retrieval_without_translator_is_identity() -> None:
    from mvr.enrichment.qna_translation import translate_for_retrieval

    assert translate_for_retrieval("đình thần", None) == "đình thần"


def test_translate_for_retrieval_falls_back_to_original_on_failure() -> None:
    """A translation outage must degrade retrieval, not break the query --
    the untranslated text still retrieves something."""
    from mvr.enrichment.qna_translation import translate_for_retrieval

    class _Broken:
        def translate(self, text: str) -> str:
            raise RuntimeError("backend down")

    assert translate_for_retrieval("đình thần", _Broken()) == "đình thần"


def test_translate_for_retrieval_falls_back_when_result_is_blank() -> None:
    from mvr.enrichment.qna_translation import translate_for_retrieval

    class _Blank:
        def translate(self, text: str) -> str:
            return "   "

    assert translate_for_retrieval("đình thần", _Blank()) == "đình thần"


def test_translate_for_retrieval_rejects_blank_input() -> None:
    from mvr.enrichment.qna_translation import translate_for_retrieval

    with pytest.raises(ValueError):
        translate_for_retrieval("   ", None)


def test_helsinki_translator_fails_closed_without_cached_weights() -> None:
    """Real (non-mocked) check: no silent model download, no hang."""
    pytest.importorskip("transformers")
    from mvr.enrichment.qna_translation import (
        HelsinkiViEnTranslator,
        TranslatorUnavailable,
    )

    with pytest.raises(TranslatorUnavailable):
        HelsinkiViEnTranslator(model_name="definitely-not-real-org/definitely-not-real-mt")

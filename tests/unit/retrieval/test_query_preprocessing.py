from mvr.retrieval.query_preprocessing import QueryPreprocessingService


def test_preprocessing_normalizes_only_the_lexical_branch() -> None:
    class Translator:
        def translate(self, query: str) -> str:
            return "dense query"

    service = QueryPreprocessingService(translator_factory=Translator)

    prepared = service.prepare("  GIA\u0301   VA\u0300NG TĂNG  ")

    assert prepared.dense == "dense query"
    assert prepared.lexical == "giá vàng tăng"

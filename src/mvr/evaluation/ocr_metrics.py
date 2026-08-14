"""OCR comparison metric extracted from Lab 6 cells 6 and 8."""

OCR_HIT_THRESHOLD = 0.60


def word_recall(ground_truth: str, prediction: str) -> float:
    words = ground_truth.lower().split()
    if not words:
        raise ValueError("ground_truth must contain at least one word")
    return sum(word in prediction.lower() for word in words) / len(words)


def ocr_hit(
    ground_truth: str,
    prediction: str,
    threshold: float = OCR_HIT_THRESHOLD,
) -> bool:
    return word_recall(ground_truth, prediction) >= threshold

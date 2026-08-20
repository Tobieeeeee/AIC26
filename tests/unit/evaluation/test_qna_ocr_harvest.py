"""Recovering OCR text from prompt batches already on disk.

571 frames have already been OCR'd across earlier runs, at ~6.7s each on
CPU. That work is sitting in `artifacts/qna*/ *.prompts.json`, embedded in
the prompts sent to the GPU host.

Recovering it turns a 1.8-hour feedback loop into a sub-second one, which is
what makes the synthetic-set rules tunable at all: the first version scored
36% precision, and diagnosing that took a full OCR run per attempt.
"""
from __future__ import annotations

from mvr.evaluation.qna_synthetic import ocr_text_from_prompt

PROMPT = (
    "You are an expert visual question-answering assistant.\n"
    "\n"
    "Detected on-screen text (OCR): Xã Giang Ly 04.08/2024\n"
    "Transcribed speech near this frame (ASR): (no speech transcribed near this frame)\n"
    "Detected objects (label: count): person: 2\n"
    "Question: Hỏi xã này có tên là gì?\n"
)


def test_the_ocr_line_is_recovered() -> None:
    assert ocr_text_from_prompt(PROMPT) == "Xã Giang Ly 04.08/2024"


def test_it_stops_at_the_end_of_the_line() -> None:
    """The ASR and objects lines follow it and must not bleed in."""
    recovered = ocr_text_from_prompt(PROMPT)

    assert "ASR" not in recovered
    assert "person" not in recovered


def test_the_no_text_placeholder_reads_as_empty() -> None:
    """A frame with no text and a frame nobody OCR'd must not look alike."""
    prompt = PROMPT.replace("Xã Giang Ly 04.08/2024", "(no OCR text detected)")

    assert ocr_text_from_prompt(prompt) == ""


def test_a_prompt_without_an_ocr_line_yields_none() -> None:
    assert ocr_text_from_prompt("Question: gì?\n") is None


def test_an_older_prompt_without_the_asr_line_still_parses() -> None:
    """Batches written before ASR existed are the bulk of what is on disk."""
    older = (
        "Detected on-screen text (OCR): TIN CHÍNH\n"
        "Detected objects (label: count): (no object detections available)\n"
        "Question: gì?\n"
    )

    assert ocr_text_from_prompt(older) == "TIN CHÍNH"

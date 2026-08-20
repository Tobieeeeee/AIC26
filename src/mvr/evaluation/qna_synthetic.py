"""Ground truth built backwards from a known frame.

Real ground truth is expensive and, at three queries, permanently capped:
the p2/p3 queries are reference queries whose answers lie outside the data
we hold (notes 3j). Of those three, only one is currently scorable, so every
decision about answering is being made on n=1 -- which is why the notes
freeze `temporal_radius` and `neighbour_cost` as defaults rather than
calling them tuned.

The usual direction is query -> search -> frame -> a human confirms, and the
expensive, error-prone step is the confirmation. This inverts it: **choose
the frame first**, read its OCR, derive a question whose answer is a span of
that text. ``video_id`` and ``frame_id`` cannot be wrong because they are
the input, not a search result.

What that does and does not buy -- the distinction decides how the numbers
may be read, so it is stated rather than implied:

**Correct by construction**
    ``video_id``, ``frame_id``, and the fact that the answer appears on that
    frame.

**Not guaranteed**
    That the answer matches what a *human* sees. OCR mangles text: on the
    real p1-15 frame it read "Khánh Vĩnh" as "Khónh Vinh". And that the
    answer is unique to that frame -- handled by ``drop_ambiguous_cases``,
    which is a filter, not a guarantee.

**Therefore this measures** answer *extraction*: given context X and
question Q, does the model return span S? OCR errors cancel out, because
both the expectation and the model see the same X.

**It does not measure** whether the system produces what a human would call
the answer, and it does not measure retrieval at all -- any query derived
from a frame's own text leaks the answer, so retrieval numbers from this set
would be meaningless. Scores from this module are never comparable with
scores on the three real queries, and must never be presented as either.

Questions come from deterministic rules rather than an LLM. A measuring
instrument has to be reproducible and inspectable; a generated question
whose provenance nobody can reconstruct is not a measurement.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from mvr.evaluation.aic_qna import QnaGroundTruth

#: Vietnamese administrative units, with the question each one asks. Ordered
#: longest-first so "thành phố" is matched before any shorter prefix of it.
_ADMIN_UNITS: tuple[tuple[str, str], ...] = (
    ("thành phố", "Thành phố"),
    ("thị trấn", "Thị trấn"),
    ("thị xã", "Thị xã"),
    ("phường", "Phường"),
    ("huyện", "Huyện"),
    ("quận", "Quận"),
    ("tỉnh", "Tỉnh"),
    ("xã", "Xã"),
)

#: A proper-noun run: capitalized words, allowing Vietnamese diacritics.
#: Capped at four words -- "Thành phố Hồ Chí Minh" is the longest shape that
#: occurs, and without a cap the rule swallowed a whole seven-word headline
#: ("SỰ CỦA TA TRONG KHÁNG CHIẾN CHỐNG PHÁP") as a district name on a real
#: frame.
_NAME = r"[A-ZÀ-ỸĐ][^\s,.;:]*(?:\s+[A-ZÀ-ỸĐ][^\s,.;:]*){0,3}"

#: Dates, tolerating the separator soup real OCR produces -- the p1-15 frame
#: came back as "04.08/2024", one dot and one slash.
_DATE_RE = re.compile(r"\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b")

#: Vietnamese function words. A place name is built from Sino-Vietnamese
#: nouns and never contains a preposition, pronoun or conjunction, so one of
#: these inside a candidate name means the match ran into a headline rather
#: than a name. Truncating at four words was not enough on its own: it
#: shortened "SỰ CỦA TA TRONG KHÁNG CHIẾN CHỐNG PHÁP" to "SỰ CỦA TA TRONG",
#: which is equally wrong and merely less obvious.
_NOT_IN_A_NAME = frozenset(
    "của và trong cho với từ đến là có ta tôi này đó các những một sự về "
    "tại bởi nhưng hoặc khi mà thì đã sẽ được người nước".split()
)


def _fold(text: str) -> str:
    """Casefold and strip diacritics, so a unit word survives OCR tone damage."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


#: Folded once at import: the stoplist is compared against folded words so
#: an all-caps or tone-damaged "CỦA" still matches "cua".
_folded_stopwords = frozenset(_fold(word) for word in _NOT_IN_A_NAME)


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", _fold(text)).strip()


@dataclass(frozen=True)
class SyntheticQnaCase:
    """One question whose answer is a known span of a known frame."""

    video_id: str
    keyframe_id: str
    frame_id: int
    question: str
    answer: str
    ocr_text: str
    template: str

    def __post_init__(self) -> None:
        # The guarantee the whole approach rests on, enforced rather than
        # hoped for: an answer that is not in the context is not ground
        # truth, it is a wrong expectation baked into the benchmark.
        if self.answer not in self.ocr_text:
            raise ValueError(
                f"answer {self.answer!r} is not a span of the frame's text"
            )

    def to_ground_truth(self) -> QnaGroundTruth:
        """Scoreable ground truth over exactly the frame it was built from.

        The interval is a single frame. Widening it would invent tolerance
        the organizer did not grant, and this set exists to measure honestly.
        """
        return QnaGroundTruth(
            video_id=self.video_id,
            start_frame=self.frame_id,
            end_frame=self.frame_id,
            answer=self.answer,
        )


def _administrative_cases(
    ocr_text: str, *, video_id: str, keyframe_id: str, frame_id: int
) -> list[SyntheticQnaCase]:
    cases: list[SyntheticQnaCase] = []
    for unit, display_unit in _ADMIN_UNITS:
        # Matched WITH diacritics, deliberately. Folding them away seemed
        # right -- OCR damages tone marks constantly -- but Vietnamese
        # diacritics carry meaning: "quân" (army) folds onto "quận"
        # (district), "tính"/"tình" onto "tỉnh" (province). On 60 real
        # frames that made every single generated case a false positive.
        #
        # For a ground-truth generator a false positive is far worse than a
        # miss: it writes a wrong expectation into the benchmark, while a
        # miss only makes the benchmark smaller. So OCR-damaged unit words
        # are lost, and that is the correct trade.
        for match in re.finditer(rf"\b{re.escape(unit)}\s+", ocr_text, re.IGNORECASE):
            tail = ocr_text[match.end() :]
            name = re.match(_NAME, tail)
            if name is None:
                continue
            answer = name.group(0).strip()
            if not answer:
                continue
            if any(_fold(word) in _folded_stopwords for word in answer.split()):
                continue
            cases.append(
                SyntheticQnaCase(
                    video_id=video_id,
                    keyframe_id=keyframe_id,
                    frame_id=frame_id,
                    question=f"Hỏi {display_unit.lower()} này có tên là gì?",
                    answer=answer,
                    ocr_text=ocr_text,
                    template="administrative-unit",
                )
            )
    return cases


def _date_cases(
    ocr_text: str, *, video_id: str, keyframe_id: str, frame_id: int
) -> list[SyntheticQnaCase]:
    return [
        SyntheticQnaCase(
            video_id=video_id,
            keyframe_id=keyframe_id,
            frame_id=frame_id,
            question="Hỏi ngày hiển thị trên màn hình là ngày nào?",
            answer=match.group(1),
            ocr_text=ocr_text,
            template="date",
        )
        for match in _DATE_RE.finditer(ocr_text)
    ]


def generate_cases(
    ocr_text: str, *, video_id: str, keyframe_id: str, frame_id: int
) -> list[SyntheticQnaCase]:
    """Derive every question this frame's text can answer, or nothing.

    Silence is a valid outcome. Text supporting no rule produces no case --
    inventing a question from it would put a wrong expectation into the
    benchmark, which is worse than a smaller benchmark.
    """
    if not ocr_text or not ocr_text.strip():
        return []
    kwargs = {"video_id": video_id, "keyframe_id": keyframe_id, "frame_id": frame_id}
    return [
        *_administrative_cases(ocr_text, **kwargs),
        *_date_cases(ocr_text, **kwargs),
    ]


def drop_ambiguous_cases(
    cases: Sequence[SyntheticQnaCase], *, max_occurrences: int = 3
) -> list[SyntheticQnaCase]:
    """Remove answers that appear on too many frames to identify one.

    The organizer's rule requires ``frame_id`` to fall inside the
    ground-truth interval, so a case whose answer is visible on hundreds of
    frames marks a *right* answer wrong -- the benchmark would understate
    performance, which is the more dangerous direction: it would have us
    chase problems that are not there.

    Counted on the normalized answer, since "GIANG LY" and "Giang Ly" score
    identically and so must be treated as one answer here too.
    """
    if max_occurrences < 1:
        raise ValueError("max_occurrences must be >= 1")
    counts = Counter(_normalize_answer(case.answer) for case in cases)
    return [
        case
        for case in cases
        if counts[_normalize_answer(case.answer)] <= max_occurrences
    ]

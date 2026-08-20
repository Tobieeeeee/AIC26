"""Official AIC Q&A (Visual Question Answering) scoring rule.

Aggregation mechanics (R@k cutoffs, Final Score = mean of R@k) are identical
to Textual KIS (see ``mvr.evaluation.aic_kis``) per the organizer document --
this is intentionally a *separate* implementation, not a parametrized call
into ``evaluate_kis()``, matching the existing KIS-vs-generic-metrics
separation recorded in ``docs/CODEX_PROGRESS.md`` (``evaluation/metrics.py``
is deliberately not reused for KIS either). ``QNA_CUTOFFS`` currently equals
``KIS_CUTOFFS`` only because both organizer rules coincide today; it is
redefined locally rather than imported so the two tracks never accidentally
couple if either rule changes independently.

The organizer's Q&A correctness rule is the KIS video+frame-interval check
ANDed with "answer matches semantically" -- a human-judge operation in the
real competition (see the organizer doc's own example: both ``"5"`` and
``"Năm"`` score correct against a ground truth of ``"5"``/``"Năm"``, i.e.
non-exact-string semantic equivalence). This repository cannot fabricate a
human judge, so ``qna_answer_matches()`` implements a documented
normalized/number-word-equivalence APPROXIMATION only:

- case-fold + whitespace/punctuation normalization for exact-after-normalize
  matches;
- digit <-> VI/EN number-word equivalence (0-20) for the organizer's own
  worked example;
- everything else scores 0, deliberately NOT attempting broader semantic
  similarity (no embedding/LLM-as-judge here) -- a false "match" would be
  worse than an honest miss for an internal benchmarking tool. If this is
  too conservative for observed real queries, widen it explicitly in a
  later task rather than silently.

Any Final Score computed with this module MUST be labeled an approximate
self-evaluation and must never be presented as the organizer's official
judged score, mirroring the same caveat ``aic_kis`` already carries for its
own aggregate.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

QNA_CUTOFFS: tuple[int, ...] = (1, 5, 20, 50, 100)

_PUNCTUATION_RE = re.compile(r"[.,!?;:'\"()\[\]]")
_WHITESPACE_RE = re.compile(r"\s+")

_VI_NUMBER_WORDS = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
    "năm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
    "mười một": "11", "mười hai": "12", "mười ba": "13", "mười bốn": "14",
    "mười lăm": "15", "mười sáu": "16", "mười bảy": "17", "mười tám": "18",
    "mười chín": "19", "hai mươi": "20",
}

_EN_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_NUMBER_WORDS = {**_VI_NUMBER_WORDS, **_EN_NUMBER_WORDS}


@dataclass(frozen=True)
class QnaGroundTruth:
    video_id: str
    start_frame: int
    end_frame: int
    answer: str


@dataclass(frozen=True)
class QnaAnswer:
    video_id: str
    frame_id: int
    answer: str


@dataclass(frozen=True)
class QnaScore:
    recall_at_k: dict[int, float]
    final_score: float

    def to_dict(self) -> dict[str, float]:
        payload = {f"recall@{k}": v for k, v in self.recall_at_k.items()}
        payload["final_score"] = self.final_score
        return payload


def _normalize_answer_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold().strip()
    normalized = _PUNCTUATION_RE.sub("", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def _to_digit_string(normalized_text: str) -> str | None:
    if normalized_text.isdigit():
        return normalized_text
    return _NUMBER_WORDS.get(normalized_text)


def _strip_diacritics(text: str) -> str:
    """Fold Vietnamese diacritics so the prefix list needs one spelling each.

    Answers reach here from OCR, which mangles tone marks routinely -- the
    run that motivated this read "huyen" as "huyen" but "Khanh" as "Khonh".
    Matching the unit word without its marks is what makes the rule hold on
    real, imperfect input. U+0111 (d-stroke) has no NFD decomposition, so it
    is mapped explicitly.
    """
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


#: Vietnamese administrative-unit words. In an answer they label the kind of
#: place, they are not part of its name -- "Xa Giang Ly" answers "what is
#: this commune called?" with the name "Giang Ly". No entry is a
#: space-delimited prefix of another, so the order here does not affect
#: which one matches.
_ADMIN_UNIT_PREFIXES = (
    "thanh pho",
    "thi tran",
    "thi xa",
    "huyen",
    "phuong",
    "quan",
    "tinh",
    "xa",
)


def _strip_leading_admin_unit(normalized_text: str) -> str:
    """Drop one leading administrative-unit word, if anything survives it.

    Leading only, and one only: the goal is to equate "Xa Giang Ly" with
    "Giang Ly", NOT to reduce a whole address to its first name. Stripping
    mid-string would make "Xa Giang Ly huyen Khanh Vinh" -- which is more
    than the answer -- score as correct, so the strict matcher stays strict
    and that case is left to the lenient bound.
    """
    folded = _strip_diacritics(normalized_text)
    for prefix in _ADMIN_UNIT_PREFIXES:
        if folded.startswith(prefix + " "):
            remainder = normalized_text[len(prefix) + 1 :].strip()
            # Never return empty: a bare unit word would then match anything.
            return remainder or normalized_text
    return normalized_text


def _match_keys(text: str) -> set[str]:
    """Every normalized form one answer may legitimately take."""
    normalized = _normalize_answer_text(text)
    if not normalized:
        return set()
    keys = {normalized, _strip_leading_admin_unit(normalized)}
    for key in tuple(keys):
        digits = _to_digit_string(key)
        if digits is not None:
            keys.add(digits)
    return keys


def qna_answer_matches(candidate: str, ground_truth: str) -> bool:
    """Approximate semantic-match check -- see module docstring for scope.

    Widened once, deliberately and with a stated rule, after the first real
    end-to-end run: the model answered "Xa Giang Ly" where ground truth said
    "Giang Ly" and scored 0, which the organizer's human judge would not
    have done. Only a *leading* administrative-unit word is normalized away;
    everything looser belongs to ``qna_answer_matches_lenient``.
    """
    candidate_keys = _match_keys(candidate)
    truth_keys = _match_keys(ground_truth)
    if not candidate_keys or not truth_keys:
        return False
    return bool(candidate_keys & truth_keys)


def qna_answer_matches_lenient(candidate: str, ground_truth: str) -> bool:
    """Upper bound on what a human judge might accept: containment.

    Deliberately NOT the default. It accepts an answer buried in a longer
    reply ("Xa Giang Ly huyen Khanh Vinh" for "Giang Ly"), which a judge
    plausibly would and plausibly would not. Reporting it *alongside* the
    strict score is the honest move -- one number cannot express a range,
    and quietly adopting the generous end would have us stop fixing things
    that still need fixing.
    """
    if qna_answer_matches(candidate, ground_truth):
        return True
    norm_candidate = _normalize_answer_text(candidate)
    norm_truth = _normalize_answer_text(ground_truth)
    if not norm_candidate or not norm_truth:
        return False
    return norm_truth in norm_candidate or norm_candidate in norm_truth


AnswerMatcher = Callable[[str, str], bool]


def qna_r_score(
    answer: QnaAnswer, gt: QnaGroundTruth, *, matcher: AnswerMatcher = qna_answer_matches
) -> float:
    if answer.video_id != gt.video_id:
        return 0.0
    if not (gt.start_frame <= answer.frame_id <= gt.end_frame):
        return 0.0
    if not matcher(answer.answer, gt.answer):
        return 0.0
    return 1.0


def evaluate_qna(
    answers: Sequence[QnaAnswer],
    gt: QnaGroundTruth,
    *,
    matcher: AnswerMatcher = qna_answer_matches,
) -> QnaScore:
    """Score up to 100 ranked answers against one ground-truth interval+answer.

    ``answers`` must already be rank-ordered (rank 1 first). Ranks beyond
    100 are dropped before scoring, matching the official rule regardless
    of how many answers the caller passes in.
    """
    considered = list(answers[:100])
    recall_at_k: dict[int, float] = {}
    for cutoff in QNA_CUTOFFS:
        top = considered[:cutoff]
        recall_at_k[cutoff] = max(
            (qna_r_score(a, gt, matcher=matcher) for a in top), default=0.0
        )
    final_score = sum(recall_at_k[cutoff] for cutoff in QNA_CUTOFFS) / len(QNA_CUTOFFS)
    return QnaScore(recall_at_k=recall_at_k, final_score=final_score)

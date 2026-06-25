from __future__ import annotations

import re


NORMALIZATION_REPLACEMENTS = [
    (re.compile(r"\bм\s*г\s*р\s*п\b", re.IGNORECASE), "МГРП"),
    (re.compile(r"\bв\s*к\s*р\b", re.IGNORECASE), "ВКР"),
    (re.compile(r"\bп\s*в\s*т\b", re.IGNORECASE), "PVT"),
    (re.compile(r"\bигрек\b", re.IGNORECASE), "Y"),
    (re.compile(r"\bикс\b", re.IGNORECASE), "X"),
    (re.compile(r"\br\s*квадрат\b", re.IGNORECASE), "R²"),
    (re.compile(r"\bсемь\s+тридцать\b", re.IGNORECASE), "7:30"),
    (re.compile(r"\b7[.,]30\b"), "7:30"),
    (re.compile(r"\bчетверг\s+именно\b", re.IGNORECASE), "четверг"),
]

LEADING_FILLER_RE = re.compile(
    r"^\s*(?:ну|вот|это\s+самое|как\s+бы|значит|короче)\b[\s,]*",
    re.IGNORECASE,
)

TRAILING_MIDWORD_CUTOFF_RE = re.compile(r"\s+\S*-\s*$")


def _strip_leading_fillers(value: str) -> str:
    while True:
        stripped = LEADING_FILLER_RE.sub("", value)
        if stripped == value:
            return value
        value = stripped


def normalize_text_for_nlp(text: str) -> str:
    value = text or ""
    value = re.sub(r"\s+", " ", value).strip()
    value = _strip_leading_fillers(value)
    value = re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", value, flags=re.IGNORECASE)
    value = TRAILING_MIDWORD_CUTOFF_RE.sub("", value)
    value = re.sub(r"([.,!?;:])\1+", r"\1", value)
    value = re.sub(r"[.,]{2,}", ".", value)
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    value = value.strip()
    for pattern, replacement in NORMALIZATION_REPLACEMENTS:
        value = pattern.sub(replacement, value)
    if re.search(r"\b(параметр|модель|формул|коэффициент|значени|X|Y)\b", value, re.IGNORECASE):
        value = re.sub(r"\b(аш|аж)\b", "h", value, flags=re.IGNORECASE)
    return value


def normalize_text_for_display(text: str) -> str:
    value = text or ""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    return value

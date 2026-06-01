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


def normalize_text_for_nlp(text: str) -> str:
    value = text or ""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", value, flags=re.IGNORECASE)
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

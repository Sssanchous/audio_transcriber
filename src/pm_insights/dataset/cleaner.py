from __future__ import annotations

import re
import unicodedata


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
MOJIBAKE_RE = re.compile(r"(Р[А-Яа-яЁёA-Za-z]|С[А-Яа-яЁёA-Za-z]|вЂ|Гђ|Г‘)")
USEFUL_LATIN_TERMS = {"api", "backend", "frontend", "ui", "ux", "mvp", "json", "jsonl", "csv", "postgres", "sql"}

SERVICE_PATTERNS = [
    r"^протокол( встречи)?$",
    r"^рабочая встреча$",
    r"^ход встречи$",
    r"^повестка$",
    r"^дата\s*[:\-]",
    r"^дата и время\s*[:\-]",
    r"^дата встречи\s*[:\-]",
    r"^следующая встреча\s*[:\-]?",
    r"^участники\s*[:\-]?$",
    r"^тема\s*[:\-]?$",
    r"^страница\s+\d+$",
    r"^\d+\s*/\s*\d+$",
    r"^\d{1,3}$",
]

SHORT_STOP_PHRASES = {"да", "нет", "ок", "окей", "ага", "угу", "ну да", "ладно", "хорошо", "понятно", "спасибо"}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -–—•\t")


def has_mojibake(text: str) -> bool:
    return bool(MOJIBAKE_RE.search(text))


def has_language_signal(text: str) -> bool:
    lower = text.lower()
    return bool(CYRILLIC_RE.search(text)) or any(term in lower for term in USEFUL_LATIN_TERMS)


def drop_reason(text: str, min_length: int = 10) -> str | None:
    normalized = normalize_text(text)
    lower = normalized.lower().strip(" .,!?:;")

    if not normalized:
        return "empty"
    if len(normalized) < min_length:
        return "too_short"
    if len(normalized.split()) <= 1 and lower not in {"готово"}:
        return "too_short"
    if lower in SHORT_STOP_PHRASES:
        return "too_short"
    if has_mojibake(normalized):
        return "mojibake"
    if not has_language_signal(normalized):
        return "no_language_signal"
    if sum(ch.isalnum() for ch in normalized) < max(3, len(normalized) * 0.35):
        return "garbage"
    if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in SERVICE_PATTERNS):
        return "service"
    return None


def clean_fragment(text: str, min_length: int = 10) -> tuple[str | None, str | None]:
    normalized = normalize_text(text)
    reason = drop_reason(normalized, min_length=min_length)
    if reason:
        return None, reason
    return normalized, None

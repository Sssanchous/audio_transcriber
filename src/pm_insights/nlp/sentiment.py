from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from pm_insights import settings


POSITIVE_RE = re.compile(
    r"\b(хорошо|отлично|успешно|получилось|без\s+проблем|готово|нормально|супер|корректно|"
    r"все\s+нормально|всё\s+нормально|стабильно|согласовано)\b",
    re.IGNORECASE,
)
NEGATION_GOOD_RE = re.compile(
    r"\b(ошибок\s+нет|критических\s+ошибок\s+нет|рисков\s+нет|без\s+критических\s+ошибок|"
    r"сейчас\s+все\s+нормально|сейчас\s+всё\s+нормально|все\s+стабильно|всё\s+стабильно)\b",
    re.IGNORECASE,
)
MIXED_RECOVERY_RE = re.compile(
    r"\b(сбой|ошибка).{0,80}\b(но|сейчас).{0,80}\b(нормально|исправлено|работает|стабильно)\b",
    re.IGNORECASE,
)
ANSWER_RECOVERY_RE = re.compile(
    r"^\s*[\wА-Яа-яЁё]+[,.]\s*(да|хорошо).{0,120}\b(исправлю|проверю|обновлю|соберу|пришлю)\b"
    r".{0,120}\b(ошиб\w*|сбой\w*|рис\w*)\b|"
    r"^\s*[\wА-Яа-яЁё]+[,.]\s*(да|хорошо).{0,120}\b(ошиб\w*|сбой\w*|рис\w*)\b"
    r".{0,120}\b(исправлю|проверю|обновлю|соберу|пришлю)\b",
    re.IGNORECASE,
)
NEGATIVE_RE = re.compile(
    r"\b(плохо|ошиб\w*|сбой\w*|сбоев|сбоя|сбои|не\s+работает|не\s+успеваем|риск\w*|задержка|блокер|"
    r"сложно|не\s+получилось|сломалось|некорректно)\b|"
    r"\b(возникла|есть|обнаружена|выявлена)\s+проблема\b|\bпроблема\s+(с|в|на|при)\b",
    re.IGNORECASE,
)


def _problem_signal(text: str, sentiment: str) -> bool:
    if NEGATION_GOOD_RE.search(text or "") or MIXED_RECOVERY_RE.search(text or ""):
        return False
    return sentiment == "negative" and bool(NEGATIVE_RE.search(text or ""))


def _rule_based_sentiment(text: str) -> tuple[str, float]:
    text = text or ""
    if NEGATION_GOOD_RE.search(text):
        return "positive", 0.6
    if MIXED_RECOVERY_RE.search(text) or ANSWER_RECOVERY_RE.search(text):
        return "neutral", 0.0
    if NEGATIVE_RE.search(text):
        return "negative", -0.5
    if POSITIVE_RE.search(text):
        return "positive", 1.0
    return "neutral", 0.0


@lru_cache(maxsize=1)
def _load_rubert_pipeline() -> Any:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RuBERT sentiment engine requires optional dependencies from requirements-ml.txt "
            "(transformers and torch)."
        ) from exc
    return pipeline("text-classification", model=settings.RUBERT_SENTIMENT_MODEL)


def _map_rubert_label(raw_label: str) -> str:
    label = (raw_label or "").lower()
    if "neg" in label or "negative" in label:
        return "negative"
    if "pos" in label or "positive" in label:
        return "positive"
    return "neutral"


def _rubert_sentiment(text: str) -> tuple[str, float]:
    pipe = _load_rubert_pipeline()
    output = pipe(text or "")[0]
    label = _map_rubert_label(str(output.get("label", "")))
    confidence = float(output.get("score", 0.0) or 0.0)
    if label == "negative":
        return label, -confidence
    if label == "positive":
        return label, confidence
    return label, 0.0


def classify_sentiment(text: str, engine: str | None = None) -> dict[str, object]:
    selected = (engine or settings.SENTIMENT_ENGINE or "rule_based").lower()
    if selected == "auto":
        selected = "rubert"

    if selected == "rubert":
        try:
            label, score = _rubert_sentiment(text)
            return {"sentiment": label, "score": score, "engine": "rubert", "problem_signal": _problem_signal(text, label)}
        except Exception as exc:
            if not settings.ENABLE_MODEL_FALLBACK:
                raise
            label, score = _rule_based_sentiment(text)
            return {
                "sentiment": label,
                "score": score,
                "engine": "rule_based",
                "problem_signal": _problem_signal(text, label),
                "fallback_reason": str(exc),
            }

    label, score = _rule_based_sentiment(text)
    return {"sentiment": label, "score": score, "engine": "rule_based", "problem_signal": _problem_signal(text, label)}


def analyze_text_sentiment(text: str) -> tuple[str, float]:
    result = classify_sentiment(text, engine="rule_based")
    return str(result["sentiment"]), float(result["score"])


def analyze_sentiment(fragments: list[dict], engine: str | None = None) -> list[dict]:
    result = []
    for fragment in fragments:
        text = fragment.get("text", "")
        item = classify_sentiment(text, engine=engine)
        result.append(
            {
                "text": text,
                "sentiment": item["sentiment"],
                "score": item["score"],
                "source_fragment": fragment.get("fragment_index"),
                "engine": item["engine"],
                "problem_signal": item.get("problem_signal", False),
            }
        )
    return result

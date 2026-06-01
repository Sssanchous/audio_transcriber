from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .text_normalization import normalize_text_for_nlp


MARKERS = {
    "project_meeting": [
        "задача",
        "дедлайн",
        "ответственный",
        "статус",
        "клиент",
        "релиз",
        "презентация",
        "отчет",
        "отчёт",
        "срок",
        "до пятницы",
        "до завтра",
        "кто отвечает",
        "открытые задачи",
        "бюджет",
    ],
    "technical_research": [
        "гидродинамика",
        "скважина",
        "дебит",
        "МГРП",
        "скин-фактор",
        "параметры пласта",
        "интерпретация",
        "аппроксимация",
        "модель",
        "генерализация",
        "эталонные данные",
        "промысловые данные",
        "R²",
        "вязкость",
        "PVT",
        "безразмерные кривые",
        "проницаемость",
        "трещина",
    ],
    "education_consultation": [
        "ВКР",
        "диплом",
        "научный руководитель",
        "глава",
        "страница",
        "обоснование",
        "комиссия",
        "проверка",
        "черновик",
        "работа",
        "исследование",
    ],
}


def _iter_text(items_or_text: str | Iterable[dict]) -> str:
    if isinstance(items_or_text, str):
        return items_or_text
    return " ".join(str(item.get("normalized_text") or item.get("text") or "") for item in items_or_text)


def _marker_count(text: str, marker: str) -> int:
    marker_lower = marker.lower()
    if " " in marker_lower or marker.isupper() or marker in {"R²", "PVT"}:
        return text.count(marker_lower)
    return len(re.findall(rf"\b{re.escape(marker_lower)}\w*\b", text))


def detect_meeting_type(items_or_text: str | Iterable[dict]) -> dict:
    text = normalize_text_for_nlp(_iter_text(items_or_text)).lower()
    scores: Counter[str] = Counter()
    matched: dict[str, list[str]] = {label: [] for label in MARKERS}

    for label, markers in MARKERS.items():
        for marker in markers:
            count = _marker_count(text, marker)
            if count:
                scores[label] += count
                matched[label].append(marker)

    if not scores:
        return {
            "label": "general_discussion",
            "confidence": 0.3,
            "matched_markers": [],
            "scores": {},
        }

    project = scores["project_meeting"]
    technical = scores["technical_research"]
    education = scores["education_consultation"]
    dominant, dominant_score = scores.most_common(1)[0]
    total = sum(scores.values())

    if project >= 3 and (technical + education) >= 4:
        label = "mixed"
        confidence = min(0.9, (project + technical + education) / max(total, 1))
        markers = sorted({m for values in matched.values() for m in values})[:12]
    else:
        label = dominant
        confidence = min(0.95, max(0.45, dominant_score / max(total, 1)))
        if dominant_score >= 5:
            confidence = max(confidence, 0.78)
        markers = matched[dominant][:12]

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "matched_markers": markers,
        "scores": dict(scores),
    }

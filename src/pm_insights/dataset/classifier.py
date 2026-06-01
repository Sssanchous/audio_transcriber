from __future__ import annotations

import re
from dataclasses import dataclass


VALID_LABELS = [
    "task",
    "question",
    "answer",
    "decision",
    "deadline",
    "responsible",
    "aspect",
    "sentiment_positive",
    "sentiment_neutral",
    "sentiment_negative",
    "other",
]

PRIORITY = [
    "question",
    "decision",
    "deadline",
    "responsible",
    "task",
    "answer",
    "sentiment_negative",
    "sentiment_positive",
    "aspect",
    "sentiment_neutral",
    "other",
]


@dataclass(frozen=True)
class RuleGroup:
    label: str
    patterns: tuple[str, ...]
    confidence: float


RULES = [
    RuleGroup("question", (r"\?", r"^\s*(кто|что|когда|где|почему|зачем|как|сколько|какой|какая|какие)\b", r"\b(можем|получится|есть|готово|успеваем)\s+ли\b"), 0.95),
    RuleGroup("decision", (r"\b(решили|договорились|принимаем|утверждаем|оставляем|выбираем|фиксируем|итог|решение|согласовали)\b",), 0.9),
    RuleGroup("deadline", (r"\bсрок\s+выполнения\b", r"\bдо\s+(пятницы|завтра|конца\s+недели|конца\s+дня|вечера|утра|понедельника|вторника|среды|четверга)\b", r"\bк\s+(понедельнику|вторнику|среде|четвергу|пятнице|релизу)\b", r"\bдедлайн\b", r"\b(закончить|подготовить|сдать|отправить|завершить|доработать).{0,80}\b(к|до)\s+[А-Яа-яЁё0-9. -]+\b"), 0.88),
    RuleGroup("responsible", (r"\b(ответственный|ответственная|назначаем|бер[её]т\s+на\s+себя|за\s+это\s+отвечает|это\s+делает|поручаем|исполнитель)\b",), 0.88),
    RuleGroup("task", (r"\b(нужно|надо|необходимо|должен|должна|должны)\b", r"\b(сделай|сделать|подготовь|подготовить|проверь|проверить|исправь|исправить|реализуй|реализовать|добавь|добавить|обнови|обновить|согласуй|согласовать|отправь|отправить|оформи|оформить)\b", r"\b(задача|поручение)\b"), 0.82),
    RuleGroup("answer", (r"^\s*(да|нет|пока\s+нет|готово|сделано)\b", r"\b(я\s+посмотрю|я\s+сделал|я\s+проверил|я\s+исправил|мы\s+сделали|мы\s+проверили|сейчас\s+делаю|возьму\s+в\s+работу)\b"), 0.8),
    RuleGroup("sentiment_negative", (r"\b(плохо|ошибка|не\s+работает|не\s+успеваем|риск|задержка|блокер|сложно|не\s+получилось|сломалось|некорректно)\b", r"\b(возникла|есть|выявлена|обнаружена)\s+проблема\b", r"\bпроблема\s+(с|в|на|при)\b"), 0.82),
    RuleGroup("sentiment_positive", (r"\b(хорошо|отлично|успешно|получилось|без\s+проблем|готово|нормально|супер|корректно)\b",), 0.78),
    RuleGroup("aspect", (r"\b(сроки|срок|ресурсы|бюджет|дизайн|оплата|сервер|фронтенд|frontend|backend|бэкенд|база\s+данных|документация|тестирование|интеграция|риски|проблема|блокер|качество|коммуникация|согласование|релиз|деплой)\b",), 0.72),
]


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip()
            if value and value not in found:
                found.append(value)
    return found


def classify_fragment(text: str, include_other: bool = True) -> dict:
    labels: list[str] = []
    matched_rules: list[str] = []
    confidences: dict[str, float] = {}

    for rule in RULES:
        matched = _matches(rule.patterns, text or "")
        if matched:
            labels.append(rule.label)
            matched_rules.extend(matched)
            confidences[rule.label] = rule.confidence

    if not labels:
        label = "sentiment_neutral" if include_other else "other"
        labels = [label]
        confidences[label] = 0.55 if include_other else 0.4

    label = sorted(labels, key=lambda item: PRIORITY.index(item))[0]
    secondary = [item for item in PRIORITY if item in labels and item != label]
    return {
        "label": label,
        "secondary_labels": secondary,
        "matched_rules": matched_rules,
        "confidence": round(max(confidences.values()) if confidences else 0.4, 2),
    }

from __future__ import annotations

import re

from .entities import extract_entities


MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
DAY_WORDS = "один|два|две|тр[еи]|четыре|пять|шесть|семь|восемь|девять|десять|двух|трех|трёх|пяти"
NUMBER_OR_WORD = rf"(?:\d+|{DAY_WORDS})"

DEADLINE_PATTERNS = [
    rf"\bдо\s+\d{{1,2}}\s+(?:{MONTHS})\b",
    r"\bне\s+позже\s+\d{1,2}[- ]?(?:го|ого)?\s+числа\b",
    rf"\bоколо\s+\d{{1,2}}\s+(?:{MONTHS})\b",
    r"\bоколо\s+\d{2}[- ]?(?:го|ого)?(?:\s+числа)?\b",
    r"\b\d{1,2}\s*[–-]\s*\d{1,2}\s+числ\w*\b",
    rf"\b\d{{1,2}}\s*[–-]\s*\d{{1,2}}\s+(?:{MONTHS})\b",
    rf"\bчерез\s+{NUMBER_OR_WORD}\s+(?:календарн\w+|рабоч\w+)?\s*(?:день|дня|дней)\b",
    rf"\bза\s+{NUMBER_OR_WORD}\s+(?:календарн\w+|рабоч\w+)?\s*(?:день|дня|дней)\s+до\b",
    rf"\bминимум\s+за\s+{NUMBER_OR_WORD}\s+(?:календарн\w+|рабоч\w+)?\s*(?:день|дня|дней)\b",
    rf"\bв\s+течение\s+{NUMBER_OR_WORD}\s+(?:рабоч\w+\s+)?(?:час|часа|часов|день|дня|дней)\b",
    r"\bсегодня[- ]завтра\b",
    r"\bзавтра\s+до\s+конца\s+дня\b",
    r"\bдо\s+подписания\s+(?:контракта|договора|term sheet|термшита)\b",
    r"\bпосле\s+подписания\s+(?:контракта|договора|term sheet|термшита)\b",
    r"\bдо\s+первой\s+отгрузки\b",
    r"\bдва\s+раза\s+в\s+неделю\b",
    r"\bза\s+пять\s+дней\s+до\s+отгрузки\b",
    r"\bсегодня\s+до\s+\d{1,2}(?:[:.]\d{2})?\s*(?:час(?:а|ов|ам)?|вечера|утра)?\b",
    r"\bдо\s+завтра\s+к\s+\d{1,2}(?:[:.]\d{2})?\s*(?:час(?:а|ов|ам)?)?\b",
    r"\bдо\s+(?:пятницы|завтра|конца\s+недели|конца\s+дня|вечера|утра|понедельника|вторника|среды|четверга)\b",
    r"\bк\s+(?:понедельнику|вторнику|среде|четвергу|пятнице|демо|следующей\s+встрече)\b",
    r"\bна\s+следующей\s+неделе\b",
    r"\bследующ(?:ая|ей)\s+недел[яе]\b",
    r"\b(?:срок|дедлайн|срок\s+выполнения)\b",
]
ORG_TIME_PATTERNS = [
    r"\bчетверг\b",
    r"\bпосле\s+7\b",
    r"\b7[.:]30\b",
    r"\b\d{1,2}[.:]\d{2}\b",
]
SERVICE_DATE_RE = re.compile(
    r"\b(?:дата\s+встречи|дата\s+и\s+время|следующая\s+встреча|дата\s+следующей\s+встречи)\b",
    re.IGNORECASE,
)
TECH_TIME_RE = re.compile(
    r"\b(?:тысяч[аи]?\s+часов|тысяча\s+часов|две\s+тысячи\s+часов|часов\s+такта|n\s+часов|котировочных\s+дней)\b",
    re.IGNORECASE,
)
PAGE_OR_PARAMETER_RE = re.compile(
    r"\b(?:страниц[аые]?\s+\d+|\d+\s*(?:метр(?:а|ов)?|процент(?:а|ов)?|%))\b",
    re.IGNORECASE,
)
ORG_CONTEXT_RE = re.compile(
    r"\b(встреч\w*|созвон\w*|будние\s+дни|свободные|согласовать\s+время|поставим|давайте)\b",
    re.IGNORECASE,
)
TASK_MARKERS_RE = re.compile(
    r"\b(подготовь|проверь|исправь|сделай|отправь|согласуй|обнови|собери|пришли|"
    r"направить|предоставить|подтвердить|прописать|нужно|надо|задача)\b",
    re.IGNORECASE,
)
ANSWER_MARKERS_RE = re.compile(
    r"\b(подготовлю|отправлю|исправлю|проверю|соберу|пришлю|сделаю|уточню|планируем|да|нет)\b",
    re.IGNORECASE,
)


def find_deadlines(text: str) -> list[str]:
    text = text or ""
    if SERVICE_DATE_RE.search(text) or TECH_TIME_RE.search(text) or PAGE_OR_PARAMETER_RE.search(text):
        return []
    matches: list[str] = []
    for pattern in DEADLINE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(0).strip(" .,:;")
            if value.lower().startswith("около") and re.search(
                rf"{re.escape(value)}\s+(?:цент|доллар|баррел|%)", text, flags=re.IGNORECASE
            ):
                continue
            if value and value not in matches:
                matches.append(value)

    if ORG_CONTEXT_RE.search(text):
        for pattern in ORG_TIME_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip(" .,:;")
                if value and value not in matches:
                    matches.append(value)

    lower = text.lower()
    if any(
        word in lower
        for word in [
            "закончить",
            "подготовить",
            "подготовлю",
            "сдать",
            "отправить",
            "отправлю",
            "направить",
            "направлю",
            "предоставить",
            "подтвердить",
            "дедлайн",
            "срок",
        ]
    ):
        for date in extract_entities(text).get("dates", []):
            if date not in matches:
                matches.append(date)
    return matches


def classify_deadline_kind(text: str) -> str:
    if ORG_CONTEXT_RE.search(text or ""):
        return "meeting_time"
    if TASK_MARKERS_RE.search(text or ""):
        return "task_deadline"
    if ANSWER_MARKERS_RE.search(text or ""):
        return "answer_deadline"
    return "mention"


def extract_deadlines(fragments: list[dict]) -> list[dict]:
    result = []
    for fragment in fragments:
        text = fragment.get("text", "")
        matches = find_deadlines(text)
        if matches:
            result.append(
                {
                    "text": text,
                    "deadlines": matches,
                    "deadline_normalized": None,
                    "kind": classify_deadline_kind(text),
                    "source_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                }
            )
    return result

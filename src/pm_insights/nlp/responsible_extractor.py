from __future__ import annotations

import re
from collections import Counter

from .entities import extract_entities


KNOWN_NAMES = {
    "Алексей",
    "Анна",
    "Иван",
    "Мария",
    "Илья",
    "Ольга",
    "Дмитрий",
    "Сергей",
    "Екатерина",
    "Павел",
    "Николай",
}
KNOWN_NAMES.update(
    {
        "Андрей",
        "Александр",
        "Михаил",
        "Максим",
        "Денис",
        "Роман",
        "Владимир",
        "Анастасия",
        "Дарья",
        "Елена",
        "Наталья",
    }
)
STOPWORDS = {
    "и",
    "а",
    "в",
    "с",
    "по",
    "из",
    "мы",
    "можем",
    "не",
    "так",
    "то",
    "уже",
    "может",
    "лучшие",
    "данных",
    "неделю",
    "просто",
    "пакетом",
    "границей",
    "плане",
    "том",
    "нам",
    "аж",
    "певицы",
    "вы",
    "он",
    "она",
    "они",
    "параметр",
    "модель",
    "скважина",
    "встреча",
    "задача",
}
NAME_RE = r"[А-ЯЁ][а-яё]{2,}"
IMPERATIVE_RE = re.compile(
    r"\b(подготовь|проверь|исправь|сделай|отправь|согласуй|обнови|собери|пришли|"
    r"добавь|оформи|реализуй|раздели|опиши|скинь|скидывай|посмотри)\b",
    re.IGNORECASE,
)
ANSWER_SELF_RE = re.compile(
    r"\b(сделаю|исправлю|проверю|подготовлю|отправлю|соберу|пришлю|"
    r"уточню|обновлю|отвечаю\s+я)\b",
    re.IGNORECASE,
)


def _participant_aliases(participants: list[str] | None) -> set[str]:
    aliases = set()
    for name in participants or []:
        clean = (name or "").strip()
        if not clean:
            continue
        aliases.add(clean)
        aliases.add(clean.split()[0])
    return aliases


def _is_valid_person(value: str, participants: list[str] | None = None, ner_people: set[str] | None = None) -> bool:
    name = (value or "").strip(" ,.:;")
    if not name:
        return False
    if name.lower() in STOPWORDS:
        return False
    return name in KNOWN_NAMES or name in _participant_aliases(participants) or name in (ner_people or set())


def _leading_names(text: str, participants: list[str] | None = None, ner_people: set[str] | None = None) -> list[str]:
    parts = re.split(r"[,.:;]\s*", text or "")[:3]
    return [part.strip() for part in parts if _is_valid_person(part.strip(), participants, ner_people)]


def _canonical_name(candidate: str, participants: list[str] | None = None) -> str:
    clean = (candidate or "").strip(" ,.:;")
    for participant in participants or []:
        name = participant.strip()
        if clean == name or clean == name.split()[0]:
            return clean if clean in KNOWN_NAMES else name
    return clean


def _leading_name_before_imperative(
    text: str,
    participants: list[str] | None = None,
    ner_people: set[str] | None = None,
) -> str | None:
    aliases = _participant_aliases(participants) | KNOWN_NAMES | (ner_people or set())
    for alias in sorted(aliases, key=len, reverse=True):
        if not _is_valid_person(alias, participants, ner_people):
            continue
        pattern = rf"^\s*{re.escape(alias)}\s+{IMPERATIVE_RE.pattern}"
        if re.search(pattern, text or "", flags=re.IGNORECASE):
            return _canonical_name(alias, participants)
    return None


def find_responsibles(
    text: str,
    participants: list[str] | None = None,
    assume_task: bool = False,
) -> list[str]:
    text = text or ""
    names: list[str] = []
    entities = extract_entities(text)
    ner_people = set(entities.get("people", []))
    leading = _leading_names(text, participants, ner_people)
    leading_without_punctuation = _leading_name_before_imperative(text, participants, ner_people)
    if leading_without_punctuation:
        names.append(leading_without_punctuation)

    if len(leading) >= 2:
        parts = re.split(r"[,.:;]\s*", text, maxsplit=2)
        tail = parts[2] if len(parts) > 2 else ""
        if IMPERATIVE_RE.search(tail):
            names.append(_canonical_name(leading[1], participants))
    elif len(leading) == 1:
        parts = re.split(r"[,.:;]\s*", text, maxsplit=1)
        tail = parts[1] if len(parts) > 1 else ""
        is_capitalized_name = leading[0][:1].isupper()
        if (
            IMPERATIVE_RE.search(tail)
            or ANSWER_SELF_RE.search(tail)
            or re.search(r"\bза\b.+\bотвечаю\s+я\b", tail, re.IGNORECASE)
            or (assume_task and is_capitalized_name)
        ):
            names.append(_canonical_name(leading[0], participants))

    patterns = [
        rf"\b({NAME_RE})\s+отвечает\s+за\b",
        rf"\bза\s+.+?\s+отвечает\s+({NAME_RE})\b",
        rf"\b({NAME_RE})\s+бер[её]т\s+на\s+себя\b",
        rf"\bпоручаем\s+({NAME_RE})\b",
        rf"\bисполнитель\s*[-—:]?\s*({NAME_RE})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            candidate = match.group(1)
            if _is_valid_person(candidate, participants, ner_people):
                names.append(_canonical_name(candidate, participants))

    if re.search(r"\b(ответствен|поручаем|исполнитель|бер[её]т\s+на\s+себя)\b", text, re.IGNORECASE):
        for candidate in entities.get("people", []):
            if _is_valid_person(candidate, participants, ner_people):
                names.append(_canonical_name(candidate, participants))

    return sorted({name for name in names if _is_valid_person(name, participants, ner_people)})


OPENING_PARTICIPANTS_HEAD_RATIO = 0.1
FREQUENT_RESPONSIBLE_THRESHOLD = 3
CONFIDENCE_BASE = 0.7
CONFIDENCE_BOOSTED = 0.9


def infer_opening_participants(
    fragments: list[dict],
    head_ratio: float = OPENING_PARTICIPANTS_HEAD_RATIO,
) -> list[str]:
    if not fragments:
        return []
    head_count = max(1, round(len(fragments) * head_ratio))
    names: set[str] = set()
    for fragment in fragments[:head_count]:
        text = fragment.get("text", "")
        names.update(extract_entities(text).get("people", []))
    return sorted(names)


def extract_responsibles(fragments: list[dict], participants: list[str] | None = None) -> list[dict]:
    merged_participants = sorted({*(participants or []), *infer_opening_participants(fragments)})

    rows = []
    for fragment in fragments:
        text = fragment.get("text", "")
        names = find_responsibles(text, participants=merged_participants)
        if names:
            rows.append({"text": text, "responsibles": names, "source_fragment": fragment.get("fragment_index")})

    name_counts = Counter(name for row in rows for name in row["responsibles"])
    result = []
    for row in rows:
        boosted = any(name_counts[name] > FREQUENT_RESPONSIBLE_THRESHOLD for name in row["responsibles"])
        result.append({**row, "confidence": CONFIDENCE_BOOSTED if boosted else CONFIDENCE_BASE})
    return result

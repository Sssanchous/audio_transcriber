from __future__ import annotations

import re

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


def _is_valid_person(value: str, participants: list[str] | None = None) -> bool:
    name = (value or "").strip(" ,.:;")
    if not name:
        return False
    if name.lower() in STOPWORDS:
        return False
    return name in KNOWN_NAMES or name in _participant_aliases(participants)


def _leading_names(text: str, participants: list[str] | None = None) -> list[str]:
    parts = re.split(r"[,.:;]\s*", text or "")[:3]
    return [part.strip() for part in parts if _is_valid_person(part.strip(), participants)]


def _canonical_name(candidate: str, participants: list[str] | None = None) -> str:
    clean = (candidate or "").strip(" ,.:;")
    for participant in participants or []:
        name = participant.strip()
        if clean == name or clean == name.split()[0]:
            return clean if clean in KNOWN_NAMES else name
    return clean


def find_responsibles(text: str, participants: list[str] | None = None) -> list[str]:
    text = text or ""
    names: list[str] = []
    leading = _leading_names(text, participants)

    if len(leading) >= 2:
        parts = re.split(r"[,.:;]\s*", text, maxsplit=2)
        tail = parts[2] if len(parts) > 2 else ""
        if IMPERATIVE_RE.search(tail):
            names.append(_canonical_name(leading[1], participants))
    elif len(leading) == 1:
        parts = re.split(r"[,.:;]\s*", text, maxsplit=1)
        tail = parts[1] if len(parts) > 1 else ""
        if IMPERATIVE_RE.search(tail) or ANSWER_SELF_RE.search(tail) or re.search(r"\bза\b.+\bотвечаю\s+я\b", tail, re.IGNORECASE):
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
            if _is_valid_person(candidate, participants):
                names.append(_canonical_name(candidate, participants))

    if re.search(r"\b(ответствен|поручаем|исполнитель|бер[её]т\s+на\s+себя)\b", text, re.IGNORECASE):
        for candidate in extract_entities(text).get("people", []):
            if _is_valid_person(candidate, participants):
                names.append(_canonical_name(candidate, participants))

    return sorted({name for name in names if _is_valid_person(name, participants)})


def extract_responsibles(fragments: list[dict], participants: list[str] | None = None) -> list[dict]:
    result = []
    for fragment in fragments:
        text = fragment.get("text", "")
        names = find_responsibles(text, participants=participants)
        if names:
            result.append({"text": text, "responsibles": names, "source_fragment": fragment.get("fragment_index")})
    return result

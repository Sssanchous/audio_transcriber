from __future__ import annotations

import re


KNOWN_SPEAKERS = "Алексей|Анна|Иван|Мария|Илья|Ольга|Дмитрий|Сергей|Екатерина|Павел|Николай"
SPEAKER_PREFIX_RE = re.compile(rf"^\s*(?:{KNOWN_SPEAKERS})\s*[,.:]\s*", re.IGNORECASE)
QUESTION_START_RE = re.compile(
    r"^\s*(кто|что|когда|где|почему|зачем|сколько|какой|какая|какие|какое)\b|"
    r"^\s*(можно|нужно|есть|готово|правильно|успеваем)\s+ли\b|"
    r"^\s*(я\s+правильно\s+понимаю|хотел(?:а)?\s+уточнить|подскажите|скажите|можно\s+уточнить)\b",
    re.IGNORECASE,
)
QUESTION_STOP_RE = re.compile(
    r"\b(как\s+правило|как\s+вариант|как\s+раз|как\s+раз-таки|как\s+конструкция|"
    r"как\s+первичный\s+подход|как\s+для\s+инженера|как\s+в\s+сетках|"
    r"где\s+меньше\s+ошибку\s+получим|почему\s+я\s+спрашиваю|в\s+чем\s+смысл\s+есть|"
    r"так,\s*где-то\s+было|где-то\s+было|какой-то|что\s+мы\s+тогда|"
    r"когда\s+мы\s+обратную\s+задачу\s+решаем|что\s+оно\s+только)\b",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(
    r"^\s*(да|нет|готово|сделано|пока\s+нет|хорошо|понял|поняла|окей)\b|"
    r"\b(я\s+отправлю|отправлю|уточню|лежит|отвечаю\s+я|открыты|закрыты|планируем|"
    r"сделаю|исправлю|проверю|подготовлю|соберу|пришлю|обновлю|проверил|проверила|"
    r"исправил|исправила|уже|в\s+процессе|я\s+сделал|мы\s+сделали|возьму\s+в\s+работу|"
    r"стабильно|ошибок\s+нет|готова|готов|для\s+этого|в\s+таком\s+случае|это\s+означает)\b",
    re.IGNORECASE,
)


def strip_speaker_prefix(text: str) -> str:
    return SPEAKER_PREFIX_RE.sub("", text or "").strip()


def _words_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or ""))


def is_question(text: str) -> bool:
    clean = strip_speaker_prefix(text)
    if not clean:
        return False
    lower = clean.lower()
    if QUESTION_STOP_RE.search(lower):
        return False
    if "?" in clean:
        if _words_count(clean) < 3 and not QUESTION_START_RE.search(clean):
            return False
        if re.search(r"^[А-Яа-яЁёA-Za-z-]+\?\s*(ну\s+да|да|нет)\b", clean, re.IGNORECASE):
            return False
        return True
    return bool(QUESTION_START_RE.search(clean))


def is_answer(text: str) -> bool:
    clean = strip_speaker_prefix(text)
    return bool(ANSWER_RE.search(clean))


def split_question_spans(text: str) -> list[str]:
    clean = strip_speaker_prefix(text)
    if not is_question(clean):
        return []
    if clean.count("?") <= 1:
        return [clean]

    spans: list[str] = []
    start = 0
    for match in re.finditer(r"\?", clean):
        part = clean[start : match.end()].strip(" ,.;")
        start = match.end()
        if is_question(part):
            spans.append(part)
    tail = clean[start:].strip(" ,.;")
    if tail and is_question(tail):
        spans.append(tail)
    return spans or [clean]


def _is_plausible_answer(candidate_text: str) -> bool:
    clean = strip_speaker_prefix(candidate_text)
    if not clean or is_question(clean):
        return False
    if is_answer(clean):
        return True
    return _words_count(clean) >= 5 and not QUESTION_STOP_RE.search(clean)


def _trim_answer(answer: str | None, limit: int = 1000) -> tuple[str | None, bool]:
    if not answer or len(answer) <= limit:
        return answer, False
    clipped = answer[:limit]
    sentence_end = max(clipped.rfind("."), clipped.rfind("?"), clipped.rfind("!"))
    if sentence_end > 150:
        return clipped[: sentence_end + 1].strip(), True
    return clipped.rstrip(" ,.;") + "...", True


def extract_qa_pairs(fragments: list[dict], max_gap: int = 5) -> list[dict]:
    pairs = []
    for index, fragment in enumerate(fragments):
        text = fragment.get("text", "")
        questions = split_question_spans(text)
        if not questions:
            continue

        answer_parts: list[str] = []
        answer_fragments: list[int] = []
        status = "not_answered"
        for offset in range(1, max_gap + 1):
            if index + offset >= len(fragments):
                break
            candidate = fragments[index + offset]
            candidate_text = candidate.get("text", "")
            if is_question(candidate_text):
                break
            if _is_plausible_answer(candidate_text):
                answer_parts.append(candidate_text)
                answer_fragments.append(candidate.get("fragment_index") or candidate.get("block_index"))
                status = "answered" if is_answer(candidate_text) else "partial"
                if is_answer(candidate_text):
                    # A direct answer can be extended with one nearby explanatory
                    # fragment, but should not absorb a whole discussion.
                    continue
                break
            elif answer_parts:
                break

        answer_text = " ".join(part for part in answer_parts if part).strip() or None
        answer_text, was_trimmed = _trim_answer(answer_text)
        if was_trimmed and status == "answered":
            status = "partial"

        for question in questions:
            pairs.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "status": status,
                    "question_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                    "answer_fragment": answer_fragments[0] if answer_fragments else None,
                    "answer_fragments": [item for item in answer_fragments if item is not None],
                }
            )
    return pairs

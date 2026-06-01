from __future__ import annotations

import re

from .domain_profiles import get_domain_profile
from .text_normalization import normalize_text_for_display, normalize_text_for_nlp


QUESTION_START_RE = re.compile(
    r"^\s*(?:кто|что|когда|где|почему|зачем|сколько|какой|какая|какие|можно\s+ли|нужно\s+ли|есть\s+ли|"
    r"готово\s+ли|правильно\s+ли|я\s+правильно\s+понимаю|хотел(?:а)?\s+уточнить|подскажите|скажите|можно\s+уточнить)\b",
    re.IGNORECASE,
)
ACTION_START_RE = re.compile(
    r"^\s*(?:[А-ЯЁ][а-яё]{2,}\s*[,.:]\s*)?(?:подготовь|проверь|исправь|сделай|отправь|собери|обнови|"
    r"согласуй|нужно|надо|необходимо|давайте|из\s+ближайших\s+задач)\b",
    re.IGNORECASE,
)
ORG_RE = re.compile(r"\b(встреч|созвон|четверг|будние\s+дни|после\s+7|7[:.]30|следующ\w+\s+недел)\b", re.IGNORECASE)
CONTINUATION_RE = re.compile(
    r"^\s*(?:то\s+есть|ну|собственно|получается|в\s+таком\s+случае|поэтому|и\s+тогда|а\s+если|если|"
    r"при\s+этом|соответственно)\b",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(
    r"^\s*(?:да|нет|верно|не\s+совсем|для\s+этого|для\s+нефтяного\s+кейса|в\s+таком\s+случае|как\s+правило)\b",
    re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:Алексей|Анна|Иван|Мария|Илья|Ольга|Дмитрий|Сергей|Екатерина|Павел|Николай|"
    r"Андрей|Александр|Михаил|Максим|Денис|Роман|Владимир|Анастасия|Дарья|Елена|Наталья)\s*[,.:]",
    re.IGNORECASE,
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or ""))


def _has_final_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?…]\s*$", text or ""))


def _block_hint(text: str) -> str:
    stripped = text or ""
    if "?" in stripped or QUESTION_START_RE.search(stripped):
        return "question"
    if ACTION_START_RE.search(stripped):
        return "action_item"
    if ORG_RE.search(stripped):
        return "organization"
    return "discussion"


def _fragment_text(fragment: dict) -> str:
    return str(fragment.get("normalized_text") or fragment.get("text") or "")


def _should_break(current: list[dict], next_fragment: dict, max_words: int) -> bool:
    if not current:
        return False
    current_text = _fragment_text(current[-1])
    next_text = _fragment_text(next_fragment)
    words = sum(_word_count(_fragment_text(item)) for item in current) + _word_count(next_text)
    if words > max_words:
        return True
    pause = float(next_fragment.get("start", 0.0) or 0.0) - float(current[-1].get("end", 0.0) or 0.0)
    if pause > 6.0:
        return True
    if SPEAKER_PREFIX_RE.search(next_text) and _word_count(current_text) >= 5:
        return True
    if "?" in current_text or QUESTION_START_RE.search(current_text):
        return True
    if QUESTION_START_RE.search(next_text) and current:
        return True
    if ACTION_START_RE.search(next_text) and _word_count(current_text) >= 8:
        return True
    if ORG_RE.search(next_text) and _word_count(current_text) >= 10:
        return True
    if pause <= 4.0:
        if _word_count(current_text) < 18 or not _has_final_punctuation(current_text) or CONTINUATION_RE.search(next_text):
            return False
    return _has_final_punctuation(current_text) and not CONTINUATION_RE.search(next_text)


def _make_block(index: int, fragments: list[dict]) -> dict:
    normalized_parts = [_fragment_text(item) for item in fragments]
    display_parts = [normalize_text_for_display(str(item.get("text") or "")) for item in fragments]
    text = normalize_text_for_nlp(" ".join(normalized_parts))
    display_text = normalize_text_for_display(" ".join(display_parts))
    return {
        "block_index": index,
        "fragment_index": index,
        "text": text,
        "display_text": display_text,
        "start": fragments[0].get("start"),
        "end": fragments[-1].get("end"),
        "source_fragments": [item.get("fragment_index") for item in fragments],
        "word_count": _word_count(text),
        "has_question": "?" in text or bool(QUESTION_START_RE.search(text)),
        "speaker": None,
        "block_type_hint": _block_hint(text),
    }


def build_semantic_blocks(fragments: list[dict], meeting_type_label: str | None = None) -> list[dict]:
    if not fragments:
        return []
    profile = get_domain_profile(meeting_type_label)
    max_words = int(profile["max_block_words"])
    blocks: list[dict] = []
    current: list[dict] = []

    for fragment in fragments:
        normalized = dict(fragment)
        normalized["normalized_text"] = normalize_text_for_nlp(_fragment_text(fragment))
        if _should_break(current, normalized, max_words):
            blocks.append(_make_block(len(blocks) + 1, current))
            current = [normalized]
        else:
            current.append(normalized)
    if current:
        blocks.append(_make_block(len(blocks) + 1, current))

    enriched: list[dict] = []
    for idx, block in enumerate(blocks):
        if block["has_question"]:
            answer_context = []
            for follow in blocks[idx + 1 : idx + 5]:
                if follow["has_question"]:
                    break
                if ANSWER_RE.search(follow["text"]) or follow["word_count"] <= 35:
                    answer_context.append(follow["block_index"])
                if len(answer_context) >= 4:
                    break
            if answer_context:
                block = dict(block)
                block["answer_context_blocks"] = answer_context
        enriched.append(block)
    return enriched

from __future__ import annotations

import re

from pm_insights.dataset.classifier import classify_fragment

from .deadline_extractor import find_deadlines
from .qa_extractor import is_answer, is_question, strip_speaker_prefix
from .responsible_extractor import find_responsibles
from .responsible_side import find_responsible_side


NAME = r"[А-ЯЁ][а-яё]{2,}"

IMPERATIVE_RE = re.compile(
    r"\b(подготовь|проверь|исправь|сделай|отправь|согласуй|обнови|собери|пришли|"
    r"добавь|оформи|реализуй|раздели|опиши|скинь|скидывай|посмотри|"
    r"подтверди|уточни|забронируй|закрепи|передай|рассчитай|внеси|пропиши)\b",
    re.IGNORECASE,
)
PERSON_IMPERATIVE_RE = re.compile(rf"^\s*(?:{NAME}\s*[,.:]\s*){{1,3}}.*" + IMPERATIVE_RE.pattern, re.IGNORECASE)

TASK_ACTION_PATTERNS = (
    "подготовить",
    "направить",
    "отправить",
    "прислать",
    "предоставить",
    "согласовать",
    "подтвердить",
    "проверить",
    "прописать",
    "внести",
    "рассчитать",
    "уточнить",
    "бронировать",
    "забронировать",
    "закрепить",
    "оформить",
    "передать",
    "сделать",
    "добавить",
    "разделить",
    "описать",
    "скинуть",
    "сдать",
    "провести",
    "сформировать",
    "определить пул",
    "разработать",
    "реализовать",
    "протестировать",
    "дать комментарии",
    "будут направлены",
    "можем отправить",
    "должен подтвердить",
    "должна подтвердить",
    "должны подтвердить",
    "нужно прописать",
    "надо прописать",
    "необходимо прописать",
    "нужно указать",
    "надо указать",
    "нужно направить",
    "нужно отправить",
    "нужно предоставить",
    "нужно подтвердить",
    "нужно согласовать",
    "нужно проверить",
    "нужно бронировать",
    "надо бронировать",
)

BAD_TASK_PATTERNS = (
    "нужно понимать",
    "надо понимать",
    "важно понимать",
    "нужно определить",
    "надо определить",
    "можно сделать",
    "можно согласовать",
    "можно обсудить",
    "можно посмотреть",
    "можно разделить",
    "это можно прописать",
    "хотелось бы",
    "было бы хорошо",
    "ситуация зависит",
    "ситуация менее жесткая",
    "это приемлемо",
    "это разумно",
    "это понятно",
    "это логично",
    "можем подтвердить",
    "мы видим риск",
    "мы не хотим",
    "мы предпочитаем",
    "поставщику удобнее",
    "покупателю удобнее",
    "нужно смотреть причину",
    "должен отражаться в цене",
    "риск должен быть распределен",
    "риск был справедливо распределен",
    "качество должно отражаться в цене",
    "давайте делить 50 на 50",
    "делим 50 на 50",
    "того чтобы проверить",
    "того, чтобы проверить",
    "можно будет",
    "далеко не бесполезно",
    "себя поискать кейсы",
)

ACTION_ITEM_RE = re.compile(
    r"\b(из\s+ближайших\s+задач|тогда\s+нужно\s+сделать|вам\s+нужно\s+сделать|"
    r"нужно\s+(подготовить|проверить|добавить|разделить|описать|согласовать|скинуть|сделать|прописать|указать|направить|отправить|подтвердить)|"
    r"нужно\s+(?:\w+\s+){0,4}бронировать|"
    r"надо\s+.+\bсделать\b|необходимо\s+(подготовить|проверить|добавить|разделить|описать|согласовать|сделать|прописать|указать|бронировать))\b",
    re.IGNORECASE,
)
ORG_ACTION_RE = re.compile(
    r"\b(давайте\s+.+(четверг|встреч|созвон)|нужно\s+согласовать\s+время|"
    r"скинуть\s+промежуточные\s+результаты|скидывать\s+промежуточные\s+результаты|"
    r"подготовить\s+к\s+следующей\s+встрече|проверить\s+к\s+следующей\s+встрече)\b",
    re.IGNORECASE,
)
GENERAL_PM_TASK_RE = re.compile(
    r"\b("
    r"сделать|подготовить|проверить|исправить|реализовать|добавить|обновить|"
    r"согласовать|отправить|оформить|собрать|направить|прислать|предоставить|"
    r"подтвердить|прописать|внести|рассчитать|уточнить|забронировать|закрепить|передать|"
    r"сдать|провести|сформировать|разработать|протестировать"
    r")\b",
    re.IGNORECASE,
)
NON_TASK_ANSWER_RE = re.compile(r"^\s*(да|нет)\b|переносить\s+не\s+нужно|\bне\s+нужно\b", re.IGNORECASE)
TASK_STOP_RE = re.compile(
    r"\b(обратная\s+задача|задача\s+интерпретации|задача\s+эксперта|постановка\s+задач|"
    r"решение\s+задачи|математическая\s+задача|необходимо\s+время|необходимы\s+параметры|должно\s+быть\s+соответствие|"
    r"может\s+присутствовать|может\s+быть\s+предоставлено|можно\s+рассматривать|"
    r"как\s+правило|в\s+теории|в\s+принципе|как\s+гипотеза|как\s+вариант|получается|"
    r"я\s+правильно\s+понимаю|для\s+корректной\s+интерпретации\s+необходимо|"
    r"в\s+чем\s+смысл|почему\s+я\s+спрашиваю)\b",
    re.IGNORECASE,
)
TECH_TERMS_RE = re.compile(
    r"\b(гидродинамик|дебит|скважин|трещин|мгрп|скин[-\s]?фактор|проницаем|"
    r"интерпретац|безразмерн|аппроксимац|калькулированн|эталонн|параметр|"
    r"вязкост|pvt|вкр|пласт|флюид|модель|генерализац)\b",
    re.IGNORECASE,
)
TECHNICAL_TASK_OBJECT_RE = re.compile(
    r"\b(устойчивост\w+\s+модел\w+|точност\w+|разв[её]ртк\w+|групп\w+\s+параметр\w+|"
    r"s,\s*n,\s*a,\s*l|s\s+n\s+a\s+l|кейс\w+|реализац\w+|расч[её]т\w+|"
    r"вычислительн\w+\s+эксперимент|вкр|раздел|следующ\w+\s+встреч\w+|четверг|7[:.]30|"
    r"промежуточн\w+\s+результат\w+|скин-?фактор|безразмер\w+|параметр\w+)\b",
    re.IGNORECASE,
)
TECHNICAL_WEAK_TASK_RE = re.compile(
    r"\b((?:для\s+)?того,?\s+чтобы\s+проверить|можно\s+будет|далеко\s+не\s+бесполезно|"
    r"(?:у\s+)?себя\s+поискать\s+кейс\w*|как\s+раз[-\s]?таки|что-нибудь|какие-то|что-то)\b",
    re.IGNORECASE,
)

PERSON_IMPERATIVE_NO_PUNCT_RE = re.compile(rf"^\s*{NAME}\s+" + IMPERATIVE_RE.pattern, re.IGNORECASE)

FUTURE_OBLIGATION_RE = re.compile(
    r"\b(нужно|надо|необходимо|требуется)\s+будет\s+\w+|"
    r"\bтребуется\s+(подготовить|проверить|добавить|разделить|описать|согласовать|сделать|"
    r"прописать|указать|направить|отправить|подтвердить|предоставить|завершить|обеспечить)\b",
    re.IGNORECASE,
)

PASSIVE_OBLIGATION_RE = re.compile(
    r"\b(должно|должен|должна|должны)\s+быть\s+(готов\w*|сделан\w*|подготовлен\w*|"
    r"завершен\w*|предоставлен\w*|отправлен\w*|выполнен\w*|согласован\w*)\b|"
    r"\bследует\s+(завершить|подготовить|проверить|отправить|согласовать|предоставить|оформить|обновить|сделать)\b|"
    r"\bнеобходимо\s+(предоставить|завершить|обеспечить)\b",
    re.IGNORECASE,
)

DEADLINE_NEAR_VERB_RE = re.compile(
    r"\b(к\s+(понедельнику|вторнику|среде|четвергу|пятнице|концу\s+(?:дня|недели|месяца)|релизу)|"
    r"до\s+(конца\s+(?:дня|недели|месяца)|пятницы|завтра|понедельника|вторника|среды|четверга))"
    r"\s+.{0,20}?\b(сдать|подготовить|отправить|завершить|закончить|доработать|прислать|"
    r"оформить|предоставить|сделать|согласовать|проверить)\b|"
    r"\b(сдать|подготовить|отправить|завершить|закончить|доработать|прислать|оформить|"
    r"предоставить|сделать|согласовать|проверить)\s+.{0,20}?"
    r"\b(к\s+(понедельнику|вторнику|среде|четвергу|пятнице)|до\s+(конца\s+(?:дня|недели|месяца)|пятницы|завтра))\b",
    re.IGNORECASE,
)


def _words_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or ""))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" .,:;")


def is_technical_text(text: str) -> bool:
    return len(TECH_TERMS_RE.findall(text or "")) >= 2


def is_technical_meeting(fragments: list[dict]) -> bool:
    joined = " ".join(fragment.get("text", "") for fragment in fragments[:120])
    return len(TECH_TERMS_RE.findall(joined)) >= 8


def is_real_task(text: str, technical_mode: bool | None = None) -> bool:
    clean = strip_speaker_prefix(text)
    lower = _normalize(clean)
    if not lower or _words_count(clean) < 3:
        return False
    if is_question(clean):
        return False
    has_person_imperative = bool(PERSON_IMPERATIVE_RE.search(text or "")) or bool(
        PERSON_IMPERATIVE_NO_PUNCT_RE.search(text or "")
    )
    has_action_item = (
        bool(ACTION_ITEM_RE.search(clean))
        or bool(FUTURE_OBLIGATION_RE.search(clean))
        or bool(PASSIVE_OBLIGATION_RE.search(clean))
        or bool(DEADLINE_NEAR_VERB_RE.search(clean))
    )
    has_org_action = bool(ORG_ACTION_RE.search(clean))
    has_action_pattern = any(pattern in lower for pattern in TASK_ACTION_PATTERNS)
    has_imperative = bool(IMPERATIVE_RE.search(clean))

    if NON_TASK_ANSWER_RE.search(clean) and not (has_imperative or has_action_item or has_action_pattern):
        return False
    if is_answer(clean) and not (has_imperative or has_action_item or has_action_pattern):
        return False
    if any(pattern in lower for pattern in BAD_TASK_PATTERNS):
        return False

    if TASK_STOP_RE.search(clean) and not (has_person_imperative or has_action_item or has_org_action):
        return False

    if has_person_imperative or has_action_item or has_org_action:
        return True

    technical = is_technical_text(clean) if technical_mode is None else technical_mode
    if technical:
        if TECHNICAL_WEAK_TASK_RE.search(clean):
            return False
        return (
            has_person_imperative
            or has_action_item
            or has_org_action
            or (has_action_pattern and bool(TECHNICAL_TASK_OBJECT_RE.search(clean)) and not lower.startswith(("можно ", "хотелось бы", "было бы")))
        )

    if has_imperative:
        return True
    if has_action_pattern:
        return True

    return bool(GENERAL_PM_TASK_RE.search(clean))


def is_task_fragment(text: str, technical_mode: bool | None = None) -> bool:
    return is_real_task(text, technical_mode=technical_mode)


def extract_tasks(fragments: list[dict], participants: list[str] | None = None) -> list[dict]:
    tasks = []
    seen = set()
    technical_mode = is_technical_meeting(fragments)
    for fragment in fragments:
        text = fragment.get("text", "")
        if not is_task_fragment(text, technical_mode=technical_mode):
            continue

        cls = classify_fragment(text, include_other=True)
        key = _normalize(text)
        if key in seen:
            continue
        seen.add(key)
        deadlines = find_deadlines(text)
        responsibles = find_responsibles(text, participants=participants, assume_task=True)
        tasks.append(
            {
                "text": text,
                "responsible": responsibles[0] if responsibles else None,
                "responsible_side": find_responsible_side(text),
                "deadline": deadlines[0] if deadlines else None,
                "deadline_normalized": None,
                "status": "new",
                "source_fragment": fragment.get("fragment_index") or fragment.get("block_index"),
                "start": fragment.get("start"),
                "end": fragment.get("end"),
                "matched_rules": cls.get("matched_rules", []),
                "confidence": cls.get("confidence", 0.0),
            }
        )
    return tasks


RUBERT_ANSWER_REMOVE_THRESHOLD = 0.75
RUBERT_OTHER_REMOVE_THRESHOLD = 0.80


def filter_tasks_by_classifier_confidence(tasks: list[dict]) -> list[dict]:
    kept = []
    for task in tasks:
        label = task.get("classifier_label")
        confidence = task.get("classifier_confidence")
        if label == "task" or confidence is None:
            kept.append(task)
            continue
        if label == "answer" and confidence >= RUBERT_ANSWER_REMOVE_THRESHOLD:
            continue
        if label == "other" and confidence >= RUBERT_OTHER_REMOVE_THRESHOLD:
            continue
        kept.append(task)
    return kept

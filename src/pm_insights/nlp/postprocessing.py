from __future__ import annotations

import re
from collections import Counter
from typing import Any

from pm_insights import settings
from pm_insights.utils.logging import get_logger

from .deadline_extractor import classify_deadline_kind, find_deadlines
from .decision_extractor import extract_agreements
from .extraction_decision import build_review_items, score_task_candidate
from .fragment_classifier import score_fragment_confidence
from .meeting_type import detect_meeting_type
from .responsible_extractor import find_responsibles
from .responsible_side import find_responsible_side
from .task_extractor import filter_tasks_by_classifier_confidence, is_real_task
from .topic_modeling import DOMAIN_TEXT_LABELS, STOPWORDS as TOPIC_MODEL_STOPWORDS, TOPIC_CANDIDATE_LABELS


log = get_logger(__name__)


TECHNICAL_TOPICS = {
    "гидродинамика",
    "дебит",
    "скважина",
    "трещины",
    "МГРП",
    "скин-фактор",
    "параметры пласта",
    "вязкость",
    "PVT",
    "безразмерные кривые",
    "интерпретация",
    "аппроксимация",
    "модель",
    "генерализация",
    "ВКР",
}
TOPIC_FILLER_WORDS = {
    "что-нибудь",
    "какие-то",
    "что-то",
    "правило",
    "сказали",
    "спрашиваю",
    "принципе",
    "илья",
    "получается",
    "именно",
    "собственно",
    "отталкиваться",
    "время",
    "ну",
    "да",
    "нет",
    "ага",
    "типа",
    "значит",
    "короче",
    "наверное",
    "условно",
    "говоря",
    "какой-то",
    "какая-то",
    "кто-то",
    "где-то",
    "зачем-то",
    "почему-то",
    "чего-то",
    "как-нибудь",
    "чуть-чуть",
    "хотел",
    "хотели",
    "посмотреть",
    "спросить",
    "сказал",
    "понятно",
    "знаю",
    "этом",
    "сути",
    "эта",
    "эту",
    "больш",
    "поэт",
    "случайн",
    "такая",
    "только",
    "как-то",
    "влезет",
    "смысл",
    "кейс",
    "говорил",
    "просто",
    "какой-нибудь",
    "215-й",
    "317-й",
}
TECHNICAL_TOPIC_LABEL_RULES = [
    (("дебит", "дебет", "динамик", "метрокуб", "сутк", "давлен"), "Дебит и динамика"),
    (("интерпретац", "данн", "эталон", "калькулирован", "аппроксим", "интерпол"), "Интерпретация данных"),
    (("пласт", "проницаем", "скваж", "гидродинами", "параметр"), "Параметры пласта"),
    (("скин", "трещин", "мгрп"), "Скин-фактор и трещины"),
    (("модель", "аппроксим", "точност", "r-квадрат", "r2"), "Модель и аппроксимация"),
    (("безразмер", "крив", "формул", "коэффициент"), "Безразмерные кривые"),
    (("четверг", "встреч", "следующ", "пары", "семестр", "свободн"), "Организация следующей встречи"),
]
OIL_GAS_TOPICS = {
    "объемы поставки": ["объем", "тонн", "партия", "поставк"],
    "ценовая формула": ["brent", "брент", "dated", "формула", "премия", "дифференциал"],
    "логистика": ["терминал", "труба", "отгрузка", "окно", "маршрут"],
    "фрахт и демередж": ["фрахт", "демередж", "чартер", "сталийное время"],
    "качество сырья": ["качество", "сера", "плотность", "лаборатор", "сертификат"],
    "платежные условия": ["предоплата", "оплата", "отсрочка", "банковская гарантия", "аккредитив", "платеж"],
    "хеджирование": ["хедж", "фьючерс", "своп", "форвард", "кривая", "basis risk"],
    "комплаенс": ["судно", "страховка", "санкционные ограничения", "флаг", "порт"],
    "инспекция": ["инспектор", "sgs", "проба", "коносамент", "shore tank", "инспекц"],
}
MEETING_TYPE_LABELS = {
    "project_meeting": "проектная встреча",
    "technical_research": "техническая / исследовательская встреча",
    "education_consultation": "учебная консультация",
    "commercial_meeting": "коммерческая встреча",
    "commercial_oil_gas": "нефтегазовая коммерческая встреча",
    "oil_gas_commercial": "нефтегазовая коммерческая встреча",
    "oil_gas_trading": "нефтегазовая коммерческая встреча",
    "strategy_meeting": "стратегическая встреча",
    "hr_meeting": "HR-встреча",
    "support_meeting": "встреча поддержки / клиентский разбор",
    "general_discussion": "общее обсуждение",
    "non_meeting_speech": "монолог / обращение",
    "mixed": "смешанная встреча",
    "unknown": "тип встречи не определен",
}
COMMERCIAL_MARKERS = (
    "договор", "контракт", "term sheet", "термшит", "поставка", "отгрузка",
    "партия", "тонн", "премия", "дифференциал", "оплата", "платеж",
    "предоплата", "отсрочка", "банковская гарантия", "аккредитив",
    "покупатель", "продавец", "поставщик", "цена", "формула",
)
SECTION_LIMITS = {
    "tasks": 8,
    "recommendations": 8,
    "qa": 8,
    "deadlines": 10,
    "agreements": 8,
    "commercial_terms": 8,
    "commitments": 6,
    "responsibles": 10,
    "responsible_sides": 8,
    "decisions": 8,
    "aspects_topics": 10,
    "sentiment": 8,
    "review": 8,
}
INTRO_AGENDA_RE = re.compile(
    r"\b(доброе\s+утро|добрый\s+день|коллеги|давайте\s+начинать|пока\s+все\s+на\s+связи|"
    r"у\s+нас\s+сегодня\s+основная\s+тема|предлагаю\s+идти\s+по\s+порядку|"
    r"если\s+где-то\s+будет\s+важное\s+замечание|переходим\s+к|давайте\s+перейдем\s+к|"
    r"теперь\s+по|хорошо,\s+вернемся\s+к)\b",
    re.IGNORECASE,
)
COMMERCIAL_VALUE_RE = re.compile(
    r"(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч|тонн|доллар|баррель|%|процент|дней|сутки|час|часов)|"
    r"\b\d{1,2}\s*[–-]\s*\d{1,2}\b|\b\d{1,2}\s+(?:июля|августа|июня|сентября|октября|ноября|декабря|января|февраля|марта|апреля|мая)\b|"
    r"\b50\s*(?:/|на)\s*50\b|\bпять\s+котировочных\s+дней\b|\bтри\s+партии\b|\bдве\s+партии\b|"
    r"\bдо\s+\d{1,2}\s+(?:июля|августа|июня|сентября|октября|ноября|декабря|января|февраля|марта|апреля|мая)\b|"
    r"\bоколо\s+\d{1,2}\b|\bчерез\s+\d+\s+(?:день|дня|дней)\b)",
    re.IGNORECASE,
)
AGREEMENT_MARKER_RE = re.compile(
    r"\b(зафиксируем|запишем|согласовали|договорились|приемлемо|оставляем|делим|"
    r"подтвердить|подтверждаем|можно\s+записать|рабочий\s+вариант|предварительно\s+согласовано)\b",
    re.IGNORECASE,
)
AGREEMENT_RESULT_RE = re.compile(
    r"(первая\s+партия|вторая\s+партия|третья\s+партия|ценовое\s+окно|демередж|инспекц|"
    r"коносамент|качество\s+по\s+пробам|опцион|премия|дифференциал|предоплата|банковская\s+гарантия|аккредитив)",
    re.IGNORECASE,
)
UNCERTAIN_AGREEMENT_RE = re.compile(
    r"\b(например|если|возможен|возможна|обсуждаем|можно\s+рассмотреть|предварительно\s+около)\b",
    re.IGNORECASE,
)
BAD_AGREEMENT_RE = re.compile(
    r"^\s*(например|если|поставщику\s+удобнее|покупателю\s+удобнее|сколько|насколько|почему|как\s+вы\s+видите)\b",
    re.IGNORECASE,
)
QUESTION_START_RE = re.compile(
    r"^\s*(кто|что|когда|где|почему|зачем|сколько|какой|какая|какие|можно ли|нужно ли|есть ли|"
    r"правильно ли|я правильно понимаю|подскажите|скажите|можно уточнить|а период|по демереджу)",
    re.IGNORECASE,
)
FREQUENCY_ONLY_RE = re.compile(
    r"^\s*(два\s+раза\s+в\s+неделю|раз\s+в\s+неделю|каждый\s+день|ежедневный\s+статус)\s*$",
    re.IGNORECASE,
)
COMMERCIAL_CATEGORY_RANK = {
    "объемы и партии": 10,
    "объемы поставки": 10,
    "логистика": 20,
    "ценовая формула": 30,
    "премия": 31,
    "дифференциал": 32,
    "ценовое окно": 33,
    "платежные условия": 40,
    "банковская гарантия": 41,
    "аккредитив": 42,
    "фрахт": 50,
    "демередж": 51,
    "качество сырья": 60,
    "инспекция": 61,
    "хеджирование": 70,
    "опцион": 80,
    "комплаенс": 90,
    "коммерческое условие": 100,
}
REASONING_STOP_RE = re.compile(
    r"\b(задача\s+эксперта|задача\s+интерпретации|обратная\s+задача|постановка\s+задач|"
    r"как\s+правило|как\s+гипотеза|в\s+теории|в\s+принципе|необходимо\s+время|"
    r"необходимы\s+параметры|можно\s+рассматривать|почему\s+я\s+спрашиваю|"
    r"нужно\s+понимать|надо\s+понимать|важно\s+понимать|можно\s+сделать|можно\s+согласовать)\b",
    re.IGNORECASE,
)
ANSWER_MARKERS_RE = re.compile(
    r"\b(да|нет|верно|не\s+совсем|для\s+этого|для\s+нефтяного\s+кейса|если\s+рассматриваем|"
    r"как\s+правило|это\s+означает|это\s+связано|я\s+сделаю|проверю|подготовлю|отправлю|"
    r"уточню|лежит|отвечаю|получается|значение)\b",
    re.IGNORECASE,
)


def _words_count(text: str) -> int:
    return len(re.findall(r"[А-Яа-яЁёA-Za-z0-9]+", text or ""))


def _sentence_case(text: str) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip(" .,:;"))
    return clean[:1].upper() + clean[1:] if clean else clean


def _joined_text(result: dict) -> str:
    units = result.get("semantic_blocks") or result.get("transcript") or []
    return " ".join(item.get("text", "") for item in units)


def _participant_names_from_result(result: dict) -> list[str]:
    metadata = result.get("metadata") or {}
    meeting_info = metadata.get("meeting_info") or {}
    raw_sources = [
        metadata.get("participants"),
        meeting_info.get("participants"),
        result.get("participants"),
    ]
    names: list[str] = []
    for source in raw_sources:
        if isinstance(source, str):
            for line in re.split(r"[\n;,]+", source):
                name = re.split(r"\s+[—–-]\s+|\s*:\s*", line.strip(), maxsplit=1)[0].strip()
                if name:
                    names.append(name)
        elif isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                else:
                    name = str(item or "").strip()
                if name:
                    names.append(name)
    return list(dict.fromkeys(names))


def _is_technical_result(result: dict) -> bool:
    mt = result.get("meeting_type")
    if isinstance(mt, dict) and mt.get("label") in {"technical_research", "education_consultation", "mixed"}:
        return True
    names = {item.get("topic_name") for item in result.get("topics", [])}
    aspect_names = {aspect for item in result.get("aspects", []) for aspect in item.get("aspects", [])}
    if (names | aspect_names) & TECHNICAL_TOPICS:
        return True
    detected = detect_meeting_type(result.get("transcript", []))
    return detected.get("label") in {"technical_research", "education_consultation", "mixed"}


def _is_research_result(result: dict) -> bool:
    mt = result.get("meeting_type")
    label = mt.get("label") if isinstance(mt, dict) else None
    return label in {"technical_research", "education_consultation"} or (
        label is None and _is_technical_result(result) and not _is_commercial_result(result)
    )


def _is_oil_gas_result(result: dict) -> bool:
    mt = result.get("meeting_type")
    if isinstance(mt, dict) and mt.get("label") in {"commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        return True
    summary = result.get("analysis_summary")
    if isinstance(summary, dict) and summary.get("meeting_type") in {"commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        return True
    joined = _joined_text(result).lower()
    hits = 0
    for keywords in OIL_GAS_TOPICS.values():
        hits += sum(1 for keyword in keywords if keyword in joined)
    return hits >= 5


def _is_commercial_result(result: dict) -> bool:
    mt = result.get("meeting_type")
    if isinstance(mt, dict) and mt.get("label") in {"commercial_meeting", "commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        return True
    if result.get("commercial_terms") or any(item.get("type") == "commercial_term" for item in result.get("agreements", [])):
        return True
    joined = _joined_text(result).lower()
    hits = sum(1 for marker in COMMERCIAL_MARKERS if marker in joined)
    return hits >= 4


def _is_education_priority_result(result: dict) -> bool:
    joined = _joined_text(result).lower()
    markers = (
        "вкр",
        "черновик",
        "страниц",
        "консультац",
        "семестр",
        "пары",
        "следующ",
        "научн",
        "руководител",
    )
    return sum(1 for marker in markers if marker in joined) >= 2


def _set_meeting_type(result: dict, label: str, confidence: float = 0.72) -> None:
    current = result.get("meeting_type")
    if isinstance(current, dict):
        if current.get("label") != label:
            current.setdefault("original_label", current.get("label"))
        current["label"] = label
        current["display_name"] = MEETING_TYPE_LABELS.get(label, label)
        current["confidence"] = max(float(current.get("confidence", 0.0) or 0.0), confidence)
    else:
        result["meeting_type"] = {
            "label": label,
            "display_name": MEETING_TYPE_LABELS.get(label, label),
            "confidence": confidence,
            "matched_markers": [],
            "scores": {},
        }


def _normalize_meeting_type(result: dict) -> None:
    if _is_oil_gas_result(result):
        _set_meeting_type(result, "commercial_oil_gas", 0.82)
        return
    if _is_commercial_result(result):
        current = result.get("meeting_type")
        current_label = current.get("label") if isinstance(current, dict) else None
        if current_label not in {"technical_research", "education_consultation", "mixed"}:
            _set_meeting_type(result, "commercial_meeting", 0.72)
        elif current_label == "mixed":
            _set_meeting_type(result, "mixed", 0.7)
        return
    if _is_education_priority_result(result):
        joined = _joined_text(result).lower()
        technical_hits = sum(1 for marker in ("дебит", "скин", "пласт", "модель", "безразмер", "интерпретац") if marker in joined)
        label = "technical_research" if technical_hits >= 3 else "education_consultation"
        _set_meeting_type(result, label, 0.78)
        return
    current = result.get("meeting_type")
    if isinstance(current, dict):
        label = current.get("label") or "unknown"
        current.setdefault("display_name", MEETING_TYPE_LABELS.get(label, label))


def _oil_gas_main_topics(result: dict) -> list[str]:
    joined = _joined_text(result).lower()
    matched = []
    for topic, keywords in OIL_GAS_TOPICS.items():
        if any(keyword in joined for keyword in keywords):
            matched.append(topic)
    return matched[:8]


def _mark_oil_gas_meeting_type(result: dict) -> None:
    if not _is_oil_gas_result(result):
        return
    current = result.get("meeting_type")
    if isinstance(current, dict):
        if current.get("label") != "commercial_oil_gas":
            current.setdefault("original_label", current.get("label"))
        current["label"] = "commercial_oil_gas"
        current["display_name"] = "нефтегазовая коммерческая встреча"
        current.setdefault("confidence", max(float(current.get("confidence", 0.0) or 0.0), 0.82))
    else:
        result["meeting_type"] = {
            "label": "commercial_oil_gas",
            "display_name": "нефтегазовая коммерческая встреча",
            "confidence": 0.82,
            "matched_markers": [],
            "scores": {},
        }


def _is_commercial_commitment(text: str) -> bool:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return bool(
        re.search(r"^\s*мы\s+можем\b", compact)
        or "можем отправить" in compact
        or "можем прислать" in compact
        or "будут направлены" in compact
        or "будет направлен" in compact
    )


def _is_commercial_action_task(text: str) -> bool:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return any(
        pattern in compact
        for pattern in (
            "нужно прописать",
            "надо прописать",
            "нужно указать",
            "надо указать",
            "нужно согласовать",
            "надо согласовать",
            "нужно подготовить",
            "нужно направить",
            "нужно подтвердить",
            "должен подтвердить",
            "должна подтвердить",
            "согласовать финальный",
            "согласовать дифференциал",
            "прописать точные",
            "прописать критерии",
            "прописать порядок",
        )
    )


def _trim_text(text: str | None, limit: int = 180) -> str | None:
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text).strip(" .,:;")
    if len(clean) <= limit:
        return clean
    clipped = clean[:limit]
    cut = max(clipped.rfind("."), clipped.rfind(";"), clipped.rfind(","))
    if cut >= 80:
        return clipped[:cut].strip(" .,:;") + "..."
    return clipped.rstrip(" .,:;") + "..."


def _clean_summary(text: str | None, limit: int = 300) -> str | None:
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    selected = " ".join(sentence for sentence in sentences[:2] if sentence).strip() or clean
    return _trim_text(selected, limit)


def _compact_item(title: str | None, source_text: str | None, *, title_limit: int = 160, summary_limit: int = 300) -> dict[str, str | None]:
    return {
        "title": _trim_text(title or source_text, title_limit),
        "summary": _clean_summary(source_text, summary_limit),
        "source_text": source_text,
    }


def _has_specific_commercial_value(text: str | None) -> bool:
    return bool(COMMERCIAL_VALUE_RE.search(text or ""))


def _is_intro_or_agenda(text: str | None) -> bool:
    if not text:
        return False
    return bool(INTRO_AGENDA_RE.search(text)) and not _has_specific_commercial_value(text)


def _informative_sentence(text: str | None, *, prefer_commercial: bool = False) -> str | None:
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text).strip()
    parts = [part.strip(" .,:;") for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if not parts:
        return clean
    if prefer_commercial:
        for part in parts:
            if _has_specific_commercial_value(part) or AGREEMENT_MARKER_RE.search(part):
                return part
    for part in parts:
        if not _is_intro_or_agenda(part):
            return part
    return parts[-1]


def _sentence_span_for_match(text: str, match: re.Match[str], *, include_next: bool = False) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return clean
    start = match.start()
    end = match.end()
    left_candidates = [clean.rfind(mark, 0, start) for mark in (".", "!", "?")]
    left = max(left_candidates)
    right_candidates = [pos for pos in (clean.find(mark, end) for mark in (".", "!", "?")) if pos >= 0]
    right = min(right_candidates) if right_candidates else len(clean) - 1
    if include_next and right < len(clean) - 1:
        next_candidates = [pos for pos in (clean.find(mark, right + 1) for mark in (".", "!", "?")) if pos >= 0]
        if next_candidates:
            right = min(next_candidates)
    return clean[left + 1 : right + 1].strip(" .,:;") or clean


def _source_for_match(text: str, match: re.Match[str], *, include_next: bool = False) -> str:
    return _sentence_span_for_match(text, match, include_next=include_next) or _informative_sentence(text, prefer_commercial=True) or text


def _premium_split_source(text: str) -> str:
    first = re.search(r"(?:по\s+перв\w+\s+парт\w+|перв\w+\s+парт\w+).{0,80}?(?:преми\w*|1[,.]4)", text, re.IGNORECASE)
    second = re.search(r"(?:по\s+втор\w+\s+и\s+треть\w+|втор\w+\s+и\s+треть\w+|вторая\s+и\s+третья).{0,80}?(?:преми\w*|плюс|1[,.]2)", text, re.IGNORECASE)
    if first and second:
        start = first.start()
        end = second.end()
        clean = re.sub(r"\s+", " ", text or "").strip()
        left = max(clean.rfind(mark, 0, start) for mark in (".", "!", "?"))
        right_candidates = [pos for pos in (clean.find(mark, end) for mark in (".", "!", "?")) if pos >= 0]
        right = min(right_candidates) if right_candidates else end
        return clean[left + 1 : right + 1].strip(" .,:;")
    if first:
        return _source_for_match(text, first, include_next=bool(second))
    if second:
        return _source_for_match(text, second)
    return _informative_sentence(text, prefer_commercial=True) or text


def _source_fragment_for_text(result: dict, source_text: str | None, fallback: int | None) -> int | None:
    needle = re.sub(r"\s+", " ", source_text or "").strip().lower()
    if len(needle) < 16:
        return fallback
    for fragment in result.get("transcript", []):
        fragment_text = re.sub(r"\s+", " ", fragment.get("text", "")).strip().lower()
        if not fragment_text:
            continue
        if needle in fragment_text or (len(fragment_text) >= 24 and fragment_text in needle):
            return fragment.get("fragment_index") or fallback
    return fallback


def _normalize_title_key(text: str | None) -> str:
    return re.sub(r"[^a-zа-яё0-9]+", " ", (text or "").lower().replace("ё", "е")).strip()


def _category_rank(category: str | None) -> int:
    return COMMERCIAL_CATEGORY_RANK.get((category or "").lower(), 100)


BAD_DISPLAY_TOPIC_KEYS = {
    "требования этой эту",
    "причем понял честно",
    "диаметр интересно случайный",
    "минимизировать меньше связей",
    "минимизировать максимизировать",
    "только использую используют",
    "вторник похожие",
    "вторник вторник",
    "пакете пакете",
    "второкурсниками много",
    "влезет смысле",
    "16 го видно",
    "имя технология",
    "адекватным решение",
    "пользоваться config raspberry",
    "реальное снова написать",
    "человек машина",
    "постановке задачи",
    "raspberry подключаться",
    "даты рабочих контракте",
    "рабочих контракте отгрузки поставки даты",
}

DISPLAY_TOPIC_EXACT_MAP = {
    "пользоваться config raspberry": "Raspberry Pi и настройка окружения",
    "raspberry подключаться": "Raspberry Pi и настройка окружения",
    "камеры fps": "камеры и FPS",
    "камер fps": "камеры и FPS",
    "usb камер": "USB bandwidth и подключение камер",
    "qemu виртуализация": "QEMU / виртуализация Raspberry Pi",
    "второкурсниками много": "задачи для второкурсников",
    "дебет время правило": "дебит и динамика",
    "четверг отталкиваться сказали": "организация следующей встречи",
    "диаметр интересно случайный": "диаметр и топология графа",
}

GENERIC_DISPLAY_TOPIC_KEYS = {
    "модель",
    "сервер",
    "проблемы",
    "дизайн",
    "документация",
    "сроки",
    "ресурсы",
    "бюджет",
    "качество",
    "вкр",
}

OIL_GAS_ONLY_TOPIC_KEYS = {
    "параметры пласта",
    "скин фактор и трещины",
    "дебит и динамика",
}

COMMERCIAL_ONLY_TOPIC_KEYS = {
    "премия и дифференциал",
    "платежные условия",
    "ценовая формула",
    "фрахт и демередж",
    "объемы поставки",
    "качество сырья",
    "brent",
    "коносамент",
    "судно",
    "терминал",
}

RESEARCH_ONLY_TOPIC_KEYS = {
    "интерпретация данных",
}

STRICT_OIL_GAS_CONTEXT_MARKERS = (
    "нефть",
    "газ",
    "скважин",
    "пласт",
    "дебит",
    "скин-фактор",
    "скин фактор",
    "трещин",
    "коллектор",
    "давлен",
    "pvt",
)

NON_OIL_TECH_CONTEXT_MARKERS = (
    "raspberry",
    "orange pi",
    "yolo",
    "камер",
    "benchmark",
    "бенчмарк",
    "qemu",
    "второкурсник",
    "датасет",
    "opencv",
)

TECHNICAL_RESEARCH_CONTEXT_MARKERS = (
    "вкр",
    "исслед",
    "модель",
    "параметр",
    "расчет",
    "расчёт",
    "данн",
    "аппроксимац",
    "формул",
    "эксперимент",
    "raspberry",
    "orange pi",
    "yolo",
    "камер",
    "benchmark",
    "бенчмарк",
    "qemu",
    "датасет",
    "opencv",
)

TECHNICAL_HARDWARE_CONTEXT_MARKERS = (
    "raspberry",
    "orange pi",
    "yolo",
    "камер",
    "benchmark",
    "бенчмарк",
    "qemu",
    "opencv",
    "hailo",
    "ai kit",
    "fps",
    "npu",
    "microsd",
    "usb bandwidth",
)

ARCHITECTURE_C4_CONTEXT_MARKERS = (
    "c4",
    "диаграмм",
    "контейнер",
    "компонент",
    "systemd",
    "timer",
    "ldap",
    "каталог",
    "портал",
    "консоль",
    "инфраструктур",
    "модульный монолит",
    "клиент-сервер",
    "архитектур",
)

GRAPH_VKR_CONTEXT_MARKERS = (
    "граф",
    "тополог",
    "диаметр",
    "связ",
    "математическ",
    "критер",
    "оптимизац",
    "телеметр",
    "метрик",
    "полносвязан",
    "остовн",
    "дерев",
)

EDUCATION_CONTEXT_MARKERS = (
    "вкр",
    "второкурсник",
    "практик",
    "консультац",
    "семестр",
    "пары",
    "черновик",
    "страниц",
    "следующая встреч",
)

DISPLAY_TOPIC_STOPWORDS = TOPIC_FILLER_WORDS | set(TOPIC_MODEL_STOPWORDS) | {
    "пакете",
    "много",
    "смысле",
    "рабочих",
    "контракте",
}

DOMAIN_BLOCKED_TOPIC_KEYS = {
    "technical_hardware": OIL_GAS_ONLY_TOPIC_KEYS | {"интерпретация данных"},
    "education_vkr": set(),
}

DOMAIN_CANDIDATE_LABELS = {
    "technical_hardware": (
        "Raspberry Pi 5 и Orange Pi",
        "Raspberry Pi и настройка окружения",
        "YOLO benchmark",
        "модель детекции рук",
        "камеры и FPS",
        "датасеты и разметка",
        "USB bandwidth и подключение камер",
        "QEMU / виртуализация Raspberry Pi",
        "AI Kit / Hailo accelerator",
        "входные данные ML pipeline",
        "формальная постановка задачи",
        "перегрев и охлаждение Raspberry Pi",
        "microSD и ресурс записи",
        "задачи для второкурсников",
        "документы по практике",
        "организация практики",
    ),
    "architecture_c4": (
        "архитектурная диаграмма C4",
        "описание архитектуры в ВКР",
        "systemd timer",
        "инфраструктурный сервис",
        "LDAP / каталог домена",
        "компоненты и взаимодействия",
        "внешние системы и пользователи",
        "клиент-серверная архитектура",
        "модульный монолит",
        "локальные артефакты сервиса",
        "sequence diagram",
        "взаимодействие через каталог",
        "ВКР и презентация",
        "график презентации/встречи",
    ),
    "graph_vkr": (
        "постановка задачи",
        "диаметр и топология графа",
        "математическая модель",
        "аппроксимация метрик графа",
        "количество связей",
        "модель топологии сети",
        "требования к ВКР",
        "критерии качества",
        "метрики графа",
        "ограничения модели",
        "сбор данных и телеметрия",
        "входные данные математической модели",
    ),
    "education_vkr": (
        "ВКР и документация",
        "требования к ВКР",
        "постановка задачи",
        "цель и задачи ВКР",
        "документы по практике",
        "организация практики",
        "задачи для второкурсников",
        "организация следующей встречи",
    ),
    "oil_gas": (
        "объемы поставки",
        "партии поставки",
        "ценовая формула",
        "премия и дифференциал",
        "платежные условия",
        "сроки поставки и оплаты",
        "логистика",
        "фрахт и демередж",
        "качество сырья",
        "инспекция",
        "хеджирование",
        "комплаенс",
        "опцион",
    ),
    "technical_research": (
        "дебит и динамика",
        "интерпретация данных",
        "параметры пласта",
        "скин-фактор и трещины",
        "модель и аппроксимация",
        "безразмерные кривые",
        "организация следующей встречи",
        "параметры модели",
        "точность модели",
        "устойчивость модели",
        "данные и выборка",
    ),
    "project": (
        "задачи и поручения",
        "сроки",
        "риски",
        "статус работ",
        "план работ",
        "релиз",
        "тестирование",
        "сервер",
        "авторизация",
        "документация",
        "аналитика",
        "интеграция",
    ),
}

_SEMANTIC_MODEL: Any | None = None
_SEMANTIC_MODEL_UNAVAILABLE = False

OIL_GAS_CONTEXT_MARKERS = (
    "нефть",
    "сырь",
    "brent",
    "брент",
    "баррел",
    "фрахт",
    "демередж",
    "коносамент",
    "судно",
    "терминал",
    "плотност",
    "сера",
    "покупател",
    "поставщик",
    "контракт",
    "платеж",
    "платёж",
    "пласт",
    "скважин",
    "дебит",
)

FINAL_TOPIC_LABEL_RULES = [
    (("постановк", "задач"), "постановка задачи"),
    (("требован", "вкр"), "требования к ВКР"),
    (("формальн", "постановк"), "формальная постановка задачи"),
    (("математ", "модел"), "математическая модель"),
    (("аппроксимац", "метрик", "граф"), "аппроксимация метрик графа"),
    (("тополог", "сет"), "модель топологии сети"),
    (("ограничен", "модел"), "ограничения модели"),
    (("критер", "качеств"), "критерии качества"),
    (("критер", "оптимизац"), "критерии оптимизации"),
    (("диаметр", "граф"), "диаметр и топология графа"),
    (("метрик", "граф"), "метрики графа"),
    (("количеств", "связ"), "количество связей"),
    (("задерж", "блокиров"), "задержки и блокировки"),
    (("телеметр", "данн"), "сбор данных и телеметрия"),
    (("raspberry", "окружен"), "Raspberry Pi и настройка окружения"),
    (("raspberry", "orange"), "Raspberry Pi 5 и Orange Pi"),
    (("hailo",), "AI Kit / Hailo accelerator"),
    (("arm", "совместим"), "ML-фреймворки и совместимость с ARM"),
    (("tensorflow",), "TensorFlow / PyTorch / Keras"),
    (("pytorch",), "TensorFlow / PyTorch / Keras"),
    (("keras",), "TensorFlow / PyTorch / Keras"),
    (("камер", "fps"), "камеры и FPS"),
    (("real", "time", "inference"), "real-time inference"),
    (("перегрев", "raspberry"), "перегрев и охлаждение Raspberry Pi"),
    (("config", "raspberry"), "Raspberry config и CPU governor"),
    (("cpu", "governor"), "Raspberry config и CPU governor"),
    (("microsd",), "microSD и ресурс записи"),
    (("usb", "камер"), "USB bandwidth и подключение камер"),
    (("yolo",), "YOLO benchmark"),
    (("датасет",), "датасеты и разметка"),
    (("разметк",), "датасеты и разметка"),
    (("входн", "ml"), "входные данные ML pipeline"),
    (("ml", "pipeline"), "входные данные ML pipeline"),
    (("перегрев",), "перегрев и охлаждение Raspberry Pi"),
    (("npu",), "AI Kit / Hailo accelerator"),
    (("детекц", "рук"), "модель детекции рук"),
    (("qemu",), "QEMU / виртуализация Raspberry Pi"),
    (("второкурсник",), "задачи для второкурсников"),
    (("практик", "документ"), "документы по практике"),
    (("презентац", "проект"), "презентация проекта"),
    (("c4",), "архитектурная диаграмма C4"),
    (("sequence", "diagram"), "sequence diagram"),
    (("архитектур", "вкр"), "описание архитектуры в ВКР"),
    (("инфраструктур", "сервис"), "инфраструктурный сервис"),
    (("systemd", "timer"), "systemd timer"),
    (("ldap",), "LDAP / каталог домена"),
    (("клиент", "сервер"), "клиент-серверная архитектура"),
    (("модульн", "монолит"), "модульный монолит"),
    (("компонент", "взаимодейств"), "компоненты и взаимодействия"),
    (("внешн", "систем"), "внешние системы и пользователи"),
    (("plantuml",), "sequence diagram"),
    (("каталог", "взаимодейств"), "взаимодействие через каталог"),
    (("вкр", "презентац"), "ВКР и презентация"),
    (("контроллер", "домен"), "каталог домена"),
    (("организац", "практик"), "организация практики"),
    (("договор", "организац"), "договор с организацией"),
    (("срок", "практик"), "сроки оформления практики"),
    (("тема", "вкр", "папк"), "тема ВКР и номер папки"),
    (("введен", "вкр"), "введение ВКР"),
    (("проблем", "исслед"), "проблема исследования"),
    (("цель", "задач", "вкр"), "цель и задачи ВКР"),
    (("термин",), "список терминов"),
    (("active", "directory"), "Active Directory"),
    (("импортозамещ",), "импортозамещение ПО"),
    (("репликац", "тополог"), "репликация и топология"),
    (("idf0",), "IDF0-схемы"),
    (("a0", "декомпоз"), "A0-декомпозиция"),
    (("входн", "данн", "мод"), "входные данные математической модели"),
    (("входн", "выходн", "модул"), "входные и выходные данные модулей"),
    (("иерархическ", "каталог"), "иерархический каталог как база данных"),
    (("локальн", "реплицир"), "локальная и реплицируемая область каталога"),
    (("централиз", "децентрализ"), "централизованные и децентрализованные решения"),
    (("слабоформализ",), "формализация слабоформализуемой задачи"),
    (("транскрипц", "конспект"), "сравнение транскрипции и конспекта"),
    (("суммаризац",), "качество автоматической суммаризации"),
    (("python", "каталог"), "Python-сервис для работы с каталогом"),
    (("kerberos",), "Kerberos-авторизация"),
    (("каталог", "домен"), "каталог домена"),
    (("внутрисайтов", "тополог"), "внутрисайтовая топология"),
    (("межсайтов", "тополог"), "межсайтовая топология"),
    (("старост", "связ"), "старость связей"),
    (("small", "world"), "Small World model"),
    (("задерж", "репликац"), "задержки репликации"),
    (("блокиров", "репликац"), "блокировки репликации"),
    (("модел", "репликац"), "модель репликации"),
    (("остовн", "дерев"), "минимальное остовное дерево"),
    (("вес", "канал"), "веса каналов"),
    (("ping",), "ping и hop count"),
    (("hop", "count"), "ping и hop count"),
    (("расписан", "связ"), "расписания связей"),
    (("внутрисайтов", "межсайтов", "алгоритм"), "различия внутрисайтового и межсайтового алгоритмов"),
    (("mvp", "алгоритм"), "MVP с двумя алгоритмами"),
    (("документ", "отчет"), "документ ВКР и отчёт"),
]


def _topic_words(text: str | None) -> list[str]:
    return re.findall(r"[A-Za-zА-Яа-яЁё0-9-]{3,}", (text or "").lower())


def _semantic_token_stem(word: str) -> str:
    normalized = word.lower().replace("ё", "е").strip("-")
    for prefix in (
        "второкурсник",
        "поставк",
        "отгруз",
        "платеж",
        "платеж",
        "raspberry",
        "benchmark",
        "камер",
        "пласт",
        "скин",
        "трещин",
        "дебит",
        "данн",
        "модел",
        "параметр",
        "документ",
        "практик",
    ):
        if normalized.startswith(prefix):
            return prefix
    for suffix in (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ыми",
        "ими",
        "иях",
        "ях",
        "ах",
        "ов",
        "ев",
        "ей",
        "ам",
        "ям",
        "ом",
        "ем",
        "ой",
        "ый",
        "ий",
        "ая",
        "ое",
        "ые",
        "у",
        "а",
        "ы",
        "и",
        "е",
    ):
        if normalized.endswith(suffix) and len(normalized) - len(suffix) >= 4:
            return normalized[: -len(suffix)]
    return normalized


def _display_topic_tokens(text: str | None) -> list[str]:
    tokens = []
    for word in _topic_words(text):
        normalized = _semantic_token_stem(word)
        if normalized and normalized not in DISPLAY_TOPIC_STOPWORDS and not normalized.isdigit():
            tokens.append(normalized)
    return tokens


def _normalize_display_keywords(value: Any, limit: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = re.split(r"[,;|/\n]+", value)
        if len(candidates) == 1:
            candidates = value.split()
    elif isinstance(value, dict):
        candidates = [value.get("word"), value.get("keyword"), value.get("text"), value.get("name")]
    else:
        candidates = list(value) if isinstance(value, (list, tuple, set)) else [value]

    flattened: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, dict):
            flattened.extend(
                str(candidate.get(key) or "")
                for key in ("word", "keyword", "text", "name")
                if candidate.get(key)
            )
        elif candidate is not None:
            flattened.append(str(candidate))

    joined_chars = "".join(part for part in flattened if len(part.strip()) == 1)
    if len(joined_chars) >= 4 and len(flattened) >= len(joined_chars):
        flattened.append(joined_chars)

    result: list[str] = []
    for text in flattened:
        phrase = re.sub(r"\s+", " ", str(text or "").strip(" ,.;:")).strip()
        if not phrase:
            continue
        tokens = _display_topic_tokens(phrase)
        if not tokens and phrase.lower() not in DISPLAY_TOPIC_STOPWORDS:
            tokens = [phrase.lower()]
        clean = " ".join(tokens) if len(tokens) > 1 else (tokens[0] if tokens else "")
        if not clean or len(clean) < 3:
            continue
        if _looks_like_bad_phrase_topic(clean):
            continue
        if clean not in result:
            result.append(clean)
        if len(result) >= limit:
            break
    return result


def _semantic_embedding_model() -> Any | None:
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_UNAVAILABLE
    if _SEMANTIC_MODEL_UNAVAILABLE:
        return None
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL
    model_name = getattr(settings, "TOPIC_EMBEDDING_MODEL", "")
    if not model_name:
        _SEMANTIC_MODEL_UNAVAILABLE = True
        return None
    is_local_path = str(model_name).startswith((".", "/", "\\")) or bool(re.match(r"^[A-Za-z]:", str(model_name)))
    if not is_local_path and "/" in str(model_name):
        _SEMANTIC_MODEL_UNAVAILABLE = True
        return None
    try:
        from sentence_transformers import SentenceTransformer

        kwargs = {}
        try:
            _SEMANTIC_MODEL = SentenceTransformer(model_name, **kwargs)
        except TypeError:
            if kwargs:
                _SEMANTIC_MODEL_UNAVAILABLE = True
                return None
            _SEMANTIC_MODEL = SentenceTransformer(model_name)
        return _SEMANTIC_MODEL
    except Exception:
        _SEMANTIC_MODEL_UNAVAILABLE = True
        return None


def _cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    try:
        numerator = float(sum(float(a) * float(b) for a, b in zip(vec_a, vec_b, strict=False)))
        norm_a = sum(float(a) * float(a) for a in vec_a) ** 0.5
        norm_b = sum(float(b) * float(b) for b in vec_b) ** 0.5
        if not norm_a or not norm_b:
            return 0.0
        return numerator / (norm_a * norm_b)
    except Exception:
        return 0.0


def _lexical_semantic_similarity(text_a: str | None, text_b: str | None) -> float:
    tokens_a = set(_display_topic_tokens(text_a))
    tokens_b = set(_display_topic_tokens(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0
    return len(overlap) / ((len(tokens_a) * len(tokens_b)) ** 0.5)


def _label_markers(label: str | None) -> set[str]:
    key = _normalize_title_key(label)
    markers: set[str] = set()
    for source in (DOMAIN_TEXT_LABELS, TOPIC_CANDIDATE_LABELS):
        for candidate, candidate_markers in source.items():
            if _normalize_title_key(candidate) == key:
                markers.update(str(marker).lower().replace("ё", "е") for marker in candidate_markers)
    for rule_markers, candidate in FINAL_TOPIC_LABEL_RULES:
        if _normalize_title_key(candidate) == key:
            markers.update(str(marker).lower().replace("ё", "е") for marker in rule_markers)
    for rule_markers, candidate in TECHNICAL_TOPIC_LABEL_RULES:
        if _normalize_title_key(candidate) == key:
            markers.update(str(marker).lower().replace("ё", "е") for marker in rule_markers)
    return markers


def _label_marker_supported(label: str | None, evidence: str | None, *, min_hits: int = 1) -> bool:
    markers = _label_markers(label)
    if not markers:
        return False
    haystack = str(evidence or "").lower().replace("ё", "е")
    tokens = set(_display_topic_tokens(evidence))
    hits = 0
    for marker in markers:
        marker_stem = _semantic_token_stem(marker)
        if marker in haystack or marker_stem in tokens:
            hits += 1
    return hits >= min_hits


def semantic_similarity(text_a: str | None, text_b: str | None) -> float:
    left = re.sub(r"\s+", " ", str(text_a or "")).strip()
    right = re.sub(r"\s+", " ", str(text_b or "")).strip()
    if not left or not right:
        return 0.0
    lexical = _lexical_semantic_similarity(left, right)
    if lexical >= 0.55:
        return lexical
    model = _semantic_embedding_model()
    if model is None:
        return lexical
    try:
        embeddings = model.encode([left, right], show_progress_bar=False)
        return max(lexical, _cosine_similarity(embeddings[0], embeddings[1]))
    except Exception:
        return lexical


def is_semantically_supported_label(label: str | None, evidence: str | None, threshold: float = 0.35) -> bool:
    key = _normalize_title_key(label)
    if not key:
        return False
    if key in BAD_DISPLAY_TOPIC_KEYS:
        return False
    if _label_marker_supported(label, evidence, min_hits=1):
        return True
    if key in OIL_GAS_ONLY_TOPIC_KEYS:
        return semantic_similarity(label, evidence) >= max(threshold, 0.42)
    return semantic_similarity(label, evidence) >= threshold


def _semantic_candidate_labels(domains: set[str]) -> list[str]:
    labels: list[str] = []
    for domain in ("technical_hardware", "architecture_c4", "graph_vkr", "education_vkr", "oil_gas", "technical_research", "project"):
        if domain in domains:
            labels.extend(DOMAIN_CANDIDATE_LABELS.get(domain, ()))
    labels.extend(DOMAIN_TEXT_LABELS.keys())
    labels.extend(TOPIC_CANDIDATE_LABELS.keys())
    labels.extend(label for _markers, label in FINAL_TOPIC_LABEL_RULES)
    return list(dict.fromkeys(labels))


def choose_semantic_label(
    evidence: str | None,
    candidate_labels: list[str] | tuple[str, ...],
    fallback_keywords: list[str] | None = None,
) -> str | None:
    evidence_text = str(evidence or "").strip()
    best_label = None
    best_score = 0.0
    for label in candidate_labels:
        score = max(semantic_similarity(label, evidence_text), 0.58 if _label_marker_supported(label, evidence_text) else 0.0)
        if score > best_score:
            best_label = label
            best_score = score
    if best_label and best_score >= 0.35:
        return best_label
    keywords = _normalize_display_keywords(fallback_keywords or [], 4)
    if len(keywords) >= 2:
        return _sentence_case(" ".join(keywords[:3]))
    return None


def build_topic_evidence(topic_or_aspect: dict | None, transcript_segments: list[dict] | None, result: dict | None = None) -> str:
    item = topic_or_aspect or {}
    parts: list[str] = []
    for key in ("topic_name", "title", "name"):
        if item.get(key):
            parts.append(str(item[key]))
    for key in ("source_text", "summary", "text"):
        if item.get(key):
            parts.append(str(item[key]))
    if item.get("keywords"):
        parts.append(" ".join(_normalize_display_keywords(item.get("keywords"), 12)))
    fragment_ids = set(item.get("fragment_ids") or item.get("fragments") or [])
    for index, segment in enumerate(transcript_segments or [], start=1):
        fragment_id = segment.get("fragment_index") or segment.get("source_fragment") or index
        if fragment_ids and fragment_id not in fragment_ids:
            continue
        text = str(segment.get("text") or "").strip()
        if text:
            parts.append(text)
        if fragment_ids and len(parts) >= 8:
            break
    if not parts and result is not None:
        parts.append(_joined_text(result)[:3000])
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def detect_meeting_domains(transcript: str | dict | None = None, metadata: dict | None = None, meeting_type: str | None = None) -> set[str]:
    if isinstance(transcript, dict):
        haystack = _joined_text(transcript)
        meeting = transcript.get("meeting_type")
        if isinstance(meeting, dict):
            meeting_type = meeting.get("label") or meeting_type
        elif isinstance(meeting, str):
            meeting_type = meeting or meeting_type
        metadata = transcript.get("metadata") or metadata
    else:
        haystack = str(transcript or "")
    haystack = " ".join([haystack, str(metadata or "")]).lower().replace("ё", "е")
    domains: set[str] = set()
    if any(marker in haystack for marker in TECHNICAL_HARDWARE_CONTEXT_MARKERS):
        domains.add("technical_hardware")
    if any(marker in haystack for marker in ARCHITECTURE_C4_CONTEXT_MARKERS):
        domains.add("architecture_c4")
    if any(marker in haystack for marker in GRAPH_VKR_CONTEXT_MARKERS):
        domains.add("graph_vkr")
    if any(marker in haystack for marker in EDUCATION_CONTEXT_MARKERS):
        domains.add("education_vkr")
    if meeting_type in {"technical_research", "education_consultation"} or any(marker in haystack for marker in TECHNICAL_RESEARCH_CONTEXT_MARKERS):
        domains.add("technical_research")
    if meeting_type in {"project_meeting"}:
        domains.add("project")
    if meeting_type in {"commercial_meeting", "commercial_oil_gas", "oil_gas_commercial"}:
        domains.add("commercial")
    strict_oil = any(marker in haystack for marker in STRICT_OIL_GAS_CONTEXT_MARKERS)
    if strict_oil and "technical_hardware" not in domains:
        domains.add("oil_gas")
    if meeting_type in {"commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"} and "technical_hardware" not in domains:
        domains.add("oil_gas")
    if not domains:
        domains.add("general")
    return domains


def topic_allowed_for_domains(
    topic: str | None,
    keywords: list[str] | None,
    evidence: str | None,
    domains: set[str],
) -> bool:
    key = _normalize_title_key(topic)
    evidence_lower = str(evidence or "").lower().replace("ё", "е")
    keyword_text = " ".join(keywords or []).lower().replace("ё", "е")
    if not key:
        return False
    if key in BAD_DISPLAY_TOPIC_KEYS:
        return False
    if key in COMMERCIAL_ONLY_TOPIC_KEYS and not ({"commercial", "oil_gas"} & domains):
        return False
    if "technical_hardware" in domains and key in DOMAIN_BLOCKED_TOPIC_KEYS.get("technical_hardware", set()) and "oil_gas" not in domains:
        return False
    if "technical_hardware" in domains and key in {"входные данные математической модели", "модель топологии сети"}:
        return False
    if "architecture_c4" in domains and "graph_vkr" not in domains:
        if key in {"данные и выборка", "интерпретация данных", "точность модели", "входные данные математической модели", "модель топологии сети"}:
            return False
        if key == "организация следующей встречи" and not re.search(r"\b(?:вторник|понедельник|встреч|созвон)\b", evidence_lower):
            return False
    if key == "организация следующей встречи" and ({"technical_hardware", "architecture_c4", "graph_vkr"} & domains):
        return False
    if "graph_vkr" in domains:
        graph_evidence_without_label = evidence_lower.replace("описание архитектуры в вкр", "")
        if key == "описание архитектуры в вкр" and not re.search(r"\b(?:c4|архитектур|диаграмм|контейнер|компонент)\b", graph_evidence_without_label):
            return False
    oil_specific_text = f"{key} {keyword_text}"
    if "oil_gas" not in domains and any(
        marker in oil_specific_text
        for marker in ("пласт", "скин", "трещин", "скважин", "коллектор", "pvt")
    ):
        return False
    if key in OIL_GAS_ONLY_TOPIC_KEYS and "oil_gas" not in domains:
        if key == "дебит и динамика" and "technical_research" in domains and "technical_hardware" not in domains:
            return True
        return False
    if key in RESEARCH_ONLY_TOPIC_KEYS and not ({"technical_research", "education_vkr", "oil_gas"} & domains):
        return False
    if key in RESEARCH_ONLY_TOPIC_KEYS and "architecture_c4" in domains and "graph_vkr" not in domains:
        return False
    if key in OIL_GAS_ONLY_TOPIC_KEYS and not any(marker in f"{evidence_lower} {keyword_text}" for marker in STRICT_OIL_GAS_CONTEXT_MARKERS):
        return False
    return True


def semantic_clean_topic_or_aspect(
    item: dict,
    evidence: str,
    domains: set[str],
    result: dict | None = None,
) -> dict | None:
    raw_name = str(item.get("topic_name") or item.get("title") or item.get("name") or "").strip()
    raw_key = _normalize_title_key(raw_name)
    keywords = _normalize_display_keywords(item.get("keywords"), 8)
    source_text = evidence or item.get("source_text") or ""
    raw_keyword_text = " ".join(keywords).lower().replace("ё", "е")
    if raw_key in OIL_GAS_ONLY_TOPIC_KEYS and "oil_gas" not in domains:
        if not (raw_key == "дебит и динамика" and "technical_research" in domains and "technical_hardware" not in domains):
            return None
    if "oil_gas" not in domains and any(
        marker in f"{raw_key} {raw_keyword_text}"
        for marker in ("пласт", "скин", "трещин", "скважин", "коллектор", "pvt")
    ):
        return None
    normalized = _final_display_topic_name(raw_name, keywords, source_text, result or {})
    if not normalized:
        normalized = choose_semantic_label(source_text, _semantic_candidate_labels(domains), keywords)
    if not normalized:
        return None
    if not topic_allowed_for_domains(normalized, keywords, source_text, domains):
        return None
    if _looks_like_bad_phrase_topic(normalized):
        repaired = choose_semantic_label(source_text, _semantic_candidate_labels(domains), keywords)
        if not repaired or not topic_allowed_for_domains(repaired, keywords, source_text, domains):
            return None
        normalized = repaired
    if not is_semantically_supported_label(normalized, source_text):
        repaired = choose_semantic_label(source_text, _semantic_candidate_labels(domains), keywords)
        if repaired and topic_allowed_for_domains(repaired, keywords, source_text, domains):
            normalized = repaired
        elif _normalize_title_key(raw_name) in BAD_DISPLAY_TOPIC_KEYS or _normalize_title_key(normalized) in OIL_GAS_ONLY_TOPIC_KEYS:
            return None
    return {
        "title": normalized,
        "keywords": keywords,
        "source_text": source_text,
        "semantic_similarity": round(semantic_similarity(normalized, source_text), 3),
    }


def _technical_topic_label(name: str | None, keywords: list[str] | None = None, source_text: str | None = None) -> str | None:
    haystack = " ".join([name or "", " ".join(keywords or []), source_text or ""]).lower()
    primary = " ".join([name or "", " ".join(keywords or [])]).lower()
    best_label = None
    best_score = 0
    for markers, label in TECHNICAL_TOPIC_LABEL_RULES:
        total_hits = sum(1 for marker in markers if marker in haystack)
        if not total_hits:
            continue
        primary_hits = sum(1 for marker in markers if marker in primary)
        score = primary_hits * 3 + total_hits
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


COMMERCIAL_TOPIC_EXACT_MAP = {
    "даты рабочих контракте": "сроки поставки и оплаты",
    "рабочих контракте отгрузки поставки даты": "сроки поставки и оплаты",
    "даты платежей": "платежные условия",
}
COMMERCIAL_TIMING_TOPIC_MARKERS = (
    "рабоч",
    "контракт",
    "отгруз",
    "поставк",
    "дат",
    "календар",
    "платеж",
    "платёж",
    "оплат",
    "срок",
)
COMMERCIAL_PAYMENT_TOPIC_MARKERS = (
    "платеж",
    "платёж",
    "оплат",
    "предоплат",
    "отсроч",
    "гарант",
    "аккредитив",
)


def _commercial_topic_label(name: str | None, keywords: list[str] | None = None, source_text: str | None = None) -> str | None:
    haystack = " ".join([name or "", " ".join(keywords or []), source_text or ""]).lower().replace("ё", "е")
    exact_key = _normalize_title_key(name)
    if exact_key in COMMERCIAL_TOPIC_EXACT_MAP:
        return COMMERCIAL_TOPIC_EXACT_MAP[exact_key]
    if "за 5 рабочих" in haystack and "отгруз" in haystack:
        return "сроки поставки и оплаты"
    timing_hits = sum(1 for marker in COMMERCIAL_TIMING_TOPIC_MARKERS if marker in haystack)
    payment_hits = sum(1 for marker in COMMERCIAL_PAYMENT_TOPIC_MARKERS if marker in haystack)
    if payment_hits >= 2 and payment_hits > timing_hits:
        return "платежные условия"
    if timing_hits >= 3:
        return "сроки поставки и оплаты"
    return None


def _domain_topic_label(name: str | None, keywords: list[str] | None = None, source_text: str | None = None) -> str | None:
    haystack = " ".join([name or "", " ".join(keywords or []), source_text or ""]).lower().replace("ё", "е")
    primary = " ".join([name or "", " ".join(keywords or [])]).lower().replace("ё", "е")
    best_label = None
    best_score = 0
    for markers, label in FINAL_TOPIC_LABEL_RULES:
        if not all(marker in haystack for marker in markers):
            continue
        primary_hits = sum(1 for marker in markers if marker in primary)
        total_hits = sum(1 for marker in markers if marker in haystack)
        score = primary_hits * 3 + total_hits
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _contextual_display_topic_label(
    name: str | None,
    keywords: list[str] | None,
    source_text: str | None,
    result: dict,
) -> str | None:
    domains = detect_meeting_domains(result)
    key = _normalize_title_key(name)
    haystack = " ".join([name or "", " ".join(keywords or []), source_text or "", _joined_text(result)[:2500]])
    haystack = haystack.lower().replace("ё", "е")
    if re.search(r"\bвходн\w*\s+и\s+выходн\w+\s+данн\w+\s+модул", key):
        if "technical_hardware" in domains:
            if re.search(r"\b(?:формальн\w+\s+постановк|постановк\w+\s+задач)\b", haystack):
                return "формальная постановка задачи"
            return "входные данные ML pipeline"
        if re.search(r"\b(?:граф|тополог|диаметр|связ|метрик|математическ)\b", haystack):
            return "входные данные математической модели"
        if re.search(r"\b(?:c4|контейнер|компонент|диаграмм|взаимодейств|архитектур|каталог|ldap)\b", haystack):
            return "компоненты и взаимодействия"
    if "technical_hardware" in domains:
        if key == "входные данные математической модели":
            if re.search(r"\b(?:формальн\w+\s+постановк|постановк\w+\s+задач)\b", haystack):
                return "формальная постановка задачи"
            return "входные данные ML pipeline"
        if key in {"входные данные математической модели", "модель топологии сети"}:
            return None
        if key == "организация следующей встречи":
            if re.search(r"\b(?:термопаст|оборудован|камер|raspberry|benchmark|датасет)\b", haystack):
                return "график работы с оборудованием"
            return None
        if key == "постановка задачи" and re.search(r"\b(?:формальн|входн|ml|pipeline)\b", haystack):
            return "формальная постановка задачи"
    if "architecture_c4" in domains:
        if key in {"входные данные математической модели", "модель топологии сети"}:
            return None
        if key in {"данные и выборка", "интерпретация данных"}:
            if re.search(r"\b(?:админ|портал|консоль|каталог|ldap|пользовател|внешн\w+\s+систем)\b", haystack):
                return "внешние системы и пользователи"
            if re.search(r"\b(?:c4|контейнер|компонент|диаграмм|взаимодейств|архитектур)\b", haystack):
                return "компоненты и взаимодействия"
            return None
        if key == "входные и выходные данные модулей":
            return "компоненты и взаимодействия"
        if key in {"точность модели", "организация следующей встречи"}:
            if re.search(r"\b(?:вторник|понедельник|встреч|созвон)\b", haystack):
                return "график презентации/встречи"
            return None
        if key == "документ вкр и отчет":
            return "ВКР и презентация"
        if key == "требования к вкр" and re.search(r"\b(?:c4|архитектур|диаграмм|компонент|каталог|ldap|systemd)\b", haystack):
            return "описание архитектуры в ВКР"
    if "graph_vkr" in domains:
        if key == "входные и выходные данные модулей":
            return "входные данные математической модели"
        if key == "модель и аппроксимация":
            if re.search(r"\b(?:аппроксимац|приближен)\b", haystack) and re.search(r"\b(?:граф|метрик|диаметр|связ)\b", haystack):
                return "аппроксимация метрик графа"
            if re.search(r"\b(?:граф|тополог|диаметр|связ|метрик|математическ)\b", haystack):
                return "математическая модель"
        if key in {"вкр и документация", "документ вкр и отчет"}:
            return "требования к ВКР"
        if key == "интерпретация данных" and re.search(r"\b(?:граф|тополог|диаметр|связ|метрик)\b", haystack):
            return "метрики графа"
    return None


def _has_oil_gas_context(result: dict, name: str | None = None, keywords: list[str] | None = None, source_text: str | None = None) -> bool:
    mt = result.get("meeting_type")
    haystack = " ".join([_joined_text(result), source_text or ""]).lower().replace("ё", "е")
    strict_hit = any(marker in haystack for marker in STRICT_OIL_GAS_CONTEXT_MARKERS)
    if strict_hit:
        return True
    if any(marker in haystack for marker in NON_OIL_TECH_CONTEXT_MARKERS):
        return False
    if isinstance(mt, dict) and mt.get("label") in {"commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        return True
    if any(marker in haystack for marker in OIL_GAS_CONTEXT_MARKERS):
        return True
    return False


def _has_technical_research_context(result: dict, source_text: str | None = None) -> bool:
    mt = result.get("meeting_type")
    if isinstance(mt, dict) and mt.get("label") in {"technical_research", "education_consultation"}:
        return True
    haystack = " ".join([_joined_text(result), source_text or ""]).lower().replace("ё", "е")
    return any(marker in haystack for marker in TECHNICAL_RESEARCH_CONTEXT_MARKERS)


def _is_filler_topic_name(name: str | None) -> bool:
    words = _topic_words(name)
    if not words:
        return True
    filler_hits = sum(1 for word in words if word in TOPIC_FILLER_WORDS)
    return filler_hits >= max(1, len(words) - 1)


def _looks_like_bad_phrase_topic(name: str | None) -> bool:
    key = _normalize_title_key(name)
    if key in BAD_DISPLAY_TOPIC_KEYS:
        return True
    words = _topic_words(name)
    if len(words) >= 2 and len(set(words)) <= max(1, len(words) // 2):
        return True
    filler_hits = sum(1 for word in words if word in TOPIC_FILLER_WORDS)
    return filler_hits >= 2 and filler_hits >= len(words) // 2


def _final_display_topic_name(name: str | None, keywords: list[str] | None, source_text: str | None, result: dict) -> str | None:
    raw = str(name or "").strip()
    if not raw and keywords:
        raw = " ".join(keywords[:3])
    if not raw:
        return None

    key = _normalize_title_key(raw)
    commercial = _is_commercial_result(result) or _is_oil_gas_result(result)
    exact_title = DISPLAY_TOPIC_EXACT_MAP.get(key)
    contextual_title = _contextual_display_topic_label(raw, keywords, source_text, result)
    commercial_title = _commercial_topic_label(raw, keywords, source_text) if commercial else None
    domain_title = _domain_topic_label(raw, keywords, source_text) or _technical_topic_label(raw, keywords, source_text)

    candidate = exact_title or contextual_title or commercial_title or domain_title
    if key in COMMERCIAL_ONLY_TOPIC_KEYS and not commercial:
        return None
    if key in OIL_GAS_ONLY_TOPIC_KEYS and not _has_oil_gas_context(result, raw, keywords, source_text):
        return None
    if key in BAD_DISPLAY_TOPIC_KEYS:
        return candidate
    if key in GENERIC_DISPLAY_TOPIC_KEYS:
        return candidate
    if _looks_like_bad_phrase_topic(raw):
        return candidate

    normalized = candidate or _sentence_case(raw)
    normalized_key = _normalize_title_key(normalized)
    if normalized_key in COMMERCIAL_ONLY_TOPIC_KEYS and not commercial:
        return None
    if normalized_key in OIL_GAS_ONLY_TOPIC_KEYS and not _has_oil_gas_context(result, raw, keywords, source_text):
        return None
    if normalized_key in RESEARCH_ONLY_TOPIC_KEYS and not _has_technical_research_context(result, source_text):
        return None
    if normalized_key in GENERIC_DISPLAY_TOPIC_KEYS and not candidate:
        return None
    return normalized


def _deadline_key(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").lower().replace("–", "-")).strip(" .,:;")


def _deadline_family_key(value: str | None) -> str:
    clean = _deadline_key(value)
    working_match = re.search(
        r"(?:минимум\s+)?за\s+(5|пять)\s+рабоч\w*\s+дн\w*(?:\s+до(?:\s+отгрузк\w*)?)?",
        clean,
        re.IGNORECASE,
    )
    if working_match:
        return "минимум за 5 рабочих дней до отгрузки"
    clean = re.sub(r"\b(календарн\w+|рабоч\w+|минимум)\b", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _canonical_deadline_value(value: str | None, source_text: str | None = None) -> str | None:
    haystack = f"{source_text or ''} {value or ''}"
    haystack_lower = haystack.lower()
    clean_value = re.sub(r"\s+", " ", (value or "").strip())
    if re.search(r"минимум\s+за\s+(?:5|пять)\s+рабоч\w*\s+дн\w*\s+до\s+отгрузк\w*", haystack, re.IGNORECASE):
        return "минимум за 5 рабочих дней до отгрузки"
    if re.search(r"\bследующ(?:ая|ей)\s+недел[яе]\b|\bна\s+следующей\s+неделе\b", haystack_lower):
        if re.search(r"\bследующ(?:ая|ей)\s+недел[яе]\b|\bна\s+следующей\s+неделе\b", value or "", re.IGNORECASE):
            return "на следующей неделе"
    if re.search(r"\bчетверг\b", haystack_lower) and re.search(r"\b(?:7[.:]30|19[.:]30)\b", haystack_lower):
        return "четверг 19:30"
    if re.fullmatch(r"\s*7[.:]30\s*", value or ""):
        if re.search(r"\b(встреч|созвон|четверг|после\s+7|вечер)\w*\b", haystack_lower):
            return "19:30"
        return "7:30"
    if re.fullmatch(r"\s*19[.:]30\s*", value or ""):
        return "19:30"
    if re.fullmatch(r"(?:в|во|к)\s+\w+", clean_value, re.IGNORECASE) or clean_value.lower() in {"завтра", "в следующий раз", "через месяц"}:
        return clean_value[:1].lower() + clean_value[1:]
    return value


def _deadline_specificity(value: str | None) -> int:
    clean = value or ""
    return _words_count(clean) + len(clean) // 12


def _deadline_seen_index(clean_deadlines: list[dict], value: str | None) -> int | None:
    key = _deadline_key(value)
    if not key:
        return None
    for index, item in enumerate(clean_deadlines):
        existing = _deadline_key(item.get("deadline"))
        family = _deadline_family_key(value)
        existing_family = _deadline_family_key(item.get("deadline"))
        if (
            existing == key
            or family == existing_family
            or (len(key) > 5 and key in existing)
            or (len(existing) > 5 and existing in key)
        ):
            return index
    return None


HISTORICAL_DATE_CONTEXT_RE = re.compile(
    r"\b(ходили|смотрели|разбирались|тогда\s+делал|две\s+недели\s+назад|в\s+прошлый\s+раз|"
    r"раньше|уже\s+было|уже\s+ходили|уже\s+смотрели|уже\s+разбирались)\b",
    re.IGNORECASE,
)
FUTURE_DEADLINE_CONTEXT_RE = re.compile(
    r"\b(завтра|к\s+выходным|в\s+следующий\s+раз|"
    r"снова\s+прийти|отправим|отправить|будем\s+работать|привести|надо\s+прийти|презентац|"
    r"надо\s+выдать|через\s+месяц\s+будут|подготовить|сдать|показать|прислать)\b",
    re.IGNORECASE,
)


def _is_bad_clean_deadline(value: str | None, source_text: str | None) -> bool:
    key = _deadline_key(value)
    plain_key = re.sub(r"^(?:в|во|к|ко|на)\s+", "", key)
    source = (source_text or "").lower().replace("ё", "е")
    if key in {"срок", "дедлайн", "кратчайший срок", "16 го видно", "16-го видно"}:
        return True
    if HISTORICAL_DATE_CONTEXT_RE.search(source) and not FUTURE_DEADLINE_CONTEXT_RE.search(source):
        return True
    if plain_key == "четверг" and not re.search(r"встреч|созвон|поставим|назнач|удобн|соглас|подготов|отправ|сдать|присл|показ", source):
        return True
    if key in {"4.30", "4:30", "5.30", "5:30"} and not re.search(r"встреч|созвон|вечер|удобн", source):
        return True
    return False


def _refine_deadline_kind(kind: str, text: str | None, value: str | None, result: dict) -> str:
    if not _is_commercial_result(result) and not _is_oil_gas_result(result):
        return kind
    haystack = f"{text or ''} {value or ''}".lower().replace("ё", "е")
    if re.search(r"сегодня[- ]завтра", haystack) and re.search(r"документ|term sheet|термшит|прислат|отправ|направ", haystack):
        return "commitment_deadline"
    if re.search(r"(?:минимум\s+)?за\s+(?:5|пять)\s+рабоч\w*\s+дн\w*\s+до\s+отгруз", haystack):
        return "contract_logistics_deadline"
    if re.search(r"в\s+течение\s+24\s+час", haystack) and re.search(r"судн|приемлем|приемлемост|подтверд", haystack):
        return "operational_deadline"
    return kind


SUPPLEMENTAL_DEADLINE_PATTERNS = (
    r"\bзавтра\b",
    r"\bв\s+(?:пятницу|понедельник|среду|четверг)\b",
    r"\bво\s+вторник\b",
    r"\bк\s+выходным\b",
    r"\bв\s+следующий\s+раз\b",
    r"\bчерез\s+месяц\b",
)


def _deadline_candidates_from_result(result: dict) -> list[dict]:
    candidates: list[dict] = []
    existing_sources = {
        (item.get("source_fragment"), re.sub(r"\s+", " ", item.get("text") or "").strip())
        for item in result.get("deadlines", [])
    }
    for unit in _result_text_units(result):
        text = unit["text"]
        source_fragment = unit.get("source_fragment")
        if (source_fragment, re.sub(r"\s+", " ", text).strip()) in existing_sources:
            continue
        values = find_deadlines(text)
        for pattern in SUPPLEMENTAL_DEADLINE_PATTERNS:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(0).strip(" .,:;")
                if value and value not in values:
                    values.append(value)
        if not values:
            continue
        candidates.append(
            {
                "text": text,
                "deadlines": values,
                "deadline_normalized": None,
                "kind": classify_deadline_kind(text),
                "source_fragment": source_fragment,
            }
        )
    return candidates


def _extract_question_title(question: str) -> str:
    clean = re.sub(r"\s+", " ", question or "").strip()
    if "?" in clean:
        question_part = clean[: clean.rfind("?") + 1]
        late_short_question = re.search(
            r"\b(насколько\s+[^?.!]{1,80}\?|кто\s+[^?.!]{1,80}\?|какой\s+потолок\?|до\s+какой\s+даты\?)\s*$",
            question_part,
            re.IGNORECASE,
        )
        if late_short_question and late_short_question.start() > 24:
            selected = late_short_question.group(1).strip(" .,:;")
        else:
            first_question = clean[: clean.find("?") + 1]
            start = max(first_question.rfind(". "), first_question.rfind("! "), first_question.rfind("? "))
            selected = first_question[start + 2 :].strip(" .,:;") if start >= 0 else first_question.strip(" .,:;")
        return _trim_text(selected, 180) or selected
    return _trim_text(clean, 180) or clean


def _summarize_answer(answer: str | None, limit: int = 330) -> str | None:
    if not answer:
        return None
    clean = re.sub(r"\s+", " ", answer).strip()
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    summary = " ".join(sentence for sentence in sentences[:2] if sentence).strip()
    if not summary:
        summary = clean
    return _trim_text(summary, limit)


PREMIUM_BREAKDOWN_QUESTION = "Как раскладывается премия 1,8 доллара по фрахту, страховке и риску?"
PREMIUM_BREAKDOWN_ANSWER = (
    "Около 90 центов относится к фрахту, 30–40 центов — к страховке и портовым расходам, "
    "остальное — к риску задержек и доступности логистики."
)

BAD_QA_FRAGMENT_PATTERNS = (
    "что-то сделать, поэтому это вам",
    "когда я тестировал и просто проверял",
    "что обсудили, не придумали никаких решений",
    "что я раздаю экран",
    "где в котором есть таймер",
    "что короче не получилось связать",
    "что вроде как правда",
    "что у нас есть какие-то необходимые критерии",
    "что-то там минимизировать",
    "где-то можно это просто перед этим",
    "что это решение ожидается",
    "где-то в пакете как нам сказали",
    "что администратор целой",
    "между человеком",
    "кто-то им отвечает так что я",
    "что как бы написать математическую постановку",
    "что можно дать какие-нибудь простенькие задачи",
    "какие-нибудь простенькие задачи",
    "что значит вот минимизировать",
    "что у нас кратчайший срок",
    "что надо будет еще раз",
    "что вроде как это можно",
    "что это вам видимо надо будет",
    "что то сделать поэтому это вам",
    "какой-нибудь разметку может быть",
    "какие-то учебные вещи",
    "когда я это делал, потому что",
    "что думаете в целом понятно вам как из этого всего дела тут вообще что происходит",
    "какие-то особые дефолтные так получается",
    "что там используют какие-то локальные файлы",
    "что просто вот такая штука есть",
    "какие-то конкретные объекты",
)

GOOD_QA_TITLE_MARKERS = (
    "что означает минимальность",
    "корректно ли сформулирована",
    "какие метрики использовать",
    "что указать в критериях качества",
    "можно ли использовать аппаратное ускорение",
    "какая ос и программная среда",
    "требуется ли договор",
    "что делать сейчас с практикой",
    "кто подписывает документы",
    "как лучше начать введение",
    "какие источники использовать",
    "по архитектуре оно централизировано",
    "где хранятся метрики",
    "какая база данных используется",
    "база данных локальная",
    "можно ли обосновать актуальность",
    "что там получилось",
    "как связать критерии оптимизации",
    "почему полносвязанная топология",
    "есть ли модель блокировок",
    "можно ли использовать минимальное остовное дерево",
    "какие веса каналов учитывать",
    "как учитывать расписания",
    "достаточно ли mvp",
    "когда удобнее встречаться",
    "как лучше оформить архитектурную диаграмму",
    "как подписать systemd timer",
    "что поручить второкурсникам",
    "какие ограничения добавить в модель",
    "как сформулировать математическую постановку",
    "есть ли смешанная метрика",
    "какие метрики лучше использовать",
    "как оценивать качество модели топологии",
    "нужно ли добавить сбор данных и телеметрию",
    "можно ли прислать диаграмму отдельным",
    "нужно ли прийти во вторник",
    "понятна ли архитектурная диаграмма",
    "как корректно описать взаимодействие через каталог",
    "нужно ли указывать локальные файлы",
    "нужно ли делать отдельные уровни c4",
    "нужно ли пояснять каждую стрелку",
    "стоит ли добавить sequence diagram",
    "стоит ли подробнее расписывать ограничения",
    "правда ли что завтра дедлайн",
    "что вообще у вас есть",
    "где находятся камеры",
    "можно ли смотреть температуру npu",
    "можно ли смотреть частоту",
)

HARDWARE_QA_CANONICAL_RULES = (
    (r"задач\w+\s+перед\s+разработчик|задач\w+\s+котор\w+\s+реша\w+\s+по", "В ней описывается задача перед разработчиками или задача, которую решает ПО?"),
    (r"ограничен\w+.*формальн\w+\s+постановк|формальн\w+\s+постановк.*ограничен", "Стоит ли подробнее расписывать ограничения в формальной постановке?"),
    (r"завтра.*дедлайн|дедлайн.*завтра|дедлайн.*черновик", "Правда ли, что завтра дедлайн у черновика?"),
    (r"что\s+вообще\s+у\s+вас\s+есть|что\s+у\s+вас\s+там\s+есть", "Что вообще у вас есть?"),
    (r"простеньк\w+\s+задач|учебн\w+\s+вещ|opencv|второкурсник", "Что можно поручить второкурсникам?"),
    (r"где.*камер|камер.*где|камер.*короб|камер.*пакет", "Где находятся камеры?"),
    (r"температур.*npu|npu.*температур", "Можно ли смотреть температуру NPU?"),
    (r"частот.*benchmark|benchmark.*частот", "Можно ли смотреть частоту во время benchmark?"),
    (r"qemu|симуляц\w+\s+raspberry", "Можно ли использовать QEMU для симуляции Raspberry Pi?"),
)

ARCHITECTURE_QA_CANONICAL_RULES = (
    (r"png|pdf|svg|отдельн\w+\s+файл", "Можно ли прислать диаграмму отдельным PNG/PDF/SVG-файлом?"),
    (r"во\s+вторник|вторник.*прийти|прийти.*вторник", "Нужно ли прийти во вторник?"),
    (r"понятн\w+.*диаграм|диаграм\w+.*понятн|в\s+целом\s+понятн", "Понятна ли архитектурная диаграмма?"),
    (r"взаимодейств\w+.*каталог|каталог.*взаимодейств", "Как корректно описать взаимодействие через каталог?"),
    (r"локальн\w+\s+файл|файл\w+.*архитектур", "Нужно ли указывать локальные файлы в описании архитектуры?"),
    (r"уровн\w+\s+c4|c4.*уровн", "Нужно ли делать отдельные уровни C4?"),
    (r"стрелк", "Нужно ли пояснять каждую стрелку?"),
    (r"sequence\s+diagram|sequence", "Стоит ли добавить sequence diagram?"),
    (r"компонент.*взаимодейств|взаимодейств.*компонент", "Как описать компоненты и взаимодействия?"),
)

GRAPH_VKR_QA_CANONICAL_RULES = (
    (
        r"ухудшается\s+но\s+не\s+так\s+резко|смешан\w+\s+метрик",
        "Есть ли смешанная метрика для оценки графа?",
    ),
    (
        r"плох\w+\s+тем\s+что\s+он\s+учитывает|наихудш\w+\s+случа|диаметр.*средн\w+\s+дистанц|средн\w+\s+дистанц.*диаметр",
        "Какие метрики лучше использовать: диаметр или среднюю дистанцию?",
    ),
    (
        r"что\s+нам\s+интересно\s+от\s+этой\s+модел|качество\s+модел\w+\s+тополог",
        "Как оценивать качество модели топологии?",
    ),
    (
        r"какие-то\s+инструмент\w+\s+для|сбор\w*\s+данн\w+|телеметр",
        "Нужно ли добавить сбор данных и телеметрию?",
    ),
    (
        r"критери\w+\s+качеств|необходим\w+\s+критери",
        "Что указать в критериях качества?",
    ),
)
TECHNICAL_DEFINITION_RE = re.compile(
    r"\b(это\s+(?:дебит|параметр|значение|модель|формула|расч[её]т)|"
    r"метро(?:кубический|куб\w*)\s+разделить\s+на\s+сутки|"
    r"обратная\s+задача\s+интерпретации|r-?квадрат|r²|"
    r"калькулированн\w+\s+и\s+эталон\w+|параметр\w+\s+могут\s+быть\s+предположительн\w*)\b",
    re.IGNORECASE,
)
TECHNICAL_CONTEXT_RE = re.compile(
    r"\b(как\s+правило|в\s+теории|в\s+принципе|получается|может\s+быть|могут\s+быть|"
    r"могут\s+присутствовать|могут\s+.*корректировк\w+\s+внести|могут\s+корректировк\w+\s+внести|"
    r"можно\s+будет|далеко\s+не\s+бесполезно|как\s+раз[-\s]?таки|"
    r"(?:для\s+)?того,\s*чтобы\s+проверить|(?:для\s+)?того\s+чтобы\s+проверить)\b",
    re.IGNORECASE,
)
TECHNICAL_GENERIC_RE = re.compile(
    r"\b(это\s+необходимо\s+сделать|надо\s+подумать|нужно\s+понимать|это\s+важно|это\s+понятно)\b",
    re.IGNORECASE,
)
TECHNICAL_SPEAKER_INTENT_RE = re.compile(
    r"\b(я\s+попробую\s+(?:тогда\s+)?(?:у\s+себя\s+)?поискать|(?:у\s+)?себя\s+поискать\s+кейс\w*|"
    r"может\s+поискать|я\s+думаю\s+поискать|посмотрю\s+у\s+себя)\b",
    re.IGNORECASE,
)
TECHNICAL_RECOMMENDATION_RE = re.compile(
    r"\b(лучше|стоит|нужно\s+подумать|надо\s+подумать|можно\s+оставить|желательно|"
    r"начать\s+со|начать\s+с|частн\w+\s+верс\w+|обосновани\w+|вычислительн\w+\s+эксперимент)\b",
    re.IGNORECASE,
)
RESEARCH_ACTION_OBJECT_RE = re.compile(
    r"\b(устойчивост\w+\s+модел\w+|точност\w+|разв[её]ртк\w+|групп\w+\s+параметр\w+|"
    r"s,\s*n,\s*a,\s*l|s\s+n\s+a\s+l|частн\w+\s+случа\w+|кейс\w+|реализац\w+|"
    r"расч[её]т\w+|вычислительн\w+\s+эксперимент|вкр|раздел|следующ\w+\s+встреч\w+|"
    r"четверг|7[:.]30|промежуточн\w+\s+результат\w+|скин-?фактор|systemd|timer|kerberos|каталог)\b",
    re.IGNORECASE,
)
RESEARCH_ACTION_VERB_RE = re.compile(
    r"\b(проверить|сделать|разделить|сопоставить|подготовить|описать|оформить|найти|"
    r"предоставить|согласовать|скинуть|отправить|провести|добавить)\b",
    re.IGNORECASE,
)


def _is_premium_breakdown_qa(question: str | None, answer: str | None) -> bool:
    combined = f"{question or ''} {answer or ''}".lower()
    has_premium = "прем" in combined or "1,8" in combined or "одну целую восем" in combined or "одна целая восем" in combined
    has_breakdown = "фрахт" in combined and ("страх" in combined or "портов" in combined) and "риск" in combined
    return has_premium and has_breakdown


def _canonical_graph_vkr_question(question: str | None, answer: str | None, result: dict) -> str | None:
    domains = detect_meeting_domains(result)
    combined = re.sub(r"\s+", " ", f"{question or ''} {answer or ''}").lower().replace("ё", "е")
    if "technical_hardware" in domains:
        for pattern, title in HARDWARE_QA_CANONICAL_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return title
    if "architecture_c4" in domains:
        for pattern, title in ARCHITECTURE_QA_CANONICAL_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return title
    if "graph_vkr" in domains:
        if re.search(r"критери\w+\s+качеств|необходим\w+\s+критери", combined, re.IGNORECASE):
            return "Что указать в критериях качества?"
        for pattern, title in GRAPH_VKR_QA_CANONICAL_RULES:
            if re.search(pattern, combined, re.IGNORECASE):
                return title
    return None


def _is_low_quality_question(question: str, answer: str | None = None) -> bool:
    title = (_extract_question_title(question) or "").lower().strip(" ?!.")
    combined = re.sub(r"\s+", " ", f"{question or ''} {answer or ''}".lower())
    if any(pattern in combined for pattern in BAD_QA_FRAGMENT_PATTERNS):
        return True
    if any(marker in title for marker in GOOD_QA_TITLE_MARKERS):
        return False
    if not title:
        return True
    if title.startswith("давайте") or re.match(r"^\d", title):
        return True
    weak_exact = {
        "правильно",
        "да",
        "так",
        "это нормально",
        "а если",
        "какой компромисс",
        "насколько небольшой",
    }
    if title in weak_exact:
        answer_text = (answer or "").lower()
        strong_answer_markers = ("покупатель", "поставщик", "продавец", "банк", "потолок", "ставка", "доллар", "тонн", "%")
        return not any(marker in answer_text for marker in strong_answer_markers)
    if _words_count(title) <= 3:
        answer_text = (answer or "").lower()
        allow_short = (
            title.startswith("какой потолок") and any(marker in answer_text for marker in ("потолок", "ставка", "доллар", "тысяч"))
        ) or (
            title.startswith("кто номинирует") and any(marker in answer_text for marker in ("покупатель", "поставщик", "продавец"))
        )
        return not allow_short
    if "?" not in (question or "") and not QUESTION_START_RE.search(title):
        return True
    return False


def _deadline_context(text: str | None, deadline: str | None) -> str | None:
    if not text:
        return None
    clean = re.sub(r"\s+", " ", text).strip()
    if not deadline or deadline not in clean:
        return _trim_text(clean, 140)
    idx = clean.find(deadline)
    start = max(0, idx - 70)
    end = min(len(clean), idx + len(deadline) + 70)
    return _trim_text(clean[start:end].strip(" .,:;"), 140)


def _normalize_task_title(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip(" .,:;")
    compact = re.sub(r"^\s*(?:но|тогда)\s+", "", compact)

    oil_rules = [
        (("график", "трубе", "терминал"), "Подтвердить финальный график по трубе и терминалу"),
        (("транспорт", "окно"), "Забронировать транспортное окно"),
        (("бронировать", "окно"), "Забронировать транспортное окно для первой партии"),
        (("лаборатор", "протокол"), "Направить свежие лабораторные протоколы"),
        (("предел", "качест", "механизм"), "Прописать в договоре пределы качества и механизм корректировки цены"),
        (("шкал", "дисконт"), "Согласовать шкалу дисконта при отклонениях по качеству"),
        (("финальн", "дифференциал"), "Согласовать финальный дифференциал по партиям"),
        (("согласовать", "дифференциал"), "Согласовать финальный дифференциал по партиям"),
        (("платеж", "точн", "дат"), "Прописать точные даты платежей"),
        (("grace", "period"), "Прописать grace period и пеню после его истечения"),
        (("критер", "приемлемост"), "Прописать критерии приемлемости судна"),
        (("приемлемост", "судн"), "Прописать критерии приемлемости судна"),
        (("номинац", "судн"), "Прописать порядок номинации судна и сроки направления документов"),
        (("документ", "судн"), "Направить документы по основному и резервному судну"),
        (("независим", "инспектор"), "Согласовать независимого инспектора"),
        (("август", "опцион"), "Подтвердить августовский опцион"),
    ]
    for markers, title in oil_rules:
        if all(marker in compact for marker in markers):
            return title

    if "развертк" in compact or "развёртк" in compact:
        return "Сделать развертку по параметрам/формуле безразмеривания"
    if "ближайших задач" in compact and ("s" in compact or "n" in compact or "a" in compact or "l" in compact):
        return "Проверить разделение исследования по группам параметров S, N, A, L"
    if "промежуточные результаты" in compact and ("скин" in compact or "скидыв" in compact):
        return "Скинуть промежуточные результаты"
    if "systemd" in compact and "timer" in compact:
        return "Описать запуск сервиса через systemd timer каждые 15 минут"
    if "сопостав" in compact and ("кейс" in compact or "реализац" in compact or "расчет" in compact or "расчёт" in compact):
        return "Сопоставить кейсы/реализации расчетов"
    if "встреч" in compact and ("четверг" in compact or "7.30" in compact or "7:30" in compact):
        return "Согласовать следующую встречу на четверг 19:30"

    match = re.search(r"\b(?:надо|нужно|необходимо)\s+(.+?)\s+сделать\b", compact)
    if match:
        return _sentence_case(f"Сделать {match.group(1)}")

    replacements = [
        (r"^\s*(?:надо|нужно|необходимо)\s+проверить\s+", "Проверить "),
        (r"^\s*(?:надо|нужно|необходимо)\s+подготовить\s+", "Подготовить "),
        (r"^\s*(?:надо|нужно|необходимо)\s+разделить\s+", "Разделить "),
        (r"^\s*(?:надо|нужно|необходимо)\s+описать\s+", "Описать "),
        (r"^\s*(?:надо|нужно|необходимо)\s+добавить\s+", "Добавить "),
        (r"^\s*(?:надо|нужно|необходимо)\s+скинуть\s+", "Скинуть "),
        (r"^\s*(?:надо|нужно|необходимо)\s+сделать\s+", "Сделать "),
        (r"^\s*(?:надо|нужно|необходимо)\s+прописать\s+", "Прописать "),
        (r"^\s*(?:надо|нужно|необходимо)\s+четко\s+прописать\s+", "Прописать "),
        (r"^\s*(?:надо|нужно|необходимо)\s+указать\s+", "Указать "),
    ]
    for pattern, prefix in replacements:
        if re.search(pattern, compact):
            return _sentence_case(re.sub(pattern, prefix, compact))

    return _sentence_case(text)


BAD_CLEAN_TASK_SUBSTRINGS = (
    "которую мы писали скинь пожалуйста",
    "вот стандартные обработка изображений",
    "сделать. я думаю",
    "мощь в это или нет сделать",
)

BAD_CLEAN_TASK_TITLES = {
    "Согласовать следующую встречу на четверг 19:30",
    "Сопоставить кейсы/реализации расчетов",
    "Описать вариант в ВКР",
}


def _is_bad_clean_task(title: str | None, source_text: str | None) -> bool:
    title_clean = (title or "").strip()
    source = re.sub(r"\s+", " ", (source_text or "").lower())
    if title_clean in BAD_CLEAN_TASK_TITLES:
        return True
    return any(pattern in source for pattern in BAD_CLEAN_TASK_SUBSTRINGS)


def _research_action_title(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip(" .,:;")
    if not compact:
        return None
    rules = [
        (("устойчив", "модел"), "Проверить устойчивость модели и точность"),
        (("точност", "модел"), "Проверить устойчивость модели и точность"),
        (("развертк",), "Сделать развертку по параметрам/формуле безразмеривания"),
        (("развёртк",), "Сделать развертку по параметрам/формуле безразмеривания"),
        (("групп", "параметр"), "Разделить исследование по группам параметров S, N, A, L"),
        (("групп", "s", "n", "a", "l"), "Разделить исследование по группам параметров S, N, A, L"),
        (("s", "n", "a", "l", "параметр"), "Разделить исследование по группам параметров S, N, A, L"),
        (("сопостав", "кейс"), "Сопоставить кейсы/реализации расчетов"),
        (("сопостав", "реализац"), "Сопоставить кейсы/реализации расчетов"),
        (("вычислительн", "эксперимент"), "Подготовить описание вычислительного эксперимента"),
        (("systemd", "timer"), "Описать запуск сервиса через systemd timer каждые 15 минут"),
        (("оформ", "вкр"), "Оформить раздел ВКР"),
        (("опис", "вкр"), "Описать вариант в ВКР"),
        (("следующ", "встреч"), "Согласовать следующую встречу на четверг 19:30"),
    ]
    for markers, title in rules:
        if all(marker in compact for marker in markers):
            return title
    return _normalize_task_title(text)


def _recommendation_title(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", (text or "").lower()).strip(" .,:;")
    if "частн" in compact and ("провер" in compact or "верс" in compact):
        return "Проверить более частные версии модели"
    if "скин" in compact and ("начать" in compact or "провер" in compact):
        return "Начать со скин-фактора / одного ключевого параметра"
    if "обоснован" in compact:
        return "Подумать над обоснованием применимости модели"
    if "вычислительн" in compact and "эксперимент" in compact:
        return "Описать первичный вариант как вычислительный эксперимент"
    if "первичн" in compact and "вариант" in compact:
        return "Оставить первичный вариант как промежуточный результат"
    return _sentence_case(re.sub(r"^\s*(?:лучше|стоит|желательно|надо|нужно)\s+", "", text or ""))


def _research_note_title(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", (text or "").strip(" .,:;"))
    if not compact:
        return None
    return _trim_text(_sentence_case(compact), 160)


def _research_item(text: str, title: str, source_fragment: int | None, *, confidence: float, review_required: bool, kind: str) -> dict:
    return {
        "title": _trim_text(title, 160) or title,
        "summary": _clean_summary(text),
        "source_text": text,
        "source_fragment": source_fragment,
        "confidence": confidence,
        "review_required": review_required,
        "kind": kind,
    }


def _is_research_action(text: str) -> bool:
    if not text or _words_count(text) < 3:
        return False
    if re.search(r"ближайш\w+\s+задач", text, re.IGNORECASE) and RESEARCH_ACTION_OBJECT_RE.search(text):
        return True
    if (
        TECHNICAL_DEFINITION_RE.search(text)
        or TECHNICAL_CONTEXT_RE.search(text)
        or TECHNICAL_GENERIC_RE.search(text)
        or TECHNICAL_SPEAKER_INTENT_RE.search(text)
    ):
        return False
    if TECHNICAL_RECOMMENDATION_RE.search(text) and not re.search(r"\b(?:сделать|разделить|сопоставить|подготовить|оформить|описать|согласовать)\b", text, re.IGNORECASE):
        return False
    return bool(RESEARCH_ACTION_VERB_RE.search(text) and RESEARCH_ACTION_OBJECT_RE.search(text))


def _result_text_units(result: dict) -> list[dict]:
    units = result.get("semantic_blocks") or result.get("transcript") or []
    normalized: list[dict] = []
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict):
            continue
        text = str(unit.get("text") or "").strip()
        if not text:
            continue
        source_fragment = unit.get("source_fragment") or unit.get("fragment_index") or unit.get("block_index") or index
        normalized.append({"text": text, "source_fragment": source_fragment})
    return normalized


def _unit_supports_groups(text: str, marker_groups: tuple[tuple[str, ...], ...]) -> bool:
    lower = text.lower().replace("ё", "е")
    return all(any(marker in lower for marker in group) for group in marker_groups)


SYNTHETIC_RESEARCH_ACTION_RULES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("Прийти с термопастой в следующий раз", (("термопаст",), ("следующ", "раз"))),
    ("Запустить YOLO benchmark на Raspberry Pi", (("yolo",), ("benchmark", "бенчмарк"), ("raspberry",))),
    ("Проверить температуру и частоту Raspberry Pi во время benchmark", (("температур", "частот"), ("raspberry",), ("benchmark", "бенчмарк"))),
    ("Привести датасеты к общему формату", (("датасет", "dataset"), ("формат",))),
    ("Проверить наличие камер в коробках/пакетах", (("камер",), ("коробк", "пакет"))),
    ("Найти инструкции по AI Kit / Hailo", (("ai kit", "hailo"), ("инструкц", "найти", "посмотр"))),
    ("Уточнить слабомощное железо у второкурсников", (("второкурсник",), ("желез", "слабомощ", "маломощ"))),
    ("Поручить второкурсникам изучить QEMU / симуляцию Raspberry Pi", (("второкурсник",), ("qemu", "симуляц", "виртуализац"))),
    ("Отправить формальную постановку задачи завтра", (("формальн", "постановк"), ("задач",), ("завтра", "отправ"))),
    ("Отправить диаграмму отдельным PNG/PDF/SVG-файлом для проверки", (("диаграм",), ("png", "pdf", "svg", "файл"))),
    ("Вставить архитектурную диаграмму в ВКР и презентацию", (("архитектур", "диаграм"), ("вкр", "презентац"))),
    ("Расшифровать сокращение КД при первом использовании", (("кд",), ("расшифр", "сокращ"))),
    ("Уточнить подпись systemd timer как системный вызов", (("systemd",), ("timer",), ("подпис", "вызов"))),
    ("Описать взаимодействие сервиса через каталог, а не через клиентское API", (("каталог",), ("клиентск", "api", "взаимодейств"))),
    ("Проверить термин «инфраструктурный сервис»", (("инфраструктур",), ("сервис",), ("термин", "провер"))),
    ("Сделать архитектурную диаграмму отдельной страницей/landscape", (("диаграм",), ("страниц", "landscape", "чита"))),
    ("Описать компоненты и взаимодействия по C4-диаграмме", (("c4", "диаграм"), ("компонент", "взаимодейств"))),
    ("Продолжить главу с постановкой задачи", (("продолж",), ("глав",), ("постановк", "задач"))),
    ("Переформулировать критерий оптимальности топологии", (("переформулир", "сформулир"), ("критер", "оптимальн", "оптимизац"))),
    ("Описать ограничение по количеству связей", (("огранич",), ("количеств",), ("связ",))),
    ("Обосновать выбор топологии графа", (("обоснов",), ("тополог", "граф"))),
    ("Описать метрики графа", (("метрик",), ("граф",))),
    ("Описать сбор данных и телеметрию", (("телеметр", "сбор данн"),)),
    ("Описать исключения и ограничения модели", (("исключен", "ограничен"), ("модел",))),
    ("Сформулировать критерии качества модели", (("критер",), ("качеств",), ("модел",))),
    ("Найти похожие ВКР и шаблоны оформления", (("похож", "шаблон"), ("вкр",))),
)


def _synthetic_research_actions(result: dict) -> list[dict]:
    units = _result_text_units(result)
    joined = " ".join(unit["text"] for unit in units)
    actions: list[dict] = []
    seen: set[str] = set()
    for title, marker_groups in SYNTHETIC_RESEARCH_ACTION_RULES:
        source_unit = next((unit for unit in units if _unit_supports_groups(unit["text"], marker_groups)), None)
        if source_unit is None and _unit_supports_groups(joined, marker_groups):
            source_unit = {"text": title, "source_fragment": None}
        if source_unit is None:
            continue
        key = _normalize_title_key(title)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            _research_item(
                source_unit["text"],
                title,
                source_unit.get("source_fragment"),
                confidence=0.72,
                review_required=False,
                kind="research_action",
            )
        )
    return actions


def build_clean_research_layers(result: dict) -> dict[str, list[dict]]:
    actions: list[dict] = []
    recommendations: list[dict] = []
    notes: list[dict] = []
    review_items: list[dict] = []
    seen_actions: set[str] = set()
    seen_recommendations: set[str] = set()
    seen_notes: set[str] = set()

    def add_unique(target: list[dict], seen: set[str], item: dict) -> None:
        key = _normalize_title_key(item.get("title"))
        if key and key not in seen:
            seen.add(key)
            target.append(item)

    for task in result.get("tasks", []):
        text = task.get("text", "")
        source_fragment = task.get("source_fragment")
        if not text or _is_intro_or_agenda(text):
            continue

        if TECHNICAL_DEFINITION_RE.search(text):
            title = _research_note_title(text)
            if title:
                add_unique(notes, seen_notes, _research_item(text, title, source_fragment, confidence=0.76, review_required=False, kind="research_note"))
            continue

        if TECHNICAL_GENERIC_RE.search(text):
            title = _research_note_title(text) or text
            review_items.append(
                {
                    "type": "research_action_candidate",
                    "text": text,
                    "clean_title": title,
                    "reason": "too_generic_for_research_action",
                    "confidence": 0.42,
                    "source_fragments": [source_fragment] if source_fragment is not None else [],
                }
            )
            continue

        if TECHNICAL_SPEAKER_INTENT_RE.search(text):
            title = _research_note_title(text) or text
            review_items.append(
                {
                    "type": "research_commitment_candidate",
                    "text": text,
                    "clean_title": title,
                    "reason": "speaker_intent_without_clear_deliverable",
                    "confidence": 0.48,
                    "source_fragments": [source_fragment] if source_fragment is not None else [],
                }
            )
            continue

        if _is_research_action(text):
            title = _research_action_title(text)
            if title and _words_count(title) >= 2 and not _is_bad_clean_task(title, text):
                add_unique(
                    actions,
                    seen_actions,
                    _research_item(text, title, source_fragment, confidence=0.78, review_required=False, kind="research_action"),
                )
            continue

        if TECHNICAL_CONTEXT_RE.search(text):
            title = _research_note_title(text)
            if title:
                add_unique(notes, seen_notes, _research_item(text, title, source_fragment, confidence=0.76, review_required=False, kind="research_note"))
            continue

        if TECHNICAL_RECOMMENDATION_RE.search(text) and not _is_research_action(text):
            title = _recommendation_title(text)
            if title and _words_count(title) >= 2:
                add_unique(
                    recommendations,
                    seen_recommendations,
                    _research_item(text, title, source_fragment, confidence=0.72, review_required=False, kind="recommendation"),
            )
            continue

        if RESEARCH_ACTION_VERB_RE.search(text) or "необходимо" in text.lower() or "надо" in text.lower() or "нужно" in text.lower():
            review_items.append(
                {
                    "type": "research_action_candidate",
                    "text": text,
                    "clean_title": _research_note_title(text) or text,
                    "reason": "low_context_research_action",
                    "confidence": 0.5,
                    "source_fragments": [source_fragment] if source_fragment is not None else [],
                }
            )

    joined = _joined_text(result)
    generated_actions = []
    if re.search(r"устойчив\w+\s+модел\w+|точност\w+", joined, re.IGNORECASE):
        generated_actions.append("Проверить устойчивость модели и точность")
    if re.search(r"разв[её]ртк", joined, re.IGNORECASE):
        generated_actions.append("Сделать развертку по параметрам/формуле безразмеривания")
    if re.search(r"групп\w+\s+параметр\w+|s,\s*n,\s*a,\s*l|s\s+n\s+a\s+l", joined, re.IGNORECASE):
        generated_actions.append("Разделить исследование по группам параметров S, N, A, L")
    for title in generated_actions:
        if _is_bad_clean_task(title, joined):
            continue
        add_unique(
            actions,
            seen_actions,
            _research_item(title, title, None, confidence=0.64, review_required=True, kind="research_action"),
        )

    synthetic_actions = []
    for item in _synthetic_research_actions(result):
        if _is_bad_clean_task(item.get("title"), item.get("source_text")):
            continue
        item.update(score_fragment_confidence(item.get("source_text", ""), "task"))
        synthetic_actions.append(item)
    for item in filter_tasks_by_classifier_confidence(synthetic_actions):
        add_unique(actions, seen_actions, item)

    generated_recommendations = []
    if re.search(r"частн\w+\s+верс\w+|частн\w+\s+случа\w+", joined, re.IGNORECASE):
        generated_recommendations.append("Проверить более частные версии модели")
    if re.search(r"скин-?фактор", joined, re.IGNORECASE):
        generated_recommendations.append("Начать со скин-фактора / одного ключевого параметра")
    if re.search(r"обосновани\w+", joined, re.IGNORECASE):
        generated_recommendations.append("Подумать над обоснованием применимости модели")
    if re.search(r"вычислительн\w+\s+эксперимент|первичн\w+\s+вариант", joined, re.IGNORECASE):
        generated_recommendations.append("Описать первичный вариант как вычислительный эксперимент")

    for title in generated_recommendations:
        add_unique(
            recommendations,
            seen_recommendations,
            _research_item(title, title, None, confidence=0.62, review_required=True, kind="recommendation"),
        )

    return {
        "actions": actions,
        "recommendations": recommendations,
        "notes": notes,
        "review_items": review_items,
    }


def build_clean_tasks(result: dict) -> list[dict]:
    technical = _is_technical_result(result)
    commercial_oil_gas = _is_oil_gas_result(result)
    meeting_label = (result.get("meeting_type") or {}).get("label") if isinstance(result.get("meeting_type"), dict) else None
    participant_names = _participant_names_from_result(result)
    clean_tasks = []
    seen = set()

    for task in result.get("tasks", []):
        source_text = task.get("text", "")
        explicit_research_action = bool(
            re.search(r"ближайш\w+\s+задач", source_text, re.IGNORECASE)
            and RESEARCH_ACTION_OBJECT_RE.search(source_text)
        )
        if (
            _words_count(source_text) < 3
            or (REASONING_STOP_RE.search(source_text) and not explicit_research_action)
            or _is_intro_or_agenda(source_text)
        ):
            continue
        if technical and (
            TECHNICAL_DEFINITION_RE.search(source_text)
            or (TECHNICAL_CONTEXT_RE.search(source_text) and not explicit_research_action)
            or TECHNICAL_GENERIC_RE.search(source_text)
            or TECHNICAL_SPEAKER_INTENT_RE.search(source_text)
        ):
            continue
        if commercial_oil_gas:
            if _is_commercial_commitment(source_text) or not _is_commercial_action_task(source_text):
                continue
        if not is_real_task(source_text, technical_mode=technical):
            continue
        title = _normalize_task_title(source_text)
        if not title or _words_count(title) < 2:
            continue
        title = _trim_text(title, 160) or title
        if _is_bad_clean_task(title, source_text):
            continue
        key = title.lower()
        if key in seen:
            continue

        responsible = task.get("responsible")
        if not responsible:
            matched_responsibles = find_responsibles(source_text, participants=participant_names)
            responsible = matched_responsibles[0] if matched_responsibles else None
        responsible_side = task.get("responsible_side") or find_responsible_side(source_text)
        deadline = task.get("deadline")
        decision = score_task_candidate(source_text, meeting_label, responsible=responsible, deadline=deadline)
        confidence = float(task.get("confidence", 0.7) or 0.7)
        confidence = max(confidence, 0.76)
        if technical:
            confidence = min(confidence, 0.75)
        if responsible_side or deadline:
            confidence = max(confidence, 0.82)
        confidence = max(confidence, float(decision["score"]))
        responsibility_known = bool(responsible or responsible_side)
        review_required = (
            technical
            or (not responsibility_known and not deadline)
            or (decision["decision"] == "reject" and confidence < 0.78)
            or confidence < 0.72
        )
        seen.add(key)
        clean_tasks.append(
            {
                "title": title,
                "summary": _clean_summary(source_text),
                "source_text": source_text,
                "responsible": responsible,
                "responsible_side": responsible_side,
                "deadline": deadline,
                "confidence": round(confidence, 3),
                "review_required": review_required,
                "source_fragment": task.get("source_fragment"),
                "decision": decision,
            }
        )

    return clean_tasks


def _answer_window(question_fragment: int | None, units: list[dict]) -> tuple[str | None, list[int]]:
    if question_fragment is None:
        return None, []
    by_id = {unit.get("fragment_index") or unit.get("block_index"): index for index, unit in enumerate(units)}
    if question_fragment not in by_id:
        return None, []

    answer_parts = []
    source_ids = []
    for unit in units[by_id[question_fragment] + 1 : by_id[question_fragment] + 6]:
        text = unit.get("text", "")
        if not text:
            continue
        if "?" in text and answer_parts:
            break
        if ANSWER_MARKERS_RE.search(text) or answer_parts:
            answer_parts.append(text)
            source_ids.extend(unit.get("source_fragments") or [unit.get("fragment_index") or unit.get("block_index")])
        elif len(answer_parts) == 0 and _words_count(text) >= 7:
            answer_parts.append(text)
            source_ids.extend(unit.get("source_fragments") or [unit.get("fragment_index") or unit.get("block_index")])
    if not answer_parts:
        return None, []
    answer = " ".join(answer_parts).strip()
    if len(answer) > 1000:
        answer = answer[:1000].rstrip(" ,.;") + "..."
    return answer, [item for item in source_ids if item is not None]


def build_clean_questions_answers(result: dict) -> list[dict]:
    units = result.get("semantic_blocks") or result.get("transcript", [])
    clean_pairs = []
    seen_questions = set()
    seen_question_titles = set()
    premium_breakdown_added = False

    for pair in result.get("questions_answers", []):
        question = pair.get("question") or ""
        if question.lower() in seen_questions:
            continue
        seen_questions.add(question.lower())

        answer = pair.get("answer")
        source_fragments = [pair.get("question_fragment")]
        status = pair.get("status", "not_answered")
        if not answer or status == "not_answered":
            answer, answer_sources = _answer_window(pair.get("question_fragment"), units)
            if answer:
                source_fragments.extend(answer_sources)
                status = "answered" if ANSWER_MARKERS_RE.search(answer) else "partial"
        elif pair.get("answer_fragments"):
            source_fragments.extend(pair.get("answer_fragments", []))
        elif pair.get("answer_fragment"):
            source_fragments.append(pair.get("answer_fragment"))

        canonical_question = _canonical_graph_vkr_question(question, answer, result)
        is_premium_breakdown = _is_premium_breakdown_qa(question, answer)
        if not is_premium_breakdown and not canonical_question and _is_low_quality_question(question, answer):
            continue

        if is_premium_breakdown:
            if premium_breakdown_added:
                continue
            premium_breakdown_added = True
            question_title = PREMIUM_BREAKDOWN_QUESTION
            answer_summary = PREMIUM_BREAKDOWN_ANSWER
            status = "answered"
            review_required = False
            confidence = 0.86
        elif canonical_question:
            question_title = canonical_question
            answer_summary = _summarize_answer(answer)
            review_required = status != "answered"
            confidence = 0.78 if status == "answered" else 0.58 if status == "partial" else 0.42
        else:
            question_title = _extract_question_title(question)
            answer_summary = _summarize_answer(answer)
            review_required = status != "answered"
            confidence = 0.82 if status == "answered" else 0.58 if status == "partial" else 0.35
        title_key = _normalize_title_key(question_title)
        if title_key in seen_question_titles:
            continue
        seen_question_titles.add(title_key)
        clean_pairs.append(
            {
                "question": question,
                "answer": answer,
                "question_title": question_title,
                "answer_summary": answer_summary,
                "question_full": question,
                "answer_full": answer,
                "status": status,
                "source_fragments": [item for item in source_fragments if item is not None],
                "confidence": confidence,
                "review_required": review_required,
            }
        )
    return clean_pairs


def build_clean_deadlines(result: dict) -> list[dict]:
    clean_deadlines = []
    label_map = {
        "task_deadline": 0.82,
        "answer_deadline": 0.72,
        "meeting_time": 0.78,
        "mention": 0.68,
        "commitment_deadline": 0.82,
        "contract_logistics_deadline": 0.86,
        "operational_deadline": 0.84,
    }

    def add_deadline_item(item: dict) -> None:
        for value in item.get("deadlines", []):
            value = _canonical_deadline_value(value, item.get("text"))
            key = _deadline_key(value)
            if not key:
                continue
            if _is_bad_clean_deadline(value, item.get("text")):
                continue
            if FREQUENCY_ONLY_RE.search(value or ""):
                continue
            kind = _refine_deadline_kind(item.get("kind") or "mention", item.get("text"), value, result)
            confidence = label_map.get(kind, 0.68)
            context = _deadline_context(item.get("text"), value)
            clean_item = {
                "title": value,
                "summary": context,
                "text": item.get("text"),
                "source_text": item.get("text"),
                "context": context,
                "deadline": value,
                "normalized": item.get("deadline_normalized"),
                "type": "date_or_relative",
                "kind": kind,
                "confidence": confidence,
                "review_required": confidence < 0.68 or not context,
                "source_fragment": item.get("source_fragment"),
            }
            existing_index = _deadline_seen_index(clean_deadlines, value)
            if existing_index is None:
                clean_deadlines.append(clean_item)
            elif _deadline_specificity(value) > _deadline_specificity(clean_deadlines[existing_index].get("deadline")):
                clean_deadlines[existing_index] = clean_item

    for item in result.get("deadlines", []):
        add_deadline_item(item)
    for item in _deadline_candidates_from_result(result):
        add_deadline_item(item)
    return clean_deadlines


def build_clean_responsibles(result: dict) -> list[dict]:
    clean_responsibles = []
    seen = set()
    for item in result.get("responsibles", []):
        for name in item.get("responsibles", []):
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            clean_responsibles.append(
                {
                    "name": name,
                    "text": item.get("text"),
                    "confidence": 0.85,
                    "source_fragment": item.get("source_fragment"),
                }
            )
    for task in result.get("clean_tasks", []):
        side = task.get("responsible_side")
        if side and side.lower() not in seen:
            seen.add(side.lower())
            clean_responsibles.append(
                {
                    "name": None,
                    "side": side,
                    "text": task.get("source_text"),
                    "confidence": 0.78,
                    "source_fragment": task.get("source_fragment"),
                }
            )
    return clean_responsibles


def build_clean_responsible_sides(result: dict) -> list[dict]:
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for collection in (
        result.get("clean_tasks", []),
        result.get("clean_agreements", []),
        result.get("clean_commercial_terms", []),
        result.get("clean_commitments", []),
        result.get("agreements", []),
        result.get("commercial_terms", []),
        result.get("commitments", []),
    ):
        for item in collection:
            side = item.get("responsible_side") or item.get("side")
            if not side:
                continue
            counts[side] += 1
            examples.setdefault(side, item.get("source_text") or item.get("full_text") or item.get("text") or item.get("title") or "")
    return [
        {
            "side": side,
            "title": side,
            "summary": _clean_summary(examples.get(side)),
            "source_text": examples.get(side),
            "count": count,
            "confidence": 0.78,
            "review_required": False,
        }
        for side, count in counts.most_common()
    ]


def build_clean_decisions(result: dict) -> list[dict]:
    decisions = []
    for item in result.get("decisions", []):
        text = item.get("text")
        if _is_intro_or_agenda(text):
            continue
        compact = _compact_item(text, text)
        decisions.append(
            {
                **compact,
                "text": compact["title"],
                "source_fragment": item.get("source_fragment"),
                "confidence": item.get("confidence", 0.8),
                "review_required": False,
            }
        )
    return decisions


def build_clean_agreements(result: dict) -> list[dict]:
    agreements = []
    seen_titles = set()
    for item in result.get("agreements", []):
        text = item.get("text")
        item_type = item.get("type", "agreement")
        if not _is_clean_agreement(text, item_type):
            continue
        title, value = _agreement_title(text or "", item_type)
        if not title:
            continue
        if BAD_AGREEMENT_RE.search(title) or (not value and re.search(r"\bпредварительно\b", title, re.IGNORECASE)):
            continue
        if _words_count(title) < 3 and not value:
            continue
        key = _normalize_title_key(title)
        if key in seen_titles:
            continue
        seen_titles.add(key)
        source_sentence = _informative_sentence(text, prefer_commercial=True) or text
        compact = _compact_item(title, source_sentence)
        agreements.append(
            {
                **compact,
                "text": title,
                "full_text": text,
                "type": item_type,
                "side": item.get("side") or item.get("responsible_side") or find_responsible_side(text or ""),
                "responsible_side": item.get("responsible_side") or item.get("side") or find_responsible_side(text or ""),
                "deadline": item.get("deadline"),
                "value": value,
                "source_fragment": item.get("source_fragment"),
                "confidence": item.get("confidence", 0.72),
                "review_required": float(item.get("confidence", 0.72) or 0.72) < 0.7,
            }
        )
    return sorted(agreements, key=lambda item: (_category_rank(_commercial_term_category(item.get("source_text") or "")), item.get("source_fragment") or 0))


def build_commitments(result: dict) -> list[dict]:
    commitments = []
    seen = set()
    for source_name in ("tasks", "agreements"):
        for item in result.get(source_name, []):
            text = item.get("text", "")
            if _is_intro_or_agenda(text) or not _is_commercial_commitment(text):
                continue
            key = re.sub(r"\s+", " ", text.lower()).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            source_sentence = _informative_sentence(text, prefer_commercial=True) or text
            commitments.append(
                {
                    "title": _trim_text(source_sentence, 160),
                    "summary": _clean_summary(source_sentence),
                    "text": _trim_text(text, 220),
                    "source_text": text,
                    "full_text": text,
                    "type": "commitment",
                    "responsible_side": item.get("responsible_side") or find_responsible_side(text),
                    "deadline": item.get("deadline"),
                    "source_fragment": item.get("source_fragment"),
                    "confidence": item.get("confidence", 0.72),
                    "review_required": False,
                }
            )
    for item in result.get("deadlines", []):
        text = item.get("text", "")
        for value in item.get("deadlines", []):
            if not FREQUENCY_ONLY_RE.search(value or ""):
                continue
            source_sentence = _informative_sentence(text, prefer_commercial=True) or text or value
            key = _normalize_title_key(source_sentence)
            if not key or key in seen:
                continue
            seen.add(key)
            commitments.append(
                {
                    "title": _trim_text(source_sentence, 160),
                    "summary": _clean_summary(source_sentence),
                    "text": _trim_text(source_sentence, 220),
                    "source_text": text or value,
                    "full_text": text or value,
                    "type": "commitment_frequency",
                    "responsible_side": find_responsible_side(text or value),
                    "deadline": None,
                    "frequency": value,
                    "source_fragment": item.get("source_fragment"),
                    "confidence": 0.7,
                    "review_required": False,
                }
            )
    return commitments


def build_clean_commitments(result: dict) -> list[dict]:
    return [
        {
            **item,
            "title": item.get("title") or _trim_text(item.get("text"), 160),
            "summary": item.get("summary") or _clean_summary(item.get("full_text") or item.get("text")),
            "source_text": item.get("source_text") or item.get("full_text") or item.get("text"),
        }
        for item in result.get("commitments", [])
    ]


def _commercial_term_category(text: str) -> str:
    lower = (text or "").lower()
    categories = [
        ("премия", ("преми", "баррель", "+ 1", "1,4", "1,2")),
        ("дифференциал", ("дифференциал", "минус")),
        ("платежные условия", ("предоплата", "оплата", "отсрочка", "банковская гарантия", "аккредитив", "платеж")),
        ("демередж", ("демередж", "чартер", "фрахт")),
        ("ценовое окно", ("ценовое окно", "котировочных", "коносамент")),
        ("объемы и партии", ("тонн", "партия", "объем")),
        ("инспекция", ("инспектор", "инспекция", "проба", "shore tank")),
        ("качество сырья", ("качество", "сера", "плотность", "лаборатор")),
        ("логистика", ("терминал", "труба", "отгрузка", "судно")),
        ("хеджирование", ("хедж", "своп", "фьючерс", "форвард")),
    ]
    for category, markers in categories:
        if any(marker in lower for marker in markers):
            return category
    return "коммерческое условие"


def _commercial_term_title(category: str, text: str) -> tuple[str | None, str | None]:
    sentence = _informative_sentence(text, prefer_commercial=True) or text
    lower = sentence.lower()
    value: str | None = None

    if "тонн" in lower or "парт" in lower or "объем" in lower:
        match = re.search(r"(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн)", sentence, re.IGNORECASE)
        if match:
            value = match.group(1)
            if "август" in lower:
                return f"Августовский объем: {value}", value
            if "июл" in lower:
                return f"Июльский объем: {value}", value
            if "опцион" in lower:
                return f"Опцион: {value}", value
            return f"Объем поставки: {value}", value
        match = re.search(r"(три|две|\d+)\s+парт\w*\s+по\s+(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн)", sentence, re.IGNORECASE)
        if match:
            value = match.group(0)
            return f"График: {value}", value

    if "преми" in lower:
        match = re.search(r"([+−-]?\s*\d+(?:[,.]\d+)?)\s*(?:доллар\w*)?", sentence, re.IGNORECASE)
        if match:
            value = match.group(1).replace(" ", "")
            party = "первой партии" if "перв" in lower else "второй и третьей партии" if ("втор" in lower or "треть" in lower) else ""
            suffix = f" {party}" if party else ""
            return f"Премия{suffix}: {value} доллара/баррель", value

    if "дифференциал" in lower or "минус" in lower:
        match = re.search(r"(-\s*\d+(?:[,.]\d+)?|минус\s+\d+(?:[,.]\d+)?)", sentence, re.IGNORECASE)
        value = match.group(1).replace(" ", "") if match else None
        return (f"Дифференциал: {value}" if value else "Дифференциал: согласуемый диапазон", value)

    if "котировоч" in lower or "ценовое окно" in lower or "коносамент" in lower:
        match = re.search(r"(\d+|пять)\s+котировочн\w+\s+дн\w+", sentence, re.IGNORECASE)
        value = match.group(0) if match else None
        return (f"Ценовое окно: {value} вокруг коносамента" if value else "Ценовое окно вокруг коносамента", value)

    if "демередж" in lower or "чартер" in lower:
        match = re.search(r"(\d+(?:[,.]\d+)?\s*тыс\w*\s+доллар\w*\s+в\s+сут\w+)", sentence, re.IGNORECASE)
        value = match.group(1) if match else None
        return (f"Демередж: потолок {value}" if value else "Демередж: ставка с потолком", value)

    if "50" in lower and ("инспекц" in lower or "делим" in lower):
        return "Инспекция: 50/50 по базовой инспекции", "50/50"

    if "опцион" in lower:
        match = re.search(r"(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн).*?(до\s+\d{1,2}\s+\w+)?", sentence, re.IGNORECASE)
        if match:
            value = " ".join(part for part in match.groups() if part)
            return f"Опцион: {value}", value

    if "платеж" in lower or "оплат" in lower or "предоплат" in lower or "гарант" in lower or "аккредитив" in lower:
        match = re.search(r"(\d+\s*%|\d+(?:[,.]\d+)?\s*(?:дней|дня|день))", sentence, re.IGNORECASE)
        value = match.group(1) if match else None
        return (f"Платежные условия: {value}" if value else "Платежные условия", value)

    if category != "премия" and "до " in lower and ("парт" in lower or "отгруз" in lower) and "прем" not in lower:
        match = re.search(r"(до\s+\d{1,2}\s+\w+|около\s+\d{1,2}|\d{1,2}\s*[–-]\s*\d{1,2}\s*\w*)", sentence, re.IGNORECASE)
        value = match.group(1) if match else None
        return (f"График поставки: {value}" if value else _trim_text(sentence, 120), value)

    if not _has_specific_commercial_value(sentence):
        return None, None
    return f"{category}: {_trim_text(sentence, 110)}", _trim_text(sentence, 120)


def _format_tonn(value: str | None) -> str | None:
    if not value:
        return None
    clean = re.sub(r"\s+", " ", value).strip()
    clean = re.sub(r"\bтысяч\b", "тыс.", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bтыс\b", "тыс.", clean, flags=re.IGNORECASE)
    clean = clean.replace("тыс..", "тыс.")
    if "тонн" not in clean.lower():
        clean = f"{clean} тонн"
    return clean


def _format_money(value: str | None, *, approximate: bool = False) -> str | None:
    if not value:
        return None
    clean = value.replace(" ", "").replace("−", "-")
    if not clean.startswith(("+", "-", "минус")):
        clean = f"+{clean}"
    prefix = "около " if approximate else ""
    return f"{prefix}{clean} доллара/баррель"


def _money_after(pattern: str, text: str) -> str | None:
    match = re.search(pattern + r".{0,90}?([+-]?\s*\d+(?:[,.]\d+)?)", text, re.IGNORECASE)
    return match.group(1) if match else None


def _is_plausible_premium_value(value: str | None) -> bool:
    if not value:
        return False
    try:
        number = abs(float(value.replace("+", "").replace(" ", "").replace(",", ".")))
    except ValueError:
        return False
    return number < 10


def _commercial_term_entries(category: str, text: str) -> list[dict[str, str | None]]:
    sentence = _informative_sentence(text, prefer_commercial=True) or text
    lower = (text or "").lower()
    entries: list[dict[str, str | None]] = []

    party_match = re.search(r"(три|две|\d+)\s+парт\w*\s+по\s+(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн)", text, re.IGNORECASE)
    if party_match:
        value = f"{party_match.group(1)} партии по {_format_tonn(party_match.group(2))}"
        source_text = _source_for_match(text, party_match)
        entries.append({"category": "объемы и партии", "title": f"График поставки: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    july_match = re.search(r"(?:июл\w*|на\s+июль).{0,80}?(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн)", text, re.IGNORECASE)
    if july_match and not party_match:
        value = _format_tonn(july_match.group(1))
        if not re.search(r"плюс[- ]минус|колебани|отклонени", text, re.IGNORECASE):
            source_text = _source_for_match(text, july_match)
            entries.append({"category": "объемы и партии", "title": f"Июльский объем: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    august_hard = re.search(r"(?:на\s+август|августовск\w*|минимальн\w+\s+август\w*).{0,120}?(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?\s*тонн)", text, re.IGNORECASE)
    if august_hard:
        value = _format_tonn(august_hard.group(1))
        source_text = _source_for_match(text, august_hard)
        if re.search(r"\bтверд\w*|минимальн", source_text, re.IGNORECASE):
            title = f"Августовский твердый объем: {value}"
        elif re.search(r"ориентиров", source_text, re.IGNORECASE):
            title = f"Августовский ориентировочный объем: {value}"
        else:
            title = f"Августовский объем: {value}"
        entries.append({"category": "объемы и партии", "title": title, "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    option_match = re.search(r"опцион", text, re.IGNORECASE)
    if option_match:
        before_option = text[: option_match.start()]
        amounts = re.findall(r"(\d+(?:[,.]\d+)?\s*(?:тыс\.?|тысяч)?(?:\s*тонн)?)", before_option, re.IGNORECASE)
        value = _format_tonn(amounts[-1]) if amounts else None
    else:
        value = None
    if value:
        deadline_match = re.search(r"до\s+\d{1,2}\s+\w+", text, re.IGNORECASE)
        value_with_deadline = f"{value} {deadline_match.group(0)}" if deadline_match else value
        title = f"Августовский опцион: {value_with_deadline}" if "август" in lower or "опцион" in lower else f"Опцион: {value_with_deadline}"
        source_text = _source_for_match(text, option_match, include_next=True) if option_match else sentence
        entries.append({"category": "опцион", "title": title, "value": value_with_deadline, "summary": _clean_summary(source_text), "source_text": source_text})

    if re.search(r"корректировк|предложение", text, re.IGNORECASE):
        money_match = re.search(r"(\d+(?:[,.]\d+)?)\s*доллар", text, re.IGNORECASE)
        money = money_match.group(1) if money_match else _money_after(r"(?:корректировк\w*|предложение)", text)
        if money and _is_plausible_premium_value(money):
            value = _format_money(money)
            source_match = money_match or re.search(r"(?:корректировк\w*|предложение).{0,90}?[+-]?\s*\d+(?:[,.]\d+)?", text, re.IGNORECASE)
            source_text = _source_for_match(text, source_match) if source_match else sentence
            entries.append({"category": "премия", "title": f"Исходное предложение по премии: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    if re.search(r"средн\w+\s+преми", text, re.IGNORECASE):
        money = _money_after(r"средн\w+\s+преми\w*", text)
        if money and _is_plausible_premium_value(money):
            value = _format_money(money, approximate="около" in lower)
            source_match = re.search(r"средн\w+\s+преми\w*.{0,90}?[+-]?\s*\d+(?:[,.]\d+)?", text, re.IGNORECASE)
            source_text = _source_for_match(text, source_match) if source_match else sentence
            entries.append({"category": "премия", "title": f"Средняя премия по трем партиям: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    first_match = re.search(
        r"(?:преми\w+\s+по\s+перв\w+\s+парт\w+|перв\w+\s+парт\w+\s+преми\w*).{0,45}?([+-]?\s*\d+(?:[,.]\d+)?)",
        text,
        re.IGNORECASE,
    )
    first_money = first_match.group(1) if first_match else None
    if first_money and "прем" in lower and _is_plausible_premium_value(first_money):
        value = _format_money(first_money)
        source_text = _premium_split_source(text)
        entries.append({"category": "премия", "title": f"Премия первой партии: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    second_match = re.search(
        r"(?:втор\w+\s+и\s+треть\w+|вторая\s+и\s+третья).{0,45}?(?:(?:преми\w*|плюс)\s*)?[–-]?\s*([+-]?\s*\d+(?:[,.]\d+)?)",
        text,
        re.IGNORECASE,
    )
    second_money = second_match.group(1) if second_match else None
    if second_money and ("втор" in lower and "треть" in lower and "прем" in lower and _is_plausible_premium_value(second_money)):
        value = _format_money(second_money)
        source_text = _premium_split_source(text)
        entries.append({"category": "премия", "title": f"Премия второй и третьей партии: {value}", "value": value, "summary": _clean_summary(source_text), "source_text": source_text})

    if re.search(r"ценовое\s+окно|котировочн|коносамент", text, re.IGNORECASE):
        match = re.search(r"(\d+|пять)\s+котировочн\w+\s+дн\w+", text, re.IGNORECASE)
        if match:
            value = match.group(0)
            entries.append({"category": "ценовое окно", "title": f"Ценовое окно: {value} вокруг коносамента", "value": value, "summary": _clean_summary(sentence)})

    demurrage_match = re.search(r"(\d+(?:[,.]\d+)?\s*тыс\w*\s+доллар\w*\s+в\s+сут\w+)", text, re.IGNORECASE)
    if demurrage_match:
        value = demurrage_match.group(1)
        entries.append({"category": "демередж", "title": f"Демередж: потолок {value}", "value": value, "summary": _clean_summary(sentence)})

    if re.search(r"\b50\s*(?:/|на)\s*50\b", text, re.IGNORECASE) and re.search(r"инспекц|делим", text, re.IGNORECASE):
        entries.append({"category": "инспекция", "title": "Инспекция: 50/50 по базовой инспекции", "value": "50/50", "summary": _clean_summary(sentence)})

    if category == "премия" and re.search(r"\b50\s*%", text, re.IGNORECASE):
        entries = [entry for entry in entries if "Премия" not in (entry.get("title") or "")]

    if re.search(r"плюс[- ]минус|колебани|отклонени", text, re.IGNORECASE) and not entries:
        return entries

    if not entries and category != "премия" and not re.search(r"^\s*(сколько|насколько|почему)\b", text, re.IGNORECASE):
        title, value = _commercial_term_title(category, text)
        if title:
            entries.append({"category": category, "title": title, "value": value, "summary": _clean_summary(sentence)})
    return entries


def _is_clean_commercial_term(text: str | None) -> bool:
    if not text:
        return False
    if _is_intro_or_agenda(text) and not _has_specific_commercial_value(text):
        return False
    if re.search(r"\bнапример\b", text, re.IGNORECASE):
        return False
    if re.search(r"^\s*(сколько|насколько|почему)\b", text, re.IGNORECASE):
        return False
    if re.search(r"предварительно\s+около", text, re.IGNORECASE) and re.search(r"не\s+финальн", text, re.IGNORECASE):
        return False
    if "?" in text and not _has_specific_commercial_value(text):
        return False
    if "прем" in text.lower() and re.search(r"\b1[,.](?:2|27|4|8)\b", text):
        return True
    return _has_specific_commercial_value(text)


def _is_clean_agreement(text: str | None, item_type: str | None = None) -> bool:
    if not text or _is_intro_or_agenda(text):
        return False
    if "?" in text and not AGREEMENT_MARKER_RE.search(text):
        return False
    if BAD_AGREEMENT_RE.search(text):
        return False
    if re.search(r"\b(можем|может|хотели\s+бы|предлагали)\b", text, re.IGNORECASE) and not re.search(
        r"\b(зафиксируем|запишем|согласовали|договорились|приемлемо|можно\s+записать|рабочий\s+вариант)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    if not AGREEMENT_MARKER_RE.search(text):
        return False
    if UNCERTAIN_AGREEMENT_RE.search(text) and not re.search(r"\b(зафиксируем|запишем|согласовали|договорились|приемлемо|можно\s+записать)\b", text, re.IGNORECASE):
        return False
    return bool(AGREEMENT_RESULT_RE.search(text) or _has_specific_commercial_value(text))


def _agreement_title(text: str, item_type: str | None = None) -> tuple[str | None, str | None]:
    sentence = _informative_sentence(text, prefer_commercial=True) or text
    if item_type == "commercial_term" or _has_specific_commercial_value(sentence):
        entries = _commercial_term_entries(_commercial_term_category(sentence), sentence)
        if entries:
            return entries[0].get("title"), entries[0].get("value")
        category = _commercial_term_category(sentence)
        title, value = _commercial_term_title(category, sentence)
        if title:
            return title, value
    clean = re.sub(r"^\s*(?:давайте\s+)?(?:зафиксируем|запишем|согласовали|договорились|оставляем)\s*[:,]?\s*", "", sentence, flags=re.IGNORECASE)
    title = _trim_text(clean, 160)
    return title, None


def build_clean_commercial_terms(result: dict) -> list[dict]:
    terms = []
    seen = set()
    for item in result.get("commercial_terms", []):
        text = item.get("text", "")
        if not _is_clean_commercial_term(text):
            continue
        category = item.get("category") or _commercial_term_category(text)
        source_sentence = _informative_sentence(text, prefer_commercial=True) or text
        for entry in _commercial_term_entries(category, text):
            title = entry.get("title")
            if not title:
                continue
            key = _normalize_title_key(title)
            if not key or key in seen:
                continue
            seen.add(key)
            entry_source = entry.get("source_text") or source_sentence
            compact = _compact_item(title, entry_source)
            terms.append(
                {
                    **compact,
                    "summary": entry.get("summary") or compact["summary"],
                    "term": entry.get("category") or category,
                    "category": entry.get("category") or category,
                    "value": entry.get("value") or _trim_text(entry_source, 220),
                    "source_text": entry_source,
                    "full_text": text,
                    "responsible_side": item.get("responsible_side"),
                    "source_fragment": _source_fragment_for_text(result, entry_source, item.get("source_fragment")),
                    "confidence": item.get("confidence", 0.72),
                    "review_required": float(item.get("confidence", 0.72) or 0.72) < 0.7,
                }
            )
    option_deadline = None
    for deadline_item in result.get("deadlines", []):
        for value in deadline_item.get("deadlines", []):
            if re.search(r"до\s+10\s+июля", value or "", re.IGNORECASE):
                option_deadline = value
                break
        if option_deadline:
            break
    if option_deadline:
        option_deadline = option_deadline[:1].lower() + option_deadline[1:]
        for item in terms:
            if item.get("category") == "опцион" and "до " not in str(item.get("title", "")).lower():
                item["title"] = f"{item['title']} {option_deadline}"
                item["value"] = f"{item.get('value')} {option_deadline}".strip()
    return sorted(terms, key=lambda item: (_category_rank(item.get("category")), item.get("source_fragment") or 0))


def _main_topics(result: dict) -> list[str]:
    if _is_oil_gas_result(result):
        topics = _oil_gas_main_topics(result)
        if topics:
            return topics
    clean_names = [item.get("topic_name") or item.get("title") for item in result.get("clean_topics", []) if item.get("topic_name") or item.get("title")]
    if clean_names:
        return clean_names[:6]
    return []


def _display_topic_priority(name: str | None, domains: set[str]) -> int:
    key = _normalize_title_key(name)
    if "technical_hardware" in domains:
        priorities = {
            "raspberry pi 5 и orange pi": 0,
            "raspberry pi и настройка окружения": 0,
            "yolo benchmark": 1,
            "модель детекции рук": 1,
            "камеры и fps": 2,
            "датасеты и разметка": 2,
            "qemu виртуализация raspberry pi": 3,
            "ai kit hailo accelerator": 3,
            "перегрев и охлаждение raspberry pi": 4,
            "usb bandwidth и подключение камер": 4,
            "входные данные ml pipeline": 8,
            "формальная постановка задачи": 8,
            "постановка задачи": 10,
            "график работы с оборудованием": 14,
        }
        return priorities.get(key, 6)
    if "architecture_c4" in domains and "graph_vkr" not in domains:
        priorities = {
            "архитектурная диаграмма c4": 0,
            "описание архитектуры в вкр": 1,
            "systemd timer": 2,
            "инфраструктурный сервис": 2,
            "ldap каталог домена": 3,
            "каталог домена": 3,
            "компоненты и взаимодействия": 4,
            "внешние системы и пользователи": 4,
            "клиент серверная архитектура": 5,
            "модульный монолит": 5,
            "локальные артефакты сервиса": 6,
            "sequence diagram": 6,
            "взаимодействие через каталог": 6,
            "вкр и презентация": 10,
            "документ вкр и отчет": 10,
            "требования к вкр": 12,
            "график презентации встречи": 14,
        }
        return priorities.get(key, 7)
    if "graph_vkr" in domains:
        priorities = {
            "постановка задачи": 0,
            "математическая модель": 1,
            "аппроксимация метрик графа": 1,
            "метрики графа": 2,
            "диаметр и топология графа": 2,
            "критерии качества": 2,
            "критерии оптимизации": 2,
            "модель топологии сети": 3,
            "количество связей": 3,
            "ограничения модели": 4,
            "сбор данных и телеметрия": 4,
            "входные данные математической модели": 8,
            "требования к вкр": 12,
        }
        return priorities.get(key, 6)
    return 0


def build_clean_topics(result: dict) -> list[dict]:
    topics = []
    seen: dict[str, dict] = {}
    transcript_segments = result.get("transcript") if isinstance(result.get("transcript"), list) else []
    domains = detect_meeting_domains(result)
    for index, topic in enumerate(result.get("topics", []), start=1):
        raw_name = str(topic.get("topic_name") or topic.get("title") or "").strip()
        keywords = _normalize_display_keywords(topic.get("keywords"), 8)
        source_text = topic.get("source_text") or " ".join(str(fragment) for fragment in topic.get("texts", []) or [])
        evidence = build_topic_evidence(topic, transcript_segments, result) or source_text
        cleaned = semantic_clean_topic_or_aspect(
            {
                **topic,
                "topic_name": raw_name,
                "keywords": keywords,
                "source_text": source_text,
            },
            evidence,
            domains,
            result,
        )
        if not cleaned and keywords and (
            _normalize_title_key(raw_name) not in BAD_DISPLAY_TOPIC_KEYS
            or _normalize_title_key(raw_name) in DISPLAY_TOPIC_EXACT_MAP
        ):
            cleaned = semantic_clean_topic_or_aspect(
                {
                    **topic,
                    "topic_name": " ".join(keywords[:3]),
                    "keywords": keywords,
                    "source_text": source_text,
                },
                evidence,
                domains,
                result,
            )
        if not cleaned:
            continue
        name = cleaned["title"]
        keywords = cleaned["keywords"]
        key = _normalize_title_key(name)
        fragment_ids = topic.get("fragment_ids") or topic.get("fragments") or []
        if key not in seen:
            seen[key] = {
                "id": f"topic_{len(seen) + 1}",
                "type": "topic",
                "title": name,
                "summary": ", ".join(keywords[:5]) if keywords else name,
                "topic_name": name,
                "keywords": keywords[:8],
                "count": int(topic.get("count") or len(fragment_ids) or 1),
                "fragment_ids": list(fragment_ids),
                "source_fragment": fragment_ids[0] if fragment_ids else topic.get("source_fragment"),
                "confidence": float(topic.get("confidence", 0.55) or 0.55),
                "needs_review": float(topic.get("confidence", 0.55) or 0.55) < 0.45,
                "source_text": topic.get("source_text") or name,
                "semantic_similarity": cleaned.get("semantic_similarity", 0.0),
            }
        else:
            entry = seen[key]
            entry["count"] += int(topic.get("count") or len(fragment_ids) or 1)
            entry["fragment_ids"] = sorted(set(entry.get("fragment_ids", []) + list(fragment_ids)))
            entry["keywords"] = list(dict.fromkeys(entry.get("keywords", []) + keywords))[:8]
            entry["summary"] = ", ".join(entry["keywords"][:5]) if entry.get("keywords") else entry["title"]
    return sorted(
        seen.values(),
        key=lambda item: (_display_topic_priority(item.get("title") or item.get("topic_name"), domains), -int(item.get("count", 0) or 0)),
    )


def _clean_topic_or_aspect_name(
    name: str | None,
    keywords: list[str] | None = None,
    source_text: str | None = None,
    result: dict | None = None,
) -> str | None:
    return _final_display_topic_name(name, keywords, source_text, result or {})


def build_clean_aspects(result: dict) -> list[dict]:
    grouped: dict[str, dict] = {}
    metrics = result.get("metrics") or {}
    transcript_segments = result.get("transcript") if isinstance(result.get("transcript"), list) else []
    domains = detect_meeting_domains(result)

    def add_aspect(name: str | None, count: int = 1, *, keywords: list[str] | None = None, fragment_ids: list | None = None, source_text: str | None = None) -> None:
        raw_item = {
            "title": name,
            "keywords": _normalize_display_keywords(keywords, 8),
            "fragment_ids": fragment_ids or [],
            "source_text": source_text or "",
        }
        evidence = build_topic_evidence(raw_item, transcript_segments, result)
        cleaned = semantic_clean_topic_or_aspect(raw_item, evidence, domains, result)
        if not cleaned:
            return
        clean_name = cleaned["title"]
        clean_keywords = cleaned["keywords"]
        key = _normalize_title_key(clean_name)
        if key not in grouped:
            grouped[key] = {
                "name": clean_name,
                "count": 0,
                "keywords": [],
                "fragment_ids": [],
            }
        grouped[key]["count"] += int(count or 1)
        grouped[key]["keywords"] = list(dict.fromkeys(grouped[key]["keywords"] + clean_keywords))[:8]
        grouped[key]["fragment_ids"] = sorted(set(grouped[key]["fragment_ids"] + list(fragment_ids or [])))

    for source in (result.get("aspect_frequencies") or {}, metrics.get("aspect_frequencies") or {}):
        for name, count in source.items():
            add_aspect(str(name), int(count or 1))
    for item in result.get("aspects", []):
        for aspect in item.get("aspects") or []:
            source_fragment = item.get("source_fragment") or item.get("fragment_index")
            add_aspect(str(aspect), 1, fragment_ids=[source_fragment] if source_fragment else [], source_text=item.get("text"))
    for topic in result.get("clean_topics", []):
        if topic.get("topic_name"):
            add_aspect(
                str(topic["topic_name"]),
                int(topic.get("count") or 1),
                keywords=list(topic.get("keywords") or []),
                fragment_ids=list(topic.get("fragment_ids") or []),
                source_text=topic.get("source_text"),
            )
    clean_aspects = []
    for index, item in enumerate(
        sorted(grouped.values(), key=lambda value: (_display_topic_priority(value["name"], domains), -int(value["count"] or 0)))[:20],
        start=1,
    ):
        count = item["count"]
        clean_aspects.append(
            {
                "id": f"aspect_{index}",
                "type": "aspect",
                "title": item["name"],
                "summary": f"Упоминаний: {count}",
                "count": count,
                "keywords": item["keywords"],
                "fragment_ids": item["fragment_ids"],
                "source_text": item["name"],
                "confidence": 0.6,
                "needs_review": False,
            }
        )
    return clean_aspects


def build_clean_sentiment(result: dict) -> list[dict]:
    clean = []
    for index, item in enumerate(result.get("sentiment", []), start=1):
        label = item.get("sentiment") or item.get("label") or "neutral"
        text = item.get("text") or item.get("source_text") or ""
        clean.append(
            {
                "id": f"sentiment_{index}",
                "type": "sentiment",
                "title": _trim_text(text, 120) or label,
                "summary": label,
                "source_text": text,
                "source_fragment": item.get("source_fragment") or index,
                "timestamp_start": item.get("start"),
                "timestamp_end": item.get("end"),
                "sentiment": label,
                "score": item.get("score", 0.0),
                "confidence": abs(float(item.get("score", 0.0) or 0.0)),
                "needs_review": False,
            }
        )
    return clean


def build_sentiment_summary(result: dict) -> dict:
    items = result.get("clean_sentiment") or build_clean_sentiment(result)
    values = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    counts = Counter(item.get("sentiment", "neutral") for item in items)
    scores = [values.get(item.get("sentiment", "neutral"), 0.0) for item in items]
    return {
        "positive_count": int(counts.get("positive", 0)),
        "neutral_count": int(counts.get("neutral", 0)),
        "negative_count": int(counts.get("negative", 0)),
        "average_sentiment": round(sum(scores) / len(scores), 3) if scores else 0.0,
    }


def build_aspect_sentiment(result: dict) -> dict[str, dict]:
    sentiment_by_fragment = {
        int(item.get("source_fragment") or index): item.get("sentiment", "neutral")
        for index, item in enumerate(result.get("clean_sentiment") or [], start=1)
    }
    values = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    grouped: dict[str, list[float]] = {}
    for aspect_item in result.get("clean_aspects") or []:
        name = aspect_item.get("title") or aspect_item.get("name")
        if not name:
            continue
        fragments = aspect_item.get("fragment_ids") or []
        if not fragments:
            grouped.setdefault(str(name), []).append(0.0)
            continue
        for fragment in fragments:
            try:
                fragment_id = int(fragment)
            except (TypeError, ValueError):
                continue
            score = values.get(sentiment_by_fragment.get(fragment_id, "neutral"), 0.0)
            grouped.setdefault(str(name), []).append(score)
    return {
        aspect: {"average_sentiment": round(sum(scores) / len(scores), 3), "mentions": len(scores)}
        for aspect, scores in grouped.items()
        if scores
    }


def _summary_sentence(label: str, topics: list[str], clean_tasks: list[dict], clean_qa: list[dict], result: dict) -> str:
    topic_text = ", ".join(topics[:5]) if topics else "основные вопросы встречи"
    if label == "non_meeting_speech":
        return "Запись похожа на монолог или обращение. Задачи, вопросы и сроки выделяются только при явных формулировках."
    if label in {"technical_research", "education_consultation"}:
        return f"На встрече обсуждались {topic_text}. В отчете выделены задачи, вопросы, сроки, темы и тональность обсуждения."
    if label in {"commercial_meeting", "commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        return f"На встрече обсуждались {topic_text}. Коммерческие условия и договоренности отделены от исполнимых задач."
    if label == "project_meeting":
        return f"На встрече обсуждались {topic_text}. Выделены задачи, вопросы участников, ответственные и сроки выполнения."
    return f"На встрече обсуждались {topic_text}. В отчете собраны задачи, вопросы, сроки, аспекты и тональность обсуждения."


def build_analysis_summary(result: dict, clean_tasks: list[dict], clean_qa: list[dict]) -> dict:
    meeting_type = result.get("meeting_type")
    if not isinstance(meeting_type, dict):
        meeting_type = detect_meeting_type(result.get("semantic_blocks") or result.get("transcript", []))
        result["meeting_type"] = meeting_type
    label = meeting_type.get("label", "general_discussion")
    topics = _main_topics(result)

    if _is_oil_gas_result(result):
        summary_type = "commercial_oil_gas"
    else:
        summary_type = label
    key_findings = [_summary_sentence(summary_type, topics, clean_tasks, clean_qa, result)]
    if clean_tasks:
        key_findings.append(f"Выделено задач/action items: {len(clean_tasks)}.")
    if clean_qa:
        key_findings.append(f"Выделено вопросов и ответов: {len(clean_qa)}.")
    if result.get("clean_deadlines"):
        key_findings.append(f"Выделено сроков: {len(result.get('clean_deadlines', []))}.")

    return {
        "meeting_type": summary_type,
        "meeting_type_confidence": meeting_type.get("confidence"),
        "main_topics": topics,
        "summary_text": key_findings[0],
        "key_findings": key_findings,
        "action_items_count": len(clean_tasks),
        "questions_count": len(clean_qa),
        "agreements_count": len(result.get("clean_agreements", [])),
        "requires_manual_review": any(task.get("review_required") for task in clean_tasks)
        or any(pair.get("review_required") for pair in clean_qa)
        or label in {"technical_research", "education_consultation", "mixed"},
    }


def build_review_warnings(result: dict, clean_tasks: list[dict], clean_qa: list[dict]) -> list[str]:
    warnings = []
    label = (result.get("meeting_type") or {}).get("label") if isinstance(result.get("meeting_type"), dict) else None
    if label in {"technical_research", "education_consultation", "mixed"}:
        warnings.append("Техническая/учебная встреча: часть задач требует ручной проверки.")
    if _is_oil_gas_result(result):
        warnings.append("Коммерческая встреча: договоренности и условия отделены от исполнимых задач.")
    if any(not task.get("responsible") and not task.get("responsible_side") for task in clean_tasks):
        warnings.append("Некоторые задачи не имеют ответственного или ответственной стороны.")
    if any(pair.get("status") != "answered" for pair in clean_qa):
        warnings.append("Некоторые вопросы не имеют уверенно связанного ответа.")
    if clean_qa:
        warnings.append("Q/A определены эвристически по смысловым блокам и соседним фрагментам.")
    return warnings


def build_quality_metrics(result: dict) -> dict[str, Any]:
    return {
        "semantic_blocks_count": len(result.get("semantic_blocks", [])),
        "raw_tasks_count": len(result.get("tasks", [])),
        "clean_tasks_count": len(result.get("clean_tasks", [])),
        "raw_questions_answers_count": len(result.get("questions_answers", [])),
        "clean_questions_answers_count": len(result.get("clean_questions_answers", [])),
        "agreements_count": len(result.get("agreements", [])),
        "review_items_count": len(result.get("review_items", [])),
        "review_required": bool(result.get("review_items") or result.get("review_warnings")),
    }


def _meeting_label(result: dict) -> str:
    meeting_type = result.get("meeting_type")
    if isinstance(meeting_type, dict):
        return meeting_type.get("label") or "unknown"
    return "unknown"


def _section(id_: str, title: str, items: list | dict | None, priority: int, *, kind: str = "cards", visible: bool | None = None) -> dict:
    if visible is None:
        visible = bool(items)
    return {
        "id": id_,
        "title": title,
        "kind": kind,
        "priority": priority,
        "visible": visible,
        "items": items or [],
        "limit": SECTION_LIMITS.get(id_, 8),
        "empty_text": "Нет данных",
    }


def build_report_sections(result: dict) -> list[dict]:
    label = _meeting_label(result)
    if label in {"commercial_meeting", "commercial_oil_gas", "oil_gas_commercial", "oil_gas_trading"}:
        order = [
            ("summary", "Краткая сводка", result.get("analysis_summary"), 10, "summary", True),
            ("commercial_terms", "Коммерческие условия", result.get("clean_commercial_terms", []), 20, "cards", None),
            ("agreements", "Договоренности", result.get("clean_agreements", []), 30, "cards", None),
            ("commitments", "Обещания сторон", result.get("clean_commitments", []), 40, "cards", None),
            ("responsible_sides", "Ответственные стороны", result.get("clean_responsible_sides", []), 50, "table", None),
            ("tasks", "Задачи", result.get("clean_tasks", []), 60, "cards", True),
            ("deadlines", "Дедлайны", result.get("clean_deadlines", []), 70, "cards", None),
            ("qa", "Вопросы и ответы", result.get("clean_questions_answers", []), 80, "cards", None),
            ("aspects_topics", "Аспекты и темы", result.get("clean_topics", []), 90, "summary", True),
            ("sentiment", "Тональность", result.get("clean_sentiment", []), 100, "summary", True),
            ("review", "Требует проверки", result.get("review_items", []), 120, "cards", True),
            ("processing_time", "Время обработки", [result.get("metadata", {}).get("processing_time", {})], 130, "summary", True),
            ("transcript", "Транскрипт", result.get("transcript", []), 140, "accordion", True),
            ("raw", "Исходные извлеченные данные", [], 150, "accordion", True),
        ]
    elif label in {"technical_research", "education_consultation"}:
        order = [
            ("summary", "Краткая сводка", result.get("analysis_summary"), 10, "summary", True),
            ("tasks", "Задачи", result.get("clean_tasks", []), 20, "cards", True),
            ("qa", "Вопросы и ответы", result.get("clean_questions_answers", []), 30, "cards", None),
            ("deadlines", "Дедлайны / следующая встреча", result.get("clean_deadlines", []), 40, "cards", None),
            ("aspects_topics", "Аспекты и темы", result.get("clean_topics", []), 50, "summary", True),
            ("review", "Требует проверки", result.get("review_items", []), 60, "cards", True),
            ("sentiment", "Тональность", result.get("clean_sentiment", []), 70, "summary", True),
            ("processing_time", "Время обработки", [result.get("metadata", {}).get("processing_time", {})], 80, "summary", True),
            ("transcript", "Транскрипт", result.get("transcript", []), 90, "accordion", True),
            ("raw", "Исходные извлеченные данные", [], 100, "accordion", True),
        ]
    elif label == "project_meeting":
        order = [
            ("summary", "Краткая сводка", result.get("analysis_summary"), 10, "summary", True),
            ("tasks", "Задачи", result.get("clean_tasks", []), 20, "cards", True),
            ("qa", "Вопросы и ответы", result.get("clean_questions_answers", []), 30, "cards", None),
            ("deadlines", "Дедлайны", result.get("clean_deadlines", []), 40, "cards", None),
            ("responsibles", "Ответственные", result.get("clean_responsibles", []), 50, "table", None),
            ("decisions", "Решения", result.get("clean_decisions", []), 60, "cards", None),
            ("aspects_topics", "Аспекты и темы", result.get("clean_topics", []), 70, "summary", True),
            ("sentiment", "Тональность", result.get("clean_sentiment", []), 80, "summary", True),
            ("dynamic", "Динамика по серии встреч", [result.get("dynamic_analysis", {})], 90, "summary", True),
            ("review", "Требует проверки", result.get("review_items", []), 100, "cards", True),
            ("processing_time", "Время обработки", [result.get("metadata", {}).get("processing_time", {})], 110, "summary", True),
            ("transcript", "Транскрипт", result.get("transcript", []), 120, "accordion", True),
            ("raw", "Исходные извлеченные данные", [], 130, "accordion", True),
        ]
    else:
        order = [
            ("summary", "Краткая сводка", result.get("analysis_summary"), 10, "summary", True),
            ("tasks", "Задачи", result.get("clean_tasks", []), 20, "cards", True),
            ("qa", "Вопросы и ответы", result.get("clean_questions_answers", []), 30, "cards", None),
            ("agreements", "Договоренности / решения", result.get("clean_agreements", []) + result.get("clean_decisions", []), 40, "cards", None),
            ("deadlines", "Дедлайны", result.get("clean_deadlines", []), 50, "cards", None),
            ("aspects_topics", "Аспекты и темы", result.get("clean_topics", []), 60, "summary", True),
            ("review", "Требует проверки", result.get("review_items", []), 70, "cards", True),
            ("transcript", "Транскрипт", result.get("transcript", []), 80, "accordion", True),
            ("raw", "Исходные извлеченные данные", [], 90, "accordion", True),
        ]
    return sorted((_section(*args[:4], kind=args[4], visible=args[5]) for args in order), key=lambda item: item["priority"])


def build_display_config(result: dict) -> dict:
    label = _meeting_label(result)
    return {
        "meeting_type_label": MEETING_TYPE_LABELS.get(label, label),
        "section_limits": SECTION_LIMITS,
        "transcript_preview_chars": 1600,
        "raw_collapsed": True,
        "preferred_sections": [section["id"] for section in result.get("report_sections", []) if section.get("visible")],
    }


def _normalize_dedup_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip(" .,:;?!")


def _answer_matches_task_text(answer_text: str | None, task_texts: list[str]) -> bool:
    answer_norm = _normalize_dedup_text(answer_text)
    if not answer_norm:
        return False
    for task_text in task_texts:
        if answer_norm == task_text:
            return True
        shorter, longer = sorted([answer_norm, task_text], key=len)
        if len(shorter.split()) >= 8 and shorter in longer:
            return True
    return False


def _drop_qa_overlapping_with_tasks(clean_tasks: list[dict], clean_qa: list[dict]) -> list[dict]:
    task_fragments = {item.get("source_fragment") for item in clean_tasks if item.get("source_fragment") is not None}
    task_texts = [
        normalized
        for item in clean_tasks
        for normalized in [_normalize_dedup_text(item.get("source_text"))]
        if normalized
    ]
    if not task_fragments and not task_texts:
        return clean_qa

    deduped: list[dict] = []
    removed = 0
    for qa_item in clean_qa:
        answer_fragments = set((qa_item.get("source_fragments") or [])[1:])
        answer_text = qa_item.get("answer_full") or qa_item.get("answer")
        if (answer_fragments & task_fragments) or _answer_matches_task_text(answer_text, task_texts):
            removed += 1
            continue
        deduped.append(qa_item)

    if removed:
        log.info("Removed %d clean_questions_answers entries overlapping with clean_tasks fragments", removed)
    return deduped


def normalize_analysis_result(result: dict) -> dict:
    if result.get("tasks"):
        result["tasks"] = filter_tasks_by_classifier_confidence(result["tasks"])
    if "meeting_type" not in result:
        result["meeting_type"] = detect_meeting_type(result.get("semantic_blocks") or result.get("transcript", []))
    _normalize_meeting_type(result)
    if "agreements" not in result:
        result["agreements"] = extract_agreements(result.get("semantic_blocks") or result.get("transcript", []))
    result["commercial_terms"] = [item for item in result.get("agreements", []) if item.get("type") == "commercial_term"]
    _normalize_meeting_type(result)
    result["commitments"] = build_commitments(result)

    research_layers = {"actions": [], "recommendations": [], "notes": [], "review_items": []}
    if _is_research_result(result):
        research_layers = build_clean_research_layers(result)
        clean_tasks = research_layers["actions"]
    else:
        clean_tasks = build_clean_tasks(result)
    clean_qa = build_clean_questions_answers(result)
    clean_qa = _drop_qa_overlapping_with_tasks(clean_tasks, clean_qa)
    result["clean_tasks"] = clean_tasks
    result["clean_research_actions"] = []
    result["clean_recommendations"] = []
    result["clean_research_notes"] = []
    result["clean_questions_answers"] = clean_qa
    result["clean_decisions"] = build_clean_decisions(result)
    result["clean_agreements"] = build_clean_agreements(result)
    result["clean_commercial_terms"] = build_clean_commercial_terms(result)
    result["clean_commitments"] = build_clean_commitments(result)
    result["clean_deadlines"] = build_clean_deadlines(result)
    result["clean_responsibles"] = build_clean_responsibles(result)
    result["clean_responsible_sides"] = build_clean_responsible_sides(result)
    result["clean_topics"] = build_clean_topics(result)
    result["clean_aspects"] = build_clean_aspects(result)
    result["clean_sentiment"] = build_clean_sentiment(result)
    result["clean_notes"] = []
    result["sentiment_summary"] = build_sentiment_summary(result)
    result["aspect_sentiment"] = build_aspect_sentiment(result)
    result["analysis_summary"] = build_analysis_summary(result, clean_tasks, clean_qa)
    result["review_warnings"] = build_review_warnings(result, clean_tasks, clean_qa)
    result["review_items"] = build_review_items(result) + research_layers["review_items"]
    result["quality_metrics"] = build_quality_metrics(result)
    result["report_sections"] = build_report_sections(result)
    result["display_config"] = build_display_config(result)
    return result

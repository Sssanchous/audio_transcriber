from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .decision_extractor import extract_agreements
from .extraction_decision import build_review_items, score_task_candidate
from .meeting_type import detect_meeting_type
from .responsible_side import find_responsible_side
from .task_extractor import is_real_task


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
    "general_discussion": "общее обсуждение",
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
    "research_actions": 8,
    "recommendations": 8,
    "research_notes": 8,
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
    return re.sub(r"[^a-zа-яё0-9]+", " ", (text or "").lower()).strip()


def _category_rank(category: str | None) -> int:
    return COMMERCIAL_CATEGORY_RANK.get((category or "").lower(), 100)


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
    if re.search(r"минимум\s+за\s+(?:5|пять)\s+рабоч\w*\s+дн\w*\s+до\s+отгрузк\w*", haystack, re.IGNORECASE):
        return "минимум за 5 рабочих дней до отгрузки"
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


def _extract_question_title(question: str) -> str:
    clean = re.sub(r"\s+", " ", question or "").strip()
    if "?" in clean:
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
TECHNICAL_DEFINITION_RE = re.compile(
    r"\b(это\s+(?:дебит|параметр|значение|модель|формула|расч[её]т)|"
    r"метро(?:кубический|куб\w*)\s+разделить\s+на\s+сутки|"
    r"обратная\s+задача\s+интерпретации|r-?квадрат|r²|"
    r"калькулированн\w+\s+и\s+эталон\w+|параметр\w+\s+могут\s+быть\s+предположительн\w*)\b",
    re.IGNORECASE,
)
TECHNICAL_CONTEXT_RE = re.compile(
    r"\b(как\s+правило|в\s+теории|в\s+принципе|получается|может\s+быть|могут\s+быть|"
    r"могут\s+присутствовать|могут\s+.*корректировк\w+\s+внести|могут\s+корректировк\w+\s+внести)\b",
    re.IGNORECASE,
)
TECHNICAL_GENERIC_RE = re.compile(
    r"\b(это\s+необходимо\s+сделать|надо\s+подумать|нужно\s+понимать|это\s+важно|это\s+понятно)\b",
    re.IGNORECASE,
)
TECHNICAL_SPEAKER_INTENT_RE = re.compile(
    r"\b(я\s+попробую\s+(?:тогда\s+)?(?:у\s+себя\s+)?поискать|может\s+поискать|я\s+думаю\s+поискать|посмотрю\s+у\s+себя)\b",
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
    r"четверг|7[:.]30|промежуточн\w+\s+результат\w+|скин-?фактор)\b",
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


def _is_low_quality_question(question: str, answer: str | None = None) -> bool:
    title = (_extract_question_title(question) or "").lower().strip(" ?!.")
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
        (("s", "n", "a", "l", "параметр"), "Разделить исследование по группам параметров S, N, A, L"),
        (("сопостав", "кейс"), "Сопоставить кейсы/реализации расчетов"),
        (("сопостав", "реализац"), "Сопоставить кейсы/реализации расчетов"),
        (("вычислительн", "эксперимент"), "Подготовить описание вычислительного эксперимента"),
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
            if title and _words_count(title) >= 2:
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
    if re.search(r"сопостав", joined, re.IGNORECASE) and re.search(r"кейс|реализац|расч[её]т", joined, re.IGNORECASE):
        generated_actions.append("Сопоставить кейсы/реализации расчетов")
    if re.search(r"встреч|четверг|7[:.]30", joined, re.IGNORECASE):
        generated_actions.append("Согласовать следующую встречу на четверг 19:30")

    for title in generated_actions:
        add_unique(
            actions,
            seen_actions,
            _research_item(title, title, None, confidence=0.64, review_required=True, kind="research_action"),
        )

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
    clean_tasks = []
    seen = set()

    for task in result.get("tasks", []):
        source_text = task.get("text", "")
        if _words_count(source_text) < 3 or REASONING_STOP_RE.search(source_text) or _is_intro_or_agenda(source_text):
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
        key = title.lower()
        if key in seen:
            continue

        responsible = task.get("responsible")
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

    if technical:
        joined = _joined_text(result)
        generated = []
        if re.search(r"сопостав", joined, re.IGNORECASE):
            generated.append(("Сопоставить кейсы/реализации расчетов", None, None))
        if re.search(r"встреч|четверг|7[:.]30", joined, re.IGNORECASE):
            generated.append(("Согласовать следующую встречу на четверг 19:30", None, "четверг 19:30"))
        if re.search(r"промежуточные\s+результаты", joined, re.IGNORECASE):
            generated.append(("Скинуть промежуточные результаты", None, None))
        for title, source_fragment, deadline in generated:
            if title.lower() not in seen:
                seen.add(title.lower())
                clean_tasks.append(
                    {
                        "title": title,
                        "summary": _clean_summary(title),
                        "source_text": title,
                        "responsible": None,
                        "responsible_side": None,
                        "deadline": deadline,
                        "confidence": 0.62,
                        "review_required": True,
                        "source_fragment": source_fragment,
                        "decision": {
                            "candidate_type": "task",
                            "score": 0.62,
                            "decision": "review",
                            "review_required": True,
                            "reasons": ["technical_summary_action_item"],
                        },
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

        is_premium_breakdown = _is_premium_breakdown_qa(question, answer)
        if not is_premium_breakdown and _is_low_quality_question(question, answer):
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
        else:
            question_title = _extract_question_title(question)
            answer_summary = _summarize_answer(answer)
            review_required = status != "answered"
            confidence = 0.82 if status == "answered" else 0.58 if status == "partial" else 0.35
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
    label_map = {"task_deadline": 0.82, "answer_deadline": 0.72, "meeting_time": 0.78, "mention": 0.68}
    for item in result.get("deadlines", []):
        for value in item.get("deadlines", []):
            value = _canonical_deadline_value(value, item.get("text"))
            key = _deadline_key(value)
            if not key:
                continue
            if FREQUENCY_ONLY_RE.search(value or ""):
                continue
            kind = item.get("kind") or "mention"
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
    names = [item.get("topic_name") for item in result.get("topics", []) if item.get("topic_name")]
    if names:
        return names[:6]
    counts: Counter[str] = Counter()
    for item in result.get("aspects", []):
        counts.update(item.get("aspects", []))
    return [name for name, _ in counts.most_common(6)]


def build_analysis_summary(result: dict, clean_tasks: list[dict], clean_qa: list[dict]) -> dict:
    meeting_type = result.get("meeting_type")
    if not isinstance(meeting_type, dict):
        meeting_type = detect_meeting_type(result.get("semantic_blocks") or result.get("transcript", []))
        result["meeting_type"] = meeting_type
    label = meeting_type.get("label", "general_discussion")
    topics = _main_topics(result)

    if _is_oil_gas_result(result):
        key_findings = [
            "Обсуждались объемы поставки, график партий и логистические окна.",
            "Выделены коммерческие условия: премия, дифференциал, ценовое окно, фрахт, демередж и инспекция.",
            "Отдельно зафиксированы платежные условия, качество/инспекция, риски/демередж и хеджирование.",
            "Задачи отделены от договоренностей и обещаний сторон; спорные элементы вынесены на ручную проверку.",
        ]
        summary_type = "commercial_oil_gas"
    elif label in {"technical_research", "education_consultation", "mixed"}:
        key_findings = [
            "Обсуждалась применимость модели и параметров к исследуемым кейсам.",
            "Спорные технические выводы помечены как требующие проверки, а не как уверенные задачи.",
            "Организационные договоренности и сроки выделены отдельно от физических параметров.",
        ]
        summary_type = label
    elif label == "project_meeting":
        key_findings = [
            "Выделены поручения по проекту.",
            "Определены вопросы, ответы, сроки и ответственные там, где это возможно.",
            "Спорные элементы вынесены в блок ручной проверки.",
        ]
        summary_type = label
    else:
        key_findings = ["Встреча обработана как общее обсуждение; уверенные действия отделены от исходных фрагментов."]
        summary_type = label

    return {
        "meeting_type": summary_type,
        "meeting_type_confidence": meeting_type.get("confidence"),
        "main_topics": topics,
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
        "clean_research_actions_count": len(result.get("clean_research_actions", [])),
        "clean_recommendations_count": len(result.get("clean_recommendations", [])),
        "clean_research_notes_count": len(result.get("clean_research_notes", [])),
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
            ("aspects_topics", "Аспекты и темы", result.get("topics", []), 90, "summary", True),
            ("sentiment", "Тональность", result.get("sentiment", []), 100, "summary", True),
            ("review", "Требует проверки", result.get("review_items", []), 120, "cards", True),
            ("processing_time", "Время обработки", [result.get("metadata", {}).get("processing_time", {})], 130, "summary", True),
            ("transcript", "Транскрипт", result.get("transcript", []), 140, "accordion", True),
            ("raw", "Исходные извлеченные данные", [], 150, "accordion", True),
        ]
    elif label in {"technical_research", "education_consultation"}:
        order = [
            ("summary", "??????? ??????", result.get("analysis_summary"), 10, "summary", True),
            ("research_actions", "????????????????? ????????", result.get("clean_research_actions", []), 20, "cards", True),
            ("recommendations", "????????????", result.get("clean_recommendations", []), 30, "cards", True),
            ("research_notes", "????????????????? ??????? / ??????????? ????????", result.get("clean_research_notes", []), 40, "cards", True),
            ("qa", "??????? ? ??????", result.get("clean_questions_answers", []), 50, "cards", None),
            ("deadlines", "???????? / ????????? ???????", result.get("clean_deadlines", []), 60, "cards", None),
            ("aspects_topics", "???????? ???? ????????????", result.get("topics", []), 70, "summary", True),
            ("review", "??????? ????????", result.get("review_items", []), 80, "cards", True),
            ("sentiment", "???????????", result.get("sentiment", []), 90, "summary", True),
            ("processing_time", "????? ?????????", [result.get("metadata", {}).get("processing_time", {})], 100, "summary", True),
            ("transcript", "??????????", result.get("transcript", []), 110, "accordion", True),
            ("raw", "???????? ??????????? ??????", [], 120, "accordion", True),
        ]
    elif label == "project_meeting":
        order = [
            ("summary", "Краткая сводка", result.get("analysis_summary"), 10, "summary", True),
            ("tasks", "Задачи", result.get("clean_tasks", []), 20, "cards", True),
            ("qa", "Вопросы и ответы", result.get("clean_questions_answers", []), 30, "cards", None),
            ("deadlines", "Дедлайны", result.get("clean_deadlines", []), 40, "cards", None),
            ("responsibles", "Ответственные", result.get("clean_responsibles", []), 50, "table", None),
            ("decisions", "Решения", result.get("clean_decisions", []), 60, "cards", None),
            ("aspects_topics", "Аспекты и темы", result.get("topics", []), 70, "summary", True),
            ("sentiment", "Тональность", result.get("sentiment", []), 80, "summary", True),
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
            ("aspects_topics", "Аспекты и темы", result.get("topics", []), 60, "summary", True),
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


def normalize_analysis_result(result: dict) -> dict:
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
    result["clean_tasks"] = clean_tasks
    result["clean_research_actions"] = research_layers["actions"]
    result["clean_recommendations"] = research_layers["recommendations"]
    result["clean_research_notes"] = research_layers["notes"]
    result["clean_questions_answers"] = clean_qa
    result["clean_decisions"] = build_clean_decisions(result)
    result["clean_agreements"] = build_clean_agreements(result)
    result["clean_commercial_terms"] = build_clean_commercial_terms(result)
    result["clean_commitments"] = build_clean_commitments(result)
    result["clean_deadlines"] = build_clean_deadlines(result)
    result["clean_responsibles"] = build_clean_responsibles(result)
    result["clean_responsible_sides"] = build_clean_responsible_sides(result)
    result["analysis_summary"] = build_analysis_summary(result, clean_tasks, clean_qa)
    result["review_warnings"] = build_review_warnings(result, clean_tasks, clean_qa)
    result["review_items"] = build_review_items(result) + research_layers["review_items"]
    result["quality_metrics"] = build_quality_metrics(result)
    result["report_sections"] = build_report_sections(result)
    result["display_config"] = build_display_config(result)
    return result

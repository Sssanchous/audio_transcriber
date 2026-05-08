
"""
task_rule_engine.py — жёсткий rule engine для отсечения ложных задач в разговорных транскрипциях.

Идея:
1) сначала быстро отсекаем мусор / обрывки / филлеры
2) затем ищем признаки реальной actionable-задачи
3) затем отдельно выжигаем ложные срабатывания:
   - "задача оптимизации"
   - "надо подумать"
   - "мне кажется"
   - "я хочу описать"
   - "можно оценить"
   - математические / исследовательские обсуждения
4) возвращаем подробный разбор: label, score, reasons

Использование в app.py:
    from task_rule_engine import is_actionable_task, analyze_task_candidate

    analysis = analyze_task_candidate(text)
    if analysis["is_task"]:
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Optional


FILLER_WORDS = {
    "ну", "вот", "типа", "как", "бы", "получается", "короче", "значит",
    "наверное", "наверно", "может", "просто", "как-то", "что-то", "какой-то",
    "какая-то", "какие-то", "то есть", "в принципе", "скажем", "допустим",
}

ACTION_VERBS = [
    "сделай", "подготовь", "проверь", "отправь", "создай", "обнови", "исправь",
    "доделай", "согласуй", "закрой", "напиши", "собери", "заверши", "вынеси",
    "оформи", "выгрузи", "перепроверь", "протестируй", "дополни", "сверь",
    "изучи", "размести", "составь", "подготовить", "проверить", "согласовать",
    "отправить", "закрыть", "обновить", "исправить", "доработать", "создать",
    "вынести", "оформить", "выгрузить", "протестировать", "дополнить",
]

WEAK_MODAL_PATTERNS = [
    r"\bнужно\b", r"\bнадо\b", r"\bнеобходимо\b", r"\bтребуется\b",
    r"\bдолжен\b", r"\bдолжна\b", r"\bдолжны\b", r"\bнужно будет\b", r"\bнадо будет\b",
]

DEADLINE_PATTERNS = [
    r"\bдо\s+\d{1,2}[:.]\d{2}\b",
    r"\bдо\s+\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?\b",
    r"\bдо\s+(понедельника|вторника|среды|четверга|пятницы|субботы|воскресенья)\b",
    r"\bдо\s+(завтра|вечера|утра|обеда|релиза|конца\s+дня|конца\s+недели|следующей\s+встречи)\b",
    r"\bк\s+(утру|вечеру|обеду|созвону|завтрашнему\s+созвону)\b",
    r"\bсегодня\b", r"\bзавтра\b", r"\bна\s+этой\s+неделе\b", r"\bдо\s+пятницы\b",
]

ASSIGNEE_PATTERNS = [
    r"^[А-ЯЁ][а-яё]+,\s+",               # "Анна, подготовь..."
    r"\bза\s+это\s+отвечает\s+[А-ЯЁ][а-яё]+\b",
    r"\b[А-ЯЁ][а-яё]+\s+проверит\b",
    r"\b[А-ЯЁ][а-яё]+\s+сделает\b",
    r"\b[А-ЯЁ][а-яё]+\s+подготовит\b",
    r"\b[А-ЯЁ][а-яё]+\s+возьм[её]т\b",
]

SHORT_BAD_PATTERNS = [
    r"^\s*да\b", r"^\s*нет\b", r"^\s*хорошо\b", r"^\s*ладно\b", r"^\s*окей\b",
    r"^\s*понял\b", r"^\s*поняла\b", r"^\s*принято\b", r"^\s*спасибо\b",
]

FALSE_POSITIVE_PATTERNS = [
    # обсуждение задач как концепции, а не поручения
    r"\bзадача\s+(оптимизации|минимизации|максимизации|активизации|исследования)\b",
    r"\bзадача\s+мст\b",
    r"\bглобальная\s+задача\b",
    r"\bматематическ\w+\s+задач\w*\b",
    r"\bпостановк\w+\s+задач\w*\b",
    r"\bформулиров\w+\s+задач\w*\b",
    r"\bпо\s+сути.{0,12}\bзадач\w*\b",
    r"\bкак\s+бы.{0,12}\bзадач\w*\b",
    r"\bвроде.{0,12}\bзадач\w*\b",
    r"\bэта\s+задача\s+(не\s+)?настолько\b",
    r"\bпохож\w+\s+задач\w*\b",

    # исследовательско-рассуждательный стиль
    r"\bя\s+думаю\b",
    r"\bмне\s+кажется\b",
    r"\bя\s+бы\s+(хотел|не\s+хотел)\b",
    r"\bя\s+хочу\b",
    r"\bя\s+решил\b",
    r"\bя\s+начал\b",
    r"\bя\s+попытался\b",
    r"\bне\s+знаю,\s+как\b",
    r"\bнаверное\b",
    r"\bможет\s+быть\b",
    r"\bможно\s+(как-то|попробовать|оценить|описать|пояснить)\b",
    r"\bнадо\s+подумать\b",
    r"\bнадо\s+поискать\b",
    r"\bнужно\s+просто\b",
    r"\bнужно\s+как-то\b",
    r"\bнадо\s+как-то\b",
    r"\bя\s+смогу\b",
    r"\bне\s+смогу\b",

    # научно-математические / концептуальные обсуждения
    r"\bминимизировать\b",
    r"\bмаксимизировать\b",
    r"\bоптимальност\w*\b",
    r"\bдиаметр\w*\b",
    r"\bграф\w*\b",
    r"\bметрик\w*\b",
    r"\bмодель\w*\b",
    r"\bэвристик\w*\b",
    r"\bалгоритм\w*\b",

    # слишком общие формулировки без адресации
    r"\bдолжна\s+быть\b",
    r"\bдолжно\s+быть\b",
    r"\bхотя\s+бы\b",
    r"\bнужно\s+такое\s+железо\b",
]

QUESTION_LIKE_PATTERNS = [
    r"\?$",
    r"^\s*(кто|что|где|когда|почему|зачем|как|какой|какая|какие|сколько)\b",
    r"\bнужно\s+ли\b", r"\bможно\s+ли\b", r"\bуспеем\s+ли\b",
]

ANSWER_LIKE_PATTERNS = [
    r"^\s*(да|нет|хорошо|ладно|понял|поняла|окей)\b",
    r"^\s*я\s+(сделаю|подготовлю|отправлю|проверю|обновлю|исправлю)\b",
    r"\bготово\b", r"\bфинальная\s+версия\s+уже\s+готова\b",
]


@dataclass
class TaskAnalysis:
    text: str
    normalized_text: str
    is_task: bool
    score: float
    assignee_found: bool
    deadline_found: bool
    action_found: bool
    filler_heavy: bool
    false_positive_hit: Optional[str]
    rejected_by: Optional[str]
    reasons: list[str]


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -–—")


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[А-Яа-яЁёA-Za-z0-9-]+", text.lower())


def is_filler_heavy(text: str, threshold: float = 0.45) -> bool:
    words = word_tokens(text)
    if len(words) < 4:
        return True
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    return (filler_count / max(len(words), 1)) >= threshold


def first_matching_pattern(text: str, patterns: list[str]) -> Optional[str]:
    lower = text.lower()
    for p in patterns:
        if re.search(p, lower):
            return p
    return None


def has_action(text: str) -> bool:
    lower = text.lower()
    return any(v in lower for v in ACTION_VERBS)


def has_deadline(text: str) -> bool:
    return first_matching_pattern(text, DEADLINE_PATTERNS) is not None


def has_assignee(text: str) -> bool:
    for p in ASSIGNEE_PATTERNS:
        if re.search(p, text):
            return True
    return False


def looks_like_question(text: str) -> bool:
    return first_matching_pattern(text, QUESTION_LIKE_PATTERNS) is not None


def looks_like_answer(text: str) -> bool:
    return first_matching_pattern(text, ANSWER_LIKE_PATTERNS) is not None


def contains_weak_modal(text: str) -> bool:
    return first_matching_pattern(text, WEAK_MODAL_PATTERNS) is not None


def analyze_task_candidate(text: str) -> dict:
    raw = text
    text = normalize_text(text)
    lower = text.lower()
    reasons: list[str] = []

    if not text:
        return asdict(TaskAnalysis(raw, "", False, 0.0, False, False, False, False, None, "empty", ["empty"]))

    if first_matching_pattern(text, SHORT_BAD_PATTERNS):
        return asdict(TaskAnalysis(raw, text, False, 0.0, False, False, False, False, None, "short_bad_pattern", ["short response"]))

    if len(text) < 20:
        return asdict(TaskAnalysis(raw, text, False, 0.0, False, False, False, False, None, "too_short", ["too short"]))

    if looks_like_question(text):
        return asdict(TaskAnalysis(raw, text, False, 0.0, False, False, False, False, None, "question_like", ["looks like question"]))

    if looks_like_answer(text):
        return asdict(TaskAnalysis(raw, text, False, 0.0, False, False, False, False, None, "answer_like", ["looks like answer"]))

    filler_heavy = is_filler_heavy(text)
    if filler_heavy:
        reasons.append("filler_heavy")

    fp = first_matching_pattern(text, FALSE_POSITIVE_PATTERNS)
    if fp:
        reasons.append(f"false_positive:{fp}")

    assignee = has_assignee(text)
    deadline = has_deadline(text)
    action = has_action(text)
    weak_modal = contains_weak_modal(text)

    score = 0.0

    if assignee:
        score += 0.40
        reasons.append("assignee")
    if deadline:
        score += 0.25
        reasons.append("deadline")
    if action:
        score += 0.35
        reasons.append("action")
    if weak_modal:
        score += 0.08
        reasons.append("weak_modal")

    # Жёсткие отсеки
    # 1) если сработал false positive и нет ассигни/дедлайна → сразу reject
    if fp and not assignee and not deadline:
        return asdict(TaskAnalysis(raw, text, False, round(score, 4), assignee, deadline, action, filler_heavy, fp, "false_positive_context", reasons))

    # 2) если филлерный текст и нет действия → reject
    if filler_heavy and not action:
        return asdict(TaskAnalysis(raw, text, False, round(score, 4), assignee, deadline, action, filler_heavy, fp, "filler_no_action", reasons))

    # 3) если есть только weak modal, но нет action/assignee/deadline → reject
    if weak_modal and not action and not assignee and not deadline:
        return asdict(TaskAnalysis(raw, text, False, round(score, 4), assignee, deadline, action, filler_heavy, fp, "modal_only", reasons))

    # 4) если есть действие, но это размышление "я хочу/мне кажется/можно..." и нет ассигни/дедлайна → reject
    if fp and action and not assignee and not deadline:
        return asdict(TaskAnalysis(raw, text, False, round(score, 4), assignee, deadline, action, filler_heavy, fp, "thought_not_task", reasons))

    # 5) реальная задача должна иметь:
    #    - действие + (ассигни или дедлайн)
    #    - или императивное действие само по себе в нормальной длине
    imperative_like = bool(re.match(r"^\s*(" + "|".join(sorted(set(ACTION_VERBS), key=len, reverse=True)) + r")\b", lower))
    if imperative_like:
        score += 0.15
        reasons.append("imperative_like")

    is_task = False
    if action and (assignee or deadline):
        is_task = True
    elif imperative_like and len(word_tokens(text)) >= 4 and not fp and not filler_heavy:
        is_task = True
    elif assignee and weak_modal and deadline:
        is_task = True

    # Финальный жёсткий бан:
    if is_task and fp and not assignee and not deadline:
        is_task = False

    return asdict(TaskAnalysis(
        text=raw,
        normalized_text=text,
        is_task=is_task,
        score=round(score, 4),
        assignee_found=assignee,
        deadline_found=deadline,
        action_found=action,
        filler_heavy=filler_heavy,
        false_positive_hit=fp,
        rejected_by=None if is_task else "final_threshold",
        reasons=reasons,
    ))


def is_actionable_task(text: str) -> bool:
    return bool(analyze_task_candidate(text)["is_task"])


if __name__ == "__main__":
    tests = [
        "Анна, протестируй интеграцию с оплатой до среды.",
        "Нужно выгрузить миграции базы данных до 12:00.",
        "ну типа надо это сделать",
        "я думаю надо посмотреть",
        "задача в том что минимизировать граф",
        "надо подумать, как это лучше написать",
        "мне кажется, такую задачу можно было бы попробовать сделать",
        "Вот этот канал, значит, надо использовать",
        "Павел, исправь выгрузку по клиентам до 12:00.",
        "Подготовь список терминов и найденных определений с источниками.",
        "надо будет еще поискать, потому что",
        "должна быть какая-то метода оценки",
    ]

    for t in tests:
        result = analyze_task_candidate(t)
        print("=" * 80)
        print(t)
        print(result)

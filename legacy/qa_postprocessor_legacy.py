from __future__ import annotations

import re


BAD_QUESTION_EXACT = {
    "да",
    "нет",
    "ага",
    "угу",
    "ок",
    "окей",
    "понятно",
    "хорошо",
    "правда",
    "может быть",
    "вы слышали",
    "как красиво",
    "патриоты",
    "мечт",
    "мечтаний",
    "коммунных",
    "как же так",
    "почему я спрашиваю",
    "почему я спрашиваю?",
    "ну и почему он собственно используется",
    "ну и почему он собственно используется?",
    "ну и почему он используется",
    "ну и почему он используется?",
}

QUESTION_PATTERNS = [
    r"^следующий\s+вопрос\b",
    r"^если\s+бы\b",
    r"^расскажи\b",
    r"^поясни\b",
    r"^объясни\b",
    r"^пожелай\b",
    r"^дай\s+совет\b",
    r"^самое\s+яркое\s+воспоминание\b",
    r"^самые\s+яркие\s+воспоминания\b",
    r"^с\s+какими\b",
    r"^о\s+ч[её]м\s+ты\b",
    r"^что\s+ты\s+думаешь\b",
    r"^что\s+ты\s+знаешь\b",
    r"^какое\s+у\s+тебя\s+мнение\b",
    r"^лена,\s+что\b",
    r"\bчто\s+для\s+тебя\b",
    r"\bна\s+что\s+бы\s+ты\b",
    r"\bгде\s+бы\s+ты\b",
    r"\bчто\s+ты\s+имеешь\s+в\s+виду\b",
    r"\bпочему\s+тебя\s+это\s+останавливает\b",
]

SHORT_QUESTIONS = {
    "почему",
    "зачем",
    "как",
    "какой",
    "какая",
    "какое",
    "какие",
    "где",
    "когда",
    "кто",
    "что",
    "где именно",
    "что именно",
    "что это такое",
    "почему дурацкую",
    "почему невозможно",
    "почему тебя это останавливает",
    "не хотят или хотят",
}

TOPIC_STARTERS = [
    r"^следующий\s+вопрос\b",
    r"^самое\s+яркое\s+воспоминание\b",
    r"^самые\s+яркие\s+воспоминания\b",
    r"^если\s+бы\b",
    r"^расскажи\b",
    r"^о\s+ч[её]м\s+ты\b",
    r"^с\s+какими\b",
    r"^что\s+ты\s+думаешь\b",
    r"^пожелай\b",
    r"^дай\s+совет\b",
    r"^так\b",
    r"^так,\s+вот\b",
    r"^так,\s+где\b",
    r"^так\s+еще\b",
    r"^так\s+ещ[её]\b",
    r"^кстати\b",
    r"^сейчас\s+найду\b",
    r"^у\s+меня\s+тут\b",
    r"^вот\s+у\s+вас\b",
    r"^нет\s+чтобы\b",
    r"^ну\s+в\s+чем\s+проблема\b",
    r"^на\s+я\s+так\s+понимаю\b",
    r"^я\s+так\s+понимаю\b",
]

BAD_ANSWER_EXACT = {
    "да",
    "нет",
    "ага",
    "угу",
    "ок",
    "окей",
    "понятно",
    "хорошо",
    "вот",
    "так",
    "супер",
}


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -–—")


def norm(text: str) -> str:
    text = normalize_text(text).lower().replace("ё", "е")
    text = re.sub(r"[^\w\sа-яa-z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_question(text: str) -> bool:
    raw = normalize_text(text)
    low = norm(raw)

    if not raw:
        return False

    if low in BAD_QUESTION_EXACT:
        return False

    if low in SHORT_QUESTIONS:
        return True

    if any(re.search(p, low) for p in QUESTION_PATTERNS):
        return True

    if raw.endswith("?"):
        return bool(re.search(
            r"\b(почему|зачем|как|какой|какая|какое|какие|где|когда|кто|что|сколько|можешь|можете|можем)\b",
            low,
        ))

    return False


def is_topic_starter(text: str) -> bool:
    low = norm(text)
    return any(re.search(p, low) for p in TOPIC_STARTERS)


def is_answer(text: str) -> bool:
    raw = normalize_text(text)
    low = norm(raw)

    if not raw:
        return False

    if low in BAD_ANSWER_EXACT:
        return False

    if is_question(raw):
        return False

    if raw.endswith("?"):
        return False

    return len(low.split()) >= 3


def split_question_and_tail(text: str) -> tuple[str, str]:
    text = normalize_text(text)

    if "?" not in text:
        return text, ""

    q, tail = text.split("?", 1)
    q = normalize_text(q + "?")
    tail = normalize_text(tail)

    return q, tail


def clean_question(text: str) -> str:
    text = normalize_text(text)

    # убираем мусор перед настоящим вопросом
    patterns = [
        r".*?(Следующий\s+вопрос\b.*)",
        r".*?(Если\s+бы\b.*)",
        r".*?(Расскажи\b.*)",
        r".*?(Самое\s+яркое\s+воспоминание\b.*)",
        r".*?(Самые\s+яркие\s+воспоминания\b.*)",
        r".*?(О\s+ч[её]м\s+ты\b.*)",
        r".*?(С\s+какими\b.*)",
        r".*?(Что\s+ты\s+думаешь\b.*)",
        r".*?(Пожелай\b.*)",
    ]

    for p in patterns:
        m = re.match(p, text, flags=re.I)
        if m:
            text = normalize_text(m.group(1))
            break

    return text


def clean_answer(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^(да|нет|ну|ага|угу|вот|хорошо|понятно)[,.! ]+", "", text, flags=re.I)
    return normalize_text(text)


def confidence_for_indexes(classified: list[dict] | None, indexes: list[int]) -> float:
    if not classified:
        return 0.93

    vals = []

    for idx in indexes:
        if 0 <= idx < len(classified):
            vals.append(float(classified[idx].get("confidence", 0.9) or 0.9))

    if not vals:
        return 0.93

    return round(sum(vals) / len(vals), 4)


def dedupe_pairs(pairs: list[dict]) -> list[dict]:
    result = []
    seen = set()

    for pair in pairs:
        q = norm(pair.get("question", ""))
        a = norm(pair.get("answer", ""))

        if not q or not a:
            continue

        if q == a:
            continue

        key = (q[:120], a[:160])

        if key in seen:
            continue

        seen.add(key)
        result.append(pair)

    return result


def build_qa_pairs_global(
    segments: list[dict],
    classified: list[dict] | None = None,
) -> list[dict]:
    pairs: list[dict] = []
    i = 0

    MAX_GAP_FROM_QUESTION = 90
    MAX_CONTEXT_WINDOW = 180
    MAX_ANSWER_WORDS = 38
    MIN_ANSWER_WORDS_BEFORE_NEXT_QUESTION = 18

    while i < len(segments):
        seg = segments[i]
        text = normalize_text(seg.get("text", ""))

        label = ""
        if classified and i < len(classified):
            label = classified[i].get("label", "")

        if not (label == "question" or is_question(text)):
            i += 1
            continue

        question_text = clean_question(text)
        question_text, tail = split_question_and_tail(question_text)

        if not is_question(question_text):
            i += 1
            continue

        answer_parts: list[str] = []
        answer_indexes: list[int] = []

        if tail and is_answer(tail):
            answer_parts.append(clean_answer(tail))
            answer_indexes.append(i)

        q_start = float(seg.get("start", 0.0))
        q_end = float(seg.get("end", 0.0))
        answer_start = None
        answer_end = None

        j = i + 1

        while j < len(segments):
            cand = segments[j]
            cand_text = normalize_text(cand.get("text", ""))
            cand_start = float(cand.get("start", 0.0))
            cand_end = float(cand.get("end", 0.0))

            cand_label = ""
            if classified and j < len(classified):
                cand_label = classified[j].get("label", "")

            gap_from_question = cand_start - q_end
            context_from_question = cand_start - q_start

            if gap_from_question > MAX_GAP_FROM_QUESTION and not answer_parts:
                break

            if context_from_question > MAX_CONTEXT_WINDOW:
                break

            is_new_question = cand_label == "question" or is_question(cand_text)

            if is_new_question:
                current_answer_words = len(" ".join(answer_parts).split())

                # Если уже есть нормальный ответ, новый вопрос завершает текущую пару.
                if current_answer_words >= MIN_ANSWER_WORDS_BEFORE_NEXT_QUESTION:
                    break

                # Если ответ слишком короткий, значит вопрос, скорее всего, был служебным
                # или риторическим. Не создаём плохую пару.
                answer_parts = []
                answer_indexes = []
                break

            if answer_parts and is_topic_starter(cand_text):
                break

            if not is_answer(cand_text):
                j += 1
                continue

            if answer_start is None:
                answer_start = cand_start

            answer_end = cand_end
            answer_parts.append(clean_answer(cand_text))
            answer_indexes.append(j)

            words_count = len(" ".join(answer_parts).split())

            if words_count >= MAX_ANSWER_WORDS:
                break

            j += 1

        answer_text = clean_answer(" ".join(answer_parts))

        if answer_text and len(answer_text.split()) >= 3:
            if answer_start is None:
                answer_start = q_end
            if answer_end is None:
                answer_end = q_end

            conf_indexes = [i] + answer_indexes

            pairs.append({
                "question": question_text,
                "answer": answer_text,
                "question_timecode": f"{float(seg.get('start', 0.0)):.2f} - {float(seg.get('end', 0.0)):.2f} сек.",
                "answer_timecode": f"{answer_start:.2f} - {answer_end:.2f} сек.",
                "confidence": confidence_for_indexes(classified, conf_indexes),
            })

            i = max(j, i + 1)
        else:
            i += 1

    return dedupe_pairs(pairs)
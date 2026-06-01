from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from pathlib import Path

INPUT_PATH = "datasets/prelabeled_clean.jsonl"
TRAIN_PATH = "datasets/train.jsonl"
VAL_PATH = "datasets/val.jsonl"
REVIEW_ANSWERS_PATH = "datasets/review_answers.jsonl"
REVIEW_OTHER_PATH = "datasets/review_other_hard_negatives.jsonl"
SUMMARY_PATH = "datasets/balance_summary.json"

RANDOM_SEED = 42
TRAIN_RATIO = 0.82

MAX_PER_CLASS = 900
MAX_OTHER = 1000
MIN_CLASS_TARGET = 80
MIN_ANSWER_WARN = 80

VALID_LABELS = {"task", "question", "answer", "other"}

random.seed(RANDOM_SEED)

TIMECODE_RE = re.compile(r"^\s*\[\d{2}:\d{2}:\d{2}\]\s*")

SPEAKER_RE = re.compile(
    r"^\s*[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2}(?:\s*\(голос\s*\d+\))?:\s*$"
)

HEADER_RE = re.compile(
    r"^\s*(протокол встречи|ход встречи|ход проведения встречи|повестка|"
    r"на встрече|рассмотренные вопросы|организация встречи|формат:|дата:|"
    r"встреча проходила|участники встречи|присутствовали|результаты встречи|"
    r"итог встречи|тема\s+\d+|следующая встреча|время:|"
    r"поставленные задачи|задачи к следующей встрече|что нужно сделать до следующей встречи|"
    r"в рамках продолжения работ определены следующие задачи|рассматриваемые варианты)\b",
    re.I,
)

SUMMARY_OTHER_STARTS = (
    "обсуждение",
    "демонстрация",
    "рассмотрение",
    "на встрече",
    "в ходе встречи",
    "итог встречи",
    "результаты встречи",
    "по итогам обсуждения",
    "по итогам показа",
    "текущая тема",
    "ход встречи",
    "присутствовали",
    "участники встречи",
    "дата следующей встречи",
    "следующая встреча",
    "требования к",
    "контур",
    "формализация",
    "математическая модель",
    "общее содержание",
    "название главы",
    "вводный абзац",
    "графики и диаграммы",
    "изменения в интерфейсе",
    "по архитектуре проекта",
    "повестка",
)

BAD_SHORT = {
    "да",
    "нет",
    "да?",
    "нет?",
    "что?",
    "где?",
    "когда?",
    "зачем?",
    "почему?",
    "и все?",
    "чего?",
    "как?",
    "а?",
    "ну?",
    "вот",
    "дальше",
    "хорошо",
    "понятно",
    "так",
    "ага",
    "ладно",
}

QUESTION_WORDS = (
    "кто", "что", "где", "когда", "почему", "зачем", "как",
    "какой", "какая", "какие", "сколько", "чей", "чья", "чьи",
    "можно ли", "нужно ли", "успеем ли", "надо ли", "в чём",
    "в чем", "что за", "а что", "куда", "откуда",
)

ANSWER_STRICT_PATTERNS = [
    r"^\s*(да|нет|хорошо|ладно|понял|поняла|понятно|ок|окей)\b",
    r"^\s*(конечно|разумеется|безусловно|точно|верно|именно)\b",
    r"^\s*(согласен|согласна|согласны|принято|принял)\b",
    r"\bя\s+(сделаю|подготовлю|отправлю|проверю|исправлю|обновлю|напишу|посмотрю|разберусь|уточню|свяжусь|перешлю|загружу|доделаю|согласую)\b",
    r"\bмы\s+(сделаем|подготовим|отправим|проверим|обновим|исправим)\b",
    r"\bуже\s+(сделал|сделала|готов|готова|отправил|отправила|написал|написала|загрузил|загрузила|проверил|проверила)\b",
    r"\b(сделано|готово|выполнено|завершено|загружено|отправлено|реализовано)\b",
    r"\b(проверил|проверила|обновил|обновила|исправил|исправила)\b",
    r"\b(подготовил|подготовила|отправил|отправила|написал|написала)\b",
    r"\b(добавил|добавила|создал|создала|загрузил|загрузила)\b",
    r"^\s*(будет сделано|принято к сведению|учту|запомню)\b",
    r"^\s*(хорошо,?\s+сделаю|ладно,?\s+подготовлю)\b",
    r"\b(возьму|беру|берусь)\b",
    r"\bна себя\b",
]

NOT_ANSWER_PATTERNS = [
    r"^\s*протокол\b",
    r"^\s*дата\s*:",
    r"^\s*участники\b",
    r"^\s*тема\b",
    r"^\s*повестка\b",
    r"^\s*время\b.*\d{1,2}:\d{2}",
    r"^\s*место\b",
    r"^\s*формат\b",
    r"^\s*присутствовали\b",
    r"^\s*№\s*\d",
    r"^\d{1,2}\.\d{1,2}\.\d{2,4}",
    r"^\s*итого\b",
    r"^\s*итог встречи\b",
    r"^\s*обсуждение\b",
    r"^\s*демонстрация\b",
    r"^\s*рассмотрение\b",
    r"^\s*результаты встречи\b",
    r"^\s*поставленные задачи\b",
    r"^\s*что нужно сделать\b",
]

TASK_STRONG_STARTS = (
    "подготовить",
    "доработать",
    "исправить",
    "проверить",
    "добавить",
    "реализовать",
    "сделать",
    "согласовать",
    "переписать",
    "оформить",
    "протестировать",
    "посмотреть",
    "уточнить",
    "сформировать",
    "описать",
    "разработать",
    "отобразить",
    "сохранить",
    "сократить",
    "переформулировать",
    "переделать",
    "подписать",
    "найти",
    "дополнить",
    "задокументировать",
    "написать",
    "провести",
    "изучить",
)

TASK_PATTERNS = [
    r"\bнужно\b",
    r"\bнадо\b",
    r"\bнеобходимо\b",
    r"\bтребуется\b",
    r"\bследует\b",
    r"\bнужно будет\b",
    r"\bнадо будет\b",
    r"\bдолжен\b",
    r"\bдолжна\b",
    r"\bдолжны\b",
    r"\bк\s+следующей\s+встрече\b",
    r"\bдо\s+следующей\s+встречи\b",
]

ACTION_INF = (
    "сделать", "подготовить", "проверить", "отправить", "создать", "обновить",
    "исправить", "доделать", "согласовать", "закрыть", "написать", "собрать",
    "завершить", "вынести", "выгрузить", "дополнить", "перепроверить", "оформить",
    "протестировать", "настроить", "описать", "сверить", "разместить", "изучить",
    "начать", "составить", "назначить", "просмотреть", "найти", "добавить",
    "реализовать", "разработать", "отобразить", "сохранить", "сократить",
    "переформулировать", "переделать", "подписать", "задокументировать",
    "провести", "уточнить", "подобрать", "переписать",
)

OTHER_HARD_NEGATIVE_PATTERNS = [
    r"\bя\s+думаю\b",
    r"\bмне\s+кажется\b",
    r"\bя\s+бы\s+(хотел|не\s+хотел)\b",
    r"\bя\s+хочу\b",
    r"\bя\s+решил\b",
    r"\bя\s+начал\b",
    r"\bя\s+попытался\b",
    r"\bне\s+знаю,?\s+как\b",
    r"\bнаверное\b",
    r"\bможет\s+быть\b",
    r"\bможно\s+(как-то|попробовать|оценить|описать|пояснить)\b",
    r"\bнадо\s+подумать\b",
    r"\bнадо\s+поискать\b",
    r"\bнужно\s+просто\b",
    r"\bнужно\s+как-то\b",
    r"\bнадо\s+как-то\b",
    r"\bнадо\s+бы\b",
    r"\bзадача\s+(оптимизации|минимизации|максимизации|активизации|исследования|сортировки)\b",
    r"\bзадача\s+мст\b",
    r"\bглобальная\s+задача\b",
    r"\bматематическ\w+\s+задач\w*\b",
    r"\bпостановк\w+\s+задач\w*\b",
    r"\bформулиров\w+\s+задач\w*\b",
    r"\bкак\s+конкретно\s+задачу\b",
    r"\bэта\s+задача\s+(не\s+)?настолько\b",
    r"\bпохож\w+\s+задач\w*\b",
    r"\bминимизировать\b",
    r"\bмаксимизировать\b",
    r"\bоптимальност\w*\b",
    r"\bдиаметр\w*\b",
    r"\bграф\w*\b",
    r"\bметрик\w*\b",
    r"\bмодель\w*\b",
    r"\bэвристик\w*\b",
    r"\bалгоритм\w*\b",
    r"\bдолжна\s+быть\b",
    r"\bдолжно\s+быть\b",
    r"\bхотя\s+бы\b",
    r"\bне\s+понятно\b",
    r"\bсложно\s+оценить\b",
    r"\bне\s+смогу\b",
    r"\bя\s+смогу\b",
]

SHORT_FILLER_PATTERNS = [
    r"^\s*ну\b",
    r"^\s*вот\b",
    r"^\s*типа\b",
    r"^\s*как\s+бы\b",
]


def normalize_text(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    text = TIMECODE_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" -–—")


def is_speaker(text: str) -> bool:
    return SPEAKER_RE.match(text.strip()) is not None


def is_header(text: str) -> bool:
    return HEADER_RE.match(text.strip()) is not None


def is_summary_other(text: str) -> bool:
    lower = normalize_text(text).lower()
    return any(lower.startswith(s) for s in SUMMARY_OTHER_STARTS)


def is_bad_short(text: str) -> bool:
    lower = normalize_text(text).lower()
    return lower in BAD_SHORT or len(lower) <= 4


def looks_question(text: str) -> bool:
    text = normalize_text(text)
    lower = text.lower()

    if is_bad_short(text):
        return False

    if text.endswith("?"):
        if len(text.split()) <= 2:
            return False
        return True

    return any(lower.startswith(w + " ") for w in QUESTION_WORDS)


def has_action_inf(text: str) -> bool:
    lower = normalize_text(text).lower()
    return any(re.search(rf"\b{re.escape(v)}\b", lower) for v in ACTION_INF)


def looks_task(text: str) -> bool:
    text = normalize_text(text)
    lower = text.lower()

    if is_header(text):
        return False

    if is_summary_other(text):
        if lower.startswith("рекомендовано") or lower.startswith("отмечено, что необходимо") or lower.startswith("отмечено, что нужно"):
            pass
        else:
            return False

    if len(text.split()) <= 2:
        return False

    if looks_question(text):
        return False

    if any(re.search(p, lower) for p in OTHER_HARD_NEGATIVE_PATTERNS):
        if not lower.startswith(TASK_STRONG_STARTS) and not lower.startswith("рекомендовано") and not lower.startswith("необходимо") and not lower.startswith("требуется"):
            return False

    if lower.startswith(TASK_STRONG_STARTS):
        return True

    if lower.startswith("рекомендовано"):
        return True

    if lower.startswith("необходимо"):
        return True

    if lower.startswith("требуется"):
        return True

    if any(re.search(p, lower) for p in TASK_PATTERNS) and has_action_inf(text):
        return True

    return False


def looks_answer_strict(text: str) -> tuple[bool, str]:
    text = normalize_text(text)
    lower = text.lower()

    if not text:
        return False, ""

    if is_bad_short(text):
        return False, ""

    if is_speaker(text):
        return False, ""

    if is_header(text):
        return False, ""

    if is_summary_other(text):
        return False, ""

    if looks_question(text):
        return False, ""

    if looks_task(text):
        return False, ""

    for p in NOT_ANSWER_PATTERNS:
        if re.search(p, lower):
            return False, ""

    for p in ANSWER_STRICT_PATTERNS:
        if re.search(p, lower):
            return True, p

    if lower.startswith("это ") and len(text.split()) >= 4:
        return True, "starts_with:это"

    if lower.startswith("вот,") and len(text.split()) >= 4:
        return True, "starts_with:вот"

    return False, ""


def is_hard_negative_other(text: str) -> tuple[bool, str]:
    lower = normalize_text(text).lower()

    if len(lower) < 12:
        return False, ""

    if is_header(lower) or is_summary_other(lower):
        return True, "header_or_summary"

    for p in SHORT_FILLER_PATTERNS:
        if re.search(p, lower):
            return True, p

    for p in OTHER_HARD_NEGATIVE_PATTERNS:
        if re.search(p, lower):
            return True, p

    return False, ""


def load_jsonl(path: str) -> list[dict]:
    items = []

    if not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = normalize_text(item.get("text", ""))
            label = str(item.get("label", "other")).strip()

            if not text:
                continue

            if label not in VALID_LABELS:
                label = "other"

            if is_speaker(text):
                continue

            if is_bad_short(text):
                continue

            items.append({
                "text": text,
                "label": label,
            })

    return items


def save_jsonl(items: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedupe_by_text(items: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for item in items:
        text = normalize_text(item.get("text", ""))
        label = item.get("label", "other")

        if not text:
            continue

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)
        out.append({
            "text": text,
            "label": label,
        })

    return out


def clean_label(item: dict) -> dict | None:
    text = normalize_text(item.get("text", ""))
    old_label = item.get("label", "other")

    if not text:
        return None

    if is_speaker(text):
        return None

    if is_bad_short(text):
        return None

    if is_header(text):
        return {"text": text, "label": "other"}

    if looks_question(text):
        return {"text": text, "label": "question"}

    if looks_task(text):
        return {"text": text, "label": "task"}

    is_ans, _ = looks_answer_strict(text)

    if is_ans:
        return {"text": text, "label": "answer"}

    if old_label in VALID_LABELS:
        if old_label == "answer":
            return {"text": text, "label": "other"}

        return {"text": text, "label": old_label}

    return {"text": text, "label": "other"}


def stratified_split(items: list[dict], train_ratio: float = TRAIN_RATIO) -> tuple[list[dict], list[dict]]:
    by_label: dict[str, list[dict]] = {}

    for item in items:
        by_label.setdefault(item["label"], []).append(item)

    train, val = [], []

    for _, group in by_label.items():
        random.shuffle(group)
        split_idx = int(len(group) * train_ratio)
        train.extend(group[:split_idx])
        val.extend(group[split_idx:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def sample_class(items: list[dict], n: int, label_name: str) -> list[dict]:
    if n <= 0:
        print(f"  {label_name}: 0")
        return []

    if len(items) <= n:
        print(f"  {label_name}: {len(items)} (все)")
        return list(items)

    sampled = random.sample(items, n)
    print(f"  {label_name}: {len(sampled)} из {len(items)}")
    return sampled


def create_fallback_tasks_from_other(items: list[dict], max_items: int = 120) -> list[dict]:
    fallback_patterns = [
        r"\b(сделать|подготовить|написать|отправить|проверить|создать|исправить|обновить|согласовать|оформить|изучить|добавить|реализовать|разработать)\b",
        r"^\s*(назначить|подготовить|проверить|создать|обновить|оформить|изучить|добавить|реализовать|разработать)\b",
        r"\bдо\s+(пятницы|понедельника|вторника|среды|четверга|конца\s+дня|конца\s+недели|завтра|следующей\s+встречи)\b",
    ]

    result = []

    for item in items:
        text = normalize_text(item["text"])
        lower = text.lower()

        if looks_question(text):
            continue

        if is_summary_other(text):
            continue

        if any(re.search(p, lower) for p in fallback_patterns):
            result.append({"text": text, "label": "task"})

        if len(result) >= max_items:
            break

    return dedupe_by_text(result)


def balance_and_split() -> None:
    if not os.path.exists(INPUT_PATH):
        print(f"Файл {INPUT_PATH} не найден. Сначала запустите prepare_dataset.py")
        return

    raw_data = load_jsonl(INPUT_PATH)
    print(f"Загружено: {len(raw_data)} записей")

    cleaned_data = []

    for item in raw_data:
        cleaned = clean_label(item)

        if cleaned:
            cleaned_data.append(cleaned)

    cleaned_data = dedupe_by_text(cleaned_data)

    by_label = {"task": [], "question": [], "answer": [], "other": []}

    for item in cleaned_data:
        by_label[item["label"]].append(item)

    print("\nРаспределение после повторной чистки:")
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {len(by_label[label])}")

    found_answers = []
    remaining_other = []

    print(f"\nОсторожный поиск скрытых answer среди {len(by_label['other'])} записей other...")

    for item in by_label["other"]:
        is_ans, reason = looks_answer_strict(item["text"])

        if is_ans:
            found_answers.append({
                "text": item["text"],
                "label": "answer",
                "auto_reason": reason,
            })
        else:
            remaining_other.append(item)

    print(f"  Найдено кандидатов в answer: {len(found_answers)}")
    print(f"  Осталось other после answer-фильтра: {len(remaining_other)}")

    save_jsonl(found_answers, REVIEW_ANSWERS_PATH)

    hard_negative_other = []
    regular_other = []

    print(f"\nПоиск hard negative other среди {len(remaining_other)} записей...")

    for item in remaining_other:
        is_hard, reason = is_hard_negative_other(item["text"])

        if is_hard:
            hard_negative_other.append({
                "text": item["text"],
                "label": "other",
                "auto_reason": reason,
            })
        else:
            regular_other.append(item)

    print(f"  Hard negative other: {len(hard_negative_other)}")
    print(f"  Regular other: {len(regular_other)}")

    save_jsonl(hard_negative_other, REVIEW_OTHER_PATH)

    all_task = dedupe_by_text(by_label["task"])
    all_question = dedupe_by_text(by_label["question"])
    all_answer = dedupe_by_text(
        by_label["answer"] + [{"text": x["text"], "label": "answer"} for x in found_answers]
    )
    all_hard_other = dedupe_by_text([{"text": x["text"], "label": "other"} for x in hard_negative_other])
    all_regular_other = dedupe_by_text(regular_other)

    if len(all_task) == 0:
        print(" task всё ещё 0 — включаю аварийный fallback из other")
        all_task = create_fallback_tasks_from_other(all_regular_other, max_items=120)
        print(f"  fallback task: {len(all_task)}")
    else:
        print(f"\nРеальные task сохранены: {len(all_task)}")

    existing_sizes = [len(all_task), len(all_question)]

    if len(all_answer) > 0:
        existing_sizes.append(len(all_answer))

    base_target = min(MAX_PER_CLASS, max(MIN_CLASS_TARGET, min(existing_sizes)))
    other_target = min(MAX_OTHER, max(int(base_target * 1.4), base_target), len(all_hard_other) + len(all_regular_other))

    print("\nЦелевой размер:")
    print(f"  task/question/answer target: {base_target}")
    print(f"  other target: {other_target}")

    final_task = sample_class(all_task, min(base_target, len(all_task)), "task")
    final_question = sample_class(all_question, min(base_target, len(all_question)), "question")
    final_answer = sample_class(all_answer, min(base_target, len(all_answer)), "answer")

    hard_take = min(len(all_hard_other), int(other_target * 0.6))
    final_other_hard = sample_class(all_hard_other, hard_take, "other_hard")

    regular_need = max(other_target - len(final_other_hard), 0)
    final_other_regular = sample_class(all_regular_other, regular_need, "other_regular")

    final_other = dedupe_by_text(final_other_hard + final_other_regular)

    if len(final_answer) < MIN_ANSWER_WARN:
        print(f"\n WARNING: answer всего {len(final_answer)} — это мало.")
        print("  Лучше вручную добавить реальные ответы: подтверждения, статусы выполнения, короткие объяснения.")

    all_data = dedupe_by_text(final_task + final_question + final_answer + final_other)
    random.shuffle(all_data)

    train_data, val_data = stratified_split(all_data, TRAIN_RATIO)

    save_jsonl(train_data, TRAIN_PATH)
    save_jsonl(val_data, VAL_PATH)

    train_counts = Counter(item["label"] for item in train_data)
    val_counts = Counter(item["label"] for item in val_data)
    total_counts = Counter(item["label"] for item in all_data)

    summary = {
        "total": len(all_data),
        "train": len(train_data),
        "val": len(val_data),
        "total_counts": dict(total_counts),
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
        "review_answers": len(found_answers),
        "review_other_hard_negatives": len(hard_negative_other),
    }

    Path(SUMMARY_PATH).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("ФИНАЛЬНОЕ распределение:")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {total_counts.get(label, 0)}")

    print("\nTrain:")
    print(f"  всего: {len(train_data)} → {TRAIN_PATH}")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {train_counts.get(label, 0)}")

    print("\nVal:")
    print(f"  всего: {len(val_data)} → {VAL_PATH}")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {val_counts.get(label, 0)}")

    print("\nФайлы проверки:")
    print(f"  {REVIEW_ANSWERS_PATH}")
    print(f"  {REVIEW_OTHER_PATH}")
    print(f"  {SUMMARY_PATH}")

    print("\nЧто дальше:")
    print("1. Проверьте review_answers.jsonl — там должны быть только реальные ответы.")
    print("2. Проверьте review_other_hard_negatives.jsonl — там должны быть сложные other-примеры.")
    print("3. Запустите: python train_classifier.py")
    print("=" * 60)


if __name__ == "__main__":
    balance_and_split()
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH = Path("datasets/prelabeled.jsonl")
OUTPUT_PATH = Path("datasets/prelabeled_clean.jsonl")


HEADER_PATTERNS = [
    r"^\s*обсуждение\b",
    r"^\s*демонстрация\b",
    r"^\s*результаты\s+встречи\b",
    r"^\s*поставленные\s+задачи\b",
    r"^\s*управление\s+статусами\b",
    r"^\s*цветовая\s+маркировка\b",
    r"^\s*учёт\s+результатов\b",
    r"^\s*организация\s+встречи\b",
    r"^\s*использование\s+сервисов\b",
    r"^\s*тестирование\s+производительности\b",
    r"^\s*рассмотренные\s+вопросы\b",
    r"^\s*ход\s+встречи\b",
    r"^\s*на\s+встрече\b",
    r"^\s*повестка\b",
    r"^\s*формат\b",
    r"^\s*дата\b",
    r"^\s*время\b",
    r"^\s*тема\s*\d*\.?\s*$",
    r"^\s*протокол\b",
    r"^\s*присутствовали\b",
    r"^\s*участники\s+встречи\b",
]

TASK_LIKE_PATTERNS = [
    r"^\s*[•\-]?\s*(подготовить|проверить|исправить|доработать|написать|создать|обновить|согласовать|оформить|изучить|провести|реализовать|разработать|завершить|задокументировать|протестировать|отображать|заменить|уточнить)\b",
    r"\b(необходимо|требуется|нужно|надо|следует)\b.+\b(подготовить|проверить|исправить|доработать|написать|создать|обновить|согласовать|оформить|изучить|провести|реализовать|разработать|завершить|задокументировать|протестировать|отображать|заменить|уточнить)\b",
    r"\bк\s+следующей\s+встрече\b",
    r"\bдо\s+(понедельника|вторника|среды|четверга|пятницы|следующей\s+встречи|конца\s+дня|конца\s+недели)\b",
]

FALSE_TASK_PATTERNS = [
    r"\bв\s+принципе,\s+задача\b",
    r"\bзадача\s+же\s+не\s+может\b",
    r"\bинтересная\s+задача\b",
    r"\bполаг\w*,?\s+задача\b",
    r"^\s*\d+,\s*а\s+задача\b",
    r"^\s*задача\s+срок\s*$",
    r"\bзадача\s+(оптимизации|минимизации|максимизации|активизации|мст)\b",
    r"\bглобальная\s+задача\b",
    r"\bматематическая\s+задача\b",
    r"\bпостановк\w*\s+задач",
    r"\bформулировк\w*\s+задач",
]

ANSWER_LIKE_PATTERNS = [
    r"^\s*(да|нет|хорошо|ладно|понял|поняла|понятно|согласен|согласна|верно|точно)\b",
    r"\b(сделано|готово|выполнено|завершено|загружено|отправлено|добавил|добавила|исправил|исправила|обновил|обновила|создал|создала|доделал|доделала)\b",
    r"\bя\s+(сделал|сделала|добавил|добавила|исправил|исправила|обновил|обновила|создал|создала|загрузил|загрузила|пытался|пыталась)\b",
    r"\bмы\s+(сделали|добавили|исправили|обновили|создали|загрузили)\b",
    r"\bполучилось\b",
    r"\bв итоге\b",
]

QUESTION_PATTERNS = [
    r"\?$",
    r"^\s*(кто|что|где|когда|почему|зачем|как|какой|какая|какие|сколько|чем|куда|откуда|можно ли|нужно ли|успеем ли)\b",
]


def norm(text: str) -> str:
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_time(text: str) -> str:
    return re.sub(r"^\s*\[\d{2}:\d{2}:\d{2}\]\s*", "", text).strip()


def match_any(patterns: list[str], text: str) -> bool:
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)


def fix_label(text: str, label: str) -> str:
    clean = strip_time(norm(text))
    lower = clean.lower()

    # Спикеры, даты, заголовки — always other
    if re.search(r"^[А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+ [А-ЯЁ][а-яё]+(\s+\(голос \d+\))?:$", clean):
        return "other"

    if match_any(HEADER_PATTERNS, clean):
        return "other"

    # Вопросы имеют приоритет
    if match_any(QUESTION_PATTERNS, clean):
        return "question"

    # Явные ложные task
    if label == "task" and match_any(FALSE_TASK_PATTERNS, clean):
        return "other"

    # Явные задачи
    if match_any(TASK_LIKE_PATTERNS, clean):
        return "task"

    # Явные ответы/отчёт о статусе
    if match_any(ANSWER_LIKE_PATTERNS, clean):
        return "answer"

    return label


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = norm(item.get("text", ""))
            label = item.get("label", "other")
            if text and label in {"task", "question", "answer", "other"}:
                items.append({"text": text, "label": label})
    return items


def dedupe(items: list[dict]) -> list[dict]:
    seen = set()
    out = []

    for item in items:
        key = norm(item["text"]).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def main() -> None:
    if not INPUT_PATH.exists():
        print(f"Не найден файл: {INPUT_PATH}")
        return

    data = load_jsonl(INPUT_PATH)

    fixed = []
    changes = Counter()

    for item in data:
        old = item["label"]
        new = fix_label(item["text"], old)

        if old != new:
            changes[f"{old}->{new}"] += 1

        fixed.append({
            "text": item["text"],
            "label": new,
        })

    fixed = dedupe(fixed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for item in fixed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Готово:")
    print(f"  {OUTPUT_PATH}")

    print("\nИзменения:")
    for k, v in changes.most_common():
        print(f"  {k}: {v}")

    print("\nРаспределение после чистки:")
    counts = Counter(item["label"] for item in fixed)
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label:8} {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
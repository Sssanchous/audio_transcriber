"""Resolve pm_dataset/protocol_structured label conflicts, drop low-confidence
needs_review rows, top up under-represented classes with synthetic examples,
and re-split into train/val/test with zero cross-split text leakage.

Usage: python scripts/clean_dataset.py
Inputs:  datasets/training_dataset_fixed.jsonl
Outputs: datasets/training_dataset_clean.jsonl, train_clean.jsonl, val_clean.jsonl,
         test_clean.jsonl, clean_dataset_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

INPUT_PATH = DATASETS_DIR / "training_dataset_fixed.jsonl"
OLD_TRAIN_PATH = DATASETS_DIR / "train.jsonl"
OLD_VAL_PATH = DATASETS_DIR / "val.jsonl"
OLD_TEST_PATH = DATASETS_DIR / "test.jsonl"

CLEAN_PATH = DATASETS_DIR / "training_dataset_clean.jsonl"
TRAIN_PATH = DATASETS_DIR / "train_clean.jsonl"
VAL_PATH = DATASETS_DIR / "val_clean.jsonl"
TEST_PATH = DATASETS_DIR / "test_clean.jsonl"
REPORT_PATH = DATASETS_DIR / "clean_dataset_report.json"

ALLOWED_LABELS = ("task", "question", "answer", "other")

# Sources with no `metadata.confidence` field that we nonetheless trust by
# default (treated as confidence=PROTECTED_CONFIDENCE_FLOOR). manual_examples
# is hand-curated synthetic training data with a deliberately fixed label, so
# a missing confidence score does not mean "low quality".
TRUSTED_SOURCES = {"manual_examples"}

NEEDS_REVIEW_CONFIDENCE_FLOOR = 0.75
PROTECTED_CONFIDENCE_FLOOR = 0.85

TARGET_PROPORTIONS = {"task": 0.22, "question": 0.22, "answer": 0.20, "other": 0.36}
IMBALANCE_THRESHOLD_PP = 5.0

SPLIT_RATIOS = (0.80, 0.10, 0.10)
SPLIT_SEED = 42


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --------------------------------------------------------------------------- #
# Confidence / review helpers
# --------------------------------------------------------------------------- #


def confidence(row: dict) -> float | None:
    return row.get("metadata", {}).get("confidence")


def effective_confidence(row: dict) -> float | None:
    """Real confidence if present, else PROTECTED_CONFIDENCE_FLOOR for trusted sources."""
    c = confidence(row)
    if c is not None:
        return c
    if row.get("source") in TRUSTED_SOURCES:
        return PROTECTED_CONFIDENCE_FLOOR
    return None


def needs_review(row: dict) -> bool:
    return bool(row.get("metadata", {}).get("needs_review"))


def is_protected(row: dict) -> bool:
    c = effective_confidence(row)
    return row.get("source") in TRUSTED_SOURCES and c is not None and c >= PROTECTED_CONFIDENCE_FLOOR


# --------------------------------------------------------------------------- #
# Step 1: duplicate / conflicting-label resolution
# --------------------------------------------------------------------------- #


def resolve_duplicates(rows: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = normalize_text(row["text"])
        if key:
            groups[key].append(row)
    groups = {k: v for k, v in groups.items() if len(v) > 1}

    def survivor(group: list[dict]) -> dict:
        protected = [r for r in group if is_protected(r)]
        if protected:
            return protected[0]

        def sort_key(r: dict):
            c = confidence(r)
            conf_key = c if c is not None else -1.0
            proto_key = 1 if r.get("source") == "protocol_structured" else 0
            return (-conf_key, -proto_key, -len(r["text"]), r["id"])

        return sorted(group, key=sort_key)[0]

    dropped_ids: set[str] = set()
    conflicting_groups = 0
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            conflicting_groups += 1
        keep = survivor(group)
        dropped_ids.update(r["id"] for r in group if r["id"] != keep["id"])

    kept = [r for r in rows if r["id"] not in dropped_ids]
    stats = {
        "duplicate_groups": len(groups),
        "conflicting_groups": conflicting_groups,
        "rows_dropped": len(dropped_ids),
    }
    return kept, stats


# --------------------------------------------------------------------------- #
# Step 2: needs_review / low-confidence filter
# --------------------------------------------------------------------------- #


def filter_needs_review(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    def should_drop(row: dict) -> bool:
        if is_protected(row):
            return False
        if not needs_review(row):
            return False
        c = effective_confidence(row)
        return c is None or c < NEEDS_REVIEW_CONFIDENCE_FLOOR

    dropped = [r for r in rows if should_drop(r)]
    kept = [r for r in rows if not should_drop(r)]
    return kept, dropped


# --------------------------------------------------------------------------- #
# Step 3: synthetic balancing (only triggered if a class is >5pp below target)
# --------------------------------------------------------------------------- #

NAMES = [
    "Анна", "Дмитрий", "Сергей", "Мария", "Павел", "Ольга",
    "Игорь", "Татьяна", "Алексей", "Наталья", "Виктор", "Елена",
]

# Kept in accusative case so they drop cleanly into "сделать/завершить/проверить/за ..." slots.
TOPICS = [
    "миграцию базы данных", "интеграцию с CRM", "тестовое покрытие API",
    "документацию для заказчика", "ревью дизайна интерфейса", "настройку CI/CD",
    "отчёт по бюджету", "анализ метрик конверсии", "доработку модуля авторизации",
    "подготовку демо для клиента", "оптимизацию запросов к базе", "обновление зависимостей",
    "сценарии нагрузочного тестирования", "интеграцию платёжного шлюза", "макет нового раздела",
    "план миграции на новую инфраструктуру", "согласование требований с заказчиком",
    "подготовку релиза", "анализ обратной связи пользователей", "вёрстку лендинга",
]

DEADLINES = [
    "до пятницы", "к концу недели", "до завтра", "к следующему спринту",
    "в течение двух дней", "до конца месяца", "к понедельнику", "до среды",
    "к 15 числу", "до начала демо", "к утру", "в течение трёх дней",
]

SYNTH_TEMPLATES: dict[str, list[str]] = {
    "task": [
        "{name}, возьми, пожалуйста, {topic} и сделай это {deadline}.",
        "Нужно завершить {topic} {deadline}.",
        "{name}, подготовь {topic} {deadline}.",
        "Прошу взять на контроль {topic}, срок — {deadline}.",
        "{name}, проверь {topic} и пришли результат {deadline}.",
        "{name}, на тебе {topic}, готовность нужна {deadline}.",
    ],
    "question": [
        "Кто отвечает за {topic}?",
        "Когда планируем закончить {topic}?",
        "Успеваем ли завершить {topic} {deadline}?",
        "Что мешает закрыть {topic}?",
        "{name}, ты успеешь сделать {topic} {deadline}?",
        "Кто возьмёт {topic} на себя?",
        "Перенесём {topic} на следующий спринт?",
    ],
    "answer": [
        "Да, {name} уже взял {topic} в работу.",
        "Пока не закончили {topic}, нужно ещё немного времени.",
        "{name} подтвердил, что закроет {topic} {deadline}.",
        "Уже почти готово, осталось протестировать {topic}.",
        "Нет, {topic} ещё не начали, приоритет был другой.",
        "{name} сказал, что возьмёт {topic} на себя {deadline}.",
        "Всё под контролем, {topic} сделаем {deadline}.",
    ],
    "other": [
        "Сегодня обсудили {topic} и зафиксировали текущий статус.",
        "Команда отметила, что {topic} требует дополнительного внимания в следующем спринте.",
        "{name} рассказал о текущем прогрессе по {topic}.",
        "Обсуждение {topic} перенесли в отдельный канал для деталей.",
        "Резюмируя встречу: основной фокус сейчас на {topic}.",
        "Коллеги поделились впечатлениями после демо {topic}.",
        "По итогам спринта {topic} остаётся в приоритете команды.",
    ],
}


def generate_synthetic(label: str, count: int, used_normalized: set[str], rng: random.Random) -> list[dict]:
    templates = SYNTH_TEMPLATES[label]
    generated: list[dict] = []
    attempts = 0
    max_attempts = count * 50 + 200
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        text = rng.choice(templates).format(
            name=rng.choice(NAMES),
            topic=rng.choice(TOPICS),
            deadline=rng.choice(DEADLINES),
        )
        key = normalize_text(text)
        if key in used_normalized:
            continue
        used_normalized.add(key)
        idx = len(generated) + 1
        generated.append(
            {
                "id": f"synthetic_{label}_{idx:04d}",
                "source_file": "synthetic_balanced",
                "text": text,
                "label": label,
                "original_label": label,
                "source": "synthetic_balanced",
                "metadata": {
                    "source": "synthetic_balanced",
                    "original_label": label,
                    "needs_review": False,
                    "confidence": 0.80,
                    "synthetic": True,
                },
            }
        )
    if len(generated) < count:
        raise RuntimeError(
            f"Could not generate enough unique synthetic examples for label={label}: "
            f"needed {count}, got {len(generated)}"
        )
    return generated


def balance_classes(
    rows: list[dict],
    targets: dict[str, float],
    threshold_pp: float,
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    counts = Counter(r["label"] for r in rows)
    total = len(rows)
    used_normalized = {normalize_text(r["text"]) for r in rows}
    added_rows: list[dict] = []
    synthetic_added = {label: 0 for label in targets}

    for label, target_frac in targets.items():
        pct = (counts[label] / total * 100) if total else 0.0
        gap = target_frac * 100 - pct
        if gap <= threshold_pp:
            continue
        n_needed = (target_frac * total - counts[label]) / (1 - target_frac)
        n_needed = max(0, math.ceil(n_needed))
        if n_needed == 0:
            continue
        new_rows = generate_synthetic(label, n_needed, used_normalized, rng)
        added_rows.extend(new_rows)
        synthetic_added[label] = len(new_rows)
        counts[label] += len(new_rows)
        total += len(new_rows)

    return rows + added_rows, synthetic_added


# --------------------------------------------------------------------------- #
# Step 4: stratified re-split, 80/10/10, zero cross-split text overlap
# --------------------------------------------------------------------------- #


def stratified_split(rows: list[dict], seed: int, ratios: tuple[float, float, float]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    train_ratio, val_ratio, _test_ratio = ratios
    for group in by_label.values():
        group = group[:]
        rng.shuffle(group)
        n = len(group)
        n_val = max(1, round(n * val_ratio))
        n_test = max(1, round(n * (1 - train_ratio - val_ratio)))
        val.extend(group[:n_val])
        test.extend(group[n_val : n_val + n_test])
        train.extend(group[n_val + n_test :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #


def label_distribution(rows: list[dict]) -> dict[str, dict]:
    total = len(rows)
    counts = Counter(r["label"] for r in rows)
    return {
        label: {
            "count": counts.get(label, 0),
            "pct": round(counts.get(label, 0) / total * 100, 2) if total else 0.0,
        }
        for label in ALLOWED_LABELS
    }


def count_duplicate_groups(rows: list[dict]) -> int:
    groups: dict[str, int] = defaultdict(int)
    for r in rows:
        key = normalize_text(r["text"])
        if key:
            groups[key] += 1
    return sum(1 for v in groups.values() if v > 1)


def normalized_overlap(a: list[dict], b: list[dict]) -> int:
    na = {normalize_text(r["text"]) for r in a}
    nb = {normalize_text(r["text"]) for r in b}
    return len(na & nb)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and re-split the PM Insights training dataset.")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    rows = load_jsonl(INPUT_PATH)
    before_dist = label_distribution(rows)
    before_dup_groups = count_duplicate_groups(rows)

    old_train = load_jsonl(OLD_TRAIN_PATH)
    old_val = load_jsonl(OLD_VAL_PATH)
    old_test = load_jsonl(OLD_TEST_PATH)
    old_leaks = {
        "train_val": normalized_overlap(old_train, old_val),
        "train_test": normalized_overlap(old_train, old_test),
        "val_test": normalized_overlap(old_val, old_test),
    }

    deduped, dup_stats = resolve_duplicates(rows)
    after_review, dropped_review = filter_needs_review(deduped)

    rng = random.Random(args.seed)
    balanced, synthetic_added = balance_classes(after_review, TARGET_PROPORTIONS, IMBALANCE_THRESHOLD_PP, rng)

    after_dist = label_distribution(balanced)
    after_dup_groups = count_duplicate_groups(balanced)

    train, val, test = stratified_split(balanced, seed=args.seed, ratios=SPLIT_RATIOS)
    new_leaks = {
        "train_val": normalized_overlap(train, val),
        "train_test": normalized_overlap(train, test),
        "val_test": normalized_overlap(val, test),
    }
    if any(new_leaks.values()):
        raise RuntimeError(f"Cross-split text leakage detected after re-split: {new_leaks}")

    save_jsonl(CLEAN_PATH, balanced)
    save_jsonl(TRAIN_PATH, train)
    save_jsonl(VAL_PATH, val)
    save_jsonl(TEST_PATH, test)

    report = {
        "input": str(INPUT_PATH.relative_to(ROOT)),
        "before": {
            "total": len(rows),
            "label_distribution": before_dist,
            "duplicate_groups": before_dup_groups,
            "needs_review_rows": sum(1 for r in rows if needs_review(r)),
            "old_split_leaks": old_leaks,
        },
        "duplicate_resolution": dup_stats,
        "needs_review_filter": {
            "rows_dropped": len(dropped_review),
            "dropped_by_source": dict(Counter(r.get("source") for r in dropped_review)),
        },
        "synthetic_examples_added": synthetic_added,
        "after": {
            "total": len(balanced),
            "label_distribution": after_dist,
            "duplicate_groups": after_dup_groups,
            "needs_review_rows": sum(1 for r in balanced if needs_review(r)),
            "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "new_split_leaks": new_leaks,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

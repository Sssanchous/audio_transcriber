from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("datasets/full_clean.jsonl")
TRAIN_PATH = Path("datasets/train.jsonl")
VAL_PATH = Path("datasets/val.jsonl")
SUMMARY_PATH = Path("datasets/dataset_summary.json")

VALID_LABELS = {"task", "question", "answer", "other"}

TRAIN_RATIO = 0.82
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def load_jsonl(path: Path) -> list[dict]:
    items = []

    if not path.exists():
        print(f"Файл не найден: {path}")
        return items

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(item.get("text", "")).strip()
            label = str(item.get("label", "")).strip()

            if not text or label not in VALID_LABELS:
                continue

            items.append({
                "text": text,
                "label": label,
            })

    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dedupe_items(items: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for item in items:
        text = item["text"].strip()
        label = item["label"]

        key = text.lower()

        if key in seen:
            continue

        seen.add(key)

        result.append({
            "text": text,
            "label": label,
        })

    return result


def stratified_split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    by_label = defaultdict(list)

    for item in items:
        by_label[item["label"]].append(item)

    train = []
    val = []

    for label, group in by_label.items():
        random.shuffle(group)

        if len(group) <= 2:
            train.extend(group)
            continue

        split_index = int(len(group) * TRAIN_RATIO)

        if split_index <= 0:
            split_index = 1

        if split_index >= len(group):
            split_index = len(group) - 1

        train.extend(group[:split_index])
        val.extend(group[split_index:])

    random.shuffle(train)
    random.shuffle(val)

    return train, val


def main() -> None:
    items = load_jsonl(INPUT_PATH)
    items = dedupe_items(items)

    if not items:
        print("Нет данных для разделения.")
        return

    train, val = stratified_split(items)

    save_jsonl(train, TRAIN_PATH)
    save_jsonl(val, VAL_PATH)

    total_counts = Counter(item["label"] for item in items)
    train_counts = Counter(item["label"] for item in train)
    val_counts = Counter(item["label"] for item in val)

    summary = {
        "total": len(items),
        "train": len(train),
        "val": len(val),
        "total_counts": dict(total_counts),
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Готово.")
    print(f"Всего записей: {len(items)}")
    print("Распределение классов:")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {total_counts.get(label, 0)}")

    print("\nTrain:")
    print(f"  файл: {TRAIN_PATH}")
    print(f"  всего: {len(train)}")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {train_counts.get(label, 0)}")

    print("\nVal:")
    print(f"  файл: {VAL_PATH}")
    print(f"  всего: {len(val)}")

    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {val_counts.get(label, 0)}")

    print(f"\nSummary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
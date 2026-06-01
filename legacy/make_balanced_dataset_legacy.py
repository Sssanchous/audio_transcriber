from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

INPUT_PATH = Path("datasets/full_clean.jsonl")

BALANCED_ALL_PATH = Path("datasets/balanced_all.jsonl")
BALANCED_TRAIN_PATH = Path("datasets/train_balanced.jsonl")
BALANCED_VAL_PATH = Path("datasets/val_balanced.jsonl")
BALANCED_SUMMARY_PATH = Path("datasets/balanced_summary.json")

VALID_LABELS = {"task", "question", "answer", "other"}

RANDOM_SEED = 42
TRAIN_RATIO = 0.82

TARGETS = {
    "task": None,       # берём все task
    "question": 300,    # ограничиваем
    "answer": 300,      # ограничиваем
    "other": 450,       # ограничиваем
}

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


def split_by_label(items: list[dict]) -> dict[str, list[dict]]:
    by_label = defaultdict(list)

    for item in items:
        by_label[item["label"]].append(item)

    return by_label


def sample_group(label: str, group: list[dict]) -> list[dict]:
    target = TARGETS.get(label)

    random.shuffle(group)

    if target is None:
        return group

    return group[:min(target, len(group))]


def stratified_split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    by_label = split_by_label(items)

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


def counts(items: list[dict]) -> dict:
    return dict(Counter(item["label"] for item in items))


def main() -> None:
    items = load_jsonl(INPUT_PATH)
    items = dedupe_items(items)

    if not items:
        print("Нет данных.")
        return

    by_label = split_by_label(items)

    balanced = []

    print("Исходное распределение:")
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {len(by_label[label])}")

    for label in ["task", "question", "answer", "other"]:
        selected = sample_group(label, by_label[label])
        balanced.extend(selected)

    random.shuffle(balanced)

    train, val = stratified_split(balanced)

    save_jsonl(balanced, BALANCED_ALL_PATH)
    save_jsonl(train, BALANCED_TRAIN_PATH)
    save_jsonl(val, BALANCED_VAL_PATH)

    summary = {
        "total": len(balanced),
        "train": len(train),
        "val": len(val),
        "targets": TARGETS,
        "total_counts": counts(balanced),
        "train_counts": counts(train),
        "val_counts": counts(val),
        "source_total_counts": counts(items),
    }

    BALANCED_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nСбалансированное распределение:")
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {counts(balanced).get(label, 0)}")

    print("\nTrain balanced:")
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {counts(train).get(label, 0)}")

    print("\nVal balanced:")
    for label in ["task", "question", "answer", "other"]:
        print(f"  {label}: {counts(val).get(label, 0)}")

    print(f"\nФайлы:")
    print(f"  {BALANCED_ALL_PATH}")
    print(f"  {BALANCED_TRAIN_PATH}")
    print(f"  {BALANCED_VAL_PATH}")
    print(f"  {BALANCED_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
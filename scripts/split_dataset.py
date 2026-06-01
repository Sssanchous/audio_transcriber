from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


STRATEGIES = {
    "seed": (0.80, 0.10, 0.10),
    "train-heavy": (0.90, 0.05, 0.05),
    "no-test": (0.90, 0.10, 0.0),
}


def split_group(rows: list[dict], ratios: tuple[float, float, float]) -> tuple[list[dict], list[dict], list[dict]]:
    n = len(rows)
    train_ratio, val_ratio, test_ratio = ratios
    if n == 1:
        return rows, [], []
    if n == 2:
        return (rows[:1], rows[1:], []) if test_ratio == 0 else (rows[:1], [], rows[1:])

    train_n = max(1, int(round(n * train_ratio)))
    val_n = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0

    if test_ratio == 0:
        if train_n + val_n > n:
            train_n = max(1, n - val_n)
        return rows[:train_n], rows[train_n:], []

    test_min = 1
    if train_n + val_n >= n:
        train_n = max(1, n - val_n - test_min)
        if train_n + val_n >= n:
            val_n = max(0, n - train_n - test_min)
    return rows[:train_n], rows[train_n : train_n + val_n], rows[train_n + val_n :]


def stratified_split(rows: list[dict], seed: int = 42, strategy: str = "seed") -> tuple[list[dict], list[dict], list[dict]]:
    ratios = STRATEGIES[strategy]
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for group in by_label.values():
        group = group[:]
        rng.shuffle(group)
        group_train, group_val, group_test = split_group(group, ratios)
        train.extend(group_train)
        val.extend(group_val)
        test.extend(group_test)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["label"] for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test split for PM Insights dataset.")
    parser.add_argument("--input", default="datasets/pm_dataset.jsonl")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="seed")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    train, val, test = stratified_split(rows, seed=args.seed, strategy=args.strategy)
    output_dir = Path(args.output_dir)
    save_jsonl(output_dir / "train.jsonl", train)
    save_jsonl(output_dir / "val.jsonl", val)
    save_jsonl(output_dir / "test.jsonl", test)

    summary = {
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_labels": counts(train),
        "val_labels": counts(val),
        "test_labels": counts(test),
        "seed": args.seed,
        "strategy": args.strategy,
        "ratios": {
            "train": STRATEGIES[args.strategy][0],
            "val": STRATEGIES[args.strategy][1],
            "test": STRATEGIES[args.strategy][2],
        },
        "strategy_note": "stratified_by_label_with_safe_small_class_handling",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

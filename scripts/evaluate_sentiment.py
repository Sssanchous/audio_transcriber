from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.nlp.sentiment import classify_sentiment


LABELS = ["positive", "neutral", "negative"]


def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    per_class = {}
    scores = []
    for label in LABELS:
        tp = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred == label)
        fp = sum(1 for true, pred in zip(y_true, y_pred) if true != label and pred == label)
        fn = sum(1 for true, pred in zip(y_true, y_pred) if true == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        scores.append(f1)
    return sum(scores) / len(scores), per_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment against expert labels.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=False)
    parser.add_argument("--engine", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    if not reference_path.exists():
        print("Sentiment evaluation skipped: no expert reference provided.")
        return
    data = json.loads(reference_path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    y_true = [item.get("label", "neutral") for item in items]
    y_pred = [str(classify_sentiment(item.get("text", ""), engine=args.engine)["sentiment"]) for item in items]
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) if y_true else 0.0
    macro, per_class = macro_f1(y_true, y_pred)
    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro, 4),
        "per_class_precision_recall": per_class,
        "true_distribution": dict(Counter(y_true)),
        "predicted_distribution": dict(Counter(y_pred)),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

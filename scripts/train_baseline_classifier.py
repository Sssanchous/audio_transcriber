from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def labels(rows: list[dict]) -> list[str]:
    return [row["label"] for row in rows]


def texts(rows: list[dict]) -> list[str]:
    return [row["text"] for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train simple TF-IDF baseline classifier.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", default="models/baseline_classifier")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    val_rows = load_jsonl(Path(args.val))
    test_rows = load_jsonl(Path(args.test))

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(texts(train_rows), labels(train_rows))

    val_pred = model.predict(texts(val_rows))
    test_pred = model.predict(texts(test_rows))
    label_order = sorted(set(labels(train_rows) + labels(val_rows) + labels(test_rows)))
    val_report = classification_report(labels(val_rows), val_pred, labels=label_order, output_dict=True, zero_division=0)
    test_report = classification_report(labels(test_rows), test_pred, labels=label_order, output_dict=True, zero_division=0)
    test_confusion = confusion_matrix(labels(test_rows), test_pred, labels=label_order).tolist()
    confused_pairs = {}
    for actual, predicted in zip(labels(test_rows), test_pred):
        if actual != predicted:
            key = f"{actual} -> {predicted}"
            confused_pairs[key] = confused_pairs.get(key, 0) + 1

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "labels.json").write_text(json.dumps(label_order, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "note": "Experimental TF-IDF baseline, not a production model.",
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "test_examples": len(test_rows),
        "labels": label_order,
        "val_report": val_report,
        "test_report": test_report,
        "test_confusion_matrix": {"labels": label_order, "matrix": test_confusion},
        "confused_pairs": dict(sorted(confused_pairs.items(), key=lambda item: -item[1])),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Validation:")
    print(classification_report(labels(val_rows), val_pred, labels=label_order, zero_division=0))
    print("Test:")
    print(classification_report(labels(test_rows), test_pred, labels=label_order, zero_division=0))


if __name__ == "__main__":
    main()

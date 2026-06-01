from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        texts.append(item["text"])
        labels.append(item["label"])
    return texts, labels


def train_baseline(train_path: Path, val_path: Path, output_dir: Path) -> dict:
    try:
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report
        from sklearn.pipeline import Pipeline
    except Exception as exc:
        raise RuntimeError("scikit-learn and joblib are required for baseline training") from exc

    train_texts, train_labels = read_jsonl(train_path)
    val_texts, val_labels = read_jsonl(val_path)
    train_counts = Counter(train_labels)
    warnings = []

    if len(train_counts) < 2:
        warnings.append("Only one class in train split; baseline training is not meaningful.")
    for label, count in train_counts.items():
        if count < 5:
            warnings.append(f"Class {label} has only {count} train examples.")
        elif count < 50:
            warnings.append(f"Class {label} is very small ({count} train examples); metrics are unstable.")
    if len(val_labels) < 50:
        warnings.append("Validation split is very small; classification report is only a technical smoke test.")

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(val_texts) if val_texts else []
    report = classification_report(val_labels, predictions, output_dict=True, zero_division=0) if val_texts else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "model.joblib")
    labels = sorted(train_counts)
    (output_dir / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = {
        "status": "experimental_baseline",
        "model": "tfidf_logistic_regression",
        "train_examples": len(train_texts),
        "val_examples": len(val_texts),
        "train_labels": dict(train_counts),
        "warnings": warnings,
        "classification_report": report,
        "not_production_ready": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train experimental TF-IDF baseline classifier.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output", default="models/baseline_classifier")
    args = parser.parse_args()

    metrics = train_baseline(Path(args.train), Path(args.val), Path(args.output))
    for warning in metrics["warnings"]:
        print(f"WARNING: {warning}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

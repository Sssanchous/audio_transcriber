from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "models/rubert_classifier"
DEFAULT_TRAIN_SPLIT = "datasets/train.jsonl"
DEFAULT_VAL_SPLIT = "datasets/val.jsonl"
DEFAULT_TEST_SPLIT = "datasets/test.jsonl"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("sentence") or row.get("source_text") or "").strip()


def get_label(row: dict[str, Any]) -> str | None:
    value = row.get("label") or row.get("class") or row.get("target")
    return str(value).strip() if value is not None else None


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    return str(label).strip().lower()


def infer_label_map(model, fallback_order: list[str]) -> dict[int, str]:
    id2label = getattr(model.config, "id2label", None) or {}

    cleaned: dict[int, str] = {}

    for k, v in id2label.items():
        try:
            idx = int(k)
        except Exception:
            idx = k if isinstance(k, int) else None

        if idx is None:
            continue

        cleaned[idx] = str(v).strip()

    if cleaned and not all(v.upper().startswith("LABEL_") for v in cleaned.values()):
        return {k: normalize_label(v) for k, v in cleaned.items()}

    return {i: normalize_label(label) for i, label in enumerate(fallback_order)}


def predict_labels(
    rows: list[dict[str, Any]],
    model_path: str,
    batch_size: int = 16,
    device: str = "auto",
    label_order: list[str] | None = None,
) -> list[str]:
    label_order = label_order or ["task", "question", "answer", "other"]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    id2label = infer_label_map(model, label_order)

    texts = [get_text(row) for row in rows]
    predictions: list[str] = []

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            logits = model(**encoded).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()

            for pred_id in pred_ids:
                predictions.append(id2label.get(int(pred_id), f"label_{pred_id}"))

    return predictions


def calculate_task_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    y_true = [1 if label == "task" else 0 for label in true_labels]
    y_pred = [1 if label == "task" else 0 for label in pred_labels]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "true_task_count": int(sum(y_true)),
        "predicted_task_count": int(sum(y_pred)),
    }


def calculate_classifier_metrics(true_labels: list[str], pred_labels: list[str]) -> dict[str, Any]:
    labels = sorted(set(true_labels) | set(pred_labels))

    report = classification_report(
        true_labels,
        pred_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": round(accuracy_score(true_labels, pred_labels) * 100, 2),
        "macro_f1": round(report["macro avg"]["f1-score"] * 100, 2),
        "weighted_f1": round(report["weighted avg"]["f1-score"] * 100, 2),
        "labels": labels,
        "classification_report": report,
    }


def evaluate_split(
    split_path: str,
    model_path: str,
    batch_size: int,
    device: str,
    label_order: list[str],
) -> dict[str, Any]:
    rows = load_jsonl(split_path)
    rows = [row for row in rows if get_text(row) and get_label(row)]

    true_labels = [normalize_label(get_label(row)) for row in rows]

    pred_labels = predict_labels(
        rows,
        model_path=model_path,
        batch_size=batch_size,
        device=device,
        label_order=label_order,
    )

    task_metrics = calculate_task_metrics(true_labels, pred_labels)
    classifier_metrics = calculate_classifier_metrics(true_labels, pred_labels)

    return {
        "split": split_path,
        "items": len(rows),
        "task": task_metrics,
        "classifier": classifier_metrics,
    }


def print_compact_result(title: str, result: dict[str, Any]) -> None:
    task = result["task"]
    classifier = result["classifier"]

    print(title)
    print(f"Task Precision: {task['precision']}%")
    print(f"Task Recall: {task['recall']}%")
    print(f"Task F1: {task['f1']}%")
    print(f"Общая точность модели: {classifier['accuracy']}%")
    print(f"Macro F1 модели: {classifier['macro_f1']}%")
    print()


def save_class_distribution_chart(
    train_result: dict[str, Any],
    val_result: dict[str, Any],
    test_result: dict[str, Any],
    output_path: str = "results/class_distribution_train_val_test.png",
) -> None:
    labels = ["task", "question", "answer", "other"]

    train_report = train_result["classifier"]["classification_report"]
    val_report = val_result["classifier"]["classification_report"]
    test_report = test_result["classifier"]["classification_report"]

    train_counts = [int(train_report.get(label, {}).get("support", 0)) for label in labels]
    val_counts = [int(val_report.get(label, {}).get("support", 0)) for label in labels]
    test_counts = [int(test_report.get(label, {}).get("support", 0)) for label in labels]

    x = list(range(len(labels)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar([i - width for i in x], train_counts, width, label="Train")
    ax.bar(x, val_counts, width, label="Validation")
    ax.bar([i + width for i in x], test_counts, width, label="Test")

    ax.set_title("Распределение классов в train / validation / test выборках")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Количество примеров")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for i, value in enumerate(train_counts):
        ax.text(i - width, value, str(value), ha="center", va="bottom", fontsize=8)

    for i, value in enumerate(val_counts):
        ax.text(i, value, str(value), ha="center", va="bottom", fontsize=8)

    for i, value in enumerate(test_counts):
        ax.text(i + width, value, str(value), ha="center", va="bottom", fontsize=8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"График сохранён: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", default=DEFAULT_TRAIN_SPLIT)
    parser.add_argument("--val", default=DEFAULT_VAL_SPLIT)
    parser.add_argument("--test", default=DEFAULT_TEST_SPLIT)
    parser.add_argument("--output", default="results/kpi_train_val_test_summary.json")
    parser.add_argument("--chart-output", default="results/class_distribution_train_val_test.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--label-order",
        default="task,question,answer,other",
        help="Fallback order if model config has LABEL_0 labels",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not save JSON file",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Do not save class distribution chart",
    )
    args = parser.parse_args()

    label_order = [x.strip() for x in args.label_order.split(",") if x.strip()]

    train_result = evaluate_split(
        split_path=args.train,
        model_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        label_order=label_order,
    )

    val_result = evaluate_split(
        split_path=args.val,
        model_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        label_order=label_order,
    )

    test_result = evaluate_split(
        split_path=args.test,
        model_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        label_order=label_order,
    )

    print("KPI")
    print()

    print_compact_result("Train", train_result)
    print_compact_result("Validation", val_result)
    print_compact_result("Test", test_result)

    if not args.no_chart:
        save_class_distribution_chart(
            train_result,
            val_result,
            test_result,
            output_path=args.chart_output,
        )

    result = {
        "model": args.model,
        "train": train_result,
        "validation": val_result,
        "test": test_result,
    }

    if not args.no_json:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
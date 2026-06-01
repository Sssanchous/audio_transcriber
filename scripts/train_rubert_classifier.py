from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights import settings


LABELS = ["task", "question", "answer", "other"]


def require_ml_dependencies():
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except Exception:
        print("RuBERT training requires requirements-ml.txt. Install it first.")
        raise SystemExit(2)
    return torch, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") in LABELS and row.get("text"):
                rows.append(row)
    return rows


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    matrix = [[0 for _ in LABELS] for _ in LABELS]
    for true, pred in zip(y_true, y_pred):
        matrix[true][pred] += 1

    per_class = {}
    f1_scores = []
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(LABELS)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(LABELS)) if col != index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_scores.append(f1)

    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true) if y_true else 0.0
    confusions = []
    for true_index, true_label in enumerate(LABELS):
        for pred_index, pred_label in enumerate(LABELS):
            if true_index != pred_index and matrix[true_index][pred_index]:
                confusions.append(
                    {"true_label": true_label, "predicted_label": pred_label, "count": matrix[true_index][pred_index]}
                )
    confusions.sort(key=lambda item: item["count"], reverse=True)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "main_confusions": confusions[:10],
    }


class JsonlDataset:
    def __init__(self, rows: list[dict], label2id: dict[str, int]) -> None:
        self.rows = rows
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {"text": row["text"], "label": self.label2id[row["label"]]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RuBERT tiny classifier for task/question/answer/other.")
    parser.add_argument("--train", default="datasets/train.jsonl")
    parser.add_argument("--val", default="datasets/val.jsonl")
    parser.add_argument("--test", default="datasets/test.jsonl")
    parser.add_argument("--output", default="models/rubert_classifier")
    parser.add_argument("--model", default=settings.RUBERT_TASK_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup = require_ml_dependencies()
    from torch.utils.data import DataLoader
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}

    train_rows = read_jsonl(Path(args.train))
    val_rows = read_jsonl(Path(args.val))
    test_rows = read_jsonl(Path(args.test))
    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("Train/val/test files must contain labeled examples.")

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        label2id=label2id,
        id2label=id2label,
    )
    model.to(device)

    def collate(batch: list[dict]) -> dict:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {key: value.to(device) for key, value in encoded.items()}

    train_loader = DataLoader(JsonlDataset(train_rows, label2id), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(JsonlDataset(val_rows, label2id), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(JsonlDataset(test_rows, label2id), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().cpu())
        print(f"epoch={epoch + 1} loss={total_loss / max(1, len(train_loader)):.4f}")

    def predict(loader: DataLoader) -> tuple[list[int], list[int]]:
        model.eval()
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels")
                logits = model(**batch).logits
                predictions = torch.argmax(logits, dim=-1)
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(predictions.detach().cpu().tolist())
        return y_true, y_pred

    val_true, val_pred = predict(val_loader)
    test_true, test_pred = predict(test_loader)
    val_metrics = compute_metrics(val_true, val_pred)
    test_metrics = compute_metrics(test_true, test_pred)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    labels_payload = {"labels": LABELS, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    (output_dir / "labels.json").write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = {
        "model": args.model,
        "device": str(device),
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "test_size": len(test_rows),
        "class_distribution": {
            "train": dict(Counter(row["label"] for row in train_rows)),
            "val": dict(Counter(row["label"] for row in val_rows)),
            "test": dict(Counter(row["label"] for row in test_rows)),
        },
        "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "per_class_precision": {label: values["precision"] for label, values in test_metrics["per_class"].items()},
        "per_class_recall": {label: values["recall"] for label, values in test_metrics["per_class"].items()},
        "per_class_f1": {label: values["f1"] for label, values in test_metrics["per_class"].items()},
        "confusion_matrix": test_metrics["confusion_matrix"],
        "main_confusions": test_metrics["main_confusions"],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

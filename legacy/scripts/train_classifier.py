from __future__ import annotations

"""
train_classifier.py — скрипт донастройки RuBERT-tiny2
для классификации сегментов проектных встреч.

Использование:
    1. Подготовить обучающую выборку:
        datasets/train_balanced.jsonl
        datasets/val_balanced.jsonl

    2. Запустить обучение:
        python train_classifier.py

    3. Дообученная модель сохранится в:
        models/classifier/

Формат датасета JSONL:
    {"text": "Подготовь отчёт до пятницы.", "label": "task"}
    {"text": "Кто отвечает за презентацию?", "label": "question"}
    {"text": "Да, я сделаю это сегодня.", "label": "answer"}
    {"text": "Коллеги, всем доброе утро.", "label": "other"}
"""

import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ============================
# Настройки
# ============================

MODEL_NAME = "cointegrated/rubert-tiny2"

DATASET_PATH = "datasets/train_balanced.jsonl"
VAL_DATASET_PATH = "datasets/val_balanced.jsonl"

OUTPUT_DIR = "models/classifier"
REPORT_DIR = "models/classifier_reports"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 4
MIN_SAMPLES_PER_CLASS_WARN = 60

LABEL_MAP = {
    "task": 0,
    "question": 1,
    "answer": 2,
    "other": 3,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
TARGET_NAMES = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]


# ============================
# Загрузка данных
# ============================

def load_jsonl(path: str) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    file_path = Path(path)

    if not file_path.exists():
        return texts, labels

    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(obj.get("text", "")).strip()
            label = str(obj.get("label", "")).strip().lower()

            if not text:
                continue

            if label not in LABEL_MAP:
                continue

            texts.append(text)
            labels.append(LABEL_MAP[label])

    return texts, labels


# ============================
# Dataset
# ============================

class MeetingDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ============================
# Utils
# ============================

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = Counter(labels)
    total = len(labels)

    weights = []

    for class_id in range(num_classes):
        count = counts.get(class_id, 1)
        weight = total / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


def validate_dataset_distribution(labels: list[int], title: str = "Распределение классов") -> None:
    counts = Counter(labels)

    print(f"\n{title}:")

    for label_name, label_id in LABEL_MAP.items():
        count = counts.get(label_id, 0)
        warn = " ⚠ МАЛО" if count < MIN_SAMPLES_PER_CLASS_WARN else ""
        print(f"  {label_name:8}: {count}{warn}")

    for label_name, label_id in LABEL_MAP.items():
        if counts.get(label_id, 0) == 0:
            raise RuntimeError(f"Класс {label_name} отсутствует в датасете.")


def safe_save_model(model, tokenizer, output_dir: str) -> None:
    """
    Безопасное сохранение модели:
    - сначала сохраняем во временную папку;
    - потом заменяем старую папку модели.
    Это снижает риск ошибки на Windows из-за блокировки файлов.
    """
    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.parent / f"{output_path.name}_tmp_{int(time.time())}"

    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)

    temp_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(temp_path)
    tokenizer.save_pretrained(temp_path)

    if output_path.exists():
        backup_path = output_path.parent / f"{output_path.name}_backup_{int(time.time())}"

        try:
            output_path.rename(backup_path)
            temp_path.rename(output_path)
            shutil.rmtree(backup_path, ignore_errors=True)
        except Exception as e:
            print(f"⚠ Не удалось заменить старую модель в {output_path}: {e}")
            print(f"Новая версия сохранена во временную папку: {temp_path}")
            return
    else:
        temp_path.rename(output_path)

    print(f"✅ Модель сохранена в: {output_path}")


def save_training_report(
    best_epoch: int,
    best_val_acc: float,
    best_macro_f1: float,
    best_weighted_f1: float,
    best_report_text: str,
    best_report_dict: dict,
    train_counts: Counter,
    val_counts: Counter,
) -> None:
    report_path = Path(REPORT_DIR)
    report_path.mkdir(parents=True, exist_ok=True)

    txt_path = report_path / "classification_report.txt"
    json_path = report_path / "classification_report.json"

    txt_content = (
        "Итоговый отчёт по донастройке RuBERT-tiny2\n"
        "=" * 60
        + "\n\n"
        f"Модель: {MODEL_NAME}\n"
        f"Train dataset: {DATASET_PATH}\n"
        f"Validation dataset: {VAL_DATASET_PATH}\n"
        f"Лучшая эпоха: {best_epoch}\n"
        f"Validation accuracy: {best_val_acc:.4f}\n"
        f"Macro F1: {best_macro_f1:.4f}\n"
        f"Weighted F1: {best_weighted_f1:.4f}\n\n"
        "Train distribution:\n"
    )

    for i, name in ID_TO_LABEL.items():
        txt_content += f"  {name}: {train_counts.get(i, 0)}\n"

    txt_content += "\nValidation distribution:\n"

    for i, name in ID_TO_LABEL.items():
        txt_content += f"  {name}: {val_counts.get(i, 0)}\n"

    txt_content += "\nClassification report:\n"
    txt_content += best_report_text

    txt_path.write_text(txt_content, encoding="utf-8")

    json_content = {
        "model": MODEL_NAME,
        "train_dataset": DATASET_PATH,
        "validation_dataset": VAL_DATASET_PATH,
        "best_epoch": best_epoch,
        "validation_accuracy": best_val_acc,
        "macro_f1": best_macro_f1,
        "weighted_f1": best_weighted_f1,
        "train_counts": {ID_TO_LABEL[k]: int(v) for k, v in train_counts.items()},
        "val_counts": {ID_TO_LABEL[k]: int(v) for k, v in val_counts.items()},
        "classification_report": best_report_dict,
    }

    json_path.write_text(
        json.dumps(json_content, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n📄 Отчёт сохранён:")
    print(f"  {txt_path}")
    print(f"  {json_path}")


# ============================
# Обучение
# ============================

def train() -> None:
    print(f"Загрузка train из {DATASET_PATH}...")
    texts, labels = load_jsonl(DATASET_PATH)

    if not texts:
        print(f"ОШИБКА: файл {DATASET_PATH} пуст или не найден.")
        return

    print(f"Train examples: {len(texts)}")
    validate_dataset_distribution(labels, "Train distribution")

    print(f"\nЗагрузка validation из {VAL_DATASET_PATH}...")
    val_texts_file, val_labels_file = load_jsonl(VAL_DATASET_PATH)

    if val_texts_file:
        train_texts = texts
        train_labels = labels
        val_texts = val_texts_file
        val_labels = val_labels_file

        print(f"Validation examples: {len(val_texts)}")
        validate_dataset_distribution(val_labels, "Validation distribution")
    else:
        print("Отдельный validation-файл не найден. Выполняю stratified split из train.")
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=VAL_SPLIT,
            random_state=42,
            stratify=labels,
        )

        print(f"Split: train={len(train_texts)}, val={len(val_texts)}")
        validate_dataset_distribution(val_labels, "Validation distribution")

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print(f"\nЗагрузка модели {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=ID_TO_LABEL,
        label2id=LABEL_MAP,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model.to(device)

    class_weights = compute_class_weights(train_labels, len(LABEL_MAP)).to(device)

    print("\nClass weights:")

    for idx, weight in enumerate(class_weights.tolist()):
        print(f"  {ID_TO_LABEL[idx]}: {weight:.4f}")

    train_dataset = MeetingDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = MeetingDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = 0.0
    best_weighted_f1 = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_report_preds: list[int] = []
    best_report_labels: list[int] = []

    print(f"\n{'=' * 60}")
    print(f"Начинаю донастройку: {EPOCHS} эпох, batch_size={BATCH_SIZE}")
    print(f"{'=' * 60}\n")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            logits = outputs.logits
            loss = criterion(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

        train_acc = correct / total if total else 0.0
        avg_loss = total_loss / len(train_loader) if train_loader else 0.0

        model.eval()

        all_preds: list[int] = []
        all_labels_list: list[int] = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels_list.extend(batch["labels"].cpu().numpy().tolist())

        val_acc = accuracy_score(all_labels_list, all_preds)
        macro_f1 = f1_score(
            all_labels_list,
            all_preds,
            labels=list(range(len(LABEL_MAP))),
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            all_labels_list,
            all_preds,
            labels=list(range(len(LABEL_MAP))),
            average="weighted",
            zero_division=0,
        )

        marker = ""

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_weighted_f1 = weighted_f1
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_report_preds = all_preds[:]
            best_report_labels = all_labels_list[:]

            safe_save_model(model, tokenizer, OUTPUT_DIR)

            marker = " *** best macro-F1 ***"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train acc: {train_acc:.3f} | "
            f"Val acc: {val_acc:.3f} | "
            f"Macro F1: {macro_f1:.3f} | "
            f"Weighted F1: {weighted_f1:.3f}"
            f"{marker}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping: {EARLY_STOPPING_PATIENCE} эпох без улучшения. "
                f"Лучшая эпоха: {best_epoch}"
            )
            break

    print(f"\n{'=' * 60}")
    print("Итоговый classification report по лучшей эпохе:")
    print(f"{'=' * 60}")

    report_text = classification_report(
        best_report_labels,
        best_report_preds,
        labels=list(range(len(LABEL_MAP))),
        target_names=TARGET_NAMES,
        zero_division=0,
    )

    report_dict = classification_report(
        best_report_labels,
        best_report_preds,
        labels=list(range(len(LABEL_MAP))),
        target_names=TARGET_NAMES,
        zero_division=0,
        output_dict=True,
    )

    print(report_text)

    print(f"Лучшая val accuracy: {best_val_acc:.4f}")
    print(f"Лучший macro F1: {best_macro_f1:.4f}")
    print(f"Лучший weighted F1: {best_weighted_f1:.4f}")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"Модель сохранена в: {OUTPUT_DIR}/")

    save_training_report(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_macro_f1=best_macro_f1,
        best_weighted_f1=best_weighted_f1,
        best_report_text=report_text,
        best_report_dict=report_dict,
        train_counts=train_counts,
        val_counts=val_counts,
    )

    print("\nДля использования в app.py укажи в .env:")
    print(f"CLASSIFIER_DIR={OUTPUT_DIR}")


if __name__ == "__main__":
    train()
from __future__ import annotations

"""
train_classifier.py — Скрипт дообучения RuBERT для классификации сегментов встреч.

Использование:
    1. Положи датасет в datasets/train.jsonl
    2. python train_classifier.py
    3. Модель сохранится в models/classifier/

Формат датасета (JSONL):
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
from sklearn.metrics import classification_report
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
DATASET_PATH = "datasets/train.jsonl"
VAL_DATASET_PATH = "datasets/val.jsonl"  # опционально
OUTPUT_DIR = "models/classifier"

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

LABEL_MAP = {"task": 0, "question": 1, "answer": 2, "other": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


# ============================
# Загрузка данных
# ============================
def load_jsonl(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    file_path = Path(path)
    if not file_path.exists():
        return texts, labels

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "").strip()
            label = obj.get("label", "").strip().lower()

            if text and label in LABEL_MAP:
                texts.append(text)
                labels.append(LABEL_MAP[label])

    return texts, labels


# ============================
# Dataset
# ============================
class MeetingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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

    weights = torch.tensor(weights, dtype=torch.float)
    return weights


def safe_save_model(model, tokenizer, output_dir: str) -> None:
    """
    Безопасное сохранение модели:
    - сначала во временную папку
    - потом замена папки целиком
    Это снижает шанс падения на Windows из-за блокировки .safetensors
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


def validate_dataset_distribution(labels: list[int]) -> None:
    counts = Counter(labels)
    print("\nПроверка распределения классов:")
    for label_name, label_id in LABEL_MAP.items():
        count = counts.get(label_id, 0)
        warn = " ⚠ МАЛО" if count < MIN_SAMPLES_PER_CLASS_WARN else ""
        print(f"  {label_name:8}: {count}{warn}")

    if counts.get(LABEL_MAP["task"], 0) == 0:
        raise RuntimeError("Класс task отсутствует! Сначала исправьте prepare_dataset.py и balance_dataset.py")


# ============================
# Обучение
# ============================
def train():
    # Загрузка данных
    print(f"Загрузка данных из {DATASET_PATH}...")
    texts, labels = load_jsonl(DATASET_PATH)

    if not texts:
        print(f"ОШИБКА: файл {DATASET_PATH} пуст или не найден!")
        return

    print(f"Всего примеров: {len(texts)}")
    validate_dataset_distribution(labels)

    # Загрузка val или разбиение
    val_texts_file, val_labels_file = load_jsonl(VAL_DATASET_PATH)

    if val_texts_file:
        train_texts, train_labels = texts, labels
        val_texts, val_labels = val_texts_file, val_labels_file
        print(f"\nИспользуется отдельный val: {len(val_texts)} примеров")
    else:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=VAL_SPLIT,
            random_state=42,
            stratify=labels,
        )
        print(f"\nРазбиение: train={len(train_texts)}, val={len(val_texts)}")

    # Модель и токенизатор
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

    # Веса классов
    class_weights = compute_class_weights(train_labels, len(LABEL_MAP)).to(device)
    print("\nClass weights:")
    for idx, w in enumerate(class_weights.tolist()):
        print(f"  {ID_TO_LABEL[idx]}: {w:.4f}")

    # Датасеты и загрузчики
    train_dataset = MeetingDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = MeetingDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Оптимизатор и scheduler
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

    # Loss
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Цикл обучения
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_report_preds = []
    best_report_labels = []

    print(f"\n{'=' * 60}")
    print(f"Начинаю обучение: {EPOCHS} эпох, batch_size={BATCH_SIZE}")
    print(f"{'=' * 60}\n")

    for epoch in range(EPOCHS):
        # Train
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
        avg_loss = total_loss / len(train_loader) if len(train_loader) else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == batch["labels"]).sum().item()
                val_total += len(batch["labels"])
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels_list.extend(batch["labels"].cpu().numpy().tolist())

        val_acc = val_correct / val_total if val_total else 0.0

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_report_preds = all_preds[:]
            best_report_labels = all_labels_list[:]
            safe_save_model(model, tokenizer, OUTPUT_DIR)
            marker = " *** best ***"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train: {train_acc:.3f} | "
            f"Val: {val_acc:.3f}{marker}"
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"\nEarly stopping: {EARLY_STOPPING_PATIENCE} эпох без улучшения. "
                f"Лучшая эпоха: {best_epoch}"
            )
            break

    # Итог
    print(f"\n{'=' * 60}")
    print("Итоговый classification report (лучшая эпоха):")
    print(f"{'=' * 60}")
    target_names = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]
    print(
        classification_report(
            best_report_labels,
            best_report_preds,
            target_names=target_names,
            zero_division=0,
        )
    )

    print(f"Лучшая val accuracy: {best_val_acc:.3f}")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"Модель сохранена в: {OUTPUT_DIR}/")
    print("\nДля использования в app.py:")
    print(f'  CLASSIFIER_DIR = "{OUTPUT_DIR}"')


if __name__ == "__main__":
    train()
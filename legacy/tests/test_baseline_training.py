import json
from pathlib import Path

from scripts.train_baseline_classifier import train_baseline


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_baseline_training_handles_small_dataset_with_warning(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    output = tmp_path / "model"
    train_rows = [
        {"text": "Нужно подготовить отчёт.", "label": "task"},
        {"text": "Подготовить презентацию.", "label": "task"},
        {"text": "Когда будет готов макет?", "label": "question"},
        {"text": "Какой срок у задачи?", "label": "question"},
    ]
    val_rows = [
        {"text": "Нужно проверить интеграцию.", "label": "task"},
        {"text": "Когда будет релиз?", "label": "question"},
    ]
    write_jsonl(train, train_rows)
    write_jsonl(val, val_rows)

    metrics = train_baseline(train, val, output)

    assert metrics["status"] == "experimental_baseline"
    assert metrics["warnings"]
    assert (output / "model.joblib").exists()
    assert (output / "metrics.json").exists()

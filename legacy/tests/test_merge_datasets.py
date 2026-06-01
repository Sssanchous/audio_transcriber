import json
from pathlib import Path

from scripts.merge_datasets import merge_datasets
from scripts.split_dataset import stratified_split


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_merge_datasets_deduplicates_and_adds_classes(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    seed = tmp_path / "seed.jsonl"
    output = tmp_path / "merged.jsonl"
    base_row = {
        "id": "base_1",
        "source_file": "base",
        "fragment_index": 1,
        "text": "Нужно подготовить отчёт.",
        "label": "task",
        "secondary_labels": [],
        "metadata": {"language": "ru"},
    }
    write_jsonl(base, [base_row])
    write_jsonl(seed, [base_row | {"id": "dup"}, base_row | {"id": "ans_1", "text": "Да, я проверил.", "label": "answer"}])

    stats = merge_datasets([base, seed], output)
    rows = [json.loads(line) for line in output.open(encoding="utf-8")]

    assert stats["total"] == 2
    assert stats["skipped"]["duplicate"] == 1
    assert {row["label"] for row in rows} == {"task", "answer"}


def test_enriched_split_keeps_all_rows():
    rows = [
        {"label": "task", "text": f"task {i}"} for i in range(10)
    ] + [
        {"label": "answer", "text": f"answer {i}"} for i in range(5)
    ]

    train, val, test = stratified_split(rows, seed=42, strategy="seed")

    assert len(train) + len(val) + len(test) == len(rows)
    assert train
    assert val
    assert test

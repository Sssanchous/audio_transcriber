import json
import subprocess
import sys
from pathlib import Path

from pm_insights.dataset.protocol_parser import parse_protocol_text


SAMPLE = """
Дата: 15.03.2026
Время: 20:00
Формат: онлайн
Присутствовали: Плотоненко Юрий Анатольевич, Сидоров Владислав Евгеньевич
Тема встречи: Рабочая встреча по ВКР

На встрече обсуждались следующие вопросы:
1. Обсуждение результатов внедрения Qdrant в подсистему
2. Обсуждение пула LLM для тестирования на сервере

До следующей встречи подготовить:
1. Распарсить данные с сайта ТОГИРРО, загрузить в векторную БД
2. Развернуть и провести тестирование LLM на сервере

Поставленные задачи по итогам встречи:
1. Сформировать перечень доработок

Принятые решения:
1. Решили разворачивать на собственных ресурсах

Итог встречи:
Работы продолжаются по согласованному плану.
"""


def test_protocol_parser_extracts_structured_sections():
    record = parse_protocol_text(SAMPLE, "demo.txt")
    assert "Плотоненко Юрий Анатольевич" in record.participants
    assert len(record.tasks) == 3
    assert any("Распарсить данные" in item["text"] for item in record.tasks)
    assert len(record.decisions) == 1
    assert record.decisions[0]["label"] == "decision"
    assert record.discussion_items[0]["label"] == "discussion_item"
    assert record.summary


def test_build_protocol_dataset_script(tmp_path):
    input_dir = tmp_path / "transcripts"
    input_dir.mkdir()
    (input_dir / "protocol.txt").write_text(SAMPLE, encoding="utf-8")
    output = tmp_path / "protocol_dataset.jsonl"
    references = tmp_path / "refs.json"
    stats = tmp_path / "stats.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_protocol_dataset.py",
            "--input",
            str(input_dir),
            "--output",
            str(output),
            "--references-output",
            str(references),
            "--stats-output",
            str(stats),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    labels = {row["label"] for row in rows}
    assert {"task", "discussion_item", "decision", "summary"} <= labels
    assert json.loads(stats.read_text(encoding="utf-8"))["participants_extracted"] == 2

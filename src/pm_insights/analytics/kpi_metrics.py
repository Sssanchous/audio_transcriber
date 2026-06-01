from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PASS_MAX_PROCESSING_1H_MINUTES = 20.0


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_nested(data: dict, *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_audio_duration_seconds(result: dict) -> float | None:
    candidates = [
        result.get("audio_duration_seconds"),
        result.get("duration_seconds"),
        _get_nested(result, "metadata", "audio_duration_seconds"),
        _get_nested(result, "metadata", "duration_seconds"),
        _get_nested(result, "processing_metrics", "audio_duration_seconds"),
        _get_nested(result, "metrics", "audio_duration_seconds"),
    ]
    for value in candidates:
        number = _to_float(value)
        if number and number > 0:
            return number
    return None


def _extract_processing_seconds(result: dict) -> float | None:
    candidates = [
        result.get("processing_time_seconds"),
        result.get("total_processing_seconds"),
        _get_nested(result, "processing_metrics", "processing_time_seconds"),
        _get_nested(result, "processing_metrics", "total_seconds"),
        _get_nested(result, "metrics", "processing_time_seconds"),
        _get_nested(result, "metrics", "total_processing_seconds"),
    ]
    for value in candidates:
        number = _to_float(value)
        if number is not None and number >= 0:
            return number
    return None


def _extract_estimated_1h_minutes(result: dict) -> float | None:
    candidates = [
        result.get("estimated_1h_processing_minutes"),
        _get_nested(result, "processing_metrics", "estimated_1h_processing_minutes"),
        _get_nested(result, "metrics", "estimated_1h_processing_minutes"),
    ]
    for value in candidates:
        number = _to_float(value)
        if number is not None and number >= 0:
            return number

    duration = _extract_audio_duration_seconds(result)
    processing = _extract_processing_seconds(result)

    if duration and processing is not None and duration > 0:
        return processing / duration * 60.0

    return None


def _average(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def load_classifier_eval_metrics(
    path: str | Path = "results/eval_active_rubert.json",
) -> dict | None:
    file_path = Path(path)
    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None

    allowed_keys = [
        "test_accuracy",
        "test_macro_f1",
        "val_accuracy",
        "val_macro_f1",
        "per_class_precision",
        "per_class_recall",
        "per_class_f1",
        "confusion_matrix",
        "main_confusions",
    ]

    result = {key: data.get(key) for key in allowed_keys if key in data}

    if not result:
        return None

    result["source"] = "prepared_dataset"
    result["source_label"] = "Оценка на подготовленном датасете"
    return result


def build_kpi_report(
    meeting_result: dict | None = None,
    evaluation_metrics: dict | None = None,
    project_results: list[dict] | None = None,
) -> dict:
    meeting_result = meeting_result or {}
    project_results = project_results or []

    processing_values: list[float] = []

    if project_results:
        for item in project_results:
            value = _extract_estimated_1h_minutes(item)
            if value is not None:
                processing_values.append(value)
    else:
        value = _extract_estimated_1h_minutes(meeting_result)
        if value is not None:
            processing_values.append(value)

    avg_processing_1h = _average(processing_values)

    if avg_processing_1h is None:
        processing_status = "not_measured"
        processing_status_label = "Нет данных"
        processing_value = None
        processing_comment = "Нет данных о длительности аудио или времени обработки."
    elif avg_processing_1h <= PASS_MAX_PROCESSING_1H_MINUTES:
        processing_status = "passed"
        processing_status_label = "Выполнено"
        processing_value = avg_processing_1h
        processing_comment = f"Средняя оценка обработки 1 часа аудио: {avg_processing_1h} мин."
    else:
        processing_status = "failed"
        processing_status_label = "Не выполнено"
        processing_value = avg_processing_1h
        processing_comment = f"Средняя оценка обработки 1 часа аудио: {avg_processing_1h} мин."

    classifier_metrics = evaluation_metrics or load_classifier_eval_metrics()

    kpi_items = [
        {
            "id": "wer",
            "name": "Точность транскрибации (WER)",
            "target": "≤ 15%",
            "value": None,
            "unit": "%",
            "status": "requires_reference",
            "status_label": "Требуется эталонный транскрипт",
            "method": "Сравнение ASR-текста с ручным эталонным транскриптом",
            "comment": "Для расчёта WER нужен ручной эталонный транскрипт. Без него значение не рассчитывается автоматически.",
        },
        {
            "id": "task_precision_recall",
            "name": "Точность выделения задач (Precision/Recall)",
            "target": "≥ 80%",
            "value": None,
            "unit": "%",
            "status": "requires_expert_annotation",
            "status_label": "Требуется экспертная разметка поручений",
            "method": "Сравнение найденных задач с экспертной разметкой",
            "comment": "Для проверки на реальных встречах нужна экспертная разметка задач. Метрики классификатора на подготовленном датасете показываются отдельно.",
            "classifier_dataset_metrics": classifier_metrics,
        },
        {
            "id": "sentiment_accuracy",
            "name": "Точность sentiment-анализа (Accuracy)",
            "target": "≥ 75%",
            "value": None,
            "unit": "%",
            "status": "requires_expert_annotation",
            "status_label": "Требуется экспертная оценка тональности",
            "method": "Сравнение автоматической тональности с экспертной оценкой для 3 классов",
            "comment": "Sentiment-модель не дообучалась на экспертной разметке встреч. Для расчёта accuracy нужна ручная оценка позитивных, нейтральных и негативных фрагментов.",
        },
        {
            "id": "processing_time_1h",
            "name": "Время обработки 1 часа аудио",
            "target": "≤ 15–20 мин",
            "value": processing_value,
            "unit": "мин",
            "status": processing_status,
            "status_label": processing_status_label,
            "method": "Автоматический расчёт по фактическому времени обработки",
            "comment": processing_comment,
        },
    ]

    measured_items = [item for item in kpi_items if item["value"] is not None]
    passed_items = [item for item in kpi_items if item["status"] == "passed"]
    requires_annotation_items = [
        item
        for item in kpi_items
        if item["status"] in {"requires_reference", "requires_expert_annotation"}
    ]

    if passed_items and requires_annotation_items:
        overall_status = "partial"
    elif len(passed_items) == len(kpi_items):
        overall_status = "passed"
    elif any(item["status"] == "failed" for item in kpi_items):
        overall_status = "failed"
    else:
        overall_status = "not_measured"

    return {
        "kpi_items": kpi_items,
        "summary": {
            "measured_count": len(measured_items),
            "passed_count": len(passed_items),
            "requires_annotation_count": len(requires_annotation_items),
            "overall_status": overall_status,
        },
    }


def build_kpi_report_from_file(path: str | Path) -> dict:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as file:
        result = json.load(file)

    return build_kpi_report(meeting_result=result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PM Insights KPI report")
    parser.add_argument("--input", required=True, help="Path to meeting result JSON")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    report = build_kpi_report_from_file(args.input)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

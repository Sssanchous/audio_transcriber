import json
from io import BytesIO
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
from fastapi.testclient import TestClient

from pm_insights.api.main import app


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_health_smoke():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_manual_examples_are_four_class_dataset():
    path = Path("datasets/sources/manual_examples.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["label"] for row in rows}
    counts = Counter(row["label"] for row in rows)
    assert labels == {"task", "question", "answer", "other"}
    assert all(counts[label] >= 500 for label in labels)
    assert not any("????" in row["text"] for row in rows)
    assert not any("Рђ" in row["text"] or "Р°" in row["text"] for row in rows)


def test_training_dataset_is_four_class_dataset():
    path = Path("datasets/training_dataset.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["label"] for row in rows}
    assert labels == {"task", "question", "answer", "other"}
    assert len(rows) > 2000
    assert not any("????" in row["text"] for row in rows)


def test_real_hard_examples_are_clean_four_class_examples():
    path = Path("datasets/sources/real_hard_examples.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {row["label"] for row in rows}
    assert rows
    assert labels <= {"task", "question", "answer", "other"}
    assert "other" in labels
    assert not any("????" in row["text"] for row in rows)
    assert not any("Рђ" in row["text"] or "Р°" in row["text"] for row in rows)


def test_saved_baseline_predicts_real_meeting_phrases():
    model_path = Path("models/baseline_classifier/model.joblib")
    assert model_path.exists()
    model = joblib.load(model_path)
    checks = [
        ("Алексей, Анна, подготовь финальный отчет по аналитике до пятницы", "task"),
        ("Алексей, когда мы отправим презентацию клиенту?", "question"),
        ("Анна, я отправлю презентацию сегодня после обеда", "answer"),
        ("Коллеги, всем доброе утро. Начинаем еженедельную встречу по проекту.", "other"),
        ("Иван. Нет, релиз переносить не нужно", "answer"),
        ("Алексей, можно ли согласовать бюджет сегодня?", "question"),
    ]
    predictions = model.predict([text for text, _ in checks])
    assert list(predictions) == [expected for _, expected in checks]


def test_baseline_training_smoke(tmp_path):
    rows = []
    examples = {
        "task": ["Анна, подготовь отчёт.", "Иван, исправь ошибку.", "Мария, проверь сборку."],
        "question": ["Когда готов отчёт?", "Кто отвечает за сборку?", "Можно ли согласовать бюджет?"],
        "answer": ["Да, я подготовлю отчёт.", "Нет, переносить не нужно.", "Файл уже лежит в папке."],
        "other": ["Коллеги, доброе утро.", "Начинаем встречу.", "Сегодня обсуждаем статус."],
    }
    for label, texts in examples.items():
        for index, text in enumerate(texts):
            rows.append({"id": f"{label}_{index}", "text": text, "label": label})

    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    test = tmp_path / "test.jsonl"
    write_jsonl(train, rows)
    write_jsonl(val, rows)
    write_jsonl(test, rows)
    output = tmp_path / "model"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_classifier.py",
            "--train",
            str(train),
            "--val",
            str(val),
            "--test",
            str(test),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "model.joblib").exists()
    assert (output / "metrics.json").exists()


def test_evaluation_scripts_skip_missing_references():
    scripts = [
        ("scripts/evaluate_asr_wer.py", "ASR evaluation skipped"),
        ("scripts/evaluate_task_extraction.py", "Task extraction evaluation skipped"),
        ("scripts/evaluate_sentiment.py", "Sentiment evaluation skipped"),
    ]
    for script, expected in scripts:
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--reference",
                "eval_data/missing_reference.json",
                "--prediction",
                "results/example_result_fixed.json",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert expected in result.stdout


def test_optional_sentiment_and_topic_fallbacks(monkeypatch):
    from pm_insights import settings
    from pm_insights.nlp.sentiment import analyze_sentiment
    from pm_insights.nlp.topic_modeling import build_topics, extract_topics

    fragments = [{"fragment_index": 1, "text": "Отлично, все получилось по серверу."}]
    assert analyze_sentiment(fragments)[0]["sentiment"] in {"positive", "neutral", "negative"}

    monkeypatch.setattr(settings, "ENABLE_MODEL_FALLBACK", True)
    rubert_result = analyze_sentiment(fragments, engine="rubert")[0]
    assert rubert_result["engine"] in {"rubert", "rule_based"}

    fallback_topics = build_topics(fragments, engine="fallback")
    bertopic_topics = build_topics(fragments, engine="bertopic")
    assert isinstance(fallback_topics, list)
    assert isinstance(bertopic_topics, list)

    topic_result = extract_topics(fragments, engine="auto", max_topics=4)
    assert topic_result["source"] in {"bertopic", "embedding_clustering", "rule_based_fallback"}
    assert isinstance(topic_result["topics"], list)
    assert isinstance(topic_result["aspect_frequencies"], dict)
    assert isinstance(topic_result["warnings"], list)


def test_unknown_domain_topic_fallback_is_not_empty():
    from pm_insights.nlp.topic_modeling import extract_topics

    fragments = [
        {"fragment_index": 1, "text": "На встрече обсуждали стерилизацию инструментов, журнал контроля и обучение персонала клиники."},
        {"fragment_index": 2, "text": "Отдельно разобрали график медицинских осмотров и подготовку кабинетов к проверке."},
        {"fragment_index": 3, "text": "Юридический блок включал согласование договора, персональные данные и хранение документов."},
    ]
    result = extract_topics(fragments, engine="rule_based", max_topics=3)
    assert result["source"] == "rule_based_fallback"
    assert result["topics"]
    assert result["aspect_frequencies"]


def test_fragment_classifier_rule_based_and_baseline_safe():
    from pm_insights.nlp.fragment_classifier import classify_fragment

    question = classify_fragment("Когда будет готов финальный отчет?", engine="rule_based")
    task = classify_fragment("Анна, подготовь отчет до пятницы.", engine="rule_based")
    assert question["label"] == "question"
    assert task["label"] == "task"

    baseline = classify_fragment("Коллеги, начинаем встречу по проекту.", engine="baseline")
    assert baseline["label"] in {"task", "question", "answer", "other"}


def test_pipeline_adds_processing_time(tmp_path):
    from pm_insights.meeting.pipeline import analyze_meeting

    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"fake-audio")
    result = analyze_meeting(audio, transcript_text="Нужно подготовить отчет до пятницы.", output_dir=tmp_path)
    processing_time = result["metadata"]["processing_time"]
    assert "total_processing_seconds" in processing_time
    assert "estimated_1h_processing_minutes" in processing_time


def test_dashboard_project_summary_payload(monkeypatch):
    from pm_insights.api import main as api_main

    monkeypatch.setattr(api_main.settings, "REQUIRE_AUTH", False)
    monkeypatch.setattr(api_main.db, "list_meetings", lambda user_id=None: [])
    response = TestClient(app).get("/dashboard")
    assert response.status_code == 200
    payload = response.json()
    assert "kpi_targets" not in payload
    assert "kpi_status" not in payload
    assert "business_kpi" not in payload
    assert "projects" in payload
    assert "summary" in payload
    assert "task_trend" in payload
    assert "sentiment_trend" in payload
    assert "aspect_word_cloud" in payload


def test_json_export_endpoint(monkeypatch):
    from pm_insights.api import main as api_main

    monkeypatch.setattr(api_main.settings, "REQUIRE_AUTH", False)
    monkeypatch.setattr(api_main.db, "get_result", lambda meeting_id, user_id=None: {"meeting_id": meeting_id, "tasks": []})
    response = TestClient(app).get("/meetings/demo/export/json")
    assert response.status_code == 200
    assert response.json()["meeting_id"] == "demo"
    assert "attachment" in response.headers["content-disposition"]


def test_binary_export_endpoints(monkeypatch):
    from docx import Document
    from openpyxl import load_workbook
    from pm_insights.api import main as api_main

    result = {
        "meeting_id": "demo",
        "metadata": {"meeting_info": {"meeting_title": "Demo", "project_name": "PM Insights"}},
        "clean_tasks": [{"title": "Подготовить отчет", "summary": "Кратко оформить результаты"}],
        "report_sections": [{"id": "tasks", "title": "Задачи", "items": [{"title": "Подготовить отчет"}]}],
        "transcript": [{"start": 0, "end": 1, "text": "Нужно подготовить отчет."}],
    }
    monkeypatch.setattr(api_main.settings, "REQUIRE_AUTH", False)
    monkeypatch.setattr(api_main.db, "get_result", lambda meeting_id, user_id=None: result | {"meeting_id": meeting_id})
    client = TestClient(app)

    xlsx = client.get("/meetings/demo/export/xlsx")
    assert xlsx.status_code == 200
    workbook = load_workbook(BytesIO(xlsx.content))
    assert "Summary" in workbook.sheetnames
    assert "Tasks" in workbook.sheetnames
    assert workbook["Summary"]["A2"].value == "Система"

    docx = client.get("/meetings/demo/export/docx")
    assert docx.status_code == 200
    document = Document(BytesIO(docx.content))
    assert any("PM Insights" in paragraph.text for paragraph in document.paragraphs)
    assert not any("????" in paragraph.text for paragraph in document.paragraphs)

    pdf = client.get("/meetings/demo/export/pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")


def test_export_recovers_corrupted_section_titles():
    from docx import Document
    from openpyxl import load_workbook
    from pm_insights.export.report_builder import (
        _section_title,
        build_report_payload,
        export_docx,
        export_pdf,
        export_xlsx,
        register_cyrillic_pdf_font,
    )

    damaged_sections = [
        {"id": "research_actions", "title": "????????????????? ????????", "items": [{"title": "Проверить модель"}]},
        {"id": "recommendations", "title": "????????????", "items": [{"title": "Начать со скин-фактора"}]},
        {"id": "research_notes", "title": "????????????????? ??????? / ??????????? ????????", "items": [{"title": "R-квадрат почти единичный"}]},
        {"id": "questions_answers", "title": "??????? ? ??????", "items": [{"question_title": "Что проверить?", "answer_summary": "Частный случай."}]},
        {"id": "deadlines", "title": "???????? / ????????? ???????", "items": [{"deadline": "четверг 19:30"}]},
        {"id": "topics", "title": "???????? ????", "items": [{"topic_name": "модель"}]},
        {"id": "review_items", "title": "??????? ????????", "items": [{"text": "спорный элемент"}]},
        {"id": "transcript", "title": "??????????", "items": []},
    ]
    assert _section_title(damaged_sections[0]) == "Исследовательские действия"
    assert _section_title(damaged_sections[2]) == "Исследовательские заметки / технический контекст"
    assert _section_title(damaged_sections[3]) == "Вопросы и ответы"

    result = {
        "meeting_id": "technical-demo",
        "meeting_type": {"label": "technical_research"},
        "metadata": {"meeting_info": {"meeting_title": "Техническая встреча", "project_name": "ВКР"}},
        "report_sections": damaged_sections,
        "transcript": [{"start": 0, "end": 1, "text": "Проверить модель."}],
    }
    report = build_report_payload(result)
    docx_bytes = export_docx(report)
    document = Document(BytesIO(docx_bytes))
    paragraphs = "\n".join(paragraph.text for paragraph in document.paragraphs)
    for expected in [
        "Задачи",
        "Вопросы и ответы",
        "Дедлайны",
        "Аспекты и темы",
        "Транскрипт",
    ]:
        assert expected in paragraphs
    assert "????" not in paragraphs

    workbook = load_workbook(BytesIO(export_xlsx(report)))
    assert workbook["Summary"]["A2"].value == "Система"
    assert workbook["Summary"]["B4"].value == "ВКР"

    assert register_cyrillic_pdf_font()
    pdf = export_pdf(report)
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


def test_rubert_training_script_help():
    result = subprocess.run(
        [sys.executable, "scripts/train_rubert_classifier.py", "--help"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "Train RuBERT tiny classifier" in result.stdout


def test_prepare_training_maps_protocol_discussion_to_other():
    from scripts.prepare_training_dataset import normalize_record

    row = {"text": "Обсуждение результатов внедрения Qdrant", "label": "discussion_item"}
    normalized = normalize_record(row, 1, "protocol_dataset")
    assert normalized["label"] == "other"
    assert normalized["original_label"] == "discussion_item"

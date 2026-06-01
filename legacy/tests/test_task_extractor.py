from pm_insights.nlp.task_extractor import extract_tasks


def test_task_extractor_extracts_russian_task():
    tasks = extract_tasks([{"fragment_index": 1, "text": "Нужно подготовить отчёт до пятницы."}])

    assert len(tasks) == 1
    assert tasks[0]["deadline"] == "до пятницы"

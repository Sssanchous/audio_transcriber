from pm_insights.nlp.deadline_extractor import extract_deadlines


def test_deadline_extractor_finds_dates_and_phrases():
    items = extract_deadlines([{"fragment_index": 1, "text": "Дедлайн 15.05.2026, закончить до пятницы."}])

    assert "15.05.2026" in items[0]["deadlines"]
    assert "до пятницы" in items[0]["deadlines"]

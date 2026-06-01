from pm_insights.nlp.responsible_extractor import extract_responsibles


def test_responsible_extractor_finds_responsible_name():
    items = extract_responsibles([{"fragment_index": 1, "text": "За это отвечает Иван."}])

    assert items[0]["responsibles"] == ["Иван"]

from pm_insights.nlp.aspects import extract_aspects


def test_aspects_extract_project_aspects():
    aspects = extract_aspects([{"fragment_index": 1, "text": "Есть проблема с сервером и сроками."}])

    assert "сервер" in aspects[0]["aspects"]
    assert "сроки" in aspects[0]["aspects"]

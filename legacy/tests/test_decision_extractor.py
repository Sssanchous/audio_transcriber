from pm_insights.nlp.decision_extractor import extract_decisions


def test_decision_extractor_finds_decision():
    decisions = extract_decisions([{"fragment_index": 1, "text": "Решили оставить текущую архитектуру."}])

    assert decisions

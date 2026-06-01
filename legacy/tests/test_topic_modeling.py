from pm_insights.nlp.topic_modeling import build_topics


def test_topic_modeling_uses_rule_based_fallback():
    topics = build_topics([{"fragment_index": 1, "text": "Дедлайн и сроки по релизу задерживаются."}])

    assert topics
    assert topics[0]["source"] == "rule_based_fallback"

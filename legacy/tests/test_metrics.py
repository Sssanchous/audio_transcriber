from pm_insights.analytics.metrics import calculate_metrics


def test_metrics_counts_main_entities():
    metrics = calculate_metrics(
        {
            "tasks": [{}],
            "questions_answers": [{"answer": "Да"}],
            "decisions": [{}],
            "deadlines": [{}],
            "responsibles": [{}],
            "sentiment": [{"sentiment": "negative", "score": -1.0}],
            "aspects": [{"aspects": ["сроки"]}],
            "topics": [{"topic_name": "сроки", "fragments": [1]}],
        }
    )

    assert metrics["tasks_count"] == 1
    assert metrics["negative_fragments_count"] == 1
    assert metrics["aspect_frequencies"]["сроки"] == 1

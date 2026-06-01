from pm_insights.nlp.sentiment import analyze_sentiment


def test_sentiment_rule_based_positive_and_negative():
    result = analyze_sentiment(
        [
            {"fragment_index": 1, "text": "Отлично, всё получилось."},
            {"fragment_index": 2, "text": "Есть риск и проблема со сроками."},
        ]
    )

    assert result[0]["sentiment"] == "positive"
    assert result[1]["sentiment"] == "negative"

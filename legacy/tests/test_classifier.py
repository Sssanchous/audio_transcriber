from pm_insights.dataset.classifier import classify_fragment


def test_classifier_priority_and_secondary_labels():
    result = classify_fragment("Нужно подготовить отчёт до пятницы.")

    assert result["label"] == "deadline"
    assert "task" in result["secondary_labels"]


def test_classifier_examples():
    assert classify_fragment("Когда будет готов макет?")["label"] == "question"
    assert classify_fragment("Да, я уже проверил.")["label"] == "answer"
    assert classify_fragment("Решили оставить текущую архитектуру.")["label"] == "decision"
    assert classify_fragment("Ответственный за задачу — Иван.")["label"] == "responsible"
    assert classify_fragment("Есть риск, что не успеваем к сроку.")["label"] == "sentiment_negative"
    assert classify_fragment("Отлично, всё получилось.")["label"] == "sentiment_positive"

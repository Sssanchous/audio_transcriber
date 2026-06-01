from __future__ import annotations

from statistics import mean

from pm_insights.nlp.aspects import aspect_frequencies


def calculate_metrics(result: dict) -> dict:
    sentiment = result.get("sentiment", [])
    topics = result.get("topics", [])
    scores = [item.get("score", 0.0) for item in sentiment]
    topic_counts = {item["topic_name"]: len(item.get("fragments", [])) for item in topics}

    return {
        "tasks_count": len(result.get("tasks", [])),
        "questions_count": len(result.get("questions_answers", [])),
        "answers_count": sum(1 for item in result.get("questions_answers", []) if item.get("answer")),
        "decisions_count": len(result.get("decisions", [])),
        "deadlines_count": len(result.get("deadlines", [])),
        "responsibles_count": len(result.get("responsibles", [])),
        "negative_fragments_count": sum(1 for item in sentiment if item.get("sentiment") == "negative"),
        "positive_fragments_count": sum(1 for item in sentiment if item.get("sentiment") == "positive"),
        "aspect_frequencies": aspect_frequencies(result.get("aspects", [])),
        "topic_frequencies": topic_counts,
        "average_sentiment": round(mean(scores), 3) if scores else 0.0,
    }

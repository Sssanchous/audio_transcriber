from __future__ import annotations


def build_trends(meetings: list[dict]) -> dict:
    ordered = sorted(meetings, key=lambda item: item.get("meeting_date") or item.get("metadata", {}).get("upload_date") or "")
    return {
        "dates": [item.get("meeting_date") for item in ordered],
        "tasks": [item.get("metrics", {}).get("tasks_count", 0) for item in ordered],
        "questions": [item.get("metrics", {}).get("questions_count", 0) for item in ordered],
        "decisions": [item.get("metrics", {}).get("decisions_count", 0) for item in ordered],
        "average_sentiment": [item.get("metrics", {}).get("average_sentiment", 0.0) for item in ordered],
    }

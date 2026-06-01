from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm_insights import db


LABEL_MAP = {
    "same_as_predicted": None,
    "accepted": None,
    "task": "task",
    "question": "question",
    "answer": "answer",
    "other": "other",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Export verified user feedback as JSONL training examples.")
    parser.add_argument("--output", default="datasets/sources/feedback_examples.jsonl")
    args = parser.parse_args()

    feedback = db.list_feedback(only_unused=False)
    rows = []
    qa_rows = []
    sentiment_rows = []
    topic_rows = []
    for item in feedback:
        corrected_label = item.get("corrected_label") or ""
        metadata = item.get("metadata") or {}
        item_type = item.get("item_type") or ""
        if item_type == "qa":
            qa_rows.append(
                {
                    "id": f"feedback_{item['id']}",
                    "question": metadata.get("question") or item.get("source_text") or "",
                    "answer": item.get("corrected_text") or metadata.get("answer") or "",
                    "status": corrected_label if corrected_label in {"answered", "partial", "not_answered"} else metadata.get("status", ""),
                    "source": "user_feedback",
                    "meeting_id": item.get("meeting_id"),
                    "user_id": str(item.get("user_id")),
                    "feedback_id": item.get("id"),
                }
            )
            continue
        if item_type == "sentiment":
            if corrected_label in {"positive", "neutral", "negative"}:
                sentiment_rows.append(
                    {
                        "id": f"feedback_{item['id']}",
                        "text": item.get("corrected_text") or item.get("source_text") or "",
                        "sentiment": corrected_label,
                        "source": "user_feedback",
                        "meeting_id": item.get("meeting_id"),
                        "user_id": str(item.get("user_id")),
                        "feedback_id": item.get("id"),
                    }
                )
            continue
        if item_type in {"topic", "aspect"}:
            topic_rows.append(
                {
                    "id": f"feedback_{item['id']}",
                    "text": item.get("source_text") or "",
                    "corrected_label": corrected_label,
                    "corrected_text": item.get("corrected_text") or "",
                    "item_type": item_type,
                    "source": "user_feedback",
                    "meeting_id": item.get("meeting_id"),
                    "user_id": str(item.get("user_id")),
                    "feedback_id": item.get("id"),
                }
            )
            continue
        label = LABEL_MAP.get(corrected_label, corrected_label)
        if corrected_label in {"same_as_predicted", "accepted"}:
            label = item.get("predicted_label")
        if label not in {"task", "question", "answer", "other"}:
            continue
        text = (item.get("corrected_text") or item.get("source_text") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "id": f"feedback_{item['id']}",
                "text": text,
                "label": label,
                "source": "user_feedback",
                "meeting_id": item.get("meeting_id"),
                "user_id": str(item.get("user_id")),
                "feedback_id": item.get("id"),
                "metadata": metadata,
            }
        )

    rows.extend(qa_rows)
    rows.extend(sentiment_rows)
    rows.extend(topic_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "feedback_rows": len(rows),
        "classifier_rows": len(rows) - len(qa_rows) - len(sentiment_rows) - len(topic_rows),
        "qa_rows": len(qa_rows),
        "sentiment_rows": len(sentiment_rows),
        "topic_rows": len(topic_rows),
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

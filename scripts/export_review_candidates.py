from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_review_candidates(result: dict) -> list[dict]:
    items: list[dict] = []
    for item in result.get("review_items", []):
        items.append({"source": "review_items", **item})

    for task in result.get("clean_tasks", []):
        confidence = float(task.get("confidence", 0.0) or 0.0)
        if task.get("review_required") or confidence < 0.75:
            items.append(
                {
                    "source": "clean_tasks",
                    "type": "task_candidate",
                    "text": task.get("source_text") or task.get("title"),
                    "clean_title": task.get("title"),
                    "confidence": confidence,
                    "reason": "review_required_task",
                    "source_fragments": [task.get("source_fragment")] if task.get("source_fragment") else [],
                }
            )

    for pair in result.get("clean_questions_answers", []):
        if pair.get("status") != "answered":
            items.append(
                {
                    "source": "clean_questions_answers",
                    "type": "qa_candidate",
                    "text": pair.get("question"),
                    "answer": pair.get("answer"),
                    "status": pair.get("status"),
                    "reason": "qa_not_fully_answered",
                    "source_fragments": pair.get("source_fragments", []),
                }
            )

    for deadline in result.get("clean_deadlines", []):
        if deadline.get("review_required"):
            items.append(
                {
                    "source": "clean_deadlines",
                    "type": "deadline_candidate",
                    "text": deadline.get("text"),
                    "deadline": deadline.get("deadline"),
                    "kind": deadline.get("kind"),
                    "confidence": deadline.get("confidence"),
                    "reason": "deadline_needs_review",
                    "source_fragments": [deadline.get("source_fragment")] if deadline.get("source_fragment") else [],
                }
            )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PM Insights review candidates from an analysis result.")
    parser.add_argument("--input", required=True, help="Path to result JSON.")
    parser.add_argument("--output", required=True, help="Path to output review JSON.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    result = load_json(input_path)
    output = {
        "source_result": input_path.name,
        "meeting_id": result.get("meeting_id"),
        "meeting_type": result.get("meeting_type"),
        "items": collect_review_candidates(result),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {len(output['items'])} review candidates to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

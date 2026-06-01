from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm_insights.dataset.protocol_parser import parse_protocol_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build structured dataset from meeting protocols.")
    parser.add_argument("--input", default="transcripts")
    parser.add_argument("--output", default="datasets/sources/protocol_dataset.jsonl")
    parser.add_argument("--references-output", default="eval_data/protocol_references.json")
    parser.add_argument("--stats-output", default="datasets/sources/protocol_dataset_stats.json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    rows: list[dict] = []
    references: list[dict] = []
    files_processed = 0
    files_failed = 0
    participants_extracted = 0

    for path in sorted(input_dir.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".docx"}:
            continue
        try:
            record = parse_protocol_file(path)
        except Exception as exc:
            files_failed += 1
            print(f"failed: {path}: {exc}")
            continue
        files_processed += 1
        participants_extracted += len(record.participants)

        for section_name, label in [
            ("discussion_items", "discussion_item"),
            ("tasks", "task"),
            ("decisions", "decision"),
        ]:
            for item in getattr(record, section_name):
                rows.append(
                    {
                        "text": item["text"],
                        "label": label,
                        "source_file": record.source_file,
                        "source_section": section_name,
                        "meeting_date": record.meeting_date,
                        "deadline": item.get("deadline"),
                        "source": "protocol_structured",
                    }
                )
        if record.summary:
            rows.append(
                {
                    "text": record.summary,
                    "label": "summary",
                    "source_file": record.source_file,
                    "source_section": "summary",
                    "meeting_date": record.meeting_date,
                    "source": "protocol_structured",
                }
            )

        references.append(
            {
                "source_file": record.source_file,
                "meeting_date": record.meeting_date,
                "meeting_time": record.meeting_time,
                "participants": record.participants,
                "discussion_items": [item["text"] for item in record.discussion_items],
                "tasks": [{"text": item["text"], "deadline": item.get("deadline")} for item in record.tasks],
                "decisions": [item["text"] for item in record.decisions],
            }
        )

    label_counts = dict(sorted(Counter(row["label"] for row in rows).items()))
    stats = {
        "files_processed": files_processed,
        "files_failed": files_failed,
        "meetings": len(references),
        "examples_total": len(rows),
        "labels": label_counts,
        "participants_extracted": participants_extracted,
    }

    write_jsonl(Path(args.output), rows)
    Path(args.references_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.references_output).write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.stats_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_output).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

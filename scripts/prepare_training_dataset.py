from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
SOURCES_DIR = DATASETS_DIR / "sources"


def source_path(filename: str) -> Path:
    preferred = SOURCES_DIR / filename
    legacy = DATASETS_DIR / filename
    return preferred if preferred.exists() else legacy


BASE_DATASET = source_path("pm_dataset.jsonl")
MANUAL_DATASET = source_path("manual_examples.jsonl")
REAL_HARD_DATASET = source_path("real_hard_examples.jsonl")
REAL_REVIEWED_DATASET = source_path("real_reviewed_examples.jsonl")
PROTOCOL_DATASET = source_path("protocol_dataset.jsonl")
FEEDBACK_DATASET = source_path("feedback_examples.jsonl")
OUTPUT_DATASET = ROOT / "datasets" / "training_dataset.jsonl"
REPORT_PATH = ROOT / "datasets" / "training_dataset_report.json"
STATS_PATH = ROOT / "datasets" / "dataset_stats.json"

ALLOWED_LABELS = {"task", "question", "answer", "other"}
LABEL_MAPPING = {
    "task": "task",
    "question": "question",
    "answer": "answer",
    "decision": "other",
    "deadline": "other",
    "responsible": "other",
    "aspect": "other",
    "sentiment_positive": "other",
    "sentiment_negative": "other",
    "sentiment_neutral": "other",
    "discussion_item": "other",
    "summary": "other",
    "other": "other",
}
SOURCE_PRIORITY = {
    "pm_dataset": 10,
    "manual_examples": 20,
    "real_hard_examples": 30,
    "real_reviewed_examples": 40,
    "protocol_dataset": 50,
    "feedback_examples": 100,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


AMBIGUOUS_AUTO_OTHER_RE = re.compile(
    r"\b(\u0440\u0435\u043a\u043e\u043c\u0435\u043d\u0434\u043e\u0432\u0430\u043d\u043e|"
    r"\u0441\u043b\u0435\u0434\u0443\u0435\u0442|\u043d\u0443\u0436\u043d\u043e|"
    r"\u043d\u0435\u043e\u0431\u0445\u043e\u0434\u0438\u043c\u043e|\u043f\u043e\u0441\u0442\u0440\u043e\u0438\u0442\u044c|"
    r"\u0438\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c|"
    r"\u043f\u043e\u043b\u0443\u0447\u0430\u0435\u0442\s+\u0443\u0432\u0435\u0434\u043e\u043c\u043b\u0435\u043d\u0438\u0435|"
    r"\u043f\u0440\u0438\u043d\u0438\u043c\u0430\u0435\u0442\u0441\u044f\s+\u0430\u0432\u0442\u043e\u043c\u0430\u0442\u0438\u0447\u0435\u0441\u043a\u0438)\b",
    re.IGNORECASE,
)
SUBORDINATE_WHEN_RE = re.compile(
    r"^\s*\u043a\u043e\u0433\u0434\u0430\s+(\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u0435\u043b\u044c|"
    r"\u0441\u0438\u0441\u0442\u0435\u043c\u0430|\u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435|"
    r"\u043c\u043e\u0434\u0443\u043b\u044c)\b",
    re.IGNORECASE,
)
MOJIBAKE_RE = re.compile(r"(\u0420\u00b0|\u0420\u00b5|\u0420\u0451|\u0420\u0455|\u0421\u201a|\u0421\u0403|\u0421\u201a|\u0432\u0402)")


def normalize_record(row: dict, index: int, source: str) -> dict | None:
    text = str(row.get("text", "")).strip()
    old_label = row.get("label")
    label = LABEL_MAPPING.get(old_label, old_label)
    if not text or label not in ALLOWED_LABELS:
        return None
    if "????" in text or MOJIBAKE_RE.search(text):
        return None
    if source == "pm_dataset":
        if label == "other" and AMBIGUOUS_AUTO_OTHER_RE.search(text):
            return None
        if label == "question" and "?" not in text and SUBORDINATE_WHEN_RE.search(text):
            return None
    return {
        "id": row.get("id") or f"{source}_{index:06d}",
        "source_file": row.get("source_file") or source,
        "text": text,
        "label": label,
        "original_label": old_label,
        "source": row.get("source") or source,
        "metadata": {
            **(row.get("metadata") or {}),
            "source": (row.get("metadata") or {}).get("source", source),
            "original_label": old_label,
        },
    }


def main() -> None:
    by_text: dict[str, dict] = {}
    priority_by_text: dict[str, int] = {}
    source_counts = {}
    duplicate_count = 0
    skipped_count = 0
    sources = [
        ("pm_dataset", BASE_DATASET),
        ("manual_examples", MANUAL_DATASET),
        ("real_hard_examples", REAL_HARD_DATASET),
        ("real_reviewed_examples", REAL_REVIEWED_DATASET),
        ("protocol_dataset", PROTOCOL_DATASET),
        ("feedback_examples", FEEDBACK_DATASET),
    ]

    for source_name, path in sources:
        source_rows = load_jsonl(path)
        source_counts[source_name] = len(source_rows)
        for index, row in enumerate(source_rows, 1):
            normalized = normalize_record(row, index, source_name)
            if not normalized:
                skipped_count += 1
                continue
            key = normalize_text(normalized["text"])
            priority = SOURCE_PRIORITY[source_name]
            if key in by_text:
                duplicate_count += 1
                if priority <= priority_by_text[key]:
                    continue
            priority_by_text[key] = priority
            by_text[key] = normalized

    rows = list(by_text.values())

    label_counts = dict(sorted(Counter(row["label"] for row in rows).items()))
    max_count = max(label_counts.values()) if label_counts else 0
    min_count = min(label_counts.values()) if label_counts else 0
    imbalance_warning = bool(min_count and max_count / min_count > 2.0)

    save_jsonl(OUTPUT_DATASET, rows)
    report = {
        "base_examples": source_counts.get("pm_dataset", 0),
        "manual_examples": source_counts.get("manual_examples", 0),
        "real_hard_examples": source_counts.get("real_hard_examples", 0),
        "real_reviewed_examples": source_counts.get("real_reviewed_examples", 0),
        "protocol_dataset": source_counts.get("protocol_dataset", 0),
        "feedback_examples": source_counts.get("feedback_examples", 0),
        "source_examples": source_counts,
        "duplicates_removed": duplicate_count,
        "skipped_invalid_or_ambiguous": skipped_count,
        "total_examples": len(rows),
        "label_counts": label_counts,
        "imbalance_warning": imbalance_warning,
        "allowed_labels": sorted(ALLOWED_LABELS),
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    STATS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

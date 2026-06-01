from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EXPECTED_LABELS = [
    "task",
    "question",
    "answer",
    "decision",
    "deadline",
    "responsible",
    "aspect",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_neutral",
    "other",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def label_status(count: int) -> str:
    if count < 50:
        return "very_weak"
    if count < 100:
        return "weak"
    if count < 300:
        return "limited"
    return "usable"


def build_readiness_report(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    total = len(rows)
    present = [label for label in EXPECTED_LABELS if counts.get(label, 0) > 0]
    missing = [label for label in EXPECTED_LABELS if counts.get(label, 0) == 0]
    statuses = {label: label_status(counts.get(label, 0)) for label in EXPECTED_LABELS}
    dominant_label, dominant_count = counts.most_common(1)[0] if counts else ("", 0)
    dominant_share = dominant_count / total if total else 0.0
    imbalance_warning = dominant_share > 0.5

    return {
        "total_examples": total,
        "labels_present": present,
        "missing_labels": missing,
        "label_counts": {label: counts.get(label, 0) for label in EXPECTED_LABELS},
        "label_status": statuses,
        "weak_labels": [label for label, status in statuses.items() if status in {"very_weak", "weak"}],
        "dominant_label": dominant_label,
        "dominant_share": round(dominant_share, 4),
        "imbalance_warning": imbalance_warning,
        "recommended_use": "rule-based MVP / seed dataset / not enough for full ML training",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PM Insights dataset readiness for ML training.")
    parser.add_argument("--input", default="datasets/pm_dataset.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = build_readiness_report(load_jsonl(Path(args.input)))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Total examples: {report['total_examples']}")
    print(f"Labels present: {', '.join(report['labels_present'])}")
    print(f"Missing labels: {', '.join(report['missing_labels'])}")
    print(f"Weak labels: {', '.join(report['weak_labels'])}")
    print(f"Dominant label: {report['dominant_label']} ({report['dominant_share']:.1%})")
    print(f"Imbalance warning: {str(report['imbalance_warning']).lower()}")
    print(f"Recommended use: {report['recommended_use']}")
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

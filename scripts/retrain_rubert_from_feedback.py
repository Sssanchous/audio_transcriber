from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm_insights import settings


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely train a RuBERT candidate from verified feedback.")
    parser.add_argument("--output", default=str(settings.RUBERT_CANDIDATE_PATH))
    parser.add_argument("--min-feedback", type=int, default=settings.MIN_FEEDBACK_EXAMPLES_FOR_RETRAIN)
    args = parser.parse_args()

    subprocess.run([sys.executable, "scripts/export_feedback_dataset.py", "--output", str(settings.FEEDBACK_DATASET_PATH)], check=False)
    feedback_rows = load_jsonl(settings.FEEDBACK_DATASET_PATH)
    counts = Counter(row.get("label") for row in feedback_rows)
    if len(feedback_rows) < args.min_feedback or any(count < 10 for count in counts.values()):
        print("Недостаточно проверенных feedback-примеров для безопасного дообучения.")
        print(json.dumps({"feedback_examples": len(feedback_rows), "label_counts": dict(counts)}, ensure_ascii=False, indent=2))
        return 0

    subprocess.run([sys.executable, "scripts/prepare_training_dataset.py"], check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/split_dataset.py",
            "--input",
            "datasets/training_dataset.jsonl",
            "--output-dir",
            "datasets",
            "--seed",
            "42",
            "--strategy",
            "seed",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/train_rubert_classifier.py",
            "--train",
            "datasets/train.jsonl",
            "--val",
            "datasets/val.jsonl",
            "--test",
            "datasets/test.jsonl",
            "--output",
            args.output,
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

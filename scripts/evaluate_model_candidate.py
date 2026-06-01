from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_metrics(path: Path) -> dict:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare active and candidate RuBERT classifiers.")
    parser.add_argument("--current", default="models/rubert_classifier")
    parser.add_argument("--candidate", default="models/rubert_classifier_candidate")
    args = parser.parse_args()

    current = read_metrics(Path(args.current))
    candidate = read_metrics(Path(args.candidate))
    current_f1 = current.get("test_macro_f1", 0.0)
    candidate_f1 = candidate.get("test_macro_f1", 0.0)
    report = {
        "current_model": {"path": args.current, "test_macro_f1": current_f1},
        "candidate_model": {"path": args.candidate, "test_macro_f1": candidate_f1},
        "recommendation": "promote_candidate" if candidate_f1 >= current_f1 and candidate else "keep_current",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

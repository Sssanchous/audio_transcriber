from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.nlp.postprocessing import normalize_analysis_result


def _titles(items: list[dict], key: str = "title") -> list[str]:
    return [str(item.get(key) or item.get("question_title") or item.get("deadline") or "") for item in items[:12]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild clean/report fields for an existing result JSON.")
    parser.add_argument("--input", required=True, help="Path to existing result JSON")
    parser.add_argument("--output", required=True, help="Path to write reprocessed result JSON")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise SystemExit(f"Input result not found: {input_path}")

    result = json.loads(input_path.read_text(encoding="utf-8"))
    before = {
        "clean_commercial_terms": _titles(result.get("clean_commercial_terms", [])),
        "clean_agreements": _titles(result.get("clean_agreements", [])),
        "clean_questions_answers": _titles(result.get("clean_questions_answers", []), "question_title"),
        "clean_deadlines": _titles(result.get("clean_deadlines", []), "deadline"),
    }

    normalized = normalize_analysis_result(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")

    after = {
        "clean_commercial_terms": _titles(normalized.get("clean_commercial_terms", [])),
        "clean_agreements": _titles(normalized.get("clean_agreements", [])),
        "clean_questions_answers": _titles(normalized.get("clean_questions_answers", []), "question_title"),
        "clean_deadlines": _titles(normalized.get("clean_deadlines", []), "deadline"),
    }

    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "meeting_id": normalized.get("meeting_id"),
        "meeting_type": (normalized.get("meeting_type") or {}).get("label"),
        "counts": {
            "clean_commercial_terms": len(normalized.get("clean_commercial_terms", [])),
            "clean_agreements": len(normalized.get("clean_agreements", [])),
            "clean_questions_answers": len(normalized.get("clean_questions_answers", [])),
            "clean_deadlines": len(normalized.get("clean_deadlines", [])),
            "review_items": len(normalized.get("review_items", [])),
        },
        "before_titles": before,
        "after_titles": after,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

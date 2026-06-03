from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.nlp.topic_modeling import extract_topics


def load_fragments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fragments = data.get("transcript") or data.get("segments") or []
    return [
        {"fragment_index": item.get("fragment_index", index), "text": item.get("text", "")}
        for index, item in enumerate(fragments, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run universal topic modeling over an existing result JSON.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--engine", default="auto", choices=["auto", "bertopic", "embedding", "rule_based", "fallback"])
    parser.add_argument("--allow-fit", action="store_true")
    parser.add_argument("--save-model", action="store_true", help="Save a fitted BERTopic model to TOPIC_MODEL_PATH.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_model and not args.allow_fit:
        raise SystemExit("--save-model requires --allow-fit.")
    fragments = load_fragments(Path(args.input))
    result = extract_topics(
        fragments,
        engine=args.engine,
        allow_fit=args.allow_fit,
        save_model=args.save_model,
    )
    if result.get("source") == "bertopic" and result.get("model_saved_to"):
        print(f"BERTopic fitted and saved to {result['model_saved_to']}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

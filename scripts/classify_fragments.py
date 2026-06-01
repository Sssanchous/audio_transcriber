from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.nlp.fragment_classifier import classify_fragments


def load_fragments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    fragments = data.get("transcript") or data.get("segments") or []
    rows = []
    for index, item in enumerate(fragments, start=1):
        rows.append({"fragment_index": item.get("fragment_index", index), "text": item.get("text", "")})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify transcript fragments with rule-based/baseline/RuBERT engine.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--engine", default=None, choices=["rule_based", "baseline", "rubert", "auto"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = classify_fragments(load_fragments(Path(args.input)), engine=args.engine)
    print(json.dumps(predictions, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

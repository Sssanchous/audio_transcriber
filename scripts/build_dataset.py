from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.dataset.builder import build_dataset, format_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PM Insights dataset from DOCX files.")
    parser.add_argument("--input", default="transcripts")
    parser.add_argument("--output", default="datasets/sources/pm_dataset.jsonl")
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, stats = build_dataset(
        input_dir=args.input,
        output_path=args.output,
        output_format=args.format,
        min_length=args.min_length,
        include_other=args.include_other,
    )
    if args.stats:
        print(format_stats(stats))


if __name__ == "__main__":
    main()

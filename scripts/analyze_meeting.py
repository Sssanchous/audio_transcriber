from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.meeting.pipeline import analyze_meeting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a prepared meeting audio file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-name", default="PM Insights")
    parser.add_argument("--meeting-date", default=None)
    parser.add_argument("--transcript-text", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_meeting(
        args.input,
        project_name=args.project_name,
        meeting_date=args.meeting_date,
        transcript_text=args.transcript_text,
        output_dir=Path(args.output).parent,
    )
    output = Path(args.output)
    generated = output.parent / f"{result['meeting_id']}.json"
    if generated.resolve() != output.resolve():
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(generated, output)
    print(json.dumps({"meeting_id": result["meeting_id"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm_insights import settings  # noqa: E402

LEGACY_TABLES = [
    "app_user",
    "meeting",
    "transcript",
    "text_segment",
    "task",
    "question_answer",
    "aspect",
    "segment_aspect",
    "audio_recording",
    "transcriptions",
]


def build_drop_statements() -> list[str]:
    return [f"DROP TABLE IF EXISTS {table} CASCADE;" for table in LEGACY_TABLES]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop the 10 orphaned legacy tables left over from an earlier project schema. "
            "These tables are not used by src/pm_insights/db.py -- the active app only uses "
            "users, meetings, transcripts, analysis_results, analysis_feedback, processing_logs."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Actually run the DROP statements. Without this flag, only prints them.")
    args = parser.parse_args()

    statements = build_drop_statements()

    if not args.execute:
        print("Dry run -- the following statements would be executed (pass --execute to run them):\n")
        for statement in statements:
            print(statement)
        return

    import psycopg

    database_url = settings.DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            for statement in statements:
                print("Executing:", statement)
                cur.execute(statement)
        conn.commit()
    print(f"\nDropped {len(statements)} legacy tables.")


if __name__ == "__main__":
    main()

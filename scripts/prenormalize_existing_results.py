from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sqlalchemy import select

from pm_insights import db
from pm_insights.nlp.postprocessing import normalize_analysis_result


def _latest_result_rows(session):
    rows = session.scalars(
        select(db.AnalysisResult).order_by(db.AnalysisResult.meeting_id, db.AnalysisResult.id.desc())
    ).all()
    latest_by_meeting: dict[str, db.AnalysisResult] = {}
    for row in rows:
        latest_by_meeting.setdefault(row.meeting_id, row)
    return list(latest_by_meeting.values())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration: run normalize_analysis_result() once on every existing "
            "meeting's stored result and mark it is_normalized=True, so /dashboard and "
            "/meetings/{id}/result stop re-normalizing it on every request."
        )
    )
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing to the DB.")
    args = parser.parse_args()

    db.init_db()
    with db.session_scope() as session:
        rows = _latest_result_rows(session)
        pending = [row for row in rows if not (row.result_json or {}).get("is_normalized")]

        print(f"Found {len(rows)} meetings with analysis results, {len(pending)} not yet pre-normalized.")
        if not pending:
            print("Nothing to do.")
            return

        total_ms = 0.0
        for row in pending:
            t0 = time.perf_counter()
            normalized = normalize_analysis_result(dict(row.result_json or {}))
            normalized["is_normalized"] = True
            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_ms += elapsed_ms

            suffix = " (dry-run, not saved)" if args.dry_run else ""
            print(f"  {row.meeting_id}: normalized in {elapsed_ms:.1f} ms{suffix}")

            if not args.dry_run:
                row.result_json = normalized

        avg_ms = total_ms / len(pending)
        print()
        print(f"{'Would normalize' if args.dry_run else 'Normalized'} {len(pending)} meetings.")
        print(f"Total time spent (paid once, now): {total_ms:.1f} ms")
        print(f"Average per meeting: {avg_ms:.1f} ms")
        print("This is exactly the cost each of these meetings used to pay on EVERY /dashboard")
        print("and /meetings/{id}/result request - now paid once here instead.")


if __name__ == "__main__":
    main()

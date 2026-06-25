from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pm_insights.db import Base  # noqa: E402


def generate_schema_sql() -> str:
    dialect = postgresql.dialect()
    tables = list(Base.metadata.sorted_tables)
    indexes = sorted(
        (index for table in tables for index in table.indexes),
        key=lambda index: index.name or "",
    )

    statements = [
        "-- Generated from SQLAlchemy models in src/pm_insights/db.py.",
        "-- Regenerate with: python scripts/create_schema.py --output schema.sql",
    ]
    statements.extend(str(CreateTable(table).compile(dialect=dialect)).rstrip() + ";" for table in tables)
    statements.extend(str(CreateIndex(index).compile(dialect=dialect)).rstrip() + ";" for index in indexes)
    return "\n\n".join(statements) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PostgreSQL schema.sql from SQLAlchemy models.")
    parser.add_argument("--output", default="schema.sql", help="Output SQL file path.")
    parser.add_argument("--print", action="store_true", help="Print generated SQL to stdout.")
    args = parser.parse_args()

    sql = generate_schema_sql()
    output_path = Path(args.output)
    output_path.write_text(sql, encoding="utf-8")
    if args.print:
        print(sql, end="")
    print(f"Schema written to {output_path}")


if __name__ == "__main__":
    main()

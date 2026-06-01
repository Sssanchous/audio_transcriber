from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.dataset.cleaner import normalize_text
from pm_insights.dataset.validator import validate_record


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalize_record(record: dict, fallback_index: int) -> dict:
    out = dict(record)
    out.setdefault("id", f"merged_{fallback_index:06d}")
    out.setdefault("source_file", "unknown")
    out.setdefault("fragment_index", fallback_index)
    out.setdefault("secondary_labels", [])
    out.setdefault("metadata", {})
    out["text"] = normalize_text(out.get("text", ""))
    return out


def merge_datasets(inputs: list[Path], output: Path) -> dict:
    merged: list[dict] = []
    seen_texts: set[str] = set()
    added_by_source: Counter[str] = Counter()
    added_by_label: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for path in inputs:
        source_name = path.name
        for raw in read_jsonl(path):
            record = normalize_record(raw, len(merged) + 1)
            valid, reason = validate_record(record)
            if not valid:
                skipped[reason or "invalid"] += 1
                continue
            key = record["text"].lower()
            if key in seen_texts:
                skipped["duplicate"] += 1
                continue
            seen_texts.add(key)
            merged.append(record)
            added_by_source[source_name] += 1
            added_by_label[record["label"]] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for record in merged:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "output": str(output),
        "total": len(merged),
        "added_by_source": dict(added_by_source),
        "labels": dict(added_by_label),
        "skipped": dict(skipped),
    }
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PM Insights auto and manual datasets.")
    parser.add_argument("--base", default="datasets/pm_dataset.jsonl")
    parser.add_argument("--manual-seed", default="datasets/manual_seed_examples.jsonl")
    parser.add_argument("--manual-labels", default="datasets/manual_labels.jsonl")
    parser.add_argument("--output", default="datasets/pm_dataset_enriched.jsonl")
    args = parser.parse_args()

    inputs = [Path(args.base), Path(args.manual_seed)]
    manual_labels = Path(args.manual_labels)
    if manual_labels.exists():
        inputs.append(manual_labels)
    stats = merge_datasets(inputs, Path(args.output))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

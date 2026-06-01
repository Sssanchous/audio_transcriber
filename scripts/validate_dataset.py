from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ALLOWED_LABELS = {"task", "question", "answer", "other"}
MOJIBAKE_RE = re.compile(r"(Р[ђ-џ]|\uFFFD|Ð|Ñ)")


def load_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    rows = []
    errors = []
    if not path.exists():
        return rows, [{"file": str(path), "line": 0, "reason": "file_not_found"}]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            errors.append({"file": str(path), "line": line_no, "reason": f"invalid_json: {exc}", "text": line[:120]})
    return rows, errors


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def validate_rows(path: Path, rows: list[dict]) -> list[dict]:
    errors = []
    seen = {}
    labels_by_text = defaultdict(set)
    for index, row in enumerate(rows, 1):
        text = str(row.get("text", "")).strip()
        label = row.get("label")
        if not text:
            errors.append({"file": str(path), "line": index, "reason": "empty_text"})
        if "????" in text or MOJIBAKE_RE.search(text):
            errors.append({"file": str(path), "line": index, "reason": "bad_encoding", "text": text[:120]})
        if label not in ALLOWED_LABELS:
            errors.append({"file": str(path), "line": index, "reason": "invalid_label", "label": label, "text": text[:120]})
        key = normalize(text)
        if key in seen:
            errors.append({"file": str(path), "line": index, "reason": "duplicate_text", "text": text[:120]})
        seen[key] = index
        labels_by_text[key].add(label)
    for key, labels in labels_by_text.items():
        if len(labels) > 1:
            errors.append({"file": str(path), "line": 0, "reason": "label_conflict", "labels": sorted(labels), "text": key[:120]})
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PM Insights JSONL datasets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", nargs="*", default=[])
    args = parser.parse_args()

    input_path = Path(args.input)
    rows, errors = load_jsonl(input_path)
    errors.extend(validate_rows(input_path, rows))
    split_sets = []
    split_counts = {}
    for split in args.splits:
        split_rows, split_errors = load_jsonl(Path(split))
        errors.extend(split_errors)
        keys = {normalize(row.get("text", "")) for row in split_rows}
        split_counts[split] = len(split_rows)
        split_sets.append((split, keys))
    for index, (name_a, keys_a) in enumerate(split_sets):
        for name_b, keys_b in split_sets[index + 1 :]:
            overlap = keys_a & keys_b
            if overlap:
                errors.append({"file": f"{name_a}|{name_b}", "line": 0, "reason": "split_overlap", "count": len(overlap)})
    counts = Counter(row.get("label") for row in rows)
    report = {"ok": not errors, "total": len(rows), "label_counts": dict(counts), "split_counts": split_counts, "errors": errors[:100]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

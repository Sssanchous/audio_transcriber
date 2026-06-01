from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .classifier import PRIORITY, classify_fragment
from .cleaner import clean_fragment, normalize_text
from .reader import find_docx_files, read_docx
from .segmenter import segment_blocks
from .validator import validate_record


def _doc_prefix(index: int) -> str:
    return f"doc_{index:03d}"


def build_dataset(
    input_dir: str | Path,
    output_path: str | Path,
    output_format: str = "jsonl",
    min_length: int = 10,
    include_other: bool = False,
) -> tuple[list[dict], dict]:
    input_root = Path(input_dir)
    records: list[dict] = []
    seen: set[str] = set()
    dropped = Counter()
    label_counts = Counter()
    errors: list[dict] = []
    extracted_fragments = 0

    files = find_docx_files(input_root)

    for doc_index, path in enumerate(files, start=1):
        try:
            blocks = read_docx(path)
        except Exception as exc:
            errors.append({"source_file": path.name, "error": str(exc)})
            dropped["read_error"] += 1
            continue

        fragments = segment_blocks(blocks)
        extracted_fragments += len(fragments)
        local_index = 0

        for fragment in fragments:
            cleaned, reason = clean_fragment(fragment["text"], min_length=min_length)
            if reason:
                dropped[reason] += 1
                continue

            assert cleaned is not None
            dedupe_key = normalize_text(cleaned).lower()
            if dedupe_key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(dedupe_key)

            classification = classify_fragment(cleaned, include_other=True)
            label = classification["label"]
            if label in {"other", "sentiment_neutral"} and not include_other:
                dropped["other"] += 1
                continue

            local_index += 1
            record = {
                "id": f"{_doc_prefix(doc_index)}_{local_index:04d}",
                "source_file": fragment["source_file"],
                "fragment_index": local_index,
                "paragraph_index": fragment.get("paragraph_index"),
                "text": cleaned,
                "label": label,
                "secondary_labels": classification["secondary_labels"],
                "metadata": {
                    "matched_rules": classification["matched_rules"],
                    "confidence": classification["confidence"],
                    "language": "ru",
                    "block_type": fragment.get("block_type", "paragraph"),
                },
            }

            valid, invalid_reason = validate_record(record)
            if not valid:
                dropped[invalid_reason or "invalid"] += 1
                continue

            records.append(record)
            label_counts[label] += 1

    stats = {
        "processed_files": len(files),
        "extracted_fragments": extracted_fragments,
        "saved_examples": len(records),
        "dropped_fragments": int(sum(dropped.values())),
        "labels": {label: int(label_counts.get(label, 0)) for label in PRIORITY},
        "dropped_reasons": dict(dropped),
        "errors": errors,
    }

    save_dataset(records, output_path, output_format)
    return records, stats


def save_dataset(records: list[dict], output_path: str | Path, output_format: str = "jsonl") -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower()

    if fmt == "jsonl":
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return

    if fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as fh:
            fieldnames = ["id", "source_file", "fragment_index", "paragraph_index", "text", "label", "secondary_labels", "metadata"]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = dict(record)
                row["secondary_labels"] = json.dumps(row.get("secondary_labels", []), ensure_ascii=False)
                row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
                writer.writerow(row)
        return

    raise ValueError("format must be jsonl or csv")


def format_stats(stats: dict) -> str:
    lines = [
        f"Processed files: {stats['processed_files']}",
        f"Extracted fragments: {stats['extracted_fragments']}",
        f"Saved examples: {stats['saved_examples']}",
        f"Dropped fragments: {stats['dropped_fragments']}",
        "",
        "Labels:",
    ]
    for label, count in stats["labels"].items():
        lines.append(f"{label}: {count}")
    lines.append("")
    lines.append("Dropped reasons:")
    for reason, count in sorted(stats["dropped_reasons"].items()):
        lines.append(f"{reason}: {count}")
    if stats.get("errors"):
        lines.append("")
        lines.append("Read errors:")
        for item in stats["errors"]:
            lines.append(f"{item['source_file']}: {item['error']}")
    return "\n".join(lines)

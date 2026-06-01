from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ALLOWED_LABELS = {"task", "question", "answer", "other"}
QUESTION_STARTS = (
    "кто",
    "что",
    "когда",
    "где",
    "почему",
    "зачем",
    "сколько",
    "какой",
    "какая",
    "какие",
    "можно ли",
    "нужно ли",
    "есть ли",
    "правильно ли",
    "я правильно понимаю",
    "подскажите",
    "скажите",
    "можно уточнить",
)
DISCUSSION_STARTS = ("обсуждение", "рассмотрение", "анализ", "проверка", "изучение")
TASK_MARKERS = (
    "подготовить",
    "сделать",
    "проверить",
    "исправить",
    "собрать",
    "отправить",
    "обновить",
    "согласовать",
    "добавить",
    "оформить",
    "прислать",
    "загрузить",
    "настроить",
    "развернуть",
    "провести",
    "сформировать",
    "определить",
    "разработать",
    "реализовать",
    "протестировать",
)
TASK_SECTIONS = {
    "tasks",
    "assignments",
    "поручения",
    "задачи",
    "до следующей встречи",
    "поставленные задачи",
}
ANSWER_STARTS = (
    "да",
    "нет",
    "хорошо",
    "понял",
    "сделаю",
    "проверю",
    "подготовлю",
    "отправлю",
    "уточню",
    "лежит",
    "отвечаю я",
    "сейчас открыты",
    "уже готово",
    "выполнено",
)
SHORT_ANSWERS = {"да", "нет", "ок", "хорошо", "понял", "готово", "сделано"}
SERVICE_RE = re.compile(
    r"^\s*(дата\s+и\s+время|дата:|время:|присутствовали:|тема\s+встречи:|"
    r"тема:|формат:|страница\b|документ\b)",
    re.IGNORECASE,
)
MOJIBAKE_MARKERS = ("????", "Рџ", "Рё", "Ð", "Ñ")
TECHNICAL_HINTS = (
    "гидродинами",
    "скважин",
    "мгрп",
    "скин",
    "проницаем",
    "аппроксимац",
    "интерпретац",
    "безразмер",
    "параметр",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def word_count(text: str) -> int:
    return len(re.findall(r"[\wА-Яа-яЁё]+", text, flags=re.UNICODE))


def source_section(row: dict) -> str:
    metadata = row.get("metadata") or {}
    return normalize_text(row.get("source_section") or metadata.get("source_section") or "")


def has_bad_encoding(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def starts_with_any(text_lower: str, prefixes: tuple[str, ...]) -> bool:
    return any(text_lower.startswith(prefix) for prefix in prefixes)


def is_question_like(text: str) -> bool:
    lower = normalize_text(text)
    if starts_with_any(lower, DISCUSSION_STARTS):
        return False
    return "?" in text or starts_with_any(lower, QUESTION_STARTS)


def is_answer_like(text: str) -> bool:
    lower = normalize_text(text).strip(" .,!?:;")
    return lower in SHORT_ANSWERS or starts_with_any(lower, ANSWER_STARTS)


def is_task_section(row: dict) -> bool:
    section = source_section(row)
    return any(marker in section for marker in TASK_SECTIONS)


def is_task_like(text: str, row: dict) -> bool:
    lower = normalize_text(text)
    if starts_with_any(lower, DISCUSSION_STARTS):
        return False
    if is_task_section(row):
        return True
    if any(marker in lower for marker in TASK_MARKERS):
        return True
    return False


def is_protocol_other(row: dict) -> bool:
    original_label = normalize_text(row.get("original_label") or (row.get("metadata") or {}).get("original_label") or "")
    section = source_section(row)
    return original_label in {"discussion_item", "summary", "decision"} or section in {
        "discussion_items",
        "summary",
        "decisions",
    }


def obvious_label(text: str, row: dict) -> str | None:
    lower = normalize_text(text)
    if SERVICE_RE.search(text) or is_protocol_other(row):
        return "other"
    if is_question_like(text):
        return "question"
    if is_answer_like(text):
        return "answer"
    if is_task_like(text, row):
        return "task"
    if starts_with_any(lower, DISCUSSION_STARTS):
        return "other"
    return None


def review_reason(text: str, current_label: str, suggested_label: str | None, row: dict) -> str | None:
    lower = normalize_text(text)
    if current_label == "task" and suggested_label != "task":
        return "task_vs_other"
    if suggested_label == "answer" and current_label == "other" and word_count(text) > 18:
        return "answer_vs_other_long_text"
    if suggested_label == "question" and "?" not in text:
        return "question_without_question_mark"
    if current_label == "task" and any(hint in lower for hint in TECHNICAL_HINTS):
        return "long_or_technical_task_candidate"
    if row.get("metadata", {}).get("needs_review"):
        return "source_marked_needs_review"
    return None


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit and safely fix a 4-class PM Insights training dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--review-output", required=True)
    parser.add_argument("--fix-output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    fixed_rows: list[dict] = []
    review_rows: list[dict] = []
    seen: dict[str, dict] = {}
    conflicts: dict[str, set[str]] = {}
    stats = Counter()
    before_labels = Counter()
    after_labels = Counter()
    auto_fixed_labels = Counter()

    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        stats["total_before"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            stats["invalid_json"] += 1
            review_rows.append(
                {
                    "line_number": line_number,
                    "text": line[:200],
                    "current_label": None,
                    "suggested_label": None,
                    "reason": f"invalid_json: {exc.msg}",
                    "source": None,
                    "source_file": None,
                }
            )
            continue

        text = str(row.get("text") or "").strip()
        label = row.get("label")
        if label in ALLOWED_LABELS:
            before_labels[label] += 1
        if not text:
            stats["removed_empty_text"] += 1
            continue
        if has_bad_encoding(text):
            stats["removed_bad_encoding"] += 1
            review_rows.append(
                {
                    "line_number": line_number,
                    "text": text,
                    "current_label": label,
                    "suggested_label": None,
                    "reason": "bad_encoding",
                    "source": row.get("source"),
                    "source_file": row.get("source_file"),
                }
            )
            continue
        if label not in ALLOWED_LABELS:
            stats["removed_invalid_label"] += 1
            review_rows.append(
                {
                    "line_number": line_number,
                    "text": text,
                    "current_label": label,
                    "suggested_label": None,
                    "reason": "invalid_label",
                    "source": row.get("source"),
                    "source_file": row.get("source_file"),
                }
            )
            continue
        if SERVICE_RE.search(text):
            stats["removed_service_lines"] += 1
            continue
        if word_count(text) < 3 and normalize_text(text).strip(" .,!?:;") not in SHORT_ANSWERS:
            stats["removed_too_short"] += 1
            continue

        suggested = obvious_label(text, row)
        reason = review_reason(text, label, suggested, row)
        if suggested and suggested != label:
            if reason in {"answer_vs_other_long_text", "question_without_question_mark", "long_or_technical_task_candidate"}:
                metadata = row.get("metadata") or {}
                row["metadata"] = {**metadata, "needs_review": True, "audit_suggested_label": suggested}
                review_rows.append(
                    {
                        "line_number": line_number,
                        "text": text,
                        "current_label": label,
                        "suggested_label": suggested,
                        "reason": reason,
                        "source": row.get("source"),
                        "source_file": row.get("source_file"),
                    }
                )
                stats["review_items"] += 1
            else:
                auto_fixed_labels[f"{label}->{suggested}"] += 1
                row["label"] = suggested
                metadata = row.get("metadata") or {}
                row["metadata"] = {**metadata, "audit_fixed_from": label}
                stats["auto_fixed_labels"] += 1
        elif reason:
            metadata = row.get("metadata") or {}
            row["metadata"] = {**metadata, "needs_review": True}
            review_rows.append(
                {
                    "line_number": line_number,
                    "text": text,
                    "current_label": label,
                    "suggested_label": suggested,
                    "reason": reason,
                    "source": row.get("source"),
                    "source_file": row.get("source_file"),
                }
            )
            stats["review_items"] += 1

        key = normalize_text(text)
        if key in seen:
            previous = seen[key]
            if previous.get("label") != row.get("label"):
                conflicts.setdefault(key, {previous.get("label")}).add(row.get("label"))
                previous_metadata = previous.get("metadata") or {}
                previous["metadata"] = {**previous_metadata, "needs_review": True, "audit_conflict_labels": sorted(conflicts[key])}
                review_rows.append(
                    {
                        "line_number": line_number,
                        "text": text,
                        "current_label": row.get("label"),
                        "suggested_label": previous.get("label"),
                        "reason": "duplicate_text_conflicting_labels",
                        "source": row.get("source"),
                        "source_file": row.get("source_file"),
                    }
                )
                stats["conflicts"] += 1
            stats["removed_duplicates"] += 1
            continue

        seen[key] = row
        fixed_rows.append(row)

    for row in fixed_rows:
        after_labels[row["label"]] += 1

    report = {
        "input": str(input_path),
        "total_before": stats["total_before"],
        "total_after": len(fixed_rows),
        "removed_duplicates": stats["removed_duplicates"],
        "removed_bad_encoding": stats["removed_bad_encoding"],
        "removed_empty_text": stats["removed_empty_text"],
        "removed_invalid_label": stats["removed_invalid_label"],
        "removed_too_short": stats["removed_too_short"],
        "removed_service_lines": stats["removed_service_lines"],
        "invalid_json": stats["invalid_json"],
        "auto_fixed_labels": stats["auto_fixed_labels"],
        "auto_fixed_label_pairs": dict(sorted(auto_fixed_labels.items())),
        "review_items": len(review_rows),
        "conflicts": stats["conflicts"],
        "label_distribution_before": dict(sorted(before_labels.items())),
        "label_distribution_after": dict(sorted(after_labels.items())),
    }

    write_jsonl(Path(args.fix_output), fixed_rows)
    write_jsonl(Path(args.review_output), review_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pm_insights.dataset.classifier import classify_fragment
from pm_insights.dataset.cleaner import clean_fragment, normalize_text
from pm_insights.dataset.reader import find_docx_files, read_docx
from pm_insights.dataset.segmenter import segment_blocks

TARGET_LABELS = {
    "answer",
    "responsible",
    "sentiment_negative",
    "sentiment_neutral",
    "other",
    "decision",
    "deadline",
    "question",
}


QUEUE_PATTERNS = [
    ("answer", (r"^\s*(да|нет|готово|сделано|пока\s+нет)\b", r"\b(я\s+проверил|я\s+сделал|мы\s+сделали|в\s+процессе|возьму\s+в\s+работу)\b")),
    ("responsible", (r"\b(ответственный|ответственная|исполнитель|за\s+это\s+отвечает|бер[её]т\s+на\s+себя|поручаем)\b",)),
    ("sentiment_negative", (r"\b(не\s+работает|ошибка|риск|не\s+успеваем|задержка|блокер|сломалось|не\s+получилось|некорректно|проблема\s+(с|в|на|при))\b",)),
    ("sentiment_neutral", (r"\b(обсудили|рассмотрели|планируется|отмечено|встреча|проект|раздел)\b",)),
    ("decision", (r"\b(решили|договорились|решение|фиксируем|утверждаем|согласовали)\b",)),
    ("deadline", (r"\b(до\s+пятницы|до\s+завтра|к\s+понедельнику|дедлайн|срок\s+выполнения|до\s+конца\s+недели)\b",)),
    ("question", (r"\?", r"^\s*(кто|что|когда|где|почему|зачем|как|сколько|какой|какая|какие)\b")),
]


def load_existing_texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    texts = set()
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        item = json.loads(line)
        texts.add(normalize_text(item.get("text", "")).lower())
    return texts


def suggest_for_queue(text: str) -> dict:
    lower = text.lower()
    matched_rules: list[str] = []
    for label, patterns in QUEUE_PATTERNS:
        for pattern in patterns:
            for match in re.finditer(pattern, lower, flags=re.IGNORECASE):
                matched_rules.append(match.group(0))
        if matched_rules:
            return {
                "suggested_label": label,
                "suggested_secondary_labels": [],
                "matched_rules": matched_rules,
            }

    auto = classify_fragment(text, include_other=True)
    label = auto["label"]
    if label not in TARGET_LABELS:
        label = "other"
    return {
        "suggested_label": label,
        "suggested_secondary_labels": auto.get("secondary_labels", []),
        "matched_rules": auto.get("matched_rules", []),
    }


def build_annotation_queue(input_dir: Path, existing_dataset: Path, limit: int | None = None) -> list[dict]:
    existing = load_existing_texts(existing_dataset)
    queue: list[dict] = []
    seen: set[str] = set()

    for path in find_docx_files(input_dir):
        try:
            blocks = read_docx(path)
        except Exception:
            continue

        for fragment in segment_blocks(blocks):
            raw_text = normalize_text(fragment["text"])
            if not raw_text:
                continue

            cleaned, reason = clean_fragment(raw_text, min_length=5)
            candidate_text = cleaned or raw_text
            key = normalize_text(candidate_text).lower()
            if key in existing or key in seen:
                continue

            suggestion = suggest_for_queue(candidate_text)
            if suggestion["suggested_label"] not in TARGET_LABELS:
                continue

            seen.add(key)
            queue.append(
                {
                    "id": f"ann_{len(queue) + 1:06d}",
                    "source_file": fragment["source_file"],
                    "fragment_index": fragment["fragment_index"],
                    "text": candidate_text,
                    "suggested_label": suggestion["suggested_label"],
                    "suggested_secondary_labels": suggestion["suggested_secondary_labels"],
                    "matched_rules": suggestion["matched_rules"],
                    "drop_reason": reason,
                    "annotation_status": "pending",
                    "manual_label": None,
                    "comment": None,
                }
            )
            if limit and len(queue) >= limit:
                return queue

    return queue


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotation queue from DOCX fragments.")
    parser.add_argument("--input", default="transcripts")
    parser.add_argument("--dataset", default="datasets/pm_dataset.jsonl")
    parser.add_argument("--output", default="datasets/annotation_queue.jsonl")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    queue = build_annotation_queue(Path(args.input), Path(args.dataset), limit=args.limit)
    save_jsonl(Path(args.output), queue)
    counts: dict[str, int] = {}
    for item in queue:
        label = item["suggested_label"]
        counts[label] = counts.get(label, 0) + 1
    print(json.dumps({"items": len(queue), "labels": counts, "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

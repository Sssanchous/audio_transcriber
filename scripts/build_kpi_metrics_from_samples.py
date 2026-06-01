from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TARGETS = {
    "wer": 15.0,
    "task_precision": 80.0,
    "task_recall": 80.0,
    "sentiment_accuracy": 75.0,
    "processing_1h_minutes": 20.0,
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    text = normalize_text(text)
    return text.split() if text else []


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev

    return prev[m]


def calculate_wer(reference: str, hypothesis: str) -> dict:
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    if not ref_tokens:
        return {
            "value": None,
            "status": "not_available",
            "comment": "Эталонный транскрипт пустой.",
        }

    distance = levenshtein_distance(ref_tokens, hyp_tokens)
    wer = round(distance / len(ref_tokens) * 100, 2)

    return {
        "value": wer,
        "unit": "%",
        "target": f"≤ {TARGETS['wer']}%",
        "passed": wer <= TARGETS["wer"],
        "status": "passed" if wer <= TARGETS["wer"] else "failed",
        "reference_words": len(ref_tokens),
        "hypothesis_words": len(hyp_tokens),
        "edit_distance": distance,
    }


def get_result_transcript(result: dict) -> str:
    candidates = [
        result.get("transcript"),
        result.get("full_transcript"),
        result.get("text"),
        result.get("asr_text"),
    ]

    for c in candidates:
        if isinstance(c, str) and c.strip():
            return c

    segments = result.get("segments") or result.get("transcript_segments") or []
    if isinstance(segments, list):
        texts = []
        for seg in segments:
            if isinstance(seg, dict):
                text = seg.get("text") or seg.get("transcript")
                if text:
                    texts.append(str(text))
        if texts:
            return " ".join(texts)

    return ""


def normalize_task_text(text: str) -> str:
    text = normalize_text(text)
    words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)


def extract_predicted_tasks(result: dict) -> list[str]:
    tasks = result.get("clean_tasks") or result.get("tasks") or []
    extracted = []
    for item in tasks:
        if isinstance(item, dict):
            text = item.get("title") or item.get("text") or item.get("summary") or item.get("source_text")
        else:
            text = str(item)
        text = normalize_task_text(text)
        if text:
            extracted.append(text)
    return extracted


def load_expert_tasks(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".jsonl":
        rows = load_jsonl(p)
    else:
        data = load_json(p)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("tasks") or data.get("expert_tasks") or []

    tasks = []
    for item in rows:
        if isinstance(item, dict):
            text = item.get("title") or item.get("text") or item.get("task") or item.get("source_text")
            label = item.get("label") or item.get("corrected_label")
            if label and label not in {"task", "true", "correct"}:
                continue
        else:
            text = str(item)
        text = normalize_task_text(text)
        if text:
            tasks.append(text)
    return tasks


def token_jaccard(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def calculate_task_precision_recall(predicted: list[str], expert: list[str], threshold: float = 0.55) -> dict:
    if not expert:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "status": "not_available",
            "comment": "Нет экспертной разметки задач.",
        }

    matched_expert = set()
    true_positive = 0

    for pred in predicted:
        best_idx = None
        best_score = 0.0
        for i, exp in enumerate(expert):
            if i in matched_expert:
                continue
            score = token_jaccard(pred, exp)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= threshold:
            true_positive += 1
            matched_expert.add(best_idx)

    false_positive = max(len(predicted) - true_positive, 0)
    false_negative = max(len(expert) - true_positive, 0)

    precision = true_positive / len(predicted) * 100 if predicted else 0.0
    recall = true_positive / len(expert) * 100 if expert else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    precision = round(precision, 2)
    recall = round(recall, 2)
    f1 = round(f1, 2)

    passed = precision >= TARGETS["task_precision"] and recall >= TARGETS["task_recall"]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "unit": "%",
        "target": f"Precision ≥ {TARGETS['task_precision']}%, Recall ≥ {TARGETS['task_recall']}%",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_count": len(predicted),
        "expert_count": len(expert),
        "matching_threshold": threshold,
    }


def normalize_sentiment(label: str | None) -> str | None:
    if label is None:
        return None
    s = str(label).lower().strip()
    mapping = {
        "positive": "positive",
        "pos": "positive",
        "позитив": "positive",
        "положительный": "positive",
        "neutral": "neutral",
        "neu": "neutral",
        "нейтрально": "neutral",
        "нейтральный": "neutral",
        "negative": "negative",
        "neg": "negative",
        "негатив": "negative",
        "отрицательный": "negative",
    }
    return mapping.get(s, s)


def extract_predicted_sentiments(result: dict) -> list[dict]:
    candidates = []

    for key in ["sentiment_fragments", "fragments_sentiment", "semantic_blocks", "segments"]:
        rows = result.get(key)
        if isinstance(rows, list):
            candidates.extend(rows)

    extracted = []
    for i, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("source_text") or item.get("transcript") or ""
        label = (
            item.get("sentiment")
            or item.get("sentiment_label")
            or item.get("tone")
            or item.get("label")
        )
        label = normalize_sentiment(label)
        if text and label in {"positive", "neutral", "negative"}:
            extracted.append({
                "id": str(item.get("id") or item.get("fragment_id") or i),
                "text": normalize_text(text),
                "label": label,
            })

    return extracted


def load_expert_sentiment(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []

    if p.suffix.lower() == ".jsonl":
        rows = load_jsonl(p)
    else:
        data = load_json(p)
        if isinstance(data, list):
            rows = data
        else:
            rows = data.get("sentiment") or data.get("items") or []

    extracted = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("source_text") or item.get("fragment") or ""
        label = normalize_sentiment(item.get("sentiment") or item.get("label") or item.get("corrected_label"))
        if text and label in {"positive", "neutral", "negative"}:
            extracted.append({
                "id": str(item.get("id") or item.get("fragment_id") or i),
                "text": normalize_text(text),
                "label": label,
            })
    return extracted


def calculate_sentiment_accuracy(predicted: list[dict], expert: list[dict], threshold: float = 0.55) -> dict:
    if not expert:
        return {
            "accuracy": None,
            "status": "not_available",
            "comment": "Нет экспертной разметки тональности.",
        }

    used_pred = set()
    total = 0
    correct = 0
    confusion = Counter()

    for exp in expert:
        best_idx = None
        best_score = 0.0

        for i, pred in enumerate(predicted):
            if i in used_pred:
                continue
            if pred["id"] == exp["id"]:
                score = 1.0
            else:
                score = token_jaccard(pred["text"], exp["text"])

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None and best_score >= threshold:
            used_pred.add(best_idx)
            pred_label = predicted[best_idx]["label"]
            exp_label = exp["label"]
            total += 1
            if pred_label == exp_label:
                correct += 1
            confusion[(exp_label, pred_label)] += 1

    if total == 0:
        return {
            "accuracy": 0.0,
            "status": "failed",
            "comment": "Не удалось сопоставить фрагменты с экспертной разметкой.",
            "matched_count": 0,
            "expert_count": len(expert),
        }

    accuracy = round(correct / total * 100, 2)
    passed = accuracy >= TARGETS["sentiment_accuracy"]

    return {
        "accuracy": accuracy,
        "unit": "%",
        "target": f"≥ {TARGETS['sentiment_accuracy']}%",
        "passed": passed,
        "status": "passed" if passed else "failed",
        "matched_count": total,
        "correct_count": correct,
        "expert_count": len(expert),
        "predicted_count": len(predicted),
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }


def extract_processing_1h_minutes(result: dict) -> float | None:
    for path in [
        ("estimated_1h_processing_minutes",),
        ("processing_metrics", "estimated_1h_processing_minutes"),
        ("metrics", "estimated_1h_processing_minutes"),
        ("performance", "estimated_1h_processing_minutes"),
        ("processing", "estimated_1h_processing_minutes"),
        ("technical_metrics", "estimated_1h_processing_minutes"),
    ]:
        current = result
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if isinstance(current, (int, float)):
            return round(float(current), 3)

    duration = None
    processing = None

    for key in ["audio_duration_seconds", "duration_seconds", "duration"]:
        if isinstance(result.get(key), (int, float)):
            duration = float(result[key])
            break

    for key in ["processing_time_seconds", "total_processing_seconds", "processing_time"]:
        if isinstance(result.get(key), (int, float)):
            processing = float(result[key])
            break

    if duration and processing:
        return round(processing / duration * 60, 3)

    return None


def calculate_processing_kpi(result: dict) -> dict:
    value = extract_processing_1h_minutes(result)
    if value is None:
        return {
            "value": None,
            "status": "not_available",
            "comment": "В JSON не найдены поля времени обработки.",
        }

    passed = value <= TARGETS["processing_1h_minutes"]

    return {
        "value": value,
        "unit": "мин",
        "target": "≤ 15–20 мин",
        "passed": passed,
        "status": "passed" if passed else "failed",
    }


def maybe_load_text(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KPI metrics from PM Insights samples")
    parser.add_argument("--result", required=True, help="Meeting result JSON")
    parser.add_argument("--reference-transcript", default=None, help="Manual reference transcript TXT")
    parser.add_argument("--expert-tasks", default=None, help="Expert tasks JSON/JSONL")
    parser.add_argument("--expert-sentiment", default=None, help="Expert sentiment JSON/JSONL")
    parser.add_argument("--output", default="results/kpi_metrics_samples.json")
    args = parser.parse_args()

    result = load_json(args.result)

    report = {
        "source_result": args.result,
        "targets": {
            "wer": "≤ 15%",
            "task_precision_recall": "≥ 80%",
            "sentiment_accuracy": "≥ 75%",
            "processing_1h": "≤ 15–20 мин",
        },
        "metrics": {},
        "presentation_summary": [],
    }

    reference = maybe_load_text(args.reference_transcript)
    hypothesis = get_result_transcript(result)

    if reference:
        report["metrics"]["wer"] = calculate_wer(reference, hypothesis)
    else:
        report["metrics"]["wer"] = {
            "value": None,
            "status": "not_available",
            "target": "≤ 15%",
            "comment": "Нет эталонного ручного транскрипта. WER не рассчитывается.",
        }

    predicted_tasks = extract_predicted_tasks(result)
    expert_tasks = load_expert_tasks(args.expert_tasks) if args.expert_tasks else []
    report["metrics"]["task_precision_recall"] = calculate_task_precision_recall(
        predicted_tasks,
        expert_tasks,
    )

    predicted_sentiment = extract_predicted_sentiments(result)
    expert_sentiment = load_expert_sentiment(args.expert_sentiment) if args.expert_sentiment else []
    report["metrics"]["sentiment_accuracy"] = calculate_sentiment_accuracy(
        predicted_sentiment,
        expert_sentiment,
    )

    report["metrics"]["processing_1h"] = calculate_processing_kpi(result)

    for key, metric in report["metrics"].items():
        status = metric.get("status")
        if status == "passed":
            report["presentation_summary"].append(f"{key}: требование выполнено")
        elif status == "failed":
            report["presentation_summary"].append(f"{key}: требование не выполнено")
        else:
            report["presentation_summary"].append(f"{key}: требуется эталонная/экспертная разметка")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
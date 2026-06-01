from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\sА-Яа-яЁё-]", " ", (text or "").lower())).strip()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize(left), normalize(right)).ratio()


def load_reference(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("tasks", [])


def load_prediction(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("tasks", [])


def evaluate(reference: list[dict], prediction: list[dict], threshold: float = 0.65) -> dict:
    used_predictions: set[int] = set()
    matched = []
    false_negatives = []

    for ref in reference:
        best_index = None
        best_score = 0.0
        for index, pred in enumerate(prediction):
            if index in used_predictions:
                continue
            score = similarity(ref.get("text", ""), pred.get("text", ""))
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= threshold:
            used_predictions.add(best_index)
            matched.append({"reference": ref, "prediction": prediction[best_index], "similarity": round(best_score, 3)})
        else:
            false_negatives.append(ref)

    false_positives = [pred for index, pred in enumerate(prediction) if index not in used_predictions]
    precision = len(matched) / len(prediction) if prediction else 0.0
    recall = len(matched) / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_tasks": len(matched),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "matched": matched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task extraction against expert annotation.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    prediction_path = Path(args.prediction)
    if not reference_path.exists():
        print("Task extraction evaluation skipped: no expert reference provided.")
        return
    if not prediction_path.exists():
        print("Task extraction evaluation skipped: prediction result not found.")
        return
    metrics = evaluate(load_reference(reference_path), load_prediction(prediction_path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

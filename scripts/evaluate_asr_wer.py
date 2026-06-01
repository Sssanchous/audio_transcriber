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


def normalize(text: str) -> list[str]:
    value = re.sub(r"[^\w\sА-Яа-яЁё-]", " ", (text or "").lower())
    return [word for word in value.split() if word]


def word_distance(reference: list[str], prediction: list[str]) -> int:
    prev = list(range(len(prediction) + 1))
    for i, ref_word in enumerate(reference, start=1):
        cur = [i]
        for j, pred_word in enumerate(prediction, start=1):
            cost = 0 if ref_word == pred_word else 1
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def load_reference(path: Path) -> str:
    texts = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
    return " ".join(texts)


def load_prediction(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    transcript = data.get("transcript", [])
    return " ".join(item.get("text", "") for item in transcript)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR WER against an expert transcript.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--prediction", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference)
    prediction_path = Path(args.prediction)
    if not reference_path.exists():
        print("ASR evaluation skipped: no reference transcript provided.")
        return
    if not prediction_path.exists():
        print("ASR evaluation skipped: prediction result not found.")
        return

    ref_words = normalize(load_reference(reference_path))
    pred_words = normalize(load_prediction(prediction_path))
    errors = word_distance(ref_words, pred_words)
    total = len(ref_words)
    wer = errors / total if total else 0.0
    metrics = {
        "wer": round(wer, 4),
        "word_accuracy": round(max(0.0, 1.0 - wer), 4),
        "word_errors": errors,
        "reference_words": total,
        "prediction_words": len(pred_words),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

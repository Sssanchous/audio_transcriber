"""
evaluate_kpi.py — Оценка качества моделей по KPI из ТЗ.

KPI из техзадания:
    1. WER транскрибации ≤ 15%
    2. Precision/Recall задач ≥ 80%
    3. Accuracy сентимент-анализа ≥ 75% (3 класса)

Использование:
    1. Подготовьте эталонные данные (см. ниже)
    2. python evaluate_kpi.py --mode all
    3. Или по отдельности:
       python evaluate_kpi.py --mode wer
       python evaluate_kpi.py --mode tasks
       python evaluate_kpi.py --mode sentiment

Эталонные данные:
    — eval_data/wer_reference.jsonl — для WER
      {"audio": "path/to/audio.wav", "reference_text": "эталонный текст..."}

    — eval_data/tasks_reference.jsonl — для задач
      {"text": "Подготовь отчёт до пятницы.", "is_task": true}
      {"text": "Коллеги, всем доброе утро.", "is_task": false}

    — eval_data/sentiment_reference.jsonl — для сентимента
      {"text": "Всё отлично, молодцы!", "label": "positive"}
      {"text": "Опять задержка, это критично.", "label": "negative"}
      {"text": "Перейдём к следующему пункту.", "label": "neutral"}
"""

import json
import os
import re
import argparse
from pathlib import Path

# ==================
# Настройки
# ==================
EVAL_DATA_DIR = "eval_data"
os.makedirs(EVAL_DATA_DIR, exist_ok=True)

# Порог KPI из ТЗ
WER_TARGET = 0.15       # ≤ 15%
TASK_PRECISION_TARGET = 0.80  # ≥ 80%
TASK_RECALL_TARGET = 0.80     # ≥ 80%
SENTIMENT_ACCURACY_TARGET = 0.75  # ≥ 75%


# ==================
# WER (Word Error Rate)
# ==================
def compute_wer(reference: str, hypothesis: str) -> float:
    """Вычисляет WER между эталонным и распознанным текстом."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Динамическое программирование (Levenshtein на словах)
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1   # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def evaluate_wer():
    """Оценка WER транскрибации."""
    ref_path = os.path.join(EVAL_DATA_DIR, "wer_reference.jsonl")

    if not os.path.exists(ref_path):
        print(f"[WER] Файл {ref_path} не найден!")
        print(f"  Создайте его в формате:")
        print(f'  {{"audio": "path/to/audio.wav", "reference_text": "эталонный текст"}}')
        _create_wer_template()
        return

    # Загружаем ASR модель
    try:
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL_NAME", "small")
        device = "cuda" if os.getenv("WHISPER_DEVICE", "").lower() == "cuda" else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        whisper = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"[WER] Не удалось загрузить Whisper: {e}")
        return

    entries = _load_jsonl(ref_path)
    if not entries:
        print("[WER] Нет данных для оценки!")
        return

    wer_scores = []
    for entry in entries:
        audio_path = entry.get("audio", "")
        ref_text = entry.get("reference_text", "")

        if not os.path.exists(audio_path):
            print(f"  ⚠ Аудио не найдено: {audio_path}")
            continue

        # Транскрибация
        segments, info = whisper.transcribe(audio_path, beam_size=5, vad_filter=True)
        hyp_text = " ".join(s.text.strip() for s in segments if s.text.strip())

        wer = compute_wer(ref_text, hyp_text)
        wer_scores.append(wer)
        status = "✅" if wer <= WER_TARGET else "❌"
        print(f"  {status} {os.path.basename(audio_path)}: WER = {wer:.2%}")

    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        status = "✅ PASSED" if avg_wer <= WER_TARGET else "❌ FAILED"
        print(f"\n  Средний WER: {avg_wer:.2%} (цель ≤ {WER_TARGET:.0%}) {status}")
    else:
        print("  Нет результатов")


# ==================
# Task Precision / Recall
# ==================
def evaluate_tasks():
    """Оценка Precision/Recall выделения задач."""
    ref_path = os.path.join(EVAL_DATA_DIR, "tasks_reference.jsonl")

    if not os.path.exists(ref_path):
        print(f"[Tasks] Файл {ref_path} не найден!")
        print(f"  Создайте его в формате:")
        print(f'  {{"text": "Подготовь отчёт.", "is_task": true}}')
        print(f'  {{"text": "Доброе утро.", "is_task": false}}')
        _create_tasks_template()
        return

    # Импортируем классификатор
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        classifier_dir = os.getenv("CLASSIFIER_DIR", "models/classifier")
        if os.path.exists(classifier_dir):
            tok = AutoTokenizer.from_pretrained(classifier_dir)
            mdl = AutoModelForSequenceClassification.from_pretrained(classifier_dir)
            mdl.eval()
            use_classifier = True
            print(f"  Используется дообученный классификатор: {classifier_dir}")
        else:
            use_classifier = False
            print("  Классификатор не найден, используются правила")
    except ImportError:
        use_classifier = False
        print("  PyTorch/Transformers не установлены, используются правила")

    entries = _load_jsonl(ref_path)
    if not entries:
        print("[Tasks] Нет данных!")
        return

    tp = fp = fn = tn = 0

    for entry in entries:
        text = entry.get("text", "").strip()
        expected = entry.get("is_task", False)

        if use_classifier:
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                probs = F.softmax(mdl(**inputs).logits, dim=1)[0]
            predicted = bool(probs[0].item() > 0.5)  # index 0 = task
        else:
            predicted = _rule_based_task_check(text)

        if expected and predicted:
            tp += 1
        elif not expected and predicted:
            fp += 1
        elif expected and not predicted:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    p_status = "✅" if precision >= TASK_PRECISION_TARGET else "❌"
    r_status = "✅" if recall >= TASK_RECALL_TARGET else "❌"

    print(f"\n  Всего примеров: {len(entries)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  {p_status} Precision: {precision:.2%} (цель ≥ {TASK_PRECISION_TARGET:.0%})")
    print(f"  {r_status} Recall:    {recall:.2%} (цель ≥ {TASK_RECALL_TARGET:.0%})")
    print(f"  F1-score: {f1:.2%}")


def _rule_based_task_check(text: str) -> bool:
    """Проверка задачи по правилам (fallback)."""
    lower = text.lower().strip()
    task_patterns = [
        r"\bнужно\b", r"\bнадо\b", r"\bнеобходимо\b",
        r"\bсделай\b", r"\bподготовь\b", r"\bпроверь\b",
        r"\bотправь\b", r"\bсогласуй\b", r"\bзакрой\b",
        r"\bсделать\b", r"\bподготовить\b", r"\bдо\s+завтра\b",
        r"\bдо\s+пятницы\b", r"\bдо\s+конца\s+недели\b",
    ]
    exclude = [r"^\s*да\b", r"^\s*нет\b", r"^\s*хорошо\b",
               r"^\s*понял\b", r"\bне нужно\b"]

    for p in exclude:
        if re.search(p, lower):
            return False
    return any(re.search(p, lower) for p in task_patterns)


# ==================
# Sentiment Accuracy
# ==================
def evaluate_sentiment():
    """Оценка точности сентимент-анализа."""
    ref_path = os.path.join(EVAL_DATA_DIR, "sentiment_reference.jsonl")

    if not os.path.exists(ref_path):
        print(f"[Sentiment] Файл {ref_path} не найден!")
        print(f"  Создайте его в формате:")
        print(f'  {{"text": "Всё отлично!", "label": "positive"}}')
        _create_sentiment_template()
        return

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        model_name = os.getenv("SENTIMENT_MODEL_NAME",
                               "seara/rubert-tiny2-russian-sentiment")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        clf = pipeline("text-classification", model=mdl, tokenizer=tok, device=-1)
        print(f"  Модель: {model_name}")
    except Exception as e:
        print(f"[Sentiment] Не удалось загрузить модель: {e}")
        return

    entries = _load_jsonl(ref_path)
    if not entries:
        print("[Sentiment] Нет данных!")
        return

    correct = 0
    total = 0
    confusion = {}  # (expected, predicted) -> count

    for entry in entries:
        text = entry.get("text", "").strip()
        expected = entry.get("label", "").strip().lower()
        if expected not in ("positive", "negative", "neutral"):
            continue

        result = clf(text, truncation=True, max_length=512)[0]
        raw_label = result.get("label", "").lower()
        if "neg" in raw_label:
            predicted = "negative"
        elif "pos" in raw_label:
            predicted = "positive"
        else:
            predicted = "neutral"

        total += 1
        if predicted == expected:
            correct += 1

        key = (expected, predicted)
        confusion[key] = confusion.get(key, 0) + 1

    accuracy = correct / total if total > 0 else 0.0
    status = "✅ PASSED" if accuracy >= SENTIMENT_ACCURACY_TARGET else "❌ FAILED"

    print(f"\n  Всего примеров: {total}")
    print(f"  Правильных: {correct}")
    print(f"  {status} Accuracy: {accuracy:.2%} (цель ≥ {SENTIMENT_ACCURACY_TARGET:.0%})")

    # Confusion matrix
    labels = ["positive", "negative", "neutral"]
    print(f"\n  Confusion matrix:")
    print(f"  {'':>12s}  {'positive':>10s}  {'negative':>10s}  {'neutral':>10s}")
    for exp in labels:
        row = []
        for pred in labels:
            row.append(str(confusion.get((exp, pred), 0)))
        print(f"  {exp:>12s}  {'':>4s}{row[0]:>6s}  {'':>4s}{row[1]:>6s}  {'':>4s}{row[2]:>6s}")


# ==================
# Helpers
# ==================
def _load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _create_wer_template():
    path = os.path.join(EVAL_DATA_DIR, "wer_reference.jsonl")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "audio": "converted/example.wav",
                "reference_text": "Вставьте сюда эталонный текст расшифровки аудио"
            }, ensure_ascii=False) + "\n")
        print(f"  Создан шаблон: {path}")


def _create_tasks_template():
    path = os.path.join(EVAL_DATA_DIR, "tasks_reference.jsonl")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            examples = [
                {"text": "Подготовь отчёт до пятницы.", "is_task": True},
                {"text": "Нужно согласовать бюджет.", "is_task": True},
                {"text": "Коллеги, всем доброе утро.", "is_task": False},
                {"text": "Да, я сделаю это сегодня.", "is_task": False},
                {"text": "Когда будет готов релиз?", "is_task": False},
            ]
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Создан шаблон: {path}")


def _create_sentiment_template():
    path = os.path.join(EVAL_DATA_DIR, "sentiment_reference.jsonl")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            examples = [
                {"text": "Всё отлично, молодцы!", "label": "positive"},
                {"text": "Опять задержка, это критично.", "label": "negative"},
                {"text": "Перейдём к следующему пункту.", "label": "neutral"},
            ]
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  Создан шаблон: {path}")


# ==================
# Main
# ==================
def main():
    parser = argparse.ArgumentParser(
        description="Оценка KPI моделей PM Insights"
    )
    parser.add_argument(
        "--mode", choices=["all", "wer", "tasks", "sentiment"],
        default="all", help="Что оценивать"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PM Insights — Оценка KPI")
    print("=" * 60)

    if args.mode in ("all", "wer"):
        print("\n📊 1. WER (Word Error Rate)")
        print("-" * 40)
        evaluate_wer()

    if args.mode in ("all", "tasks"):
        print("\n📊 2. Task Detection (Precision / Recall)")
        print("-" * 40)
        evaluate_tasks()

    if args.mode in ("all", "sentiment"):
        print("\n📊 3. Sentiment Accuracy")
        print("-" * 40)
        evaluate_sentiment()

    print("\n" + "=" * 60)
    print("Готово!")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from pm_insights import settings


LABELS = {"task", "question", "answer", "other"}
BASELINE_MODEL_PATH = settings.MODELS_DIR / "baseline_classifier" / "model.joblib"

QUESTION_RE = re.compile(
    r"\?|^\s*(кто|что|когда|где|почему|зачем|как|сколько|какой|какая|какие)\b|"
    r"\b(можно ли|нужно ли|есть ли|готово ли|успеваем ли|подскажи|скажи)\b",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(
    r"^\s*([\wА-Яа-яЁё]+[,.]\s*)?(да|нет|готово|сделано|уже|пока нет|в процессе)\b|"
    r"\b(я\s+уточню|я\s+отправлю|отправлю|уточню|проверю|исправлю|подготовлю|"
    r"лежит|открыты|закрыты|готова|готов|исправлена|согласовано|отвечаю\s+я)\b",
    re.IGNORECASE,
)
TASK_RE = re.compile(
    r"\b(нужно|надо|необходимо)\s+\w+|"
    r"\b(подготовь|исправь|проверь|собери|отправь|обнови|добавь|согласуй|оформи|"
    r"реализуй|передай|создай|заполни|уточни|сделай|пришли)\b",
    re.IGNORECASE,
)


def _rule_based_label(text: str) -> tuple[str, float]:
    value = text or ""
    if QUESTION_RE.search(value):
        return "question", 0.82
    if ANSWER_RE.search(value):
        return "answer", 0.78
    if TASK_RE.search(value):
        return "task", 0.80
    return "other", 0.60


@lru_cache(maxsize=1)
def _load_baseline_model() -> Any:
    if not BASELINE_MODEL_PATH.exists():
        raise RuntimeError(f"Baseline classifier is not found at {BASELINE_MODEL_PATH}.")
    try:
        import joblib
    except Exception as exc:
        raise RuntimeError("Baseline classifier requires joblib and scikit-learn.") from exc
    return joblib.load(BASELINE_MODEL_PATH)


@lru_cache(maxsize=1)
def _load_rubert_classifier() -> Any:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "RuBERT fragment classifier requires optional dependencies from requirements-ml.txt "
            "(transformers and torch)."
        ) from exc
    if not settings.RUBERT_CLASSIFIER_PATH.exists():
        raise RuntimeError(f"Trained RuBERT classifier is not found at {settings.RUBERT_CLASSIFIER_PATH}.")
    return pipeline(
        "text-classification",
        model=str(settings.RUBERT_CLASSIFIER_PATH),
        tokenizer=str(settings.RUBERT_CLASSIFIER_PATH),
    )


def _baseline_predict(text: str) -> tuple[str, float]:
    model = _load_baseline_model()
    label = str(model.predict([text])[0])
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = float(max(probabilities))
    elif hasattr(model, "decision_function"):
        scores = model.decision_function([text])
        if hasattr(scores, "ravel"):
            values = list(scores.ravel())
            confidence = float(max(values)) if values else 0.0
    return label if label in LABELS else "other", confidence


def _rubert_predict(text: str) -> tuple[str, float]:
    pipe = _load_rubert_classifier()
    output = pipe(text or "")[0]
    if isinstance(output, list):
        output = max(output, key=lambda item: item.get("score", 0.0))
    label = str(output.get("label", "")).lower()
    if label.startswith("label_"):
        labels_path = settings.RUBERT_CLASSIFIER_PATH / "labels.json"
        if labels_path.exists():
            import json

            mapping = json.loads(labels_path.read_text(encoding="utf-8"))
            label = str(mapping.get("id2label", {}).get(label.removeprefix("label_"), label)).lower()
    if label not in LABELS:
        label = "other"
    return label, float(output.get("score", 0.0) or 0.0)


def classify_fragment(text: str, engine: str | None = None) -> dict[str, object]:
    selected = (engine or settings.TASK_CLASSIFIER_ENGINE or "auto").lower()
    if selected == "auto":
        if settings.TASK_CLASSIFIER_ENGINE == "rubert" and settings.RUBERT_CLASSIFIER_PATH.exists():
            selected = "rubert"
        else:
            selected = "baseline" if Path(BASELINE_MODEL_PATH).exists() else "rule_based"

    if selected == "baseline":
        try:
            label, confidence = _baseline_predict(text)
            return {"label": label, "confidence": confidence, "engine": "baseline"}
        except Exception as exc:
            if not settings.ENABLE_MODEL_FALLBACK:
                raise
            label, confidence = _rule_based_label(text)
            return {"label": label, "confidence": confidence, "engine": "rule_based", "fallback_reason": str(exc)}

    if selected == "rubert":
        try:
            label, confidence = _rubert_predict(text)
            return {"label": label, "confidence": confidence, "engine": "rubert"}
        except Exception as exc:
            if not settings.ENABLE_MODEL_FALLBACK:
                raise
            label, confidence = _rule_based_label(text)
            return {"label": label, "confidence": confidence, "engine": "rule_based", "fallback_reason": str(exc)}

    label, confidence = _rule_based_label(text)
    return {"label": label, "confidence": confidence, "engine": "rule_based"}


def classify_fragments(fragments: list[dict], engine: str | None = None) -> list[dict]:
    rows = []
    for fragment in fragments:
        prediction = classify_fragment(fragment.get("text", ""), engine=engine)
        rows.append(
            {
                "fragment_index": fragment.get("fragment_index"),
                "text": fragment.get("text", ""),
                "predicted_label": prediction["label"],
                "confidence": prediction["confidence"],
                "engine": prediction["engine"],
            }
        )
    return rows

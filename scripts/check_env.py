from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm_insights import settings


def _has_sentence_transformers_static_embedding() -> bool:
    try:
        from sentence_transformers.sentence_transformer import modules  # type: ignore

        return hasattr(modules, "StaticEmbedding")
    except Exception:
        return False


REQUIRED = [
    ("DATABASE_URL", "DATABASE_URL"),
    ("REQUIRE_AUTH", "REQUIRE_AUTH"),
    ("ASR_ENGINE", "ASR_ENGINE"),
    ("WHISPER_MODEL_NAME", "WHISPER_MODEL_NAME"),
    ("WHISPER_DEVICE", "WHISPER_DEVICE"),
    ("WHISPER_COMPUTE_TYPE", "WHISPER_COMPUTE_TYPE"),
    ("WHISPER_LANGUAGE", "WHISPER_LANGUAGE"),
    ("TASK_CLASSIFIER_ENGINE", "TASK_CLASSIFIER_ENGINE"),
    ("RUBERT_CLASSIFIER_PATH", "RUBERT_CLASSIFIER_PATH"),
    ("TOPIC_MODEL_ENGINE", "TOPIC_MODEL_ENGINE"),
    ("TOPIC_MODELING_ENGINE", "TOPIC_MODELING_ENGINE"),
    ("UPLOAD_DIR", "UPLOADS_DIR"),
    ("RESULTS_DIR", "RESULTS_DIR"),
    ("AUTO_RETRAIN", "AUTO_RETRAIN"),
    ("MIN_FEEDBACK_EXAMPLES_FOR_RETRAIN", "MIN_FEEDBACK_EXAMPLES_FOR_RETRAIN"),
    ("FEEDBACK_DATASET_PATH", "FEEDBACK_DATASET_PATH"),
    ("RUBERT_CANDIDATE_PATH", "RUBERT_CANDIDATE_PATH"),
    ("UI_THEME", "UI_THEME"),
]


def main() -> int:
    warnings = []
    missing = [env_name for env_name, attr_name in REQUIRED if not hasattr(settings, attr_name)]
    optional_topic_dependencies = {
        "bertopic": importlib.util.find_spec("bertopic") is not None,
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "sentence_transformers_static_embedding": _has_sentence_transformers_static_embedding(),
        "umap": importlib.util.find_spec("umap") is not None,
        "hdbscan": importlib.util.find_spec("hdbscan") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }
    if not settings.REQUIRE_AUTH:
        warnings.append("REQUIRE_AUTH is false; demo mode is less isolated.")
    if settings.AUTO_RETRAIN:
        warnings.append("AUTO_RETRAIN is true; current MVP should only create pending training jobs, not train in HTTP requests.")
    if settings.TASK_CLASSIFIER_ENGINE == "rubert" and not settings.RUBERT_CLASSIFIER_PATH.exists():
        warnings.append("TASK_CLASSIFIER_ENGINE=rubert but RUBERT_CLASSIFIER_PATH does not exist.")
    if settings.TOPIC_MODEL_ENGINE not in {"auto", "bertopic", "embedding", "rule_based", "fallback"}:
        warnings.append("TOPIC_MODEL_ENGINE should be auto, bertopic, embedding, rule_based, or fallback.")
    if settings.TOPIC_MODEL_ENGINE in {"auto", "bertopic"}:
        missing_bertopic = [
            name
            for name in ["bertopic", "sentence_transformers", "umap", "hdbscan", "sklearn", "numpy", "torch"]
            if not optional_topic_dependencies[name]
        ]
        if not optional_topic_dependencies["sentence_transformers_static_embedding"]:
            missing_bertopic.append("sentence_transformers_static_embedding")
        if missing_bertopic:
            warnings.append(
                "BERTopic topic modeling dependencies are missing: "
                + ", ".join(missing_bertopic)
                + "; embedding/rule-based fallback should be used."
            )
    if settings.TOPIC_MODEL_ENGINE in {"auto", "embedding", "bertopic"}:
        missing_embedding = [
            name
            for name in ["sentence_transformers", "sklearn", "numpy", "torch"]
            if not optional_topic_dependencies[name]
        ]
        if missing_embedding:
            warnings.append(
                "Embedding topic modeling dependencies are missing: "
                + ", ".join(missing_embedding)
                + "; rule-based fallback should be used."
            )
    settings.ensure_runtime_dirs()
    report = {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "checked_keys": [env_name for env_name, _ in REQUIRED],
        "database_url_configured": bool(settings.DATABASE_URL),
        "upload_dir_exists": settings.UPLOADS_DIR.exists(),
        "results_dir_exists": settings.RESULTS_DIR.exists(),
        "topic_model_path": str(settings.TOPIC_MODEL_PATH),
        "topic_model_exists": settings.TOPIC_MODEL_PATH.exists(),
        "topic_dependencies": optional_topic_dependencies,
        "secrets_printed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

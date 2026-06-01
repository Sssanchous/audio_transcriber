from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pm_insights import settings


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
    if not settings.REQUIRE_AUTH:
        warnings.append("REQUIRE_AUTH is false; demo mode is less isolated.")
    if settings.AUTO_RETRAIN:
        warnings.append("AUTO_RETRAIN is true; current MVP should only create pending training jobs, not train in HTTP requests.")
    if settings.TASK_CLASSIFIER_ENGINE == "rubert" and not settings.RUBERT_CLASSIFIER_PATH.exists():
        warnings.append("TASK_CLASSIFIER_ENGINE=rubert but RUBERT_CLASSIFIER_PATH does not exist.")
    if settings.TOPIC_MODEL_ENGINE not in {"auto", "bertopic", "embedding", "rule_based", "fallback"}:
        warnings.append("TOPIC_MODEL_ENGINE should be auto, bertopic, embedding, rule_based, or fallback.")
    if settings.TOPIC_MODEL_ENGINE == "bertopic" and importlib.util.find_spec("bertopic") is None:
        warnings.append("TOPIC_MODEL_ENGINE=bertopic but BERTopic is not installed; fallback should be used.")
    settings.ensure_runtime_dirs()
    report = {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "checked_keys": [env_name for env_name, _ in REQUIRED],
        "database_url_configured": bool(settings.DATABASE_URL),
        "upload_dir_exists": settings.UPLOADS_DIR.exists(),
        "results_dir_exists": settings.RESULTS_DIR.exists(),
        "secrets_printed": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())

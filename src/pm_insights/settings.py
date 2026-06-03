from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


DATABASE_URL = os.getenv("DATABASE_URL", "")

UPLOADS_DIR = PROJECT_ROOT / os.getenv("UPLOAD_DIR", "uploads")
RESULTS_DIR = PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")
TRANSCRIPTS_DIR = PROJECT_ROOT / "transcripts"
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}

ASR_ENGINE = os.getenv("ASR_ENGINE", "whisper").lower()
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", os.getenv("WHISPER_MODEL_NAME", "large-v3"))
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").lower()
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE",
    "float16" if WHISPER_DEVICE == "cuda" else "int8",
)
WHISPER_LANGUAGE = os.getenv("ASR_LANGUAGE", os.getenv("WHISPER_LANGUAGE", "ru"))
WHISPER_BEAM_SIZE = _get_int("WHISPER_BEAM_SIZE", 5)

SPLIT_TRANSCRIPT_SEGMENTS = _get_bool("SPLIT_TRANSCRIPT_SEGMENTS", True)
MAX_SEGMENT_WORDS = _get_int("MAX_SEGMENT_WORDS", 45)
MIN_SEGMENT_CHARS = _get_int("MIN_SEGMENT_CHARS", 4)

TOPIC_MODEL_ENGINE = os.getenv("TOPIC_MODEL_ENGINE", os.getenv("TOPIC_MODELING_ENGINE", "auto")).lower()
TOPIC_MODELING_ENGINE = TOPIC_MODEL_ENGINE
TOPIC_MIN_FRAGMENTS = _get_int("TOPIC_MIN_FRAGMENTS", 8)
TOPIC_MAX_TOPICS = _get_int("TOPIC_MAX_TOPICS", 12)
TOPIC_MODEL_PATH = PROJECT_ROOT / os.getenv("TOPIC_MODEL_PATH", os.getenv("BERTOPIC_MODEL_PATH", "models/topic_model"))
TOPIC_EMBEDDING_MODEL = os.getenv("TOPIC_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
BERTOPIC_MODEL_PATH = TOPIC_MODEL_PATH

TASK_CLASSIFIER_ENGINE = os.getenv("TASK_CLASSIFIER_ENGINE", "baseline").lower()
RUBERT_TASK_MODEL = os.getenv("RUBERT_TASK_MODEL", "cointegrated/rubert-tiny2")
RUBERT_CLASSIFIER_PATH = PROJECT_ROOT / os.getenv("RUBERT_CLASSIFIER_PATH", "models/rubert_classifier")

SENTIMENT_ENGINE = os.getenv("SENTIMENT_ENGINE", "rule_based").lower()
RUBERT_SENTIMENT_MODEL = os.getenv("RUBERT_SENTIMENT_MODEL", "seara/rubert-tiny2-russian-sentiment")

ENABLE_MODEL_FALLBACK = _get_bool("ENABLE_MODEL_FALLBACK", True)

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.getenv("JWT_SECRET", "change_me"))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = _get_int(
    "JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
    _get_int("JWT_EXPIRE_MINUTES", 1440),
)
REQUIRE_AUTH = _get_bool("REQUIRE_AUTH", True)
UI_THEME = os.getenv("UI_THEME", "light_monochrome")
AUTO_RETRAIN = _get_bool("AUTO_RETRAIN", False)
MIN_FEEDBACK_EXAMPLES_FOR_RETRAIN = _get_int("MIN_FEEDBACK_EXAMPLES_FOR_RETRAIN", 50)
FEEDBACK_DATASET_PATH = PROJECT_ROOT / os.getenv("FEEDBACK_DATASET_PATH", "datasets/sources/feedback_examples.jsonl")
RUBERT_CANDIDATE_PATH = PROJECT_ROOT / os.getenv("RUBERT_CANDIDATE_PATH", "models/rubert_classifier_candidate")
BERTOPIC_MIN_TOPIC_SIZE = _get_int("BERTOPIC_MIN_TOPIC_SIZE", 3)
BERTOPIC_LANGUAGE = os.getenv("BERTOPIC_LANGUAGE", "multilingual")

ASYNC_PROCESSING = _get_bool("ASYNC_PROCESSING", False)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)


def ensure_runtime_dirs() -> None:
    for path in [UPLOADS_DIR, RESULTS_DIR, DATASETS_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def masked_database_url() -> str:
    if not DATABASE_URL:
        return ""
    if "@" not in DATABASE_URL or "://" not in DATABASE_URL:
        return DATABASE_URL
    prefix, rest = DATABASE_URL.split("://", 1)
    if "@" not in rest:
        return DATABASE_URL
    _, host = rest.rsplit("@", 1)
    return f"{prefix}://***:***@{host}"

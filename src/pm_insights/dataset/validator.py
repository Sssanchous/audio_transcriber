from __future__ import annotations

from .classifier import VALID_LABELS
from .cleaner import normalize_text


REQUIRED_FIELDS = {"id", "source_file", "fragment_index", "text", "label"}


def validate_record(record: dict) -> tuple[bool, str | None]:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        return False, "missing_fields"
    if not normalize_text(record.get("text", "")):
        return False, "empty_text"
    if record.get("label") not in VALID_LABELS:
        return False, "invalid_label"
    for label in record.get("secondary_labels", []):
        if label not in VALID_LABELS:
            return False, "invalid_secondary_label"
    return True, None

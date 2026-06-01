from __future__ import annotations

from pm_insights.dataset.cleaner import normalize_text


def clean_text(text: str) -> str:
    return normalize_text(text)

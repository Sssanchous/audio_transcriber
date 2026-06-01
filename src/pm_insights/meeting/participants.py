from __future__ import annotations

import json
import re
from typing import Any


def parse_participants(value: str | list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if isinstance(value, list):
        return [
            {"name": str(item.get("name", "")).strip(), "role": str(item.get("role", "")).strip()}
            for item in value
            if str(item.get("name", "")).strip()
        ]
    text = (value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            return parse_participants(parsed)
        except Exception:
            pass

    participants = []
    for raw_line in re.split(r"[\n;]+", text):
        line = raw_line.strip(" \t-•")
        if not line:
            continue
        parts = re.split(r"\s+[—–-]\s+|\s*:\s*", line, maxsplit=1)
        name = parts[0].strip()
        role = parts[1].strip() if len(parts) > 1 else ""
        if name:
            participants.append({"name": name, "role": role})
    return participants


def participants_to_text(participants: list[dict[str, str]]) -> str:
    lines = []
    for item in participants:
        name = item.get("name", "").strip()
        role = item.get("role", "").strip()
        if not name:
            continue
        lines.append(f"{name} — {role}" if role else name)
    return "\n".join(lines)


def participant_names(participants: list[dict[str, str]] | None) -> list[str]:
    return [item.get("name", "").strip() for item in participants or [] if item.get("name", "").strip()]

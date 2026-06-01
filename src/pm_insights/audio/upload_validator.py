from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pm_insights.settings import ALLOWED_AUDIO_EXTENSIONS


@dataclass(frozen=True)
class AudioValidationResult:
    valid: bool
    reason: str | None = None
    extension: str | None = None
    size_bytes: int = 0


def validate_audio_file(path: str | Path, original_filename: str | None = None) -> AudioValidationResult:
    audio_path = Path(path)
    filename = original_filename or audio_path.name

    if not filename or filename.strip() in {"", "."}:
        return AudioValidationResult(False, "missing_filename")

    extension = Path(filename).suffix.lower()
    if not extension:
        return AudioValidationResult(False, "missing_extension")

    if extension not in ALLOWED_AUDIO_EXTENSIONS:
        return AudioValidationResult(False, "unsupported_extension", extension=extension)

    if not audio_path.exists():
        return AudioValidationResult(False, "file_not_found", extension=extension)

    size = audio_path.stat().st_size
    if size <= 0:
        return AudioValidationResult(False, "empty_file", extension=extension, size_bytes=size)

    return AudioValidationResult(True, extension=extension, size_bytes=size)

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> dict:
        """Return text, segments, language and status for an audio file."""

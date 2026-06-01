from __future__ import annotations

from pathlib import Path

from .base import BaseTranscriber


class VoskAdapter(BaseTranscriber):
    def transcribe(self, audio_path: str | Path) -> dict:
        raise RuntimeError("Vosk adapter is not enabled in the MVP. Use StubTranscriber or install/configure Vosk later.")

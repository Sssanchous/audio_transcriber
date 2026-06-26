from __future__ import annotations

from pathlib import Path

from pm_insights import settings

from .base import BaseTranscriber
from .whisper_adapter import WhisperAdapter


class StubTranscriber(BaseTranscriber):
    def __init__(self, fallback_text: str | None = None) -> None:
        self.fallback_text = fallback_text or (
            "Транскрибация не выполнена: передайте transcript_text для теста или подключите Whisper."
        )

    def transcribe(self, audio_path: str | Path) -> dict:
        path = Path(audio_path)
        text = path.read_text(encoding="utf-8") if path.suffix.lower() == ".txt" else self.fallback_text
        return {
            "text": text,
            "segments": [{"start": 0.0, "end": 0.0, "text": text}],
            "language": "ru",
            "model": "stub",
            "duration": 0.0,
            "status": "completed",
        }


def build_transcriber(test_mode: bool = False, fallback_text: str | None = None) -> BaseTranscriber:
    if test_mode:
        return StubTranscriber(fallback_text=fallback_text)
    if settings.ASR_ENGINE != "whisper":
        raise RuntimeError(f"Unsupported ASR_ENGINE={settings.ASR_ENGINE}. Use ASR_ENGINE=whisper.")
    return WhisperAdapter()

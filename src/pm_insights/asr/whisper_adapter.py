from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pm_insights import settings
from pm_insights.dataset.cleaner import normalize_text

from .base import BaseTranscriber


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


class WhisperAdapter(BaseTranscriber):
    """Local faster-whisper adapter. The model is loaded lazily on first use."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        language: str | None = None,
        beam_size: int | None = None,
    ) -> None:
        self.model_name = model_name or settings.WHISPER_MODEL_NAME
        self.device = device or settings.WHISPER_DEVICE
        self.compute_type = compute_type or settings.WHISPER_COMPUTE_TYPE
        self.language = language or settings.WHISPER_LANGUAGE
        self.beam_size = beam_size or settings.WHISPER_BEAM_SIZE
        self._model: Any | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install requirements and keep ASR_ENGINE=whisper."
            ) from exc

        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Whisper model '{self.model_name}' on {self.device}. "
                "Check GPU/CPU resources and WHISPER_* settings."
            ) from exc
        return self._model

    def transcribe(self, audio_path: str | Path) -> dict:
        path = Path(audio_path)
        model = self._load_model()
        kwargs = {
            "beam_size": self.beam_size,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 500, "speech_pad_ms": 200},
            "condition_on_previous_text": False,
            "temperature": 0,
            "word_timestamps": False,
            "no_speech_threshold": 0.55,
            "compression_ratio_threshold": 2.6,
            "log_prob_threshold": -1.0,
        }
        if self.language:
            kwargs["language"] = self.language

        segments_iter, info = model.transcribe(str(path), **kwargs)
        segments: list[dict] = []
        for raw in segments_iter:
            text = normalize_text(getattr(raw, "text", ""))
            if not text:
                continue
            segments.extend(_split_segment(float(raw.start), float(raw.end), text))

        full_text = " ".join(item["text"] for item in segments)
        return {
            "text": full_text,
            "segments": segments,
            "language": getattr(info, "language", None) or self.language or "ru",
            "model": self.model_name,
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 2),
            "status": "completed",
        }


def _split_segment(start: float, end: float, text: str) -> list[dict]:
    if not settings.SPLIT_TRANSCRIPT_SEGMENTS:
        return [{"start": start, "end": end, "text": text}]

    words = text.split()
    if len(words) <= settings.MAX_SEGMENT_WORDS:
        return [{"start": start, "end": end, "text": text}]

    parts = [normalize_text(part) for part in SENTENCE_SPLIT_RE.split(text) if normalize_text(part)]
    if len(parts) <= 1:
        parts = [
            " ".join(words[i : i + settings.MAX_SEGMENT_WORDS])
            for i in range(0, len(words), settings.MAX_SEGMENT_WORDS)
        ]

    total_words = max(1, sum(len(part.split()) for part in parts))
    duration = max(0.0, end - start)
    cursor = start
    result = []
    for index, part in enumerate(parts):
        ratio = len(part.split()) / total_words
        part_duration = duration * ratio
        part_end = end if index == len(parts) - 1 else cursor + part_duration
        result.append({"start": round(cursor, 2), "end": round(part_end, 2), "text": part})
        cursor = part_end
    return result

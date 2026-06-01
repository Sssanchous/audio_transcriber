from __future__ import annotations

from pathlib import Path


def apply_speaker_diarization(wav_path: Path, segments: list[dict]) -> list[dict]:
    result = []

    for seg in segments:
        result.append({
            **seg,
            "speaker": seg.get("speaker") or "SPEAKER_00",
        })

    return result
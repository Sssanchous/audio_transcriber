from pathlib import Path

from pm_insights.asr.transcriber import StubTranscriber


def test_asr_stub_returns_ru_completed(tmp_path: Path):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"audio")

    result = StubTranscriber("Нужно подготовить отчёт.").transcribe(path)

    assert result["language"] == "ru"
    assert result["status"] == "completed"
    assert result["segments"][0]["text"]

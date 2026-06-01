from pathlib import Path

from pm_insights.meeting.pipeline import analyze_meeting


def test_meeting_pipeline_builds_result(tmp_path: Path):
    audio = tmp_path / "meeting.mp3"
    audio.write_bytes(b"audio")

    result = analyze_meeting(
        audio,
        output_dir=tmp_path,
        transcript_text="Когда будет готов макет? Да, я уже проверил. Нужно подготовить отчёт до пятницы.",
    )

    assert result["meeting_id"]
    assert result["tasks"]
    assert result["questions_answers"]
    assert result["metrics"]["tasks_count"] == 1

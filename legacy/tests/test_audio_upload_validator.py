from pathlib import Path

from pm_insights.audio.upload_validator import validate_audio_file


def test_audio_upload_validator_allows_supported_non_empty_file(tmp_path: Path):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"audio")

    result = validate_audio_file(path)

    assert result.valid
    assert result.extension == ".mp3"


def test_audio_upload_validator_rejects_empty_and_unsupported(tmp_path: Path):
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    bad = tmp_path / "meeting.ogg"
    bad.write_bytes(b"audio")

    assert validate_audio_file(empty).reason == "empty_file"
    assert validate_audio_file(bad).reason == "unsupported_extension"

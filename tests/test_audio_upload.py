from pm_insights.audio.upload_validator import validate_audio_file


def test_audio_validator_accepts_supported_file(tmp_path):
    path = tmp_path / "meeting.mp3"
    path.write_bytes(b"audio")
    result = validate_audio_file(path, "meeting.mp3")
    assert result.valid
    assert result.extension == ".mp3"


def test_audio_validator_rejects_empty_file(tmp_path):
    path = tmp_path / "meeting.wav"
    path.write_bytes(b"")
    result = validate_audio_file(path, "meeting.wav")
    assert not result.valid
    assert result.reason == "empty_file"


def test_audio_validator_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "meeting.opus"
    path.write_bytes(b"audio")
    result = validate_audio_file(path, "meeting.opus")
    assert not result.valid
    assert result.reason == "unsupported_extension"

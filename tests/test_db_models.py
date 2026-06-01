from pm_insights import db, settings


def test_db_models_create_with_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'pm.db'}")
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "SessionLocal", None)

    db.init_db()
    meeting = db.create_meeting(
        meeting_id="meeting_test",
        project_name="PM Insights",
        meeting_date="2026-01-01",
        participants="Иван",
        source_audio="meeting.mp3",
        original_filename="meeting.mp3",
        stored_filename="stored_meeting.mp3",
        file_extension=".mp3",
        file_size_bytes=10,
    )

    assert meeting["meeting_id"] == "meeting_test"
    assert db.get_meeting("meeting_test")["processing_status"] == "uploaded"

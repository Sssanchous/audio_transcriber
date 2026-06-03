from __future__ import annotations

from pathlib import Path
from typing import Any

from pm_insights import db, settings
from pm_insights.meeting.pipeline import analyze_meeting

try:  # Celery is optional when ASYNC_PROCESSING=false.
    from celery import Celery
except Exception:  # pragma: no cover - exercised when optional dependency is absent.
    Celery = None  # type: ignore[assignment]


celery_app = None
if Celery is not None:
    celery_app = Celery(
        "pm_insights",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
    )


def _meeting_analysis_kwargs(meeting: dict[str, Any]) -> dict[str, Any]:
    metadata = meeting.get("metadata") or {}
    return {
        "meeting_title": meeting.get("meeting_title", ""),
        "project_name": meeting.get("project_name") or "PM Insights",
        "meeting_date": meeting.get("meeting_date"),
        "participants": metadata.get("participants") or meeting.get("participants") or "",
        "output_dir": settings.RESULTS_DIR,
        "meeting_id": meeting["meeting_id"],
        "persist_db": True,
    }


def run_meeting_analysis(meeting_id: str) -> dict:
    meeting = db.get_meeting(meeting_id)
    if not meeting:
        raise ValueError(f"Meeting {meeting_id} not found")

    audio_path = settings.UPLOADS_DIR / meeting["stored_filename"]
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Uploaded audio file not found: {audio_path}")

    db.update_meeting_status(meeting_id, "started")
    db.log_processing(meeting_id, "celery", "started", "Async analysis started")
    try:
        result = analyze_meeting(audio_path, **_meeting_analysis_kwargs(meeting))
        db.log_processing(meeting_id, "celery", "success", "Async analysis completed")
        return {"status": "success", "meeting_id": meeting_id, "result": result}
    except Exception as exc:
        db.update_meeting_status(meeting_id, "failed")
        db.log_processing(meeting_id, "celery", "failure", str(exc))
        raise


if celery_app is not None:

    @celery_app.task(name="pm_insights.analyze_meeting_task", bind=True)
    def analyze_meeting_task(self, meeting_id: str) -> dict:  # type: ignore[no-untyped-def]
        db.update_meeting_metadata(meeting_id, {"async_task_id": self.request.id})
        return run_meeting_analysis(meeting_id)

else:

    def analyze_meeting_task(*_: Any, **__: Any) -> None:
        raise RuntimeError("Celery is not installed. Install celery and redis or set ASYNC_PROCESSING=false.")

from __future__ import annotations

import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from pm_insights import db, settings
from pm_insights import auth
from pm_insights.audio.upload_validator import validate_audio_file
from pm_insights.export.report_builder import build_report_payload, export_docx, export_json, export_pdf, export_xlsx
from pm_insights.meeting import storage as meeting_storage
from pm_insights.meeting.pipeline import analyze_meeting
from pm_insights.meeting.participants import parse_participants, participants_to_text
from pm_insights.nlp.postprocessing import normalize_analysis_result
from pydantic import BaseModel


settings.ensure_runtime_dirs()

KPI_TARGETS = {
    "wer_target": "<= 15%",
    "task_precision_recall_target": ">= 80%",
    "sentiment_accuracy_target": ">= 75%",
    "processing_1h_target": "15-20 minutes",
    "business_time_saving_target": "30-50%",
}
KPI_STATUS = {
    "wer": "requires_reference_transcript",
    "task_precision_recall": "requires_expert_annotation",
    "sentiment_accuracy": "requires_expert_annotation",
    "processing_time": "measured_when_audio_is_processed",
    "business_time_saving": "requires_pilot_survey",
}
BUSINESS_KPI = {"target_minutes_saved_percent": "30-50%", "status": "requires_pilot_survey"}

app = FastAPI(title="PM Insights")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str
    full_name: str | None = ""


class LoginRequest(BaseModel):
    email: str | None = None
    username: str | None = None
    password: str


class FeedbackRequest(BaseModel):
    item_type: str
    source_text: str
    predicted_label: str = ""
    corrected_label: str
    corrected_text: str | None = ""
    metadata: dict | None = None


def _db_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"Database is unavailable: {exc}")


def _current_user_id(current_user: dict | None) -> int | None:
    return int(current_user["id"]) if current_user else None


async def _save_upload(file: UploadFile) -> tuple[Path, str, int]:
    filename = Path(file.filename or "").name
    if not filename or not Path(filename).suffix:
        raise HTTPException(400, "Файл должен иметь имя и расширение.")

    extension = Path(filename).suffix.lower()
    if extension not in settings.ALLOWED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(settings.ALLOWED_AUDIO_EXTENSIONS))
        raise HTTPException(400, f"Поддерживаются только форматы: {allowed}")

    stored_path = settings.UPLOADS_DIR / f"{uuid4().hex[:12]}_{filename}"
    with stored_path.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    validation = validate_audio_file(stored_path, filename)
    if not validation.valid:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Некорректный аудиофайл: {validation.reason}")
    return stored_path, extension, validation.size_bytes


def _enqueue_meeting_analysis(meeting_id: str) -> str:
    try:
        from pm_insights.worker import analyze_meeting_task, celery_app
    except Exception as exc:
        raise HTTPException(503, f"Async processing is not available: {exc}") from exc
    if celery_app is None or not hasattr(analyze_meeting_task, "delay"):
        raise HTTPException(503, "Async processing requires celery and redis dependencies.")
    try:
        db.update_meeting_status(meeting_id, "queued")
        db.log_processing(meeting_id, "celery", "queued", "Analysis queued")
        task = analyze_meeting_task.delay(meeting_id)
        task_id = str(task.id)
        db.update_meeting_metadata(meeting_id, {"async_task_id": task_id})
        return task_id
    except HTTPException:
        raise
    except Exception as exc:
        db.update_meeting_status(meeting_id, "uploaded")
        db.log_processing(meeting_id, "celery", "queue_failed", str(exc))
        raise HTTPException(503, f"Failed to enqueue analysis: {exc}") from exc


@app.get("/")
def root() -> dict:
    return {"service": "PM Insights", "status": "running"}


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "database_configured": bool(settings.DATABASE_URL),
            "asr_engine": settings.ASR_ENGINE,
            "whisper_model": settings.WHISPER_MODEL_NAME,
        }
    )


def _public_user_payload(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "username": user["username"],
        "full_name": user.get("full_name", ""),
    }


def _token_response(user: dict) -> dict:
    token = auth.create_access_token(user)
    return {
        "access_token": token,
        "token": token,
        "token_type": "bearer",
        "user": _public_user_payload(user),
    }


@app.post("/api/auth/register")
@app.post("/auth/register")
def register_user(payload: RegisterRequest) -> dict:
    email = payload.email.strip().lower()
    username = payload.username.strip()
    password = payload.password

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email is required.")
    if not username:
        raise HTTPException(400, "Username is required.")
    if not password or len(password) < 6:
        raise HTTPException(400, "Password must contain at least 6 characters.")

    try:
        if db.get_user_by_email(email):
            raise HTTPException(400, "User with this email already exists.")
        if db.get_user_by_username(username):
            raise HTTPException(400, "User with this username already exists.")
        user = db.create_user(
            email=email,
            username=username,
            full_name=payload.full_name or "",
            password_hash=auth.hash_password(password),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _db_error(exc) from exc

    return _token_response(user)


@app.post("/api/auth/login")
@app.post("/auth/login")
def login_user(payload: LoginRequest) -> dict:
    login = (payload.email or payload.username or "").strip()
    if not login or not payload.password:
        raise HTTPException(401, "Invalid email/username or password.")
    try:
        user = db.get_user_auth_record(login)
    except Exception as exc:
        raise _db_error(exc) from exc
    if not user or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email/username or password.")
    return _token_response(user)


@app.get("/api/auth/me")
@app.get("/auth/me")
def auth_me(current_user: dict = Depends(auth.get_current_user)) -> dict:
    return _public_user_payload(current_user)


@app.post("/upload")
@app.post("/api/upload")
async def upload_audio(
    file: UploadFile = File(...),
    meeting_title: str = Form(""),
    meeting_date: str | None = Form(None),
    project_name: str = Form("PM Insights"),
    participants: str = Form(""),
    current_user: dict | None = Depends(auth.maybe_current_user),
) -> dict:
    meeting_title = meeting_title.strip()
    participant_items = parse_participants(participants)
    if not meeting_title:
        raise HTTPException(400, "Укажите название встречи.")

    stored_path, extension, size_bytes = await _save_upload(file)
    meeting_id = f"meeting_{uuid4().hex[:12]}"
    metadata = {
        "meeting_info": {
            "meeting_title": meeting_title,
            "meeting_date": meeting_date,
            "project_name": project_name,
            "participants": participant_items,
            "meeting_key": db.normalize_meeting_key(project_name, meeting_title),
        },
        "participants": participant_items,
    }
    try:
        meeting = db.create_meeting(
            meeting_id=meeting_id,
            meeting_title=meeting_title,
            project_name=project_name,
            meeting_date=meeting_date,
            participants=participants_to_text(participant_items),
            metadata=metadata,
            source_audio=stored_path.name,
            original_filename=Path(file.filename or stored_path.name).name,
            stored_filename=stored_path.name,
            file_extension=extension,
            file_size_bytes=size_bytes,
            user_id=_current_user_id(current_user),
        )
        db.log_processing(meeting_id, "upload", "uploaded", "Audio file uploaded")
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise _db_error(exc) from exc

    if settings.ASYNC_PROCESSING:
        task_id = _enqueue_meeting_analysis(meeting_id)
        return {
            "ok": True,
            "meeting_id": meeting_id,
            "record_id": meeting_id,
            "task_id": task_id,
            "status": "queued",
        }

    return {"ok": True, "meeting_id": meeting_id, "record_id": meeting_id, "status": meeting["processing_status"]}


@app.post("/meetings/{meeting_id}/analyze")
@app.post("/api/meetings/{meeting_id}/analyze")
def analyze_uploaded(meeting_id: str, current_user: dict | None = Depends(auth.maybe_current_user)) -> dict:
    try:
        meeting = db.get_meeting(meeting_id, user_id=_current_user_id(current_user))
    except Exception as exc:
        raise _db_error(exc) from exc
    if not meeting:
        raise HTTPException(404, "Meeting not found")

    if settings.ASYNC_PROCESSING:
        task_id = _enqueue_meeting_analysis(meeting_id)
        return {"ok": True, "meeting_id": meeting_id, "task_id": task_id, "status": "queued"}

    audio_path = settings.UPLOADS_DIR / meeting["stored_filename"]
    if not audio_path.exists():
        raise HTTPException(404, "Uploaded audio file not found")

    try:
        result = analyze_meeting(
            audio_path,
            meeting_title=meeting.get("meeting_title", ""),
            project_name=meeting.get("project_name") or "PM Insights",
            meeting_date=meeting.get("meeting_date"),
            participants=(meeting.get("metadata") or {}).get("participants") or meeting.get("participants") or "",
            output_dir=settings.RESULTS_DIR,
            meeting_id=meeting_id,
            persist_db=True,
        )
    except Exception as exc:
        raise HTTPException(500, f"Analysis failed: {exc}") from exc
    return {"ok": True, "meeting_id": meeting_id, "status": "completed", "result": result}


@app.get("/meetings")
@app.get("/api/meetings")
@app.get("/api/records")
def meetings(current_user: dict | None = Depends(auth.maybe_current_user)) -> list[dict]:
    try:
        return db.list_meetings(user_id=_current_user_id(current_user))
    except Exception as exc:
        raise _db_error(exc) from exc


@app.get("/meetings/{meeting_id}")
@app.get("/api/meetings/{meeting_id}")
@app.get("/api/records/{meeting_id}")
def meeting_detail(meeting_id: str, current_user: dict | None = Depends(auth.maybe_current_user)) -> dict:
    try:
        meeting = db.get_meeting(meeting_id, user_id=_current_user_id(current_user))
    except Exception as exc:
        raise _db_error(exc) from exc
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    return meeting


@app.get("/meetings/{meeting_id}/status")
@app.get("/api/meetings/{meeting_id}/status")
def meeting_status(meeting_id: str, current_user: dict | None = Depends(auth.maybe_current_user)) -> dict:
    try:
        meeting = db.get_meeting(meeting_id, user_id=_current_user_id(current_user))
    except Exception as exc:
        raise _db_error(exc) from exc
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    metadata = meeting.get("metadata") or {}
    status = meeting.get("processing_status") or "unknown"
    is_failed = status in {"failed", "failure"}
    public_status = "success" if status == "completed" else "failure" if is_failed else status
    return {
        "meeting_id": meeting_id,
        "status": status,
        "job_status": public_status,
        "task_id": metadata.get("async_task_id"),
        "result_ready": status == "completed",
        "error_message": db.get_last_processing_error(meeting_id) if is_failed else None,
    }


@app.delete("/meetings/{meeting_id}")
@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: str, current_user: dict | None = Depends(auth.maybe_current_user)) -> dict:
    user_id = _current_user_id(current_user)
    try:
        meeting = db.get_meeting(meeting_id, user_id=user_id)
    except Exception as exc:
        raise _db_error(exc) from exc
    if not meeting:
        raise HTTPException(404, "Meeting not found")

    try:
        deleted = db.delete_meeting(meeting_id, user_id=user_id)
        if not deleted:
            raise HTTPException(404, "Meeting not found")
        meeting_storage.delete_meeting(meeting_id, output_dir=settings.RESULTS_DIR)
    except HTTPException:
        raise
    except Exception as exc:
        raise _db_error(exc) from exc

    return {"status": "deleted", "meeting_id": meeting_id}


_NORMALIZED_RESULT_CACHE: dict[str, tuple[float, dict]] = {}
NORMALIZED_RESULT_CACHE_TTL_SECONDS = 60


def _get_normalized_result(meeting_id: str | None, result: dict) -> dict:
    # Results saved by the current pipeline already carry is_normalized=True
    # (normalize_analysis_result() ran once at write-time) - just use them as-is.
    if result.get("is_normalized"):
        return result

    # Older meetings predating that marker: normalize on-the-fly and cache it,
    # shared between /result and /dashboard so neither re-pays this per request.
    cached = _NORMALIZED_RESULT_CACHE.get(meeting_id) if meeting_id else None
    if cached is not None and (time.monotonic() - cached[0]) < NORMALIZED_RESULT_CACHE_TTL_SECONDS:
        return cached[1]

    normalized = normalize_analysis_result(result)
    if meeting_id:
        _NORMALIZED_RESULT_CACHE[meeting_id] = (time.monotonic(), normalized)
    return normalized


@app.get("/meetings/{meeting_id}/result")
@app.get("/api/meetings/{meeting_id}/result")
def meeting_result(meeting_id: str, current_user: dict | None = Depends(auth.maybe_current_user)) -> dict:
    try:
        # Re-checked on every request (cheap) so the cache below can never bypass
        # the per-user ownership check that get_result() enforces.
        result = db.get_result(meeting_id, user_id=_current_user_id(current_user))
    except Exception as exc:
        raise _db_error(exc) from exc
    if not result:
        raise HTTPException(404, "Meeting result not found")

    normalized = _get_normalized_result(meeting_id, result)
    payload = dict(normalized)
    payload["dynamic_analysis"] = _dynamic_analysis_for_result(payload, _current_user_id(current_user))
    return payload


@app.post("/meetings/{meeting_id}/feedback")
@app.post("/api/meetings/{meeting_id}/feedback")
def save_feedback(
    meeting_id: str,
    payload: FeedbackRequest,
    current_user: dict = Depends(auth.get_current_user),
) -> dict:
    try:
        meeting = db.get_meeting(meeting_id, user_id=int(current_user["id"]))
    except Exception as exc:
        raise _db_error(exc) from exc
    if not meeting:
        raise HTTPException(404, "Meeting not found")
    feedback = db.create_feedback(
        user_id=int(current_user["id"]),
        meeting_id=meeting_id,
        item_type=payload.item_type,
        source_text=payload.source_text,
        predicted_label=payload.predicted_label,
        corrected_label=payload.corrected_label,
        corrected_text=payload.corrected_text or "",
        metadata=payload.metadata or {},
    )
    return {
        "ok": True,
        "feedback": feedback,
        "learning_status": {
            "feedback_saved": True,
            "ready_for_training": True,
            "message": "Исправление сохранено для будущего дообучения модели.",
        },
    }


@app.get("/meetings/{meeting_id}/feedback")
@app.get("/api/meetings/{meeting_id}/feedback")
def list_meeting_feedback(meeting_id: str, current_user: dict = Depends(auth.get_current_user)) -> list[dict]:
    try:
        meeting = db.get_meeting(meeting_id, user_id=int(current_user["id"]))
        if not meeting:
            raise HTTPException(404, "Meeting not found")
        return db.list_feedback(meeting_id=meeting_id, user_id=int(current_user["id"]))
    except HTTPException:
        raise
    except Exception as exc:
        raise _db_error(exc) from exc


_DYNAMIC_ANALYSIS_CACHE: dict[tuple[str | None, int | None], tuple[float, dict]] = {}
DYNAMIC_ANALYSIS_CACHE_TTL_SECONDS = 60


def _dynamic_analysis_for_result(result: dict, user_id: int | None) -> dict:
    cache_key = (result.get("meeting_id"), user_id)
    cached = _DYNAMIC_ANALYSIS_CACHE.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < DYNAMIC_ANALYSIS_CACHE_TTL_SECONDS:
        return cached[1]

    payload = _compute_dynamic_analysis_for_result(result, user_id)
    _DYNAMIC_ANALYSIS_CACHE[cache_key] = (time.monotonic(), payload)
    return payload


def _compute_dynamic_analysis_for_result(result: dict, user_id: int | None) -> dict:
    metadata = result.get("metadata") or {}
    meeting_info = metadata.get("meeting_info") or {}
    key = meeting_info.get("meeting_key") or result.get("meeting_key")
    if not key:
        return {"available": False, "message": "Для этой серии встреч пока нет истории для динамического анализа."}
    try:
        completed = [item for item in db.list_meetings_with_results(user_id=user_id) if item.get("meeting_key") == key]
        completed = [item for item in completed if item.get("metrics")]
    except Exception:
        completed = []
    previous = [item for item in completed if item.get("meeting_id") != result.get("meeting_id")]
    if not previous:
        return {"available": False, "message": "Для этой серии встреч пока нет истории для динамического анализа."}

    current_metrics = result.get("metrics", {})
    last = previous[-1]
    last_metrics = last.get("metrics", {})
    topic_counts: Counter = Counter()
    for item in completed:
        normalized_item = _get_normalized_result(item.get("meeting_id"), item)
        for topic in normalized_item.get("clean_topics") or []:
            name = topic.get("topic_name") or topic.get("title")
            if name:
                topic_counts[str(name)] += int(topic.get("count") or 1)
        for aspect in normalized_item.get("clean_aspects") or []:
            name = aspect.get("title") or aspect.get("name")
            if name:
                topic_counts[str(name)] += int(aspect.get("count") or 1)
    repeated_topics = [topic for topic, count in topic_counts.most_common(5) if count > 1]
    return {
        "available": True,
        "meeting_key": key,
        "previous_meetings_count": len(previous),
        "tasks_delta": current_metrics.get("tasks_count", 0) - last_metrics.get("tasks_count", 0),
        "questions_delta": current_metrics.get("questions_count", 0) - last_metrics.get("questions_count", 0),
        "average_sentiment_delta": round(
            float(current_metrics.get("average_sentiment", 0.0)) - float(last_metrics.get("average_sentiment", 0.0)),
            3,
        ),
        "negative_fragments_delta": current_metrics.get("negative_fragments_count", 0)
        - last_metrics.get("negative_fragments_count", 0),
        "repeated_topics": repeated_topics,
        "repeated_topics_message": "" if repeated_topics else "Повторяющиеся аспекты не определены.",
        "unresolved_questions": [
            item.get("question")
            for item in result.get("clean_questions_answers", [])
            if item.get("status") in {"partial", "not_answered"}
        ][:5],
    }


@app.get("/meetings/{meeting_id}/export/json")
@app.get("/api/meetings/{meeting_id}/export/json")
def export_meeting_json(
    meeting_id: str,
    current_user: dict | None = Depends(auth.maybe_current_user),
) -> Response:
    result = _load_result_for_export(meeting_id, current_user)
    report = build_report_payload(result)
    return Response(
        content=export_json(report),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="pm_insights_{meeting_id}.json"'},
    )


def _load_result_for_export(meeting_id: str, current_user: dict | None) -> dict:
    try:
        result = db.get_result(meeting_id, user_id=_current_user_id(current_user))
    except Exception as exc:
        result_path = settings.RESULTS_DIR / f"{meeting_id}.json"
        if result_path.exists() and current_user is None:
            import json

            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            raise _db_error(exc) from exc
    if not result:
        result_path = settings.RESULTS_DIR / f"{meeting_id}.json"
        if result_path.exists() and current_user is None:
            import json

            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            raise HTTPException(404, "Meeting result not found")
    return result


@app.get("/meetings/{meeting_id}/export/xlsx")
@app.get("/api/meetings/{meeting_id}/export/xlsx")
def export_meeting_xlsx(
    meeting_id: str,
    current_user: dict | None = Depends(auth.maybe_current_user),
) -> Response:
    report = build_report_payload(_load_result_for_export(meeting_id, current_user))
    return Response(
        content=export_xlsx(report),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="pm_insights_{meeting_id}.xlsx"'},
    )


@app.get("/meetings/{meeting_id}/export/docx")
@app.get("/api/meetings/{meeting_id}/export/docx")
def export_meeting_docx(
    meeting_id: str,
    current_user: dict | None = Depends(auth.maybe_current_user),
) -> Response:
    report = build_report_payload(_load_result_for_export(meeting_id, current_user))
    return Response(
        content=export_docx(report),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="pm_insights_{meeting_id}.docx"'},
    )


@app.get("/meetings/{meeting_id}/export/pdf")
@app.get("/api/meetings/{meeting_id}/export/pdf")
def export_meeting_pdf(
    meeting_id: str,
    current_user: dict | None = Depends(auth.maybe_current_user),
) -> Response:
    report = build_report_payload(_load_result_for_export(meeting_id, current_user))
    return Response(
        content=export_pdf(report),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="pm_insights_{meeting_id}.pdf"'},
    )


def _date_value(item: dict) -> str:
    metadata = item.get("metadata") or {}
    meeting_info = metadata.get("meeting_info") or {}
    value = meeting_info.get("meeting_date") or item.get("meeting_date") or item.get("created_at") or item.get("upload_date") or ""
    return str(value)[:10]


def _project_value(item: dict) -> str:
    metadata = item.get("metadata") or {}
    meeting_info = metadata.get("meeting_info") or {}
    return meeting_info.get("project_name") or item.get("project_name") or item.get("meeting_title") or "Без проекта"


def _title_value(item: dict) -> str:
    metadata = item.get("metadata") or {}
    meeting_info = metadata.get("meeting_info") or {}
    return meeting_info.get("meeting_title") or item.get("meeting_title") or item.get("source_audio") or item.get("meeting_id") or ""


def _meeting_type_value(item: dict) -> str:
    return (item.get("meeting_type") or {}).get("label") or (item.get("analysis_summary") or {}).get("meeting_type") or "unknown"


def _clean_items(item: dict, section_id: str, clean_field: str, raw_field: str | None = None) -> list:
    aliases = {
        "questions_answers": {"questions_answers", "qa"},
        "topics": {"topics", "aspects_topics"},
    }
    section_ids = aliases.get(section_id, {section_id})
    for section in item.get("report_sections") or []:
        if section.get("id") in section_ids and isinstance(section.get("items"), list):
            return section["items"]
    if isinstance(item.get(clean_field), list):
        return item[clean_field]
    if raw_field and isinstance(item.get(raw_field), list):
        return item[raw_field]
    return []


def _dashboard_task_count(item: dict) -> int:
    if item.get("report_sections") or item.get("clean_tasks") is not None:
        count = len(_clean_items(item, "tasks", "clean_tasks"))
        if count:
            return count
        if item.get("tasks") and (item.get("metrics") or {}).get("tasks_count") is not None:
            return int((item.get("metrics") or {}).get("tasks_count") or 0)
        return 0
    return len(item.get("tasks") or [])


def _dashboard_sentiment(item: dict) -> tuple[float, int, int, int]:
    summary = item.get("analysis_summary") or {}
    metrics = item.get("metrics") or {}
    sentiment = item.get("sentiment") or []
    if summary.get("average_sentiment") is not None:
        average = float(summary["average_sentiment"])
    elif metrics.get("average_sentiment") is not None:
        average = float(metrics["average_sentiment"])
    else:
        values = [{"positive": 1.0, "neutral": 0.0, "negative": -1.0}.get(row.get("sentiment"), 0.0) for row in sentiment]
        average = round(sum(values) / len(values), 3) if values else 0.0
    positive = int(metrics.get("positive_fragments_count") or sum(1 for row in sentiment if row.get("sentiment") == "positive"))
    neutral = int(sum(1 for row in sentiment if row.get("sentiment") == "neutral"))
    negative = int(metrics.get("negative_fragments_count") or sum(1 for row in sentiment if row.get("sentiment") == "negative"))
    return round(average, 3), positive, neutral, negative


def _dashboard_aspects(item: dict) -> Counter:
    # `item` is already normalize_analysis_result() output by the time this is called
    # from _compute_dashboard_payload() - re-normalizing here would double the NLP cost.
    stopwords = {"и", "в", "на", "по", "для", "это", "как", "что", "нет", "да", "встреча"}
    counter: Counter = Counter()
    for topic in item.get("clean_topics") or []:
        name = str(topic.get("topic_name") or topic.get("title") or "").strip().lower()
        if name and name not in stopwords:
            counter[name] += int(topic.get("count") or 1)
    for aspect in item.get("clean_aspects") or []:
        name = str(aspect.get("title") or aspect.get("name") or "").strip().lower()
        if name and name not in stopwords:
            counter[name] += int(aspect.get("count") or 1)
    return counter


_DASHBOARD_CACHE: dict[tuple[int | None, str | None], tuple[float, dict]] = {}
DASHBOARD_CACHE_TTL_SECONDS = 60


def _build_dashboard_payload(current_user: dict | None, project: str | None) -> dict:
    user_id = _current_user_id(current_user)
    cache_key = (user_id, project)
    cached = _DASHBOARD_CACHE.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < DASHBOARD_CACHE_TTL_SECONDS:
        return cached[1]

    payload = _compute_dashboard_payload(user_id, project)
    _DASHBOARD_CACHE[cache_key] = (time.monotonic(), payload)
    return payload


def _compute_dashboard_payload(user_id: int | None, project: str | None) -> dict:
    try:
        enriched = db.list_meetings_with_results(user_id=user_id)
    except Exception as exc:
        raise _db_error(exc) from exc

    normalized_results = [
        _get_normalized_result(item.get("meeting_id"), item)
        for item in enriched
        if item.get("metrics") or item.get("report_sections")
    ]
    projects_map: dict[str, dict] = {}
    for item in enriched:
        project_name = _project_value(item)
        projects_map[project_name] = {"project_key": project_name, "project_name": project_name}
    selected_project = project or (sorted(projects_map)[0] if projects_map else None)
    filtered = []
    for item in normalized_results:
        if selected_project and _project_value(item) != selected_project:
            continue
        filtered.append(item)

    processing_times = []
    estimated_1h_times = []
    aspect_counter: Counter = Counter()
    sentiment_trend = []
    task_trend = []
    for item in filtered:
        average, positive, neutral, negative = _dashboard_sentiment(item)
        date_value = _date_value(item)
        sentiment_trend.append(
            {
                "date": date_value,
                "meeting_id": item.get("meeting_id"),
                "meeting_title": _title_value(item),
                "average_sentiment": average,
                "positive_count": positive,
                "neutral_count": neutral,
                "negative_count": negative,
            }
        )
        task_trend.append(
            {
                "date": date_value,
                "meeting_id": item.get("meeting_id"),
                "meeting_title": _title_value(item),
                "tasks_count": _dashboard_task_count(item),
            }
        )
        aspect_counter.update(_dashboard_aspects(item))
        metrics = item.get("metrics", {})
        processing_time = item.get("metadata", {}).get("processing_time", {}) or metrics.get("processing_time", {})
        if processing_time.get("total_processing_seconds") is not None:
            processing_times.append(float(processing_time["total_processing_seconds"]))
        if processing_time.get("estimated_1h_processing_minutes") is not None:
            estimated_1h_times.append(float(processing_time["estimated_1h_processing_minutes"]))

    def _average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    sentiment_trend.sort(key=lambda row: row.get("date") or "")
    task_trend.sort(key=lambda row: row.get("date") or "")
    aspect_word_cloud = [{"text": key, "value": value} for key, value in aspect_counter.most_common(30)]
    total_tasks = sum(row["tasks_count"] for row in task_trend)
    questions_count = sum(len(_clean_items(item, "questions_answers", "clean_questions_answers", "questions_answers")) for item in filtered)
    answers_count = sum(
        1
        for item in filtered
        for qa in _clean_items(item, "questions_answers", "clean_questions_answers", "questions_answers")
        if qa.get("status") in {"answered", "partial"} and (qa.get("answer_summary") or qa.get("answer_full") or qa.get("answer"))
    )
    decisions_count = sum(len(_clean_items(item, "decisions", "clean_decisions", "decisions")) for item in filtered)
    responsibles_count = sum(len(_clean_items(item, "responsibles", "clean_responsibles", "responsibles")) for item in filtered)
    deadlines_count = sum(len(_clean_items(item, "deadlines", "clean_deadlines", "deadlines")) for item in filtered)
    completed_count = sum(1 for item in filtered if item.get("processing_status") == "completed")
    average_sentiment = round(
        sum(row["average_sentiment"] for row in sentiment_trend) / len(sentiment_trend), 3
    ) if sentiment_trend else 0.0
    negative_count = sum(row.get("negative_count", 0) for row in sentiment_trend)
    average_processing = _average(processing_times)
    average_1h = _average(estimated_1h_times)
    summary = {
        "meetings_count": len(filtered),
        "completed_count": completed_count,
        "tasks_count": total_tasks,
        "questions_count": questions_count,
        "answers_count": answers_count,
        "decisions_count": decisions_count,
        "responsibles_count": responsibles_count,
        "deadlines_count": deadlines_count,
        "average_sentiment": average_sentiment,
        "negative_count": negative_count,
        "average_processing_seconds": average_processing,
        "average_estimated_1h_minutes": average_1h,
    }
    return {
        "projects": list(projects_map.values()),
        "selected_project": selected_project,
        "summary": summary,
        "sentiment_trend": sentiment_trend,
        "task_trend": task_trend,
        "aspect_word_cloud": aspect_word_cloud,
        "technical_metrics": {
            "meetings_count": len(filtered),
            "total_tasks": total_tasks,
            "total_questions": questions_count,
            "average_processing_time_seconds": average_processing,
            "average_estimated_1h_processing_minutes": average_1h,
        },
        # Backward-compatible fields used by older tests and archive widgets.
        "meetings_count": len(enriched),
        "completed_count": sum(1 for item in enriched if item.get("processing_status") == "completed"),
        "tasks_count": total_tasks,
        "questions_count": questions_count,
        "answers_count": answers_count,
        "decisions_count": decisions_count,
        "responsibles_count": responsibles_count,
        "deadlines_count": deadlines_count,
        "average_sentiment": average_sentiment,
        "negative_count": negative_count,
        "average_processing_time_seconds": average_processing,
        "average_estimated_1h_processing_minutes": average_1h,
        "meeting_groups": [],
    }


@app.get("/dashboard")
@app.get("/api/dashboard")
def dashboard(
    current_user: dict | None = Depends(auth.maybe_current_user),
    project: str | None = Query(None),
) -> dict:
    return _build_dashboard_payload(current_user, project)


@app.get("/metrics")
@app.get("/api/metrics")
def metrics(
    current_user: dict | None = Depends(auth.maybe_current_user),
    project: str | None = Query(None),
) -> dict:
    return _build_dashboard_payload(current_user, project)

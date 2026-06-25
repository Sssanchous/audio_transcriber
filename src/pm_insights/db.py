from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, Integer, String, Text, create_engine, delete, inspect, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from pm_insights import settings


class DatabaseError(RuntimeError):
    pass


class Base(DeclarativeBase):
    pass


class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    meeting_title: Mapped[str] = mapped_column(String(255), default="")
    meeting_key: Mapped[str] = mapped_column(String(512), default="", index=True)
    project_name: Mapped[str] = mapped_column(String(255), default="PM Insights")
    meeting_date: Mapped[str | None] = mapped_column(String(32), nullable=True)
    participants: Mapped[str] = mapped_column(Text, default="")
    source_audio: Mapped[str] = mapped_column(Text, default="")
    original_filename: Mapped[str] = mapped_column(Text, default="")
    stored_filename: Mapped[str] = mapped_column(Text, default="")
    file_extension: Mapped[str] = mapped_column(String(16), default="")
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    upload_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processing_status: Mapped[str] = mapped_column(String(32), default="uploaded")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Transcript(Base):
    __tablename__ = "transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), index=True)
    text: Mapped[str] = mapped_column(Text, default="")
    segments_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    language: Mapped[str] = mapped_column(String(16), default="ru")
    asr_model: Mapped[str] = mapped_column(String(128), default="")
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), index=True)
    tasks_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    questions_answers_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    decisions_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    deadlines_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    responsibles_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    sentiment_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    aspects_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    topics_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ProcessingLog(Base):
    __tablename__ = "processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    meeting_id: Mapped[str] = mapped_column(String(64), index=True)
    step: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnalysisFeedback(Base):
    __tablename__ = "analysis_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    meeting_id: Mapped[str] = mapped_column(String(64), index=True)
    item_type: Mapped[str] = mapped_column(String(64))
    source_text: Mapped[str] = mapped_column(Text, default="")
    predicted_label: Mapped[str] = mapped_column(String(64), default="")
    corrected_label: Mapped[str] = mapped_column(String(64), default="")
    corrected_text: Mapped[str] = mapped_column(Text, default="")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False)
    training_run_id: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


_engine = None
SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    global _engine, SessionLocal
    if _engine is None:
        url = settings.DATABASE_URL
        if not url:
            raise DatabaseError("DATABASE_URL is not configured. Set it in .env.")
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url.removeprefix("postgresql://")
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)
        SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def init_db() -> None:
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        _ensure_user_columns(engine)
        _ensure_meeting_columns(engine)
        _ensure_feedback_columns(engine)
    except SQLAlchemyError as exc:
        raise DatabaseError(f"Database is unavailable or misconfigured: {exc}") from exc


def _ensure_user_columns(engine) -> None:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return

    columns = {column["name"] for column in inspector.get_columns("users")}
    statements = []
    if "email" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN email VARCHAR(255)")
    if "full_name" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN full_name VARCHAR(255) DEFAULT ''")
    if "is_active" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT TRUE")

    if not statements:
        return

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _ensure_meeting_columns(engine) -> None:
    inspector = inspect(engine)
    if "meetings" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("meetings")}
    statements = []
    if "user_id" not in columns:
        statements.append("ALTER TABLE meetings ADD COLUMN user_id INTEGER")
    if "meeting_title" not in columns:
        statements.append("ALTER TABLE meetings ADD COLUMN meeting_title VARCHAR(255) DEFAULT ''")
    if "meeting_key" not in columns:
        statements.append("ALTER TABLE meetings ADD COLUMN meeting_key VARCHAR(512) DEFAULT ''")
    if "metadata_json" not in columns:
        statements.append("ALTER TABLE meetings ADD COLUMN metadata_json JSON")
    if not statements:
        return
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def _ensure_feedback_columns(engine) -> None:
    inspector = inspect(engine)
    if "analysis_feedback" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("analysis_feedback")}
    statements = []
    if "used_for_training" not in columns:
        statements.append("ALTER TABLE analysis_feedback ADD COLUMN used_for_training BOOLEAN DEFAULT FALSE")
    if "training_run_id" not in columns:
        statements.append("ALTER TABLE analysis_feedback ADD COLUMN training_run_id VARCHAR(128) DEFAULT ''")
    if not statements:
        return
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def normalize_meeting_key(project_name: str, meeting_title: str | None = "") -> str:
    import re

    raw = f"{project_name or 'PM Insights'}::{meeting_title or project_name or 'meeting'}".lower()
    raw = re.sub(r"[^a-zа-яё0-9:]+", "_", raw, flags=re.IGNORECASE)
    raw = re.sub(r"_+", "_", raw).strip("_")
    return raw[:500]


@contextmanager
def session_scope():
    if SessionLocal is None:
        get_engine()
    assert SessionLocal is not None
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def serialize_meeting(meeting: Meeting, result: dict | None = None) -> dict:
    payload = {
        "id": meeting.meeting_id,
        "meeting_id": meeting.meeting_id,
        "user_id": meeting.user_id,
        "meeting_title": meeting.meeting_title,
        "meeting_key": meeting.meeting_key,
        "project_name": meeting.project_name,
        "meeting_date": meeting.meeting_date,
        "participants": meeting.participants,
        "source_audio": meeting.source_audio,
        "original_filename": meeting.original_filename,
        "stored_filename": meeting.stored_filename,
        "file_extension": meeting.file_extension,
        "file_size_bytes": meeting.file_size_bytes,
        "upload_date": _dt(meeting.upload_date),
        "processing_status": meeting.processing_status,
        "metadata": meeting.metadata_json or {},
        "created_at": _dt(meeting.created_at),
        "updated_at": _dt(meeting.updated_at),
    }
    if result:
        payload.update(result)
    return payload


def create_meeting(
    *,
    meeting_id: str,
    meeting_title: str = "",
    project_name: str,
    meeting_date: str | None,
    participants: str,
    metadata: dict[str, Any] | None = None,
    source_audio: str,
    original_filename: str,
    stored_filename: str,
    file_extension: str,
    file_size_bytes: int,
    user_id: int | None = None,
) -> dict:
    init_db()
    with session_scope() as session:
        meeting = Meeting(
            meeting_id=meeting_id,
            user_id=user_id,
            meeting_title=meeting_title,
            meeting_key=normalize_meeting_key(project_name, meeting_title),
            project_name=project_name,
            meeting_date=meeting_date,
            participants=participants,
            source_audio=source_audio,
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_extension=file_extension,
            file_size_bytes=file_size_bytes,
            metadata_json=metadata or {},
            processing_status="uploaded",
        )
        session.add(meeting)
        session.flush()
        return serialize_meeting(meeting)


def update_meeting_status(meeting_id: str, status: str) -> None:
    init_db()
    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.meeting_id == meeting_id))
        if meeting:
            meeting.processing_status = status
            meeting.updated_at = datetime.now(timezone.utc)


def update_meeting_metadata(meeting_id: str, metadata: dict[str, Any]) -> None:
    init_db()
    with session_scope() as session:
        meeting = session.scalar(select(Meeting).where(Meeting.meeting_id == meeting_id))
        if meeting:
            current = dict(meeting.metadata_json or {})
            current.update(metadata)
            meeting.metadata_json = current
            meeting.updated_at = datetime.now(timezone.utc)


def get_meeting(meeting_id: str, user_id: int | None = None) -> dict | None:
    init_db()
    with session_scope() as session:
        query = select(Meeting).where(Meeting.meeting_id == meeting_id)
        if user_id is not None:
            query = query.where(Meeting.user_id == user_id)
        meeting = session.scalar(query)
        if not meeting:
            return None
        analysis = session.scalar(
            select(AnalysisResult).where(AnalysisResult.meeting_id == meeting_id).order_by(AnalysisResult.id.desc())
        )
        return serialize_meeting(meeting, analysis.result_json if analysis else None)


def list_meetings(user_id: int | None = None) -> list[dict]:
    init_db()
    with session_scope() as session:
        query = select(Meeting).order_by(Meeting.created_at.desc())
        if user_id is not None:
            query = query.where(Meeting.user_id == user_id)
        rows = session.scalars(query).all()
        return [serialize_meeting(row) for row in rows]


def list_meetings_with_results(user_id: int | None = None) -> list[dict]:
    """Like list_meetings(), but also batches in the latest analysis result per
    meeting using a single extra query instead of one get_meeting() call per row."""
    init_db()
    with session_scope() as session:
        query = select(Meeting).order_by(Meeting.created_at.desc())
        if user_id is not None:
            query = query.where(Meeting.user_id == user_id)
        meetings = session.scalars(query).all()
        if not meetings:
            return []

        meeting_ids = [meeting.meeting_id for meeting in meetings]
        result_rows = session.scalars(
            select(AnalysisResult)
            .where(AnalysisResult.meeting_id.in_(meeting_ids))
            .order_by(AnalysisResult.meeting_id, AnalysisResult.id.desc())
        ).all()
        latest_result_by_meeting: dict[str, dict] = {}
        for row in result_rows:
            latest_result_by_meeting.setdefault(row.meeting_id, row.result_json)

        return [
            serialize_meeting(meeting, latest_result_by_meeting.get(meeting.meeting_id))
            for meeting in meetings
        ]


def delete_meeting(meeting_id: str, user_id: int | None = None) -> bool:
    init_db()
    with session_scope() as session:
        query = select(Meeting).where(Meeting.meeting_id == meeting_id)
        if user_id is not None:
            query = query.where(Meeting.user_id == user_id)
        meeting = session.scalar(query)
        if not meeting:
            return False

        session.execute(delete(AnalysisFeedback).where(AnalysisFeedback.meeting_id == meeting_id))
        session.execute(delete(AnalysisResult).where(AnalysisResult.meeting_id == meeting_id))
        session.execute(delete(Transcript).where(Transcript.meeting_id == meeting_id))
        session.execute(delete(ProcessingLog).where(ProcessingLog.meeting_id == meeting_id))
        session.delete(meeting)
        return True


def save_transcript(meeting_id: str, transcription: dict) -> None:
    init_db()
    with session_scope() as session:
        session.add(
            Transcript(
                meeting_id=meeting_id,
                text=transcription.get("text", ""),
                segments_json=transcription.get("segments", []),
                language=transcription.get("language", "ru"),
                asr_model=transcription.get("model", ""),
                duration_seconds=float(transcription.get("duration", 0.0) or 0.0),
            )
        )


def save_analysis_result(meeting_id: str, result: dict) -> None:
    init_db()
    with session_scope() as session:
        session.add(
            AnalysisResult(
                meeting_id=meeting_id,
                tasks_json=result.get("tasks", []),
                questions_answers_json=result.get("questions_answers", []),
                decisions_json=result.get("decisions", []),
                deadlines_json=result.get("deadlines", []),
                responsibles_json=result.get("responsibles", []),
                sentiment_json=result.get("sentiment", []),
                aspects_json=result.get("aspects", []),
                topics_json=result.get("topics", []),
                metrics_json=result.get("metrics", {}),
                result_json=result,
            )
        )


def get_result(meeting_id: str, user_id: int | None = None) -> dict | None:
    meeting = get_meeting(meeting_id, user_id=user_id)
    if not meeting:
        return None
    result = dict(meeting)
    metadata = dict(result.get("metadata") or {})
    metadata.setdefault(
        "meeting_info",
        {
            "meeting_title": meeting.get("meeting_title", ""),
            "meeting_date": meeting.get("meeting_date"),
            "project_name": meeting.get("project_name"),
            "participants": metadata.get("participants", []),
            "meeting_key": meeting.get("meeting_key", ""),
        },
    )
    result["metadata"] = metadata
    return result


def create_feedback(
    *,
    user_id: int,
    meeting_id: str,
    item_type: str,
    source_text: str,
    predicted_label: str,
    corrected_label: str,
    corrected_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict:
    init_db()
    with session_scope() as session:
        feedback = AnalysisFeedback(
            user_id=user_id,
            meeting_id=meeting_id,
            item_type=item_type,
            source_text=source_text,
            predicted_label=predicted_label,
            corrected_label=corrected_label,
            corrected_text=corrected_text,
            metadata_json=metadata or {},
        )
        session.add(feedback)
        session.flush()
        return serialize_feedback(feedback)


def serialize_feedback(feedback: AnalysisFeedback) -> dict:
    return {
        "id": feedback.id,
        "user_id": feedback.user_id,
        "meeting_id": feedback.meeting_id,
        "item_type": feedback.item_type,
        "source_text": feedback.source_text,
        "predicted_label": feedback.predicted_label,
        "corrected_label": feedback.corrected_label,
        "corrected_text": feedback.corrected_text,
        "metadata": feedback.metadata_json or {},
        "used_for_training": feedback.used_for_training,
        "training_run_id": feedback.training_run_id,
        "created_at": _dt(feedback.created_at),
    }


def list_feedback(meeting_id: str | None = None, user_id: int | None = None, only_unused: bool = False) -> list[dict]:
    init_db()
    with session_scope() as session:
        query = select(AnalysisFeedback).order_by(AnalysisFeedback.created_at.desc())
        if meeting_id is not None:
            query = query.where(AnalysisFeedback.meeting_id == meeting_id)
        if user_id is not None:
            query = query.where(AnalysisFeedback.user_id == user_id)
        if only_unused:
            query = query.where(AnalysisFeedback.used_for_training.is_(False))
        return [serialize_feedback(row) for row in session.scalars(query).all()]


def log_processing(meeting_id: str, step: str, status: str, message: str = "") -> None:
    try:
        init_db()
        with session_scope() as session:
            session.add(ProcessingLog(meeting_id=meeting_id, step=step, status=status, message=message[:1000]))
    except Exception:
        pass


def get_last_processing_error(meeting_id: str) -> str | None:
    init_db()
    with session_scope() as session:
        log = session.scalar(
            select(ProcessingLog)
            .where(ProcessingLog.meeting_id == meeting_id, ProcessingLog.status.in_(["failed", "failure"]))
            .order_by(ProcessingLog.id.desc())
        )
        return log.message if log and log.message else None


def serialize_user(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name,
        "is_active": user.is_active,
        "created_at": _dt(user.created_at),
    }


def get_user_by_email(email: str) -> dict | None:
    init_db()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email.lower().strip()))
        return serialize_user(user) if user else None


def get_user_by_username(username: str) -> dict | None:
    init_db()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username.strip()))
        return serialize_user(user) if user else None


def get_user_by_id(user_id: int) -> dict | None:
    init_db()
    with session_scope() as session:
        user = session.scalar(select(User).where(User.id == user_id))
        return serialize_user(user) if user else None


def get_user_auth_record(login: str) -> dict | None:
    init_db()
    login_clean = login.strip()
    with session_scope() as session:
        user = session.scalar(
            select(User).where((User.email == login_clean.lower()) | (User.username == login_clean))
        )
        if not user:
            return None
        payload = serialize_user(user)
        payload["password_hash"] = user.password_hash
        return payload


def create_user(*, email: str, username: str, password_hash: str, full_name: str = "") -> dict:
    init_db()
    with session_scope() as session:
        user = User(
            email=email.lower().strip(),
            username=username.strip(),
            full_name=full_name.strip(),
            password_hash=password_hash,
            is_active=True,
        )
        session.add(user)
        try:
            session.flush()
        except IntegrityError as exc:
            raise DatabaseError("User with this email or username already exists.") from exc
        return serialize_user(user)

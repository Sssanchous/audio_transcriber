from fastapi.testclient import TestClient

from pm_insights import db, settings
from pm_insights.api.main import app


def _use_sqlite_auth_db(tmp_path, monkeypatch):
    from pm_insights.api import main as api_main

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'auth.db'}")
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "test-secret")
    monkeypatch.setattr(settings, "REQUIRE_AUTH", True)
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    monkeypatch.setattr(settings, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(db, "_engine", None)
    monkeypatch.setattr(db, "SessionLocal", None)
    monkeypatch.setattr(api_main, "_DASHBOARD_CACHE", {})
    db.init_db()


def test_auth_register_login_and_me(tmp_path, monkeypatch):
    _use_sqlite_auth_db(tmp_path, monkeypatch)
    client = TestClient(app)

    register = client.post(
        "/api/auth/register",
        json={"email": "user@example.com", "username": "demo", "password": "password123"},
    )
    assert register.status_code == 200
    payload = register.json()
    assert payload["token_type"] == "bearer"
    assert payload["access_token"]
    assert payload["user"]["email"] == "user@example.com"
    assert "password" not in payload["user"]

    duplicate = client.post(
        "/api/auth/register",
        json={"email": "user@example.com", "username": "demo2", "password": "password123"},
    )
    assert duplicate.status_code == 400

    login = client.post("/api/auth/login", json={"email": "user@example.com", "password": "password123"})
    assert login.status_code == 200
    token = login.json()["access_token"]

    wrong_password = client.post("/api/auth/login", json={"email": "user@example.com", "password": "badpass"})
    assert wrong_password.status_code == 401

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["username"] == "demo"

    no_token = client.get("/api/auth/me")
    assert no_token.status_code == 401


def _register(client: TestClient, email: str, username: str) -> str:
    response = client.post(
        "/api/auth/register",
        json={"email": email, "username": username, "password": "password123"},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_upload_and_history_are_user_scoped(tmp_path, monkeypatch):
    _use_sqlite_auth_db(tmp_path, monkeypatch)
    client = TestClient(app)
    token_a = _register(client, "a@example.com", "user_a")
    token_b = _register(client, "b@example.com", "user_b")

    unauthorized = client.post(
        "/api/upload",
        files={"file": ("meeting.mp3", b"audio-bytes", "audio/mpeg")},
    )
    assert unauthorized.status_code == 401

    upload_a = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {token_a}"},
        data={"meeting_title": "A meeting", "project_name": "A", "participants": "Анна — исполнитель"},
        files={"file": ("a.mp3", b"audio-bytes-a", "audio/mpeg")},
    )
    assert upload_a.status_code == 200
    meeting_a = upload_a.json()["meeting_id"]

    upload_b = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {token_b}"},
        data={"meeting_title": "B meeting", "project_name": "B", "participants": "Иван — исполнитель"},
        files={"file": ("b.mp3", b"audio-bytes-b", "audio/mpeg")},
    )
    assert upload_b.status_code == 200
    meeting_b = upload_b.json()["meeting_id"]

    meetings_a = client.get("/api/meetings", headers={"Authorization": f"Bearer {token_a}"}).json()
    meetings_b = client.get("/api/meetings", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert [item["meeting_id"] for item in meetings_a] == [meeting_a]
    assert [item["meeting_id"] for item in meetings_b] == [meeting_b]
    assert meetings_a[0]["user_id"] != meetings_b[0]["user_id"]

    db.save_analysis_result(meeting_a, {"meeting_id": meeting_a, "tasks": [{"text": "Анна, подготовь отчет"}], "metrics": {"tasks_count": 1}})
    db.save_analysis_result(meeting_b, {"meeting_id": meeting_b, "tasks": [], "metrics": {"tasks_count": 0}})

    own_result = client.get(f"/api/meetings/{meeting_a}/result", headers={"Authorization": f"Bearer {token_a}"})
    foreign_result = client.get(f"/api/meetings/{meeting_b}/result", headers={"Authorization": f"Bearer {token_a}"})
    assert own_result.status_code == 200
    assert foreign_result.status_code == 404

    dashboard_a = client.get("/api/dashboard", headers={"Authorization": f"Bearer {token_a}"}).json()
    dashboard_b = client.get("/api/dashboard", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert dashboard_a["meetings_count"] == 1
    assert dashboard_b["meetings_count"] == 1
    assert dashboard_a["tasks_count"] == 1
    assert dashboard_b["tasks_count"] == 0


def test_feedback_is_user_scoped(tmp_path, monkeypatch):
    _use_sqlite_auth_db(tmp_path, monkeypatch)
    client = TestClient(app)
    token_a = _register(client, "feedback-a@example.com", "feedback_a")
    token_b = _register(client, "feedback-b@example.com", "feedback_b")

    upload = client.post(
        "/api/upload",
        headers={"Authorization": f"Bearer {token_a}"},
        data={"meeting_title": "Feedback meeting", "project_name": "Demo", "participants": "Анна — исполнитель"},
        files={"file": ("meeting.mp3", b"audio-bytes", "audio/mpeg")},
    )
    assert upload.status_code == 200
    meeting_id = upload.json()["meeting_id"]

    feedback = client.post(
        f"/api/meetings/{meeting_id}/feedback",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "item_type": "task",
            "source_text": "надо сделать развертку",
            "predicted_label": "task",
            "corrected_label": "task",
            "corrected_text": "Сделать развертку",
        },
    )
    assert feedback.status_code == 200
    own = client.get(f"/api/meetings/{meeting_id}/feedback", headers={"Authorization": f"Bearer {token_a}"})
    foreign = client.get(f"/api/meetings/{meeting_id}/feedback", headers={"Authorization": f"Bearer {token_b}"})
    assert len(own.json()) == 1
    assert foreign.status_code == 404

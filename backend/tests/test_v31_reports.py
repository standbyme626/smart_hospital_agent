import asyncio
import io
import json
import zipfile
from pathlib import Path

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_gateway import ChatResult

PATIENT_PASSWORD = "patient123456"
ADMIN_PASSWORD = "admin123456"
USER_PASSWORD = "secret123"
USER_A = "user_a"
USER_B = "user_b"


class FakeGateway:
    def __init__(self) -> None:
        self.fail = False
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> ChatResult:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("vision model unavailable")
        return ChatResult(
            content=json.dumps(
                {
                    "summary": "未见明显异常，建议结合临床。",
                    "abnormal_values": [],
                    "questions": ["需要复查吗？"],
                },
                ensure_ascii=False,
            ),
            provider="fake",
            model="fake-vision",
        )


@pytest.fixture()
def client(tmp_path):
    with TestClient(app) as test_client:
        state = test_client.app.state
        state.settings.AUTH_ENABLED = True
        original_uploads = state.settings.UPLOADS_DIR
        state.settings.UPLOADS_DIR = str(tmp_path / "uploads")
        state.gateway = FakeGateway()
        try:
            yield test_client
        finally:
            state.settings.AUTH_ENABLED = False
            state.settings.UPLOADS_DIR = original_uploads


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _admin(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


def _patient(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "patient", PATIENT_PASSWORD))


def _create_user(client: TestClient, username: str, password: str = USER_PASSWORD) -> dict[str, str]:
    response = client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": password, "role": "patient"},
        headers=_admin(client),
    )
    assert response.status_code == 201, response.text
    return _auth(_token(client, username, password))


def _dsn(client: TestClient) -> str:
    return client.app.state.settings.DATABASE_URL


def _sql(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.execute(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _fetch(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.fetch(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _fetchval(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.fetchval(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _uploads(client: TestClient) -> Path:
    return Path(client.app.state.settings.UPLOADS_DIR)


def _add_report(
    client: TestClient,
    report_id: str,
    user_id: str,
    data: bytes = b"\xff\xd8\xff\xe0fake-jpeg",
    suffix: str = ".jpg",
) -> Path:
    path = _uploads(client) / "reports" / f"{report_id}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _sql(
        client,
        "INSERT INTO reports (id, user_id, title, image_path, summary) VALUES ($1,$2,$3,$4,$5)",
        report_id,
        user_id,
        "血常规报告",
        str(path),
        "摘要",
    )
    return path


def _add_health_event(client: TestClient, user_id: str, report_id: str) -> None:
    _sql(
        client,
        """
        INSERT INTO health_events (user_id, kind, title, detail)
        VALUES ($1, 'report', $2, $3::jsonb)
        """,
        user_id,
        "报告解读",
        json.dumps({"report_id": report_id}),
    )


def test_report_image_owner_only_and_no_path_leak(client):
    patient = _patient(client)
    other = _create_user(client, USER_B)
    image_bytes = b"\xff\xd8\xff\xe0owner-only"
    _add_report(client, "rep_owned", "patient", data=image_bytes)

    owned = client.get("/api/v1/reports/rep_owned/image", headers=patient)
    assert owned.status_code == 200, owned.text
    assert owned.content == image_bytes
    assert owned.headers["content-type"] == "image/jpeg"
    assert "private" in owned.headers["cache-control"]

    assert client.get("/api/v1/reports/rep_owned/image", headers=other).status_code == 404
    assert client.get("/api/v1/reports/rep_owned", headers=other).status_code == 404

    listing = client.get("/api/v1/reports", headers=patient)
    assert listing.status_code == 200, listing.text
    item = next(entry for entry in listing.json()["reports"] if entry["id"] == "rep_owned")
    assert item["has_image"] is True
    assert "image_path" not in item

    detail = client.get("/api/v1/reports/rep_owned", headers=patient)
    assert detail.status_code == 200
    assert detail.json()["report"]["has_image"] is True
    assert "image_path" not in detail.json()["report"]


def test_report_image_rejects_traversal_and_missing_file(client):
    patient = _patient(client)
    outside = Path(client.app.state.settings.UPLOADS_DIR).parent / "outside.jpg"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"\xff\xd8outside")
    _sql(
        client,
        "INSERT INTO reports (id, user_id, title, image_path) VALUES ($1,$2,$3,$4)",
        "rep_traversal",
        "patient",
        "越界图片",
        str(outside),
    )
    _sql(
        client,
        "INSERT INTO reports (id, user_id, title, image_path) VALUES ($1,$2,$3,$4)",
        "rep_missing",
        "patient",
        "丢失图片",
        str(_uploads(client) / "reports" / "does_not_exist.jpg"),
    )

    assert client.get("/api/v1/reports/rep_traversal/image", headers=patient).status_code == 404
    missing = client.get("/api/v1/reports/rep_missing/image", headers=patient)
    assert missing.status_code == 404

    detail = client.get("/api/v1/reports/rep_missing", headers=patient)
    assert detail.status_code == 200
    assert detail.json()["report"]["has_image"] is False
    assert outside.exists()


def test_delete_report_removes_row_file_and_event(client):
    patient = _patient(client)
    path = _add_report(client, "rep_delete", "patient")
    _add_health_event(client, "patient", "rep_delete")
    # an unrelated report event must survive
    _sql(
        client,
        """
        INSERT INTO health_events (user_id, kind, title, detail)
        VALUES ($1, 'report', $2, $3::jsonb)
        """,
        "patient",
        "其他报告",
        json.dumps({"report_id": "rep_other"}),
    )

    deleted = client.delete("/api/v1/reports/rep_delete", headers=patient)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"ok": True}
    assert not path.exists()
    assert _fetchval(client, "SELECT count(*) FROM reports WHERE id='rep_delete'") == 0
    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM health_events WHERE kind='report' AND detail->>'report_id'='rep_delete'",
        )
        == 0
    )
    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM health_events WHERE kind='report' AND detail->>'report_id'='rep_other'",
        )
        == 1
    )
    assert _fetchval(client, "SELECT count(*) FROM audit_log WHERE action='report.delete'") == 1
    assert client.get("/api/v1/reports/rep_delete", headers=patient).status_code == 404
    assert client.get("/api/v1/reports/rep_delete/image", headers=patient).status_code == 404


def test_upload_failure_leaves_no_row_or_orphan_file(client):
    patient = _patient(client)
    client.app.state.gateway.fail = True

    response = client.post(
        "/api/v1/reports/interpret",
        files={"file": ("report.jpg", b"\xff\xd8broken", "image/jpeg")},
        headers=patient,
    )
    assert response.status_code == 502, response.text
    assert _fetchval(client, "SELECT count(*) FROM reports WHERE user_id='patient'") == 0
    report_dir = _uploads(client) / "reports"
    assert not report_dir.exists() or list(report_dir.iterdir()) == []


def test_interpret_success_returns_has_image(client):
    patient = _patient(client)
    response = client.post(
        "/api/v1/reports/interpret",
        files={"file": ("report.jpg", b"\xff\xd8good", "image/jpeg")},
        data={"title": "血常规"},
        headers=patient,
    )
    assert response.status_code == 200, response.text
    report = response.json()["report"]
    assert report["has_image"] is True
    assert "image_path" not in report
    assert (Path(client.app.state.settings.UPLOADS_DIR) / "reports" / f"{report['id']}.jpg").is_file()
    image = client.get(f"/api/v1/reports/{report['id']}/image", headers=patient)
    assert image.status_code == 200
    assert image.content == b"\xff\xd8good"


def test_export_zip_contains_only_own_data(client):
    headers_a = _create_user(client, USER_A)
    _create_user(client, USER_B)
    _add_report(client, "rep_a", USER_A, data=b"\xff\xd8alpha")
    _add_report(client, "rep_b", USER_B, data=b"\xff\xd8beta")
    _sql(
        client,
        "INSERT INTO family_members (id, user_id, name) VALUES ($1,$2,$3)",
        "mem_a",
        USER_A,
        "甲的家人",
    )
    _sql(
        client,
        "INSERT INTO family_members (id, user_id, name) VALUES ($1,$2,$3)",
        "mem_b",
        USER_B,
        "乙的家人",
    )
    _sql(
        client,
        "INSERT INTO sessions (id, user_id, title) VALUES ($1,$2,$3)",
        "sess_a",
        USER_A,
        "甲的会话",
    )
    _sql(
        client,
        "INSERT INTO messages (session_id, user_id, role, content) VALUES ($1,$2,$3,$4)",
        "sess_a",
        USER_A,
        "user",
        "甲的问题",
    )

    response = client.get("/api/v1/me/export", headers=headers_a)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = set(archive.namelist())
    assert "data.json" in names
    assert "reports/rep_a.jpg" in names
    assert "reports/rep_b.jpg" not in names
    assert archive.read("reports/rep_a.jpg") == b"\xff\xd8alpha"

    data = json.loads(archive.read("data.json"))
    assert data["user"]["id"] == USER_A
    assert "password_hash" not in data["user"]
    assert [item["id"] for item in data["reports"]] == ["rep_a"]
    assert all("image_path" not in item for item in data["reports"])
    assert [item["id"] for item in data["family"]] == ["mem_a"]
    assert [item["id"] for item in data["sessions"]] == ["sess_a"]
    assert [item["content"] for item in data["messages"]] == ["甲的问题"]


def test_delete_account_removes_data_and_invalidates_token(client):
    headers = _create_user(client, USER_A)
    auth_token = headers["Authorization"].split(" ", 1)[1]
    path = _add_report(client, "rep_del", USER_A)
    _add_health_event(client, USER_A, "rep_del")
    _sql(
        client,
        "INSERT INTO family_members (id, user_id, name) VALUES ($1,$2,$3)",
        "mem_del",
        USER_A,
        "家人",
    )
    _sql(
        client,
        "INSERT INTO medications (id, user_id, name) VALUES ($1,$2,$3)",
        "med_del",
        USER_A,
        "药",
    )
    _sql(
        client,
        "INSERT INTO followups (id, user_id, title) VALUES ($1,$2,$3)",
        "fu_del",
        USER_A,
        "随访",
    )
    _sql(
        client,
        "INSERT INTO bookings (id, patient_id, user_id, department) VALUES ($1,$2,$3,$4)",
        "bk_del",
        USER_A,
        USER_A,
        "内科",
    )
    _sql(
        client,
        "INSERT INTO sessions (id, user_id, title) VALUES ($1,$2,$3)",
        "sess_del",
        USER_A,
        "会话",
    )
    _sql(
        client,
        "INSERT INTO events (session_id, event_id, type) VALUES ($1,$2,$3)",
        "sess_del",
        "evt_del",
        "final",
    )
    _sql(
        client,
        "INSERT INTO messages (session_id, user_id, role, content) VALUES ($1,$2,$3,$4)",
        "sess_del",
        USER_A,
        "user",
        "内容",
    )
    _sql(
        client,
        "INSERT INTO review_requests (id, kind, ref_id, user_id) VALUES ($1,$2,$3,$4)",
        "rev_del",
        "payment",
        "bk_del",
        USER_A,
    )
    _sql(
        client,
        "INSERT INTO analytics_events (user_id, name) VALUES ($1,$2)",
        USER_A,
        "report_view",
    )
    _sql(
        client,
        "INSERT INTO audit_log (actor, action) VALUES ($1,$2)",
        USER_A,
        "report.delete",
    )
    _sql(
        client,
        "INSERT INTO model_calls (session_id, provider, model) VALUES ($1,$2,$3)",
        "sess_del",
        "fake-reports",
        "fake-reports",
    )

    wrong = client.request("DELETE", "/api/v1/me", json={"password": "nope"}, headers=headers)
    assert wrong.status_code == 403
    assert _fetchval(client, "SELECT count(*) FROM users WHERE id=$1", USER_A) == 1

    deleted = client.request(
        "DELETE",
        "/api/v1/me",
        json={"password": USER_PASSWORD},
        headers=headers,
    )
    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"ok": True}

    assert client.get("/api/v1/auth/me", headers=_auth(auth_token)).status_code == 401
    assert _fetchval(client, "SELECT count(*) FROM users WHERE id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM reports WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM family_members WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM medications WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM followups WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM health_events WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM bookings WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM sessions WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM messages WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM review_requests WHERE user_id=$1", USER_A) == 0
    assert _fetchval(client, "SELECT count(*) FROM events WHERE session_id='sess_del'") == 0
    assert not path.exists()

    anonymized_actor = _fetchval(client, "SELECT actor FROM audit_log WHERE action='report.delete'")
    assert anonymized_actor.startswith("deleted-") and USER_A not in anonymized_actor
    anonymized_analytics = _fetchval(client, "SELECT user_id FROM analytics_events WHERE name='report_view'")
    assert anonymized_analytics.startswith("deleted-") and USER_A not in anonymized_analytics
    anonymized_model = _fetchval(client, "SELECT session_id FROM model_calls WHERE provider='fake-reports'")
    assert anonymized_model.startswith("deleted-") and USER_A not in anonymized_model
    assert (
        _fetchval(client, "SELECT count(*) FROM audit_log WHERE action='account.delete' AND actor='anonymous'")
        == 1
    )

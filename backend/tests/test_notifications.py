from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

PATIENT_PASSWORD = "patient123456"


@pytest.fixture()
def client():
    # 用上下文管理器触发 lifespan：初始化 schema 与 app.state
    with TestClient(app) as test_client:
        yield test_client


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_followup_creates_notification_and_mark_read(client) -> None:
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))

    created = client.post(
        "/api/v1/followups",
        headers=headers,
        json={"title": "两周后复诊心内科", "due_date": None, "notes": "带化验单"},
    )
    assert created.status_code == 200, created.text

    listed = client.get("/api/v1/notifications", headers=headers)
    assert listed.status_code == 200, listed.text
    payload = listed.json()
    assert payload["unread"] >= 1
    assert any(item["kind"] == "followup" for item in payload["notifications"])

    marked = client.post("/api/v1/notifications/read-all", headers=headers)
    assert marked.status_code == 200, marked.text

    again = client.get("/api/v1/notifications", headers=headers).json()
    assert again["unread"] == 0
    assert any(item["read_at"] for item in again["notifications"])

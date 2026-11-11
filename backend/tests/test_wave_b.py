from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

PATIENT_PASSWORD = "patient123456"


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_feedback_create_and_list(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))

    bad = client.post("/api/v1/feedback", headers=headers, json={"category": "无关分类", "content": "x"})
    assert bad.status_code == 422

    created = client.post(
        "/api/v1/feedback",
        headers=headers,
        json={"category": "排队等候", "content": "候诊时间过长"},
    )
    assert created.status_code == 200, created.text
    assert created.json()["ticket"]["status"] == "submitted"

    listed = client.get("/api/v1/feedback", headers=headers)
    assert listed.status_code == 200
    assert any(item["category"] == "排队等候" for item in listed.json()["items"])


def test_consent_scope_upsert_and_list(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))

    member = client.post("/api/v1/family", headers=headers, json={"name": "授权测试", "relation": "父母"})
    assert member.status_code == 200, member.text
    member_id = member.json()["member"]["id"]

    saved = client.put(
        "/api/v1/consent/scopes",
        headers=headers,
        json={"member_id": member_id, "can_view": True, "can_act": False},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["scope"]["member_id"] == member_id

    updated = client.put(
        "/api/v1/consent/scopes",
        headers=headers,
        json={"member_id": member_id, "can_view": True, "can_act": True},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["scope"]["can_act"] is True

    listed = client.get("/api/v1/consent/scopes", headers=headers)
    assert listed.status_code == 200
    assert any(item["member_id"] == member_id for item in listed.json()["items"])


def test_patient_side_lists_return_shapes(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))

    for path in ("/api/v1/dispensing", "/api/v1/insurance/settlements", "/api/v1/queue/today"):
        response = client.get(path, headers=headers)
        assert response.status_code == 200, (path, response.text)

    queue = client.get("/api/v1/queue/today", headers=headers).json()
    assert "mine" in queue and "waiting_ahead" in queue

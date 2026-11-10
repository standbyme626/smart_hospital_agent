from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
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


def test_patient_cannot_read_clinician_queue(client) -> None:
    # 默认测试环境关闭鉴权（role=admin），这个用例要验证真实角色门禁
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    response = client.get("/api/v1/clinician/queue", headers=headers)
    assert response.status_code == 403


def test_admin_can_read_queue_and_patient(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))

    queue = client.get("/api/v1/clinician/queue", headers=admin)
    assert queue.status_code == 200, queue.text
    assert "items" in queue.json()

    patient = _token(client, "patient", PATIENT_PASSWORD)
    member = client.post(
        "/api/v1/family",
        headers=_auth(patient),
        json={"name": "王小明", "relation": "子女"},
    )
    assert member.status_code == 200, member.text
    member_id = member.json()["member"]["id"]

    detail = client.get(f"/api/v1/clinician/patients/{member_id}", headers=admin)
    assert detail.status_code == 200, detail.text
    payload = detail.json()
    assert payload["member"]["id"] == member_id
    assert isinstance(payload["reports"], list)
    assert isinstance(payload["medications"], list)

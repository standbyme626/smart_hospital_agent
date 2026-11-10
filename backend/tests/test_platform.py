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


def test_platform_requires_admin(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    response = client.get("/api/v1/admin/platform", headers=headers)
    assert response.status_code == 403


def test_platform_overview_for_admin(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))

    response = client.get("/api/v1/admin/platform", headers=admin)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["app"]["version"]
    assert payload["app"]["schema_version"]
    assert "auth_enabled" in payload["flags"]
    assert "clinician_write_enabled" in payload["flags"]
    for key in ("users", "sessions", "reports", "followups", "notifications", "documents"):
        assert key in payload["counts"]
    assert isinstance(payload["capabilities"], list)
    assert any(item["owner"] == "HIS" for item in payload["capabilities"])

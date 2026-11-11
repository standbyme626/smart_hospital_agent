from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"

PATHS = [
    "/api/v1/ops/reviews",
    "/api/v1/ops/audit",
    "/api/v1/ops/analytics",
    "/api/v1/ops/insurance-recon",
    "/api/v1/ops/crisis",
    "/api/v1/ops/notifications",
]


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


def test_ops_wave_d_lists(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))
    for path in PATHS:
        response = client.get(path, headers=admin)
        assert response.status_code == 200, (path, response.text)

    analytics = client.get("/api/v1/ops/analytics", headers=admin).json()
    assert {"sessions", "reports", "bookings", "users"} <= set(analytics["totals"])

    insurance = client.get("/api/v1/ops/insurance-recon", headers=admin).json()
    assert isinstance(insurance["items"], list)
    assert "diff_count" in insurance


def test_ops_wave_d_actions_missing(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))

    decide = client.post("/api/v1/ops/reviews/not-exists/decide", headers=admin, json={"action": "approve"})
    assert decide.status_code == 404

    bad = client.post("/api/v1/ops/reviews/not-exists/decide", headers=admin, json={"action": "unknown"})
    assert bad.status_code == 422

    retry = client.post("/api/v1/ops/notifications/not-exists/retry", headers=admin)
    assert retry.status_code == 404

    resolve = client.post("/api/v1/ops/crisis/not-exists/resolve", headers=admin)
    assert resolve.status_code == 404


def test_ops_wave_d_requires_ops(client) -> None:
    get_settings().AUTH_ENABLED = True
    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
    for path in PATHS:
        assert client.get(path, headers=patient).status_code == 403, path

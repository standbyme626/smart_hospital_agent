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


def test_ops_requires_role(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    response = client.get("/api/v1/ops/summary", headers=headers)
    assert response.status_code == 403


def test_ops_summary_and_lists(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))

    summary = client.get("/api/v1/ops/summary", headers=admin)
    assert summary.status_code == 200, summary.text
    payload = summary.json()
    for key in ("pending_reviews", "open_findings", "handoff_waiting", "paid_today"):
        assert key in payload
        assert isinstance(payload[key], int)

    findings = client.get("/api/v1/ops/reconciliation", headers=admin)
    assert findings.status_code == 200, findings.text
    assert isinstance(findings.json()["items"], list)

    handoff = client.get("/api/v1/ops/handoff", headers=admin)
    assert handoff.status_code == 200, handoff.text
    assert isinstance(handoff.json()["items"], list)

    missing = client.post("/api/v1/ops/handoff/not-exists/claim", headers=admin)
    assert missing.status_code == 404

    missing_finding = client.post("/api/v1/ops/reconciliation/not-exists/resolve", headers=admin)
    assert missing_finding.status_code == 404

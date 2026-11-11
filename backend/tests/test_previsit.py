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


def test_previsit_create_and_list(client) -> None:
    get_settings().AUTH_ENABLED = True
    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))

    created = client.post(
        "/api/v1/previsit",
        headers=patient,
        json={
            "chief_complaint": "胸痛 30 分钟，活动后加重",
            "duration": "30 分钟",
            "medications": "氨氯地平 5mg",
            "allergies": "青霉素",
            "history": "高血压 8 年",
            "questions": "需要做冠脉 CT 吗？",
        },
    )
    assert created.status_code == 200, created.text
    previsit = created.json()["previsit"]
    assert previsit["chief_complaint"].startswith("胸痛")

    listed = client.get("/api/v1/previsit", headers=patient)
    assert listed.status_code == 200, listed.text
    items = listed.json()["items"]
    assert items and items[0]["id"] == previsit["id"]


def test_clinician_sees_previsit(client) -> None:
    get_settings().AUTH_ENABLED = True
    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))

    member = client.post(
        "/api/v1/family",
        headers=patient,
        json={"name": "测试就诊人", "relation": "父母"},
    )
    assert member.status_code == 200, member.text
    member_id = member.json()["member"]["id"]

    created = client.post(
        "/api/v1/previsit",
        headers=patient,
        json={"member_id": member_id, "chief_complaint": "头晕两天"},
    )
    assert created.status_code == 200, created.text

    detail = client.get(f"/api/v1/clinician/patients/{member_id}", headers=admin)
    assert detail.status_code == 200, detail.text
    assert detail.json()["previsit"]["chief_complaint"] == "头晕两天"


def test_doctors_directory(client) -> None:
    get_settings().AUTH_ENABLED = True
    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
    response = client.get("/api/v1/doctors", headers=patient)
    assert response.status_code == 200, response.text
    assert isinstance(response.json()["doctors"], list)

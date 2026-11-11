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


def test_clinician_notes_orders_labs_and_consult(client) -> None:
    get_settings().AUTH_ENABLED = True
    admin = _auth(_token(client, "admin", ADMIN_PASSWORD))
    patient = _auth(_token(client, "patient", PATIENT_PASSWORD))

    member = client.post("/api/v1/family", headers=patient, json={"name": "病历测试", "relation": "父母"})
    assert member.status_code == 200, member.text
    member_id = member.json()["member"]["id"]

    note = client.post(
        "/api/v1/clinician/notes",
        headers=admin,
        json={
            "member_id": member_id,
            "chief_complaint": "胸痛 30 分钟",
            "present_illness": "活动后出现",
            "physical_exam": "BP 152/94",
            "diagnosis": "冠心病待查",
        },
    )
    assert note.status_code == 200, note.text
    note_id = note.json()["note"]["id"]

    listed = client.get(f"/api/v1/clinician/notes?member_id={member_id}", headers=admin)
    assert listed.status_code == 200
    assert any(item["id"] == note_id for item in listed.json()["items"])

    completed = client.post(f"/api/v1/clinician/notes/{note_id}/complete", headers=admin)
    assert completed.status_code == 200, completed.text
    assert completed.json()["note"]["status"] == "completed"

    order = client.post(
        "/api/v1/clinician/orders",
        headers=admin,
        json={"member_id": member_id, "kind": "medication", "name": "氨氯地平", "dose": "5mg", "usage": "qd"},
    )
    assert order.status_code == 200, order.text
    order_id = order.json()["order"]["id"]

    get_settings().CLINICIAN_WRITE_ENABLED = False
    blocked = client.post(f"/api/v1/clinician/orders/{order_id}/sign", headers=admin)
    assert blocked.status_code == 409

    get_settings().CLINICIAN_WRITE_ENABLED = True
    try:
        signed = client.post(f"/api/v1/clinician/orders/{order_id}/sign", headers=admin)
        assert signed.status_code == 200, signed.text
        assert signed.json()["order"]["status"] == "signed"

        reviewed = client.post(
            f"/api/v1/clinician/orders/{order_id}/review",
            headers=admin,
            json={"status": "approved", "note": "无相互作用"},
        )
        assert reviewed.status_code == 200, reviewed.text
        assert reviewed.json()["order"]["review_status"] == "approved"
    finally:
        get_settings().CLINICIAN_WRITE_ENABLED = False

    labs = client.get(f"/api/v1/clinician/labs?member_id={member_id}", headers=admin)
    assert labs.status_code == 200
    assert isinstance(labs.json()["items"], list)

    schedules = client.get("/api/v1/clinician/schedules", headers=admin)
    assert schedules.status_code == 200
    assert isinstance(schedules.json()["items"], list)

    consult = client.post(
        "/api/v1/clinician/consultations",
        headers=admin,
        json={"member_id": member_id, "target_department": "神经内科", "reason": "需排除卒中"},
    )
    assert consult.status_code == 200, consult.text

    consults = client.get(f"/api/v1/clinician/consultations?member_id={member_id}", headers=admin)
    assert consults.status_code == 200
    assert any(item["target_department"] == "神经内科" for item in consults.json()["items"])


def test_clinician_wave_c_requires_role(client) -> None:
    get_settings().AUTH_ENABLED = True
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    for path in ("/api/v1/clinician/schedules", "/api/v1/clinician/notes?member_id=x"):
        response = client.get(path, headers=headers)
        assert response.status_code == 403, path

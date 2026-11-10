from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.main import app

PATIENT_PASSWORD = "patient123456"
ADMIN_PASSWORD = "admin123456"
# OpenHIS V1.3 只允许当日挂号，测试用 Asia/Shanghai 的今天
TODAY = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


class FakeAdapterMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None, str | None]] = []
        self.patient_id = "pat_v2_1"
        self.appointment_id = "appt_v2_1"
        self.encounter_id = "enc_v2_1"
        self.state = "quoted"
        self.payment_status = "unpaid"

    async def call_tool(self, name, arguments=None, idempotency_key=None, *, scope=None):
        args = dict(arguments or {})
        self.calls.append((name, args, idempotency_key, scope))
        if name == "schedule.list":
            data = {
                "departments": [{"org_id": "101", "name": "心血管内科"}],
                "doctors": [{"practitioner_id": "201", "name": "张医生", "org_id": "101"}],
                "services": [
                    {
                        "service_id": "301",
                        "definition_id": "301",
                        "name": "心血管内科普通门诊",
                        "org_id": "101",
                        "price": 50,
                    }
                ],
                "date": args.get("date"),
                "availability_mode": "catalog_only",
                "slots": [],
                "source": "fake",
                "stale": False,
            }
        elif name == "patient.resolve":
            data = {
                "found": False,
                "his_patient_id": None,
                "his_bus_no": None,
                "match_method": None,
                "candidates": [],
                "source": "fake",
                "stale": False,
            }
        elif name == "patient.create":
            data = {
                "his_patient_id": self.patient_id,
                "his_bus_no": "PB0001",
                "created": True,
                "match_method": "created",
            }
        elif name == "appointment.create":
            data = {
                "appointment_id": self.appointment_id,
                "quote_id": "QB0001",
                "amount": 50.0,
                "currency": "CNY",
                "status": "quoted",
                "expires_at": "2031-01-01T00:00:00+00:00",
                "payment_mode": "cashier",
            }
        elif name == "payment.request":
            data = {
                "task_id": "task_v2_1",
                "status": "requested",
                "instructions": "请到收费窗口支付 50.00 元。",
                "sla_due_at": "2031-01-01T00:30:00+00:00",
            }
        elif name == "payment.simulate":
            self.state = "registered"
            self.payment_status = "paid"
            data = {
                "encounter_id": self.encounter_id,
                "payment_status": "paid",
                "amount": 50.0,
                "paid_at": "2031-01-01T00:10:00+00:00",
                "payment_id": "pay_v2_1",
                "invoice_no": "INV0001",
            }
        elif name == "visit.status":
            data = {
                "state": self.state,
                "encounter_id": self.encounter_id if self.state != "quoted" else None,
                "payment_status": self.payment_status,
                "appointment_date": "2031-01-01",
                "doctor": "张医生",
                "department": "心血管内科",
                "guide": "请提前 20 分钟到院。",
                "next_action": "visit" if self.state == "registered" else "payment.request",
                "source": "fake",
                "stale": False,
            }
        elif name == "payment.status":
            data = {
                "payment_status": self.payment_status,
                "amount": 50.0,
                "paid_at": None,
                "payment_id": None,
                "invoice_no": None,
                "source": "fake",
                "stale": False,
            }
        elif name == "prescription.list":
            data = {
                "prescriptions": [
                    {
                        "his_prescription_no": "RX0001",
                        "status": "ACTIVE",
                        "issued_at": "2031-01-01T00:20:00+00:00",
                        "items": [{"name": "阿司匹林", "quantity": 1}],
                    }
                ],
                "source": "fake",
                "stale": False,
            }
        elif name == "appointment.cancel":
            self.state = "refunded" if args.get("refund", True) else "cancelled"
            data = {
                "status": "refunded" if args.get("refund", True) else "cancelled",
                "encounter_id": self.encounter_id,
                "refund_id": "ref_v2_1",
            }
        else:
            data = {}
        envelope = {"ok": True, "data": data, "error": None, "meta": {"source": "fake", "stale": False}}
        return {"is_error": False, "text": json.dumps(envelope), "data": envelope}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        test_client.app.state.settings.AUTH_ENABLED = True
        fake = FakeAdapterMCP()
        test_client.app.state.mcp = fake
        test_client.app.state.booking.mcp = fake
        try:
            yield test_client, fake
        finally:
            test_client.app.state.settings.AUTH_ENABLED = False


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_his_appointment_flow_never_marks_local_paid_without_his(client):
    test_client, fake = client
    token = _token(test_client, "patient", PATIENT_PASSWORD)
    headers = _auth(token)

    family = test_client.get("/api/v1/family", headers=headers)
    assert family.status_code == 200, family.text
    self_member = next(item for item in family.json()["members"] if item["is_self"])
    updated = test_client.patch(
        f"/api/v1/family/{self_member['id']}",
        headers=headers,
        json={
            "name": self_member["name"],
            "relation": "self",
            "is_self": True,
            "id_card": "310101198801019876",
            "birth_date": "1988-01-01",
            "gender": "male",
        },
    )
    assert updated.status_code == 200, updated.text

    catalog = test_client.get("/api/v1/booking/catalog?department=心血管内科", headers=headers)
    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["availability_mode"] == "catalog_only"

    created = test_client.post(
        "/api/v1/booking/appointments",
        headers=headers,
        json={
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": TODAY,
        },
    )
    assert created.status_code == 200, created.text
    appointment = created.json()["appointment"]
    assert appointment["payment_status"] == "unpaid"
    assert appointment["status"] == "quoted"
    appointment_id = appointment["appointment_id"]
    assert appointment_id == fake.appointment_id

    requested = test_client.post(
        f"/api/v1/booking/appointments/{appointment_id}/request-payment", headers=headers
    )
    assert requested.status_code == 200, requested.text
    assert requested.json()["task"]["status"] == "requested"
    assert requested.json()["appointment"]["payment_status"] == "unpaid"

    # No local pay endpoint; only admin mock simulation moves HIS state.
    paid = test_client.post(
        f"/api/v1/admin/appointments/{appointment_id}/simulate-payment",
        headers=_auth(_token(test_client, "admin", ADMIN_PASSWORD)),
    )
    assert paid.status_code == 200, paid.text

    detail = test_client.get(f"/api/v1/booking/appointments/{appointment_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    body = detail.json()
    assert body["appointment"]["status"] == "registered"
    assert body["appointment"]["payment_status"] == "paid"
    assert body["visit"]["state"] == "registered"

    prescriptions = test_client.get(
        f"/api/v1/booking/appointments/{appointment_id}/prescriptions", headers=headers
    )
    assert prescriptions.status_code == 200, prescriptions.text
    assert prescriptions.json()["prescriptions"][0]["his_prescription_no"] == "RX0001"

    cancelled = test_client.post(
        f"/api/v1/booking/appointments/{appointment_id}/cancel",
        headers=headers,
        json={"reason": "test", "refund": True},
    )
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["cancel"]["status"] == "refunded"
    assert cancelled.json()["appointment"]["status"] == "refunded"


def test_appointment_requires_id_card_for_his_patient_mapping(client):
    test_client, _ = client
    token = _token(test_client, "patient", PATIENT_PASSWORD)
    headers = _auth(token)
    member = test_client.post(
        "/api/v1/family",
        headers=headers,
        json={"name": "无证件家人", "relation": "父母"},
    )
    assert member.status_code == 200, member.text
    member_id = member.json()["member"]["id"]
    created = test_client.post(
        "/api/v1/booking/appointments",
        headers=headers,
        json={
            "member_id": member_id,
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": TODAY,
        },
    )
    assert created.status_code == 409
    assert "身份证" in created.json()["detail"]


def test_family_member_id_card_is_encrypted_and_masked(client):
    test_client, _ = client
    token = _token(test_client, "patient", PATIENT_PASSWORD)
    headers = _auth(token)
    created = test_client.post(
        "/api/v1/family",
        headers=headers,
        json={
            "name": "有证件家人",
            "relation": "父母",
            "id_card": "310101198801019876",
            "birth_date": "1988-01-01",
            "gender": "male",
        },
    )
    assert created.status_code == 200, created.text
    member = created.json()["member"]
    assert member["has_id_card"] is True
    assert member["id_card_last4"] == "9876"
    assert "id_card_ciphertext" not in member
    assert "310101198801019876" not in json.dumps(member)

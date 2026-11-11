"""医生端收尾：文书打印 / 诊毕与随访 / 审方聚合。

覆盖：角色门禁、模板渲染（服务端存正文）、重复诊毕冲突、随访计划生成、
审方聚合的 HIS 口径标注。
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import date, timedelta

import asyncpg
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


def _sql(query: str, *args):
    async def run():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetch(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def doctor(client: TestClient) -> dict[str, str]:
    get_settings().AUTH_ENABLED = True
    get_settings().CLINICIAN_WRITE_ENABLED = True
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


@pytest.fixture()
def patient(client: TestClient) -> dict[str, str]:
    get_settings().AUTH_ENABLED = True
    return _auth(_token(client, "patient", PATIENT_PASSWORD))


@pytest.fixture()
def member(client: TestClient, patient: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/family",
        json={"name": "测试就诊人" + uuid.uuid4().hex[:4], "relation": "self", "is_self": True},
        headers=patient,
    )
    assert response.status_code == 200, response.text
    return response.json()["member"]["id"]


@pytest.fixture(autouse=True)
def cleanup_clinician_ext():
    yield
    for statement in (
        "DELETE FROM print_jobs WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '测试就诊人%')",
        "DELETE FROM visit_closures WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '测试就诊人%')",
        "DELETE FROM followups WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '测试就诊人%')",
        "DELETE FROM clinical_orders WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '测试就诊人%')",
        "DELETE FROM clinical_notes WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '测试就诊人%')",
        "DELETE FROM family_members WHERE name LIKE '测试就诊人%'",
    ):
        try:
            _sql(statement)
        except Exception:
            continue


def _create_note(client: TestClient, doctor: dict[str, str], member: str) -> dict:
    response = client.post(
        "/api/v1/clinician/notes",
        json={
            "member_id": member,
            "chief_complaint": "头痛三天",
            "present_illness": "三天前受凉后头痛，无发热",
            "physical_exam": "神清，颈软",
            "diagnosis": "紧张性头痛",
        },
        headers=doctor,
    )
    assert response.status_code == 200, response.text
    return response.json()["note"]


# ------------------------------------------------------------------ 权限
def test_requires_clinician(client: TestClient, patient: dict[str, str]) -> None:
    get_settings().AUTH_ENABLED = True
    for path in (
        "/api/v1/clinician/print/templates",
        "/api/v1/clinician/print",
        "/api/v1/clinician/followup-plans",
        "/api/v1/clinician/prescription-reviews",
    ):
        assert client.get(path, headers=patient).status_code == 403, path
    assert (
        client.post(
            "/api/v1/clinician/notes/nope/close-visit", json={"followup_days": 7}, headers=patient
        ).status_code
        == 403
    )


# ------------------------------------------------------------------ 文书打印
def test_print_templates(client: TestClient, doctor: dict[str, str]) -> None:
    response = client.get("/api/v1/clinician/print/templates", headers=doctor)
    assert response.status_code == 200, response.text
    payload = response.json()
    keys = {item["key"] for item in payload["items"]}
    assert keys == {"outpatient_note", "prescription", "exam_request"}
    assert all(item["name"] for item in payload["items"])


def test_print_outpatient_note_and_mark(client: TestClient, doctor: dict[str, str], member: str) -> None:
    note = _create_note(client, doctor, member)
    created = client.post(
        "/api/v1/clinician/print",
        json={"template": "outpatient_note", "note_id": note["id"]},
        headers=doctor,
    )
    assert created.status_code == 200, created.text
    payload = created.json()
    content = payload["content"]
    assert "门诊病历" in content
    assert "头痛三天" in content
    assert "紧张性头痛" in content
    assert "草稿" in content, "未诊毕的病历打印必须标注草稿"
    job_id = payload["job"]["id"]
    assert payload["job"]["status"] == "queued"

    detail = client.get("/api/v1/clinician/print/" + job_id, headers=doctor)
    assert detail.status_code == 200
    assert "头痛三天" in detail.json()["content"]

    marked = client.post("/api/v1/clinician/print/" + job_id + "/mark-printed", headers=doctor)
    assert marked.status_code == 200
    assert marked.json()["job"]["status"] == "printed"
    assert marked.json()["job"]["printed_at"]

    listed = client.get("/api/v1/clinician/print?member_id=" + member, headers=doctor).json()
    assert any(item["id"] == job_id and item["status"] == "printed" for item in listed["items"])


def test_print_prescription_only_signed(
    client: TestClient, doctor: dict[str, str], member: str
) -> None:
    order = client.post(
        "/api/v1/clinician/orders",
        json={"member_id": member, "kind": "medication", "name": "布洛芬", "dose": "0.3g", "usage": "口服 tid"},
        headers=doctor,
    )
    assert order.status_code == 200, order.text
    order_id = order.json()["order"]["id"]

    draft_print = client.post(
        "/api/v1/clinician/print", json={"template": "prescription", "member_id": member}, headers=doctor
    )
    assert "没有已签名的药品医嘱" in draft_print.json()["content"], "未签名医嘱不能出现在处方上"

    signed = client.post("/api/v1/clinician/orders/" + order_id + "/sign", headers=doctor)
    assert signed.status_code == 200, signed.text

    after = client.post(
        "/api/v1/clinician/print", json={"template": "prescription", "member_id": member}, headers=doctor
    )
    assert "布洛芬" in after.json()["content"]


def test_print_validation(client: TestClient, doctor: dict[str, str], member: str) -> None:
    assert (
        client.post(
            "/api/v1/clinician/print", json={"template": "nope", "member_id": member}, headers=doctor
        ).status_code
        == 422
    )
    assert client.post("/api/v1/clinician/print", json={"template": "prescription"}, headers=doctor).status_code == 422
    assert (
        client.post(
            "/api/v1/clinician/print",
            json={"template": "outpatient_note", "note_id": "no-such-note"},
            headers=doctor,
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/clinician/print",
            json={"template": "prescription", "member_id": "no-such-member"},
            headers=doctor,
        ).status_code
        == 404
    )
    assert client.get("/api/v1/clinician/print/pj_missing", headers=doctor).status_code == 404
    assert client.post("/api/v1/clinician/print/pj_missing/mark-printed", headers=doctor).status_code == 404


# ------------------------------------------------------------------ 诊毕 / 随访
def test_close_visit_creates_followup(client: TestClient, doctor: dict[str, str], member: str) -> None:
    note = _create_note(client, doctor, member)
    closed = client.post(
        "/api/v1/clinician/notes/" + note["id"] + "/close-visit",
        json={"advice": "两周后复查，头痛加重随时就诊", "followup_days": 14},
        headers=doctor,
    )
    assert closed.status_code == 200, closed.text
    payload = closed.json()
    assert payload["note"]["status"] == "completed"
    assert payload["followup"]["member_id"] == member
    assert payload["followup"]["status"] == "pending"
    expected = (date.today() + timedelta(days=14)).isoformat()
    assert payload["followup"]["due_date"] == expected
    assert "诊后随访" in payload["followup"]["title"]

    again = client.post(
        "/api/v1/clinician/notes/" + note["id"] + "/close-visit",
        json={"followup_days": 7},
        headers=doctor,
    )
    assert again.status_code == 409, "重复诊毕必须被拒绝"

    plans = client.get("/api/v1/clinician/followup-plans", headers=doctor).json()
    assert any(item["note_id"] == note["id"] and item["followup_id"] == payload["followup"]["id"] for item in plans["items"])

    bad_days = client.post(
        "/api/v1/clinician/notes/" + _create_note(client, doctor, member)["id"] + "/close-visit",
        json={"followup_days": 0},
        headers=doctor,
    )
    assert bad_days.status_code == 422

    missing = client.post("/api/v1/clinician/notes/no-note/close-visit", json={}, headers=doctor)
    assert missing.status_code == 404


# ------------------------------------------------------------------ 审方
def test_prescription_reviews(client: TestClient, doctor: dict[str, str], member: str) -> None:
    order = client.post(
        "/api/v1/clinician/orders",
        json={"member_id": member, "kind": "medication", "name": "氨氯地平", "dose": "5mg"},
        headers=doctor,
    )
    order_id = order.json()["order"]["id"]
    client.post("/api/v1/clinician/orders/" + order_id + "/sign", headers=doctor)

    payload = client.get("/api/v1/clinician/prescription-reviews?review_status=pending", headers=doctor).json()
    assert "his_note" in payload and "HIS" in payload["his_note"]
    assert payload["summary"].get("pending", 0) >= 1
    assert any(item["id"] == order_id and item["review_status"] == "pending" for item in payload["items"])

    approved = client.post(
        "/api/v1/clinician/orders/" + order_id + "/review",
        json={"status": "approved", "note": "无相互作用"},
        headers=doctor,
    )
    assert approved.status_code == 200, approved.text

    after = client.get(
        "/api/v1/clinician/prescription-reviews?review_status=approved", headers=doctor
    ).json()
    assert any(item["id"] == order_id for item in after["items"])

    assert (
        client.get("/api/v1/clinician/prescription-reviews?review_status=bad", headers=doctor).status_code
        == 422
    )

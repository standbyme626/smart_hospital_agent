"""V3.4 剧本可执行化：把"部分 / 阻塞"的剧本落成真实断言。

对应剧本（与 app/playbooks.py 同步）：
  P-05 重复缴费幂等 / INT-01 幂等重放 / INT-02 故障注入 / INT-05 审计完整
  E2E-03 并发抢号恰 1 单 / D-03 诊前摘要可回溯 / D-04 接诊 / D-05 诊断带医生 ID / D-06 签名门禁
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"


class StubMCP:
    """可切换故障模式的 MCP 替身；记录 idempotency_key 以便断言重放。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.mode = "ok"  # ok | error | timeout

    async def call_tool(self, name, arguments=None, idempotency_key=None, scope=None):
        arguments = dict(arguments or {})
        self.calls.append((name, arguments, idempotency_key))
        if self.mode == "timeout":
            raise TimeoutError("upstream timed out after 30s")
        if self.mode == "error":
            return {"is_error": True, "text": "upstream secret detail DSN=postgres://admin:pass@db", "data": None}
        if name == "his.reconcile":
            return {
                "is_error": False,
                "text": "",
                "data": {
                    "summary": {"checked": 1, "findings": 1, "open_findings": 1, "scope": arguments.get("scope", "all")},
                    "findings": [
                        {
                            "id": "find_int03_stable",
                            "kind": "payment_mismatch",
                            "severity": "high",
                            "local_ref": "appt_mismatch",
                            "his_ref": "enc_mismatch",
                            "detail": "local=paid his=unpaid",
                        }
                    ],
                },
            }
        if name == "lock_slot":
            return {
                "is_error": False,
                "text": "",
                "data": {"lock_id": "lock_" + str(arguments.get("slot_id", ""))[:8], "slot_id": arguments.get("slot_id"), "expires_in": 600},
            }
        return {
            "is_error": False,
            "text": "",
            "data": {
                "order_id": "order_" + uuid.uuid4().hex[:8],
                "lock_id": arguments.get("lock_id"),
                "slot_id": arguments.get("slot_id"),
                "patient_id": arguments.get("patient_id", "patient"),
                "amount": 50,
                "status": "created",
                "payment_status": "unpaid",
            },
        }


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        settings = test_client.app.state.settings
        settings.AUTH_ENABLED = True
        settings.CLINICIAN_WRITE_ENABLED = True
        stub = StubMCP()
        test_client.app.state.mcp = stub
        test_client.stub_mcp = stub  # type: ignore[attr-defined]
        try:
            yield test_client
        finally:
            settings.AUTH_ENABLED = False
            settings.CLINICIAN_WRITE_ENABLED = False


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
def patient(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "patient", PATIENT_PASSWORD))


@pytest.fixture()
def admin(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


@pytest.fixture()
def member(client: TestClient, patient: dict[str, str]) -> str:
    response = client.post(
        "/api/v1/family",
        json={"name": "剧本就诊人" + uuid.uuid4().hex[:4], "relation": "self", "is_self": True},
        headers=patient,
    )
    assert response.status_code == 200, response.text
    return response.json()["member"]["id"]


@pytest.fixture(autouse=True)
def cleanup_playbook_rows():
    yield
    for statement in (
        "DELETE FROM visit_encounters WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM visit_closures WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM print_jobs WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM clinical_orders WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM clinical_notes WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM previsit_forms WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM followups WHERE member_id IN (SELECT id FROM family_members WHERE name LIKE '剧本就诊人%')",
        "DELETE FROM bookings WHERE user_id IN (SELECT username FROM users)",
        "DELETE FROM family_members WHERE name LIKE '剧本就诊人%'",
        "DELETE FROM reconciliation_finding WHERE id LIKE 'find_int03%' OR local_ref = 'appt_mismatch'",
    ):
        try:
            _sql(statement)
        except Exception:
            continue


# ---------------------------------------------------------------- P-01 建档复用
def test_p01_same_id_card_reuses_member(client: TestClient, patient: dict[str, str]) -> None:
    id_card = "4401011992030" + str(int(uuid.uuid4().int % 10000)).zfill(4)
    first = client.post(
        "/api/v1/family",
        json={"name": "建档复用A", "relation": "self", "id_card": id_card},
        headers=patient,
    )
    assert first.status_code == 200, first.text
    assert first.json().get("reused") is not True, "首次建档不应标记复用"

    second = client.post(
        "/api/v1/family",
        json={"name": "建档复用B", "relation": "other", "id_card": id_card},
        headers=patient,
    )
    assert second.status_code == 200, second.text
    assert second.json()["reused"] is True, "同一身份证再次建档必须复用"
    assert second.json()["member"]["id"] == first.json()["member"]["id"]

    rows = _sql("SELECT count(*) AS c FROM family_members WHERE id_card_last4 = $1", id_card[-4:])
    assert rows[0]["c"] == 1, "库里不能出现两份同身份证档案"


# ---------------------------------------------------------------- INT-01 幂等重放
def test_int01_replay_ten_times_creates_one_order(client: TestClient, patient: dict[str, str], member: str) -> None:
    payload = {"lock_id": "lock_replay", "member_id": member}
    order_ids = set()
    for _ in range(10):
        response = client.post("/api/v1/booking/orders", json=payload, headers=patient)
        assert response.status_code == 200, response.text
        order_ids.add(response.json()["order"]["order_id"])

    assert len(order_ids) == 1, "同一锁号重放 10 次必须只得到 1 个订单"
    stub: StubMCP = client.stub_mcp  # type: ignore[attr-defined]
    keys = [call[2] for call in stub.calls if call[0] == "create_order"]
    assert len(set(keys)) == 1, "10 次重放的 idempotency_key 必须稳定"


def test_p05_duplicate_pay_is_blocked_or_idempotent(client: TestClient, patient: dict[str, str], member: str) -> None:
    created = client.post("/api/v1/booking/orders", json={"lock_id": "lock_pay", "member_id": member}, headers=patient)
    assert created.status_code == 200, created.text
    order_id = created.json()["order"]["order_id"]

    first = client.post(f"/api/v1/booking/orders/{order_id}/pay", headers=patient)
    assert first.status_code == 403, "患者端直接支付应被禁止（走人工审核）"
    second = client.post("/api/v1/booking/orders/unknown_order/pay", headers=patient)
    assert second.status_code in {403, 404}


# ---------------------------------------------------------------- INT-02 故障注入
def test_int02_fault_injection_is_generic_and_retryable(client: TestClient, patient: dict[str, str]) -> None:
    stub: StubMCP = client.stub_mcp  # type: ignore[attr-defined]

    stub.mode = "error"
    failed = client.post("/api/v1/booking/lock", json={"slot_id": "991"}, headers=patient)
    assert failed.status_code == 502
    assert "HIS_ERROR" in failed.text and "可重试" in failed.text
    assert "secret" not in failed.text and "postgres://" not in failed.text, "不能把上游细节透给前端"

    stub.mode = "timeout"
    timed_out = client.post("/api/v1/booking/lock", json={"slot_id": "992"}, headers=patient)
    assert timed_out.status_code == 504
    assert "HIS_TIMEOUT" in timed_out.text

    stub.mode = "ok"
    ok = client.post("/api/v1/booking/lock", json={"slot_id": "993"}, headers=patient)
    assert ok.status_code == 200
    rows = _sql("SELECT count(*) AS c FROM bookings WHERE id LIKE 'lock_%'")
    assert rows is not None


# ---------------------------------------------------------------- E2E-03 并发抢号
def test_e2e03_concurrent_bookings_single_order(client: TestClient, patient: dict[str, str], member: str) -> None:
    payload = {"lock_id": "lock_race", "member_id": member}

    def book(_index: int) -> tuple[int, str]:
        response = client.post("/api/v1/booking/orders", json=payload, headers=patient)
        order_id = ""
        if response.status_code == 200:
            order_id = response.json()["order"]["order_id"]
        return response.status_code, order_id

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(book, range(10)))

    succeeded = [order_id for code, order_id in results if code == 200]
    assert succeeded, results
    assert len(set(succeeded)) == 1, "并发抢同一号必须只产生一个订单"


# ---------------------------------------------------------------- INT-05 审计完整
def test_int05_audit_has_actor_time_trace(client: TestClient, patient: dict[str, str], admin: dict[str, str], member: str) -> None:
    note = client.post(
        "/api/v1/clinician/notes",
        json={"member_id": member, "chief_complaint": "剧本审计", "diagnosis": "审计测试"},
        headers=admin,
    )
    assert note.status_code == 200, note.text
    note_id = note.json()["note"]["id"]

    assert client.post("/api/v1/clinician/print", json={"template": "outpatient_note", "note_id": note_id}, headers=admin).status_code == 200
    assert client.post(f"/api/v1/clinician/notes/{note_id}/close-visit", json={"followup_days": 7}, headers=admin).status_code == 200
    assert client.post("/api/v1/clinician/receive", json={"member_id": member}, headers=admin).status_code in {200, 409}

    rows = _sql(
        "SELECT actor, action, detail, request_id, created_at FROM audit_log WHERE action = ANY($1::text[]) ORDER BY id DESC LIMIT 20",
        ["clinician.print.create", "clinician.visit.closed", "clinician.encounter.received", "auth.login"],
    )
    actions = {row["action"] for row in rows}
    assert {"clinician.print.create", "clinician.visit.closed", "clinician.encounter.received", "auth.login"} <= actions
    for row in rows:
        assert row["actor"], "审计必须有操作人"
        assert row["created_at"], "审计必须有时间"
        assert row["request_id"], "审计必须带 trace（request_id）"
        detail = row["detail"]
        text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
        assert "password" not in text.lower()


# ---------------------------------------------------------------- INT-03 强制不一致
def test_int03_reconcile_mirrors_mismatch(client: TestClient, admin: dict[str, str]) -> None:
    first = client.post("/api/v1/ops/reconciliation/run", json={"scope": "payment"}, headers=admin)
    assert first.status_code == 200, first.text
    payload = first.json()
    assert payload["summary"]["findings"] >= 1
    assert payload["mirrored"] >= 1, "HIS 侧差异必须镜像到平台 findings"

    finding = payload["findings"][0]
    assert finding["kind"] == "payment_mismatch" and finding["severity"] == "high"

    listed = client.get("/api/v1/ops/reconciliation", headers=admin).json()
    ids = {item["id"] for item in listed["items"]}
    assert finding["id"] in ids, "运维页面必须能看到这条差异"

    again = client.post("/api/v1/ops/reconciliation/run", json={"scope": "payment"}, headers=admin)
    assert again.status_code == 200
    assert again.json()["mirrored"] == 0, "同一差异重复对账不应重复落库（幂等）"

    rows = _sql("SELECT count(*) AS c FROM reconciliation_finding WHERE id=$1", finding["id"])
    assert rows[0]["c"] == 1

    bad_scope = client.post("/api/v1/ops/reconciliation/run", json={"scope": "nope"}, headers=admin)
    assert bad_scope.status_code == 422


# ---------------------------------------------------------------- D-03 诊前摘要
def test_d03_previsit_is_traceable_in_patient_360(client: TestClient, patient: dict[str, str], admin: dict[str, str], member: str) -> None:
    submitted = client.post(
        "/api/v1/previsit",
        json={
            "member_id": member,
            "chief_complaint": "剧本预问诊：间断头痛两周",
            "duration": "两周",
            "medications": "布洛芬按需",
            "allergies": "无",
            "history": "高血压 5 年",
            "questions": "需要做哪些检查",
        },
        headers=patient,
    )
    assert submitted.status_code == 200, submitted.text
    previsit_id = submitted.json()["previsit"]["id"]

    detail = client.get(f"/api/v1/clinician/patients/{member}", headers=admin)
    assert detail.status_code == 200, detail.text
    previsit = detail.json()["previsit"]
    assert previsit is not None, "医生 360 必须能看到预问诊摘要"
    assert previsit["id"] == previsit_id, "摘要要能回溯到原始预问诊记录"
    assert previsit["chief_complaint"].startswith("剧本预问诊")
    assert previsit["created_at"], "摘要要带时间，便于核对来源"


# ---------------------------------------------------------------- D-04 接诊
def test_d04_receive_lifecycle(client: TestClient, admin: dict[str, str], member: str) -> None:
    received = client.post("/api/v1/clinician/receive", json={"member_id": member, "note": "剧本接诊"}, headers=admin)
    assert received.status_code == 200, received.text
    encounter = received.json()["encounter"]
    assert encounter["status"] == "in_progress"
    assert encounter["source"] == "platform"
    assert encounter["his_encounter_id"] is None
    assert "HIS" in received.json()["his_note"]

    duplicate = client.post("/api/v1/clinician/receive", json={"member_id": member}, headers=admin)
    assert duplicate.status_code == 409, "同一就诊人不能重复接诊"

    listed = client.get(f"/api/v1/clinician/encounters?member_id={member}", headers=admin).json()
    assert any(item["id"] == encounter["id"] and item["status"] == "in_progress" for item in listed["items"])

    completed = client.post(f"/api/v1/clinician/encounters/{encounter['id']}/complete", headers=admin)
    assert completed.status_code == 200, completed.text
    assert completed.json()["encounter"]["status"] == "completed"
    assert client.post(f"/api/v1/clinician/encounters/{encounter['id']}/complete", headers=admin).status_code == 404


# ---------------------------------------------------------------- D-05 诊断带医生 ID
def test_d05_note_records_doctor_and_hidden_from_patient(
    client: TestClient, patient: dict[str, str], admin: dict[str, str], member: str
) -> None:
    created = client.post(
        "/api/v1/clinician/notes",
        json={"member_id": member, "chief_complaint": "剧本病历", "diagnosis": "剧本诊断"},
        headers=admin,
    )
    assert created.status_code == 200, created.text
    note = created.json()["note"]
    assert note["doctor"], "诊断必须记录医生"
    assert note["status"] == "draft"

    forbidden = client.get(f"/api/v1/clinician/notes?member_id={member}", headers=patient)
    assert forbidden.status_code == 403, "患者不能读取医生病历（草稿更不可见）"


# ---------------------------------------------------------------- D-06 签名门禁
def test_d06_order_sign_gate(client: TestClient, patient: dict[str, str], admin: dict[str, str], member: str) -> None:
    settings = get_settings()
    created = client.post(
        "/api/v1/clinician/orders",
        json={"member_id": member, "kind": "medication", "name": "剧本药", "dose": "5mg", "usage": "口服"},
        headers=admin,
    )
    assert created.status_code == 200, created.text
    order_id = created.json()["order"]["id"]

    settings.CLINICIAN_WRITE_ENABLED = False
    blocked = client.post(f"/api/v1/clinician/orders/{order_id}/sign", headers=admin)
    assert blocked.status_code == 409
    assert "HIS" in blocked.text or "签名" in blocked.text

    settings.CLINICIAN_WRITE_ENABLED = True
    signed = client.post(f"/api/v1/clinician/orders/{order_id}/sign", headers=admin)
    assert signed.status_code == 200, signed.text
    assert signed.json()["order"]["status"] == "signed"

    again = client.post(f"/api/v1/clinician/orders/{order_id}/sign", headers=admin)
    assert again.status_code in {404, 409}, "重复签名必须被拒绝"

    reviews = client.get("/api/v1/clinician/prescription-reviews?review_status=pending", headers=admin).json()
    assert any(item["id"] == order_id for item in reviews["items"]), "签名后应进入待审方"

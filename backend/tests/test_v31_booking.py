import asyncio
import json
import os
from types import SimpleNamespace

import asyncpg
import pytest
from fastapi.testclient import TestClient

import mcp_mock.server as mock
from app.main import app
from app.mcp_client import HospitalMCPClient

PATIENT_PASSWORD = "patient123456"
ADMIN_PASSWORD = "admin123456"
SLOT_ID = "心血管内科|2031-01-01|09:00|张医生"


class CaptureMCP:
    def __init__(self):
        self.calls: list[tuple[str, dict, str | None]] = []
        self.order_ids: list[str] = []

    async def call_tool(self, name, arguments=None, idempotency_key=None):
        self.calls.append((name, dict(arguments or {}), idempotency_key))
        if name == "lock_slot":
            return {
                "is_error": False,
                "text": "",
                "data": {
                    "lock_id": "lock_fake",
                    "slot_id": (arguments or {}).get("slot_id"),
                    "expires_in": 600,
                },
            }
        if name == "create_order":
            order_id = self.order_ids.pop(0) if self.order_ids else "order_idem"
            return {
                "is_error": False,
                "text": "",
                "data": {
                    "order_id": order_id,
                    "slot_id": SLOT_ID,
                    "patient_id": "patient",
                    "amount": 50,
                    "status": "created",
                    "payment_status": "unpaid",
                },
            }
        if name == "cancel_order":
            return {
                "is_error": False,
                "text": "",
                "data": {
                    "order_id": (arguments or {}).get("order_id"),
                    "slot_id": SLOT_ID,
                    "patient_id": "patient",
                    "amount": 50,
                    "status": "cancelled",
                    "payment_status": "unpaid",
                },
            }
        return {"is_error": False, "text": "", "data": {}}


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        test_client.app.state.settings.AUTH_ENABLED = True
        test_client.app.state.mcp = CaptureMCP()
        try:
            yield test_client
        finally:
            test_client.app.state.settings.AUTH_ENABLED = False


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _payload(result):
    assert result["isError"] is False, result
    return json.loads(result["content"][0]["text"])


async def _count(query: str, *args) -> int:
    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        return await conn.fetchval(query, *args)
    finally:
        await conn.close()


async def test_mock_persists_orders_and_is_idempotent():
    token = os.getenv("HITL_CONFIRM_TOKEN", "confirm-demo")
    lock = _payload(await mock._call("lock_slot", {"slot_id": SLOT_ID, "patient_id": "u_mock"}))

    first = _payload(
        await mock._call(
            "create_order",
            {"lock_id": lock["lock_id"], "patient_id": "u_mock", "idempotency_key": "idem-1"},
        )
    )
    second = _payload(
        await mock._call(
            "create_order",
            {"lock_id": lock["lock_id"], "patient_id": "u_mock", "idempotency_key": "idem-1"},
        )
    )
    assert second["order_id"] == first["order_id"]
    assert await _count("SELECT count(*) FROM mcp_orders WHERE idempotency_key='idem-1'") == 1

    paid = _payload(
        await mock._call(
            "pay_order",
            {"order_id": first["order_id"], "confirm_token": token, "method": "mock"},
        )
    )
    assert paid["payment_status"] == "paid"
    repeated = _payload(
        await mock._call(
            "pay_order",
            {"order_id": first["order_id"], "confirm_token": token, "method": "mock"},
        )
    )
    assert repeated["payment_status"] == "paid"
    assert await _count("SELECT count(*) FROM mcp_booked WHERE slot_id=$1", SLOT_ID) == 1

    status = _payload(await mock._call("get_payment_status", {"order_id": first["order_id"]}))
    assert status["order_id"] == first["order_id"]
    assert status["payment_status"] == "paid"

    guide = _payload(await mock._call("get_visit_guide", {"order_id": first["order_id"]}))
    assert guide["order"]["order_id"] == first["order_id"]


async def test_mock_pay_requires_token_and_cancel_releases_slot():
    lock = _payload(await mock._call("lock_slot", {"slot_id": SLOT_ID, "patient_id": "u_mock"}))
    order = _payload(await mock._call("create_order", {"lock_id": lock["lock_id"]}))

    denied = await mock._call("pay_order", {"order_id": order["order_id"], "confirm_token": "wrong"})
    assert denied["isError"] is True
    assert "requires_confirmation" in json.loads(denied["content"][0]["text"])["error"]

    cancelled = _payload(await mock._call("cancel_order", {"order_id": order["order_id"]}))
    assert cancelled["status"] == "cancelled"
    assert await _count("SELECT count(*) FROM mcp_booked WHERE slot_id=$1", SLOT_ID) == 0


async def test_client_passes_idempotency_key():
    settings = SimpleNamespace(HOSPITAL_MCP_URL="http://127.0.0.1:1/mcp", HOSPITAL_MCP_TOKEN="")
    client = HospitalMCPClient(settings)
    captured: dict = {}

    async def fake_rpc(method, params=None, request_id=None):
        captured.update(params or {})
        return {"content": [{"type": "text", "text": "{}"}]}

    client._initialized = True
    client._rpc = fake_rpc
    try:
        await client.call_tool("create_order", {"lock_id": "lock_x"}, idempotency_key="idem-x")
    finally:
        await client.close()
    assert captured["name"] == "create_order"
    assert captured["arguments"]["idempotency_key"] == "idem-x"


def test_booking_route_reuses_idempotency_key(client):
    token = _token(client, "patient", PATIENT_PASSWORD)
    headers = {"Authorization": f"Bearer {token}"}

    first = client.post("/api/v1/booking/orders", json={"lock_id": "lock_x"}, headers=headers)
    second = client.post("/api/v1/booking/orders", json={"lock_id": "lock_x"}, headers=headers)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["order"]["order_id"] == second.json()["order"]["order_id"]

    fake: CaptureMCP = client.app.state.mcp
    assert [call[2] for call in fake.calls] == ["order:patient:lock_x", "order:patient:lock_x"]

    async def row():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchrow(
                "SELECT id, idempotency_key FROM bookings WHERE id='order_idem'"
            )
        finally:
            await conn.close()

    stored = asyncio.run(row())
    assert stored is not None
    assert stored["idempotency_key"] == "order:patient:lock_x"


def test_admin_login_still_works(client):
    from app.security import hash_password, reset_lockout_state

    reset_lockout_state()
    client.app.state.settings.AUTH_ENABLED = True

    async def seed_admin():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                """
                INSERT INTO users (id, username, password_hash, role, enabled)
                VALUES ('admin', 'admin', $1, 'admin', true)
                ON CONFLICT (id) DO UPDATE SET password_hash=EXCLUDED.password_hash, enabled=true
                """,
                hash_password(ADMIN_PASSWORD),
            )
        finally:
            await conn.close()

    asyncio.run(seed_admin())
    assert _token(client, "admin", ADMIN_PASSWORD)


async def test_mock_health_ok_and_reports_db_failure(monkeypatch):
    healthy = await mock.health()
    assert healthy["ok"] is True

    monkeypatch.setenv("DATABASE_URL", "postgresql://sha:sha_pwd@127.0.0.1:1/sha_test")
    broken = await mock.health()
    assert broken["ok"] is False
    assert broken["error"]


def test_mock_dsn_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError):
        mock._dsn()


def test_booking_insert_does_not_swallow_idempotency_conflict(client):
    token = _token(client, "patient", PATIENT_PASSWORD)
    headers = {"Authorization": f"Bearer {token}"}
    fake: CaptureMCP = client.app.state.mcp
    fake.order_ids = ["order_a", "order_b"]

    first = client.post("/api/v1/booking/orders", json={"lock_id": "lock_x"}, headers=headers)
    assert first.status_code == 200, first.text
    try:
        second = client.post("/api/v1/booking/orders", json={"lock_id": "lock_x"}, headers=headers)
    except asyncpg.UniqueViolationError:
        pass
    else:
        assert second.status_code >= 500

    async def count():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval("SELECT count(*) FROM bookings WHERE idempotency_key='order:patient:lock_x'")
        finally:
            await conn.close()

    assert asyncio.run(count()) == 1


def _insert_booking(
    booking_id: str,
    *,
    user_id: str = "patient",
    status: str = "created",
    payment_status: str = "unpaid",
    age_minutes: int = 0,
) -> None:
    async def run():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                """
                INSERT INTO bookings
                    (id, session_id, patient_id, user_id, department, slot_id, order_id,
                     status, payment_status, amount, created_at)
                VALUES ($1,$1,$2,$2,'心血管内科',$3,$1,$4,$5,50, now() - make_interval(mins => $6))
                """,
                booking_id,
                user_id,
                SLOT_ID,
                status,
                payment_status,
                age_minutes,
            )
        finally:
            await conn.close()

    asyncio.run(run())


def _insert_review(review_id: str, ref_id: str, status: str = "pending") -> None:
    async def run():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            await conn.execute(
                """
                INSERT INTO review_requests (id, kind, ref_id, user_id, status, payload)
                VALUES ($1, 'payment', $2, 'patient', $3, '{}'::jsonb)
                """,
                review_id,
                ref_id,
                status,
            )
        finally:
            await conn.close()

    asyncio.run(run())


def _booking_status(booking_id: str) -> str | None:
    async def run():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval("SELECT status FROM bookings WHERE id=$1", booking_id)
        finally:
            await conn.close()

    return asyncio.run(run())


def test_order_list_owner_newest_first_and_detail_review(client):
    headers = {"Authorization": f"Bearer {_token(client, 'patient', PATIENT_PASSWORD)}"}
    _insert_booking("order_old", age_minutes=120)
    _insert_booking("order_new", age_minutes=1)
    _insert_booking("order_other", user_id="someone_else")
    _insert_review("rev_latest", "order_new", status="pending")

    listed = client.get("/api/v1/booking/orders", headers=headers)
    assert listed.status_code == 200, listed.text
    ids = [item["id"] for item in listed.json()["orders"]]
    assert ids == ["order_new", "order_old"]

    detail = client.get("/api/v1/booking/orders/order_new", headers=headers)
    assert detail.status_code == 200, detail.text
    assert detail.json()["review"]["id"] == "rev_latest"
    assert detail.json()["review"]["status"] == "pending"

    assert client.get("/api/v1/booking/orders/order_other", headers=headers).status_code == 404
    assert (
        client.post("/api/v1/booking/orders/order_other/cancel", headers=headers).status_code == 404
    )


def test_lock_returns_expires_at(client):
    from datetime import UTC, datetime, timedelta

    headers = {"Authorization": f"Bearer {_token(client, 'patient', PATIENT_PASSWORD)}"}
    response = client.post(
        "/api/v1/booking/lock", json={"slot_id": SLOT_ID}, headers=headers
    )
    assert response.status_code == 200, response.text
    lock = response.json()["lock"]
    assert lock["expires_in"] == 600
    expires_at = datetime.fromisoformat(lock["expires_at"])
    assert expires_at > datetime.now(UTC) + timedelta(minutes=9)


def test_cancel_unpaid_updates_status_and_is_idempotent(client):
    headers = {"Authorization": f"Bearer {_token(client, 'patient', PATIENT_PASSWORD)}"}
    _insert_booking("order_cancel")

    first = client.post("/api/v1/booking/orders/order_cancel/cancel", headers=headers)
    assert first.status_code == 200, first.text
    assert first.json()["order"]["status"] == "cancelled"
    assert _booking_status("order_cancel") == "cancelled"
    fake: CaptureMCP = client.app.state.mcp
    assert [name for name, _, _ in fake.calls] == ["cancel_order"]

    second = client.post("/api/v1/booking/orders/order_cancel/cancel", headers=headers)
    assert second.status_code == 200, second.text
    assert second.json()["order"]["status"] == "cancelled"
    assert [name for name, _, _ in fake.calls] == ["cancel_order"]


def test_cancel_pending_review_rejects_review(client):
    headers = {"Authorization": f"Bearer {_token(client, 'patient', PATIENT_PASSWORD)}"}
    _insert_booking("order_pending", status="pending_review")
    _insert_review("rev_pending", "order_pending")

    response = client.post("/api/v1/booking/orders/order_pending/cancel", headers=headers)
    assert response.status_code == 200, response.text
    assert _booking_status("order_pending") == "cancelled"

    async def review_status():
        conn = await asyncpg.connect(os.environ["DATABASE_URL"])
        try:
            return await conn.fetchval(
                "SELECT status FROM review_requests WHERE id='rev_pending'"
            )
        finally:
            await conn.close()

    assert asyncio.run(review_status()) == "rejected"


def test_cancel_paid_order_returns_409(client):
    headers = {"Authorization": f"Bearer {_token(client, 'patient', PATIENT_PASSWORD)}"}
    _insert_booking("order_paid", status="booked", payment_status="paid")

    response = client.post("/api/v1/booking/orders/order_paid/cancel", headers=headers)
    assert response.status_code == 409
    assert _booking_status("order_paid") == "booked"
    fake: CaptureMCP = client.app.state.mcp
    assert fake.calls == []


def test_orders_survive_api_restart(client):
    _insert_booking("order_persist", status="pending_review")

    with TestClient(app) as restarted:
        restarted.app.state.settings.AUTH_ENABLED = True
        restarted.app.state.mcp = CaptureMCP()
        new_headers = {
            "Authorization": f"Bearer {_token(restarted, 'patient', PATIENT_PASSWORD)}"
        }
        response = restarted.get("/api/v1/booking/orders", headers=new_headers)
        assert response.status_code == 200, response.text
        ids = [item["id"] for item in response.json()["orders"]]
        assert "order_persist" in ids

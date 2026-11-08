import asyncio
import json

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"
SLOT_ID = "心血管内科|2030-01-01|09:00|张医生"


class FakeMCP:
    def __init__(self, settings):
        self.settings = settings
        self.calls: list[tuple[str, dict]] = []
        self.fail_pay = False
        self.paid: set[str] = set()

    def _ok(self, data):
        return {"is_error": False, "text": "", "data": data}

    async def call_tool(self, name, arguments=None, idempotency_key=None):
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "lock_slot":
            return self._ok({"lock_id": "lock_fake", "slot_id": args.get("slot_id"), "expires_in": 600})
        if name == "create_order":
            return self._ok(
                {
                    "order_id": "order_mem",
                    "slot_id": SLOT_ID,
                    "patient_id": "patient",
                    "amount": 50,
                    "status": "created",
                    "payment_status": "unpaid",
                }
            )
        if name == "pay_order":
            if self.fail_pay:
                return {"is_error": True, "text": "mcp unavailable", "data": None}
            self.paid.add(args["order_id"])
            return self._ok(
                {
                    "order_id": args["order_id"],
                    "slot_id": SLOT_ID,
                    "patient_id": "patient",
                    "amount": 50,
                    "status": "booked",
                    "payment_status": "paid",
                }
            )
        if name == "get_payment_status":
            paid = args["order_id"] in self.paid
            return self._ok(
                {
                    "order_id": args["order_id"],
                    "slot_id": SLOT_ID,
                    "patient_id": "patient",
                    "amount": 50,
                    "status": "booked" if paid else "created",
                    "payment_status": "paid" if paid else "unpaid",
                }
            )
        if name == "get_visit_guide":
            return self._ok({"order": {"order_id": args["order_id"]}, "hospital": "示范医院"})
        return self._ok({})

    def pay_calls(self) -> list[dict]:
        return [args for name, args in self.calls if name == "pay_order"]


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        test_client.app.state.settings.AUTH_ENABLED = True
        test_client.app.state.mcp = FakeMCP(test_client.app.state.settings)
        try:
            yield test_client
        finally:
            test_client.app.state.settings.AUTH_ENABLED = False


def _token(client: TestClient, username: str, password: str) -> str:
    response = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _patient(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "patient", PATIENT_PASSWORD))


def _admin(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


def _sql(dsn: str, query: str, *args, fetch: bool = False):
    async def run():
        conn = await asyncpg.connect(dsn)
        try:
            if fetch:
                return await conn.fetchval(query, *args)
            return await conn.execute(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _create_booking(client: TestClient, order_id: str = "order_rev1", user_id: str = "patient") -> None:
    dsn = client.app.state.settings.DATABASE_URL
    _sql(
        dsn,
        """
        INSERT INTO bookings
            (id, session_id, patient_id, user_id, department, slot_id, order_id, status, payment_status, amount)
        VALUES ($1, $1, $2, $2, $3, $4, $1, 'created', 'unpaid', 50)
        """,
        order_id,
        user_id,
        "心血管内科",
        SLOT_ID,
    )


def _booking_row(client: TestClient, order_id: str = "order_rev1") -> dict:
    dsn = client.app.state.settings.DATABASE_URL

    async def run():
        conn = await asyncpg.connect(dsn)
        try:
            return dict(await conn.fetchrow("SELECT * FROM bookings WHERE id=$1", order_id))
        finally:
            await conn.close()

    return asyncio.run(run())


def _review_rows(client: TestClient, order_id: str = "order_rev1") -> list[dict]:
    dsn = client.app.state.settings.DATABASE_URL

    async def run():
        conn = await asyncpg.connect(dsn)
        try:
            rows = await conn.fetch(
                "SELECT * FROM review_requests WHERE ref_id=$1 ORDER BY created_at", order_id
            )
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    return asyncio.run(run())


def test_patient_cannot_pay_directly(client):
    _create_booking(client)
    response = client.post(
        "/api/v1/booking/orders/order_rev1/pay",
        json={"confirm_token": client.app.state.settings.HITL_CONFIRM_TOKEN},
        headers=_patient(client),
    )
    assert response.status_code == 403
    assert _booking_row(client)["payment_status"] == "unpaid"
    assert client.app.state.mcp.calls == []


def test_request_payment_pending_then_admin_approve_is_idempotent(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)

    requested = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient)
    assert requested.status_code == 200, requested.text
    review = requested.json()["review"]
    assert review["status"] == "pending"
    assert review["kind"] == "payment"
    assert review["payload"]["order_id"] == "order_rev1"
    booking = _booking_row(client)
    assert booking["status"] == "pending_review"
    assert booking["review_id"] == review["id"]
    assert booking["payment_status"] == "unpaid"

    assert client.post("/api/v1/booking/orders/order_rev1/pay", headers=patient).status_code == 403

    status = client.get("/api/v1/booking/orders/order_rev1", headers=patient)
    assert status.status_code == 200
    assert status.json()["order"]["payment_status"] == "unpaid"
    assert status.json()["review"]["status"] == "pending"

    queue = client.get("/api/v1/admin/reviews", params={"status": "pending"}, headers=admin)
    assert queue.status_code == 200
    item = next(entry for entry in queue.json()["reviews"] if entry["id"] == review["id"])
    assert item["department"] == "心血管内科"
    assert item["amount"] == 50
    assert item["username"] == "patient"

    approved = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert approved.status_code == 200, approved.text
    assert approved.json()["review"]["status"] == "approved"
    assert approved.json()["mcp_called"] is True
    assert len(client.app.state.mcp.pay_calls()) == 1

    booking = _booking_row(client)
    assert booking["status"] == "booked"
    assert booking["payment_status"] == "paid"
    dsn = client.app.state.settings.DATABASE_URL
    assert _sql(dsn, "SELECT count(*) FROM followups WHERE booking_id='order_rev1'", fetch=True) == 1
    assert _sql(dsn, "SELECT count(*) FROM audit_log WHERE action='payment.approve'", fetch=True) == 1

    status = client.get("/api/v1/booking/orders/order_rev1", headers=patient)
    assert status.json()["order"]["payment_status"] == "paid"
    assert status.json()["review"]["status"] == "approved"

    repeated = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert repeated.status_code == 200
    assert repeated.json()["review"]["status"] == "approved"
    assert repeated.json()["mcp_called"] is False
    assert len(client.app.state.mcp.pay_calls()) == 1


def test_second_request_returns_same_pending_review(client):
    _create_booking(client)
    patient = _patient(client)
    first = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient)
    second = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient)
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["review"]["id"] == first.json()["review"]["id"]
    assert len(_review_rows(client)) == 1


def test_admin_reject_resets_order_and_allows_reapply(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]

    rejected = client.post(
        f"/api/v1/admin/reviews/{review['id']}/reject",
        json={"note": "材料不全，请补充"},
        headers=admin,
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["review"]["status"] == "rejected"
    assert rejected.json()["review"]["decision_note"] == "材料不全，请补充"

    booking = _booking_row(client)
    assert booking["status"] == "created"
    assert booking["payment_status"] == "unpaid"

    again = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient)
    assert again.status_code == 200
    assert again.json()["review"]["id"] != review["id"]
    assert again.json()["review"]["status"] == "pending"


def test_non_admin_cannot_use_review_admin_endpoints(client):
    _create_booking(client)
    patient = _patient(client)
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]

    assert client.get("/api/v1/admin/reviews", headers=patient).status_code == 403
    assert client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=patient).status_code == 403
    assert (
        client.post(
            f"/api/v1/admin/reviews/{review['id']}/reject",
            json={"note": "x"},
            headers=patient,
        ).status_code
        == 403
    )
    assert _booking_row(client)["payment_status"] == "unpaid"


def test_approve_mcp_failure_marks_failed_and_resets_order(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)
    client.app.state.mcp.fail_pay = True
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 502
    assert _review_rows(client)[0]["status"] == "failed"
    booking = _booking_row(client)
    assert booking["status"] == "created"
    assert booking["payment_status"] == "unpaid"


def test_approve_processing_review_returns_202_without_mcp(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]
    dsn = client.app.state.settings.DATABASE_URL
    _sql(
        dsn,
        "UPDATE review_requests SET status='processing', updated_at=now() WHERE id=$1",
        review["id"],
    )

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 202
    assert response.json()["review"]["status"] == "processing"
    assert response.json()["mcp_called"] is False
    assert client.app.state.mcp.pay_calls() == []


def test_member_persisted_and_used_for_review_and_followup(client):
    patient = _patient(client)
    admin = _admin(client)
    member = client.post(
        "/api/v1/family",
        json={"name": "王小明", "relation": "child"},
        headers=patient,
    ).json()["member"]

    allowed = client.post(
        "/api/v1/booking/lock",
        json={"slot_id": SLOT_ID, "member_id": member["id"]},
        headers=patient,
    )
    assert allowed.status_code == 200, allowed.text
    foreign = client.post(
        "/api/v1/booking/orders",
        json={"lock_id": "lock_fake", "member_id": member["id"]},
        headers=admin,
    )
    assert foreign.status_code == 404

    ordered = client.post(
        "/api/v1/booking/orders",
        json={"lock_id": "lock_fake", "member_id": member["id"]},
        headers=patient,
    )
    assert ordered.status_code == 200, ordered.text
    booking = _booking_row(client, "order_mem")
    assert booking["member_id"] == member["id"]
    assert booking["hospital_patient_id"] == f"HP-{member['id']}"
    dsn = client.app.state.settings.DATABASE_URL
    assert (
        _sql(
            dsn,
            "SELECT hospital_patient_id FROM family_members WHERE id=$1",
            member["id"],
            fetch=True,
        )
        == f"HP-{member['id']}"
    )

    review = client.post("/api/v1/booking/orders/order_mem/request-payment", headers=patient).json()["review"]
    queue = client.get("/api/v1/admin/reviews", params={"status": "pending"}, headers=admin).json()["reviews"]
    item = next(entry for entry in queue if entry["id"] == review["id"])
    assert item["member_id"] == member["id"]
    assert item["member_name"] == "王小明"

    assert client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin).status_code == 200
    followup = asyncio.run(_followup(client, "order_mem"))
    assert followup["member_id"] == member["id"]


async def _followup(client, order_id: str) -> dict:
    conn = await asyncpg.connect(client.app.state.settings.DATABASE_URL)
    try:
        return dict(await conn.fetchrow("SELECT * FROM followups WHERE booking_id=$1", order_id))
    finally:
        await conn.close()


def test_stale_processing_is_reclaimed_and_retried(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]
    dsn = client.app.state.settings.DATABASE_URL
    _sql(
        dsn,
        "UPDATE review_requests SET status='processing', updated_at=now() - interval '10 minutes' WHERE id=$1",
        review["id"],
    )

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 200, response.text
    assert response.json()["review"]["status"] == "approved"
    assert response.json()["mcp_called"] is True
    assert len(client.app.state.mcp.pay_calls()) == 1
    assert _booking_row(client)["payment_status"] == "paid"


def test_reject_audit_uses_mcp_order_id(client):
    dsn = client.app.state.settings.DATABASE_URL
    _sql(
        dsn,
        """
        INSERT INTO bookings
            (id, session_id, patient_id, user_id, department, slot_id, order_id, status, payment_status, amount)
        VALUES ('bk_audit', 'bk_audit', 'patient', 'patient', '心血管内科', $1, 'order_audit', 'created', 'unpaid', 50)
        """,
        SLOT_ID,
    )
    patient = _patient(client)
    admin = _admin(client)
    review = client.post("/api/v1/booking/orders/bk_audit/request-payment", headers=patient).json()["review"]
    assert review["payload"]["order_id"] == "order_audit"

    rejected = client.post(
        f"/api/v1/admin/reviews/{review['id']}/reject",
        json={"note": "金额有误"},
        headers=admin,
    )
    assert rejected.status_code == 200, rejected.text

    detail = _sql(dsn, "SELECT detail FROM audit_log WHERE action='payment.reject'", fetch=True)
    if isinstance(detail, str):
        detail = json.loads(detail)
    assert detail["order_id"] == "order_audit"
    assert detail["order_id"] != "bk_audit"


class CancelDuringPayMCP(FakeMCP):
    def __init__(self, settings, dsn: str):
        super().__init__(settings)
        self.dsn = dsn

    async def call_tool(self, name, arguments=None, idempotency_key=None):
        if name != "pay_order":
            return await super().call_tool(name, arguments, idempotency_key)
        args = dict(arguments or {})
        self.calls.append((name, args))
        conn = await asyncpg.connect(self.dsn)
        try:
            await conn.execute(
                "UPDATE bookings SET status='cancelled', updated_at=now() WHERE id='order_rev1'"
            )
            await conn.execute(
                "UPDATE review_requests SET status='rejected', decision_note='订单已取消', updated_at=now() "
                "WHERE ref_id='order_rev1'"
            )
        finally:
            await conn.close()
        return self._ok(
            {
                "order_id": args["order_id"],
                "slot_id": SLOT_ID,
                "patient_id": "patient",
                "amount": 50,
                "status": "booked",
                "payment_status": "paid",
            }
        )


def test_approve_race_sees_cancelled_booking_and_does_not_mark_paid(client):
    _create_booking(client)
    patient = _patient(client)
    admin = _admin(client)
    review = client.post("/api/v1/booking/orders/order_rev1/request-payment", headers=patient).json()["review"]
    client.app.state.mcp = CancelDuringPayMCP(
        client.app.state.settings, client.app.state.settings.DATABASE_URL
    )

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 409
    booking = _booking_row(client)
    assert booking["status"] == "cancelled"
    assert booking["payment_status"] == "unpaid"
    assert _review_rows(client)[0]["status"] == "failed"
    assert len(client.app.state.mcp.pay_calls()) == 1


def test_approve_for_deleted_member_is_blocked(client):
    patient = _patient(client)
    admin = _admin(client)
    member = client.post(
        "/api/v1/family", json={"name": "王小明", "relation": "child"}, headers=patient
    ).json()["member"]
    ordered = client.post(
        "/api/v1/booking/orders",
        json={"lock_id": "lock_deleted", "member_id": member["id"]},
        headers=patient,
    )
    assert ordered.status_code == 200, ordered.text
    order_id = ordered.json()["order"]["order_id"]
    review = client.post(
        f"/api/v1/booking/orders/{order_id}/request-payment", headers=patient
    ).json()["review"]
    dsn = client.app.state.settings.DATABASE_URL
    _sql(dsn, "DELETE FROM family_members WHERE id=$1", member["id"])

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 409
    assert response.json()["detail"] == "member deleted"
    assert _review_rows(client, order_id)[0]["status"] == "failed"
    assert _booking_row(client, order_id)["payment_status"] == "unpaid"
    assert client.app.state.mcp.pay_calls() == []

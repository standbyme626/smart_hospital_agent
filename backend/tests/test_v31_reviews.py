import asyncio

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

    async def call_tool(self, name, arguments=None):
        args = dict(arguments or {})
        self.calls.append((name, args))
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
    _sql(dsn, "UPDATE review_requests SET status='processing' WHERE id=$1", review["id"])

    response = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert response.status_code == 202
    assert response.json()["review"]["status"] == "processing"
    assert response.json()["mcp_called"] is False
    assert client.app.state.mcp.pay_calls() == []

import asyncio
from pathlib import Path

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.main import app

PATIENT_PASSWORD = "patient123456"
ADMIN_PASSWORD = "admin123456"
SLOT_ID = "心血管内科|2032-01-01|09:00|张医生"


class FakeMCP:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.counter = 0
        self.paid: set[str] = set()

    @staticmethod
    def _ok(data):
        return {"is_error": False, "text": "", "data": data}

    async def call_tool(self, name, arguments=None, idempotency_key=None):
        args = dict(arguments or {})
        self.calls.append((name, args))
        if name == "create_order":
            self.counter += 1
            order_id = f"order_child_{self.counter}"
            return self._ok(
                {
                    "order_id": order_id,
                    "slot_id": SLOT_ID,
                    "patient_id": args.get("patient_id"),
                    "amount": 50,
                    "status": "created",
                    "payment_status": "unpaid",
                }
            )
        if name == "pay_order":
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
            return self._ok({"order": {"order_id": args["order_id"]}})
        return self._ok({})


@pytest.fixture()
def client(tmp_path):
    with TestClient(app) as test_client:
        test_client.app.state.settings.AUTH_ENABLED = True
        original_uploads = test_client.app.state.settings.UPLOADS_DIR
        test_client.app.state.settings.UPLOADS_DIR = str(tmp_path / "uploads")
        test_client.app.state.mcp = FakeMCP()
        try:
            yield test_client
        finally:
            test_client.app.state.settings.AUTH_ENABLED = False
            test_client.app.state.settings.UPLOADS_DIR = original_uploads


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


def _dsn(client: TestClient) -> str:
    return client.app.state.settings.DATABASE_URL


def _run(client: TestClient, query: str, *args: object, fetch: bool = True):
    dsn = _dsn(client)

    async def run():
        conn = await asyncpg.connect(dsn)
        try:
            if fetch:
                return await conn.fetchval(query, *args)
            return await conn.execute(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _seed_session(client: TestClient, session_id: str, member_id: str, *, age_minutes: int = 0):
    dsn = _dsn(client)

    async def run():
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                INSERT INTO sessions (id, user_id, member_id, title, updated_at, created_at)
                VALUES ($1, 'patient', $2, $1, now() - make_interval(mins => $3), now() - make_interval(mins => $3))
                """,
                session_id,
                member_id,
                age_minutes,
            )
            await conn.execute(
                "INSERT INTO events (session_id, event_id, type, data) VALUES ($1, $1, 'final', '{}'::jsonb)",
                session_id,
            )
            await conn.execute(
                "INSERT INTO messages (session_id, user_id, member_id, role, content) VALUES ($1, 'patient', $2, 'user', 'hi')",
                session_id,
                member_id,
            )
        finally:
            await conn.close()

    asyncio.run(run())


def test_member_scoped_followups_summary_and_sessions(client):
    headers = _patient(client)
    members = client.get("/api/v1/family", headers=headers).json()["members"]
    self_member = next(item for item in members if item["is_self"])
    assert self_member["hospital_patient_id"] == f"HP-{self_member['id']}"

    child = client.post(
        "/api/v1/family", json={"name": "王小明", "relation": "child"}, headers=headers
    ).json()["member"]
    assert child["hospital_patient_id"] == f"HP-{child['id']}"

    for title, member_id in (("孩子复诊", child["id"]), ("本人体检", self_member["id"])):
        response = client.post(
            "/api/v1/followups", json={"title": title, "member_id": member_id}, headers=headers
        )
        assert response.status_code == 200, response.text

    _seed_session(client, "sess_child", child["id"], age_minutes=1)
    _seed_session(client, "sess_self", self_member["id"], age_minutes=30)

    child_followups = client.get(
        "/api/v1/followups", params={"status": "pending", "member_id": child["id"]}, headers=headers
    )
    assert child_followups.status_code == 200
    assert [item["title"] for item in child_followups.json()["followups"]] == ["孩子复诊"]

    summary = client.get(
        "/api/v1/home/summary", params={"member_id": child["id"]}, headers=headers
    )
    assert summary.status_code == 200, summary.text
    data = summary.json()
    assert data["member"]["id"] == child["id"]
    assert [item["title"] for item in data["followups"]] == ["孩子复诊"]
    assert data["latest_session"]["id"] == "sess_child"

    sessions = client.get(
        "/api/v1/sessions", params={"member_id": child["id"]}, headers=headers
    )
    assert sessions.status_code == 200
    assert [item["id"] for item in sessions.json()["sessions"]] == ["sess_child"]

    allowed = client.get(
        "/api/v1/sessions/sess_child", params={"member_id": child["id"]}, headers=headers
    )
    assert allowed.status_code == 200
    hidden = client.get(
        "/api/v1/sessions/sess_child",
        params={"member_id": self_member["id"]},
        headers=headers,
    )
    assert hidden.status_code == 404

    foreign = client.get(
        "/api/v1/followups", params={"member_id": "mem_missing"}, headers=headers
    )
    assert foreign.status_code == 404


def test_child_booking_writes_member_and_followup(client):
    headers = _patient(client)
    admin = _admin(client)
    child = client.post(
        "/api/v1/family", json={"name": "王小明", "relation": "child"}, headers=headers
    ).json()["member"]

    ordered = client.post(
        "/api/v1/booking/orders",
        json={"lock_id": "lock_child", "member_id": child["id"]},
        headers=headers,
    )
    assert ordered.status_code == 200, ordered.text
    order_id = ordered.json()["order"]["order_id"]

    booking = asyncio.run(_booking_row(client, order_id))
    assert booking["member_id"] == child["id"]
    assert booking["hospital_patient_id"] == f"HP-{child['id']}"

    review = client.post(
        f"/api/v1/booking/orders/{order_id}/request-payment", headers=headers
    ).json()["review"]
    assert review["member_id"] == child["id"]

    approved = client.post(f"/api/v1/admin/reviews/{review['id']}/approve", headers=admin)
    assert approved.status_code == 200, approved.text

    followup = asyncio.run(_followup_row(client, order_id))
    assert followup["member_id"] == child["id"]


async def _booking_row(client: TestClient, order_id: str) -> dict:
    conn = await asyncpg.connect(_dsn(client))
    try:
        return dict(await conn.fetchrow("SELECT * FROM bookings WHERE id=$1", order_id))
    finally:
        await conn.close()


async def _followup_row(client: TestClient, order_id: str) -> dict:
    conn = await asyncpg.connect(_dsn(client))
    try:
        return dict(await conn.fetchrow("SELECT * FROM followups WHERE booking_id=$1", order_id))
    finally:
        await conn.close()


def test_delete_member_cascades_rows_events_and_files(client):
    headers = _patient(client)
    members = client.get("/api/v1/family", headers=headers).json()["members"]
    self_member = next(item for item in members if item["is_self"])
    child = client.post(
        "/api/v1/family", json={"name": "王小明", "relation": "child"}, headers=headers
    ).json()["member"]

    uploads = Path(client.app.state.settings.UPLOADS_DIR) / "reports"
    uploads.mkdir(parents=True, exist_ok=True)
    image = uploads / "rep_cascade.jpg"
    image.write_bytes(b"fake-image")

    dsn = _dsn(client)

    async def seed():
        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                "INSERT INTO medications (id, user_id, member_id, name) VALUES ('med_child', 'patient', $1, '布洛芬')",
                child["id"],
            )
            await conn.execute(
                """
                INSERT INTO reports (id, user_id, member_id, title, image_path, summary)
                VALUES ('rep_cascade', 'patient', $1, '血常规', $2, 'ok')
                """,
                child["id"],
                str(image),
            )
            await conn.execute(
                "INSERT INTO followups (id, user_id, member_id, title) VALUES ('fu_child', 'patient', $1, '复诊')",
                child["id"],
            )
            await conn.execute(
                """
                INSERT INTO health_events (user_id, member_id, kind, title, detail)
                VALUES ('patient', $1, 'report', '报告解读', '{"report_id": "rep_cascade"}'::jsonb)
                """,
                child["id"],
            )
            await conn.execute(
                "INSERT INTO health_events (user_id, member_id, kind, title) VALUES ('patient', $1, 'consult', '问诊')",
                child["id"],
            )
            await conn.execute(
                """
                INSERT INTO health_events (user_id, member_id, kind, title, detail)
                VALUES ('patient', NULL, 'report', '孤儿报告事件', '{"report_id": "rep_cascade"}'::jsonb)
                """
            )
            await conn.execute(
                "INSERT INTO health_events (user_id, member_id, kind, title) VALUES ('patient', $1, 'consult', '本人问诊')",
                self_member["id"],
            )
            await conn.execute(
                "INSERT INTO sessions (id, user_id, member_id, title) VALUES ('sess_cascade', 'patient', $1, '孩子问诊')",
                child["id"],
            )
            await conn.execute(
                "INSERT INTO events (session_id, event_id, type, data) VALUES ('sess_cascade', 'e1', 'final', '{}'::jsonb)"
            )
            await conn.execute(
                "INSERT INTO messages (session_id, user_id, member_id, role, content) VALUES ('sess_cascade', 'patient', $1, 'user', 'hi')",
                child["id"],
            )
            await conn.execute(
                "INSERT INTO sessions (id, user_id, member_id, title) VALUES ('sess_self_keep', 'patient', $1, '本人问诊')",
                self_member["id"],
            )
        finally:
            await conn.close()

    asyncio.run(seed())

    response = client.delete(f"/api/v1/family/{child['id']}", headers=headers)
    assert response.status_code == 200, response.text
    assert response.json() == {"ok": True}

    counts = {
        "members": _run(client, "SELECT count(*) FROM family_members WHERE id=$1", child["id"]),
        "medications": _run(
            client, "SELECT count(*) FROM medications WHERE member_id=$1", child["id"]
        ),
        "reports": _run(client, "SELECT count(*) FROM reports WHERE member_id=$1", child["id"]),
        "followups": _run(client, "SELECT count(*) FROM followups WHERE member_id=$1", child["id"]),
        "health_events": _run(
            client, "SELECT count(*) FROM health_events WHERE member_id=$1", child["id"]
        ),
        "report_events": _run(
            client,
            "SELECT count(*) FROM health_events WHERE detail->>'report_id'='rep_cascade'",
        ),
        "sessions": _run(
            client, "SELECT count(*) FROM sessions WHERE member_id=$1", child["id"]
        ),
        "events": _run(
            client, "SELECT count(*) FROM events WHERE session_id='sess_cascade'"
        ),
        "messages": _run(
            client, "SELECT count(*) FROM messages WHERE session_id='sess_cascade'"
        ),
    }
    assert counts == {key: 0 for key in counts}
    assert not image.exists()

    assert _run(client, "SELECT count(*) FROM sessions WHERE id='sess_self_keep'") == 1
    assert (
        _run(
            client,
            "SELECT count(*) FROM health_events WHERE member_id=$1",
            self_member["id"],
        )
        == 1
    )
    assert _run(client, "SELECT count(*) FROM audit_log WHERE action='family.delete'") == 1


def test_delete_self_member_blocked(client):
    headers = _patient(client)
    members = client.get("/api/v1/family", headers=headers).json()["members"]
    self_member = next(item for item in members if item["is_self"])

    response = client.delete(f"/api/v1/family/{self_member['id']}", headers=headers)
    assert response.status_code == 400
    assert _run(client, "SELECT count(*) FROM family_members WHERE id=$1", self_member["id"]) == 1

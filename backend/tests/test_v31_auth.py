import asyncio
import json
from types import SimpleNamespace

import asyncpg
import pytest
from fastapi.testclient import TestClient

import app.security as security
from app.api.deps import create_token
from app.main import app
from app.model_gateway import ChatResult

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"
USER_PASSWORD = "secret123"


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        test_client.app.state.settings.AUTH_ENABLED = True
        try:
            yield test_client
        finally:
            test_client.app.state.settings.AUTH_ENABLED = False


def _login(client: TestClient, username: str, password: str):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _token(client: TestClient, username: str, password: str) -> str:
    response = _login(client, username, password)
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _admin_headers(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


def _create_user(
    client: TestClient,
    username: str,
    password: str = USER_PASSWORD,
    role: str = "patient",
) -> dict:
    response = client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": password, "role": role},
        headers=_admin_headers(client),
    )
    assert response.status_code == 201, response.text
    return response.json()["user"]


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


def test_login_correct_and_wrong_password(client):
    assert _login(client, "admin", "wrong-password").status_code == 401

    response = _login(client, "admin", ADMIN_PASSWORD)
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "admin"
    assert body["access_token"]

    me = client.get("/api/v1/auth/me", headers=_auth(body["access_token"]))
    assert me.status_code == 200
    assert me.json()["username"] == "admin"
    assert me.json()["role"] == "admin"


def test_disabled_user_cannot_login_or_use_token(client):
    _create_user(client, "u_disabled")
    token = _token(client, "u_disabled", USER_PASSWORD)

    response = client.patch(
        "/api/v1/admin/users/u_disabled",
        json={"enabled": False},
        headers=_admin_headers(client),
    )
    assert response.status_code == 200
    assert response.json()["user"]["enabled"] is False

    assert _login(client, "u_disabled", USER_PASSWORD).status_code == 403
    assert client.get("/api/v1/auth/me", headers=_auth(token)).status_code == 401


def test_expired_token_rejected(client):
    _create_user(client, "u_exp")
    settings = client.app.state.settings
    original_ttl = settings.JWT_TTL_MINUTES
    settings.JWT_TTL_MINUTES = -1
    try:
        token = create_token(settings, "u_exp", "patient")
    finally:
        settings.JWT_TTL_MINUTES = original_ttl

    response = client.get("/api/v1/auth/me", headers=_auth(token))
    assert response.status_code == 401
    assert response.json()["detail"] == "token expired"


def test_login_lockout_after_five_failures_and_success_resets(client):
    _create_user(client, "u_lock", "rightpass")

    for _ in range(4):
        assert _login(client, "u_lock", "badpass").status_code == 401
    assert _login(client, "u_lock", "rightpass").status_code == 200

    for _ in range(4):
        assert _login(client, "u_lock", "badpass").status_code == 401
    assert _login(client, "u_lock", "badpass").status_code == 423
    assert _login(client, "u_lock", "rightpass").status_code == 423


def test_admin_user_create_and_no_hash_leak(client):
    headers = _admin_headers(client)
    created = client.post(
        "/api/v1/admin/users",
        json={"username": "u_admin_made", "password": USER_PASSWORD},
        headers=headers,
    )
    assert created.status_code == 201
    user = created.json()["user"]
    assert user["username"] == "u_admin_made"
    assert "password_hash" not in user

    duplicate = client.post(
        "/api/v1/admin/users",
        json={"username": "u_admin_made", "password": USER_PASSWORD},
        headers=headers,
    )
    assert duplicate.status_code == 409

    listed = client.get("/api/v1/admin/users", headers=headers)
    assert listed.status_code == 200
    users = listed.json()["users"]
    assert {"admin", "patient", "u_admin_made"} <= {item["username"] for item in users}
    assert all("password_hash" not in item for item in users)

    patient_headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    assert client.get("/api/v1/admin/users", headers=patient_headers).status_code == 403


def test_bootstrap_idempotent_and_marker_prevents_reseed():
    with TestClient(app) as first:
        settings = first.app.state.settings
        settings.AUTH_ENABLED = True
        headers = _admin_headers(first)
        users = first.get("/api/v1/admin/users", headers=headers).json()["users"]
        assert {item["username"] for item in users} == {"admin", "patient"}
        dsn = settings.DATABASE_URL
        hash_before = _sql(dsn, "SELECT password_hash FROM users WHERE username='admin'", fetch=True)
        marker = _sql(dsn, "SELECT value FROM app_meta WHERE key='users_bootstrapped'", fetch=True)
        assert marker == "true"

    with TestClient(app) as second:
        second.app.state.settings.AUTH_ENABLED = True
        assert _login(second, "admin", ADMIN_PASSWORD).status_code == 200
        assert _sql(dsn, "SELECT count(*) FROM users", fetch=True) == 2
        assert _sql(dsn, "SELECT password_hash FROM users WHERE username='admin'", fetch=True) == hash_before

    _sql(dsn, "DELETE FROM users")

    with TestClient(app) as third:
        third.app.state.settings.AUTH_ENABLED = True
        assert _sql(dsn, "SELECT count(*) FROM users", fetch=True) == 0
        assert _login(third, "admin", ADMIN_PASSWORD).status_code == 401
        assert _login(third, "patient", PATIENT_PASSWORD).status_code == 401


def test_cross_user_isolation(client):
    _create_user(client, "user_a")
    _create_user(client, "user_b")
    headers_a = _auth(_token(client, "user_a", USER_PASSWORD))
    headers_b = _auth(_token(client, "user_b", USER_PASSWORD))
    dsn = client.app.state.settings.DATABASE_URL

    response = client.post(
        "/api/v1/family",
        json={"name": "A 的父亲", "relation": "father"},
        headers=headers_a,
    )
    assert response.status_code == 200
    member_id = response.json()["member"]["id"]

    _sql(
        dsn,
        "INSERT INTO reports (id, user_id, member_id, title) VALUES ($1,$2,$3,$4)",
        "rep_iso_a",
        "user_a",
        member_id,
        "A 的报告",
    )
    _sql(
        dsn,
        "INSERT INTO followups (id, user_id, member_id, title) VALUES ($1,$2,$3,$4)",
        "fu_iso_a",
        "user_a",
        member_id,
        "A 随访",
    )
    _sql(
        dsn,
        "INSERT INTO sessions (id, user_id, title) VALUES ($1,$2,$3)",
        "sess_iso_a",
        "user_a",
        "A 会话",
    )
    _sql(
        dsn,
        "INSERT INTO bookings (id, session_id, patient_id, user_id, department) VALUES ($1,$2,$3,$4,$5)",
        "order_iso_a",
        "sess_iso_a",
        "user_a",
        "user_a",
        "心血管内科",
    )
    _sql(
        dsn,
        "INSERT INTO bookings (id, patient_id, department) VALUES ($1,$2,$3)",
        "order_legacy_b",
        "user_b",
        "内科",
    )

    assert client.get("/api/v1/sessions/sess_iso_a/events", headers=headers_a).status_code == 200
    assert client.get("/api/v1/sessions/sess_iso_a/events", headers=headers_b).status_code == 404
    assert (
        client.post(
            "/api/v1/chat/stream",
            json={"message": "hi", "session_id": "sess_iso_a"},
            headers=headers_b,
        ).status_code
        == 404
    )

    assert client.get("/api/v1/reports/rep_iso_a", headers=headers_a).status_code == 200
    assert client.get("/api/v1/reports/rep_iso_a", headers=headers_b).status_code == 404

    assert client.get("/api/v1/medications", params={"member_id": member_id}, headers=headers_b).status_code == 404
    assert (
        client.post(
            "/api/v1/medications",
            json={"name": "药", "member_id": member_id},
            headers=headers_b,
        ).status_code
        == 404
    )
    assert (
        client.get("/api/v1/health/timeline", params={"member_id": member_id}, headers=headers_b).status_code
        == 404
    )

    family_a = client.get("/api/v1/family", headers=headers_a).json()["members"]
    family_b = client.get("/api/v1/family", headers=headers_b).json()["members"]
    assert member_id in {item["id"] for item in family_a}
    assert member_id not in {item["id"] for item in family_b}

    followups_b = client.get("/api/v1/followups", headers=headers_b).json()["followups"]
    assert followups_b == []

    assert client.get("/api/v1/booking/orders/order_legacy_b", headers=headers_b).status_code == 404
    assert client.get("/api/v1/booking/orders/order_iso_a", headers=headers_b).status_code == 404
    assert client.post("/api/v1/booking/orders/order_iso_a/cancel", headers=headers_b).status_code == 404
    assert (
        client.post(
            "/api/v1/booking/orders/order_iso_a/pay",
            json={"confirm_token": "x"},
            headers=headers_b,
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/v1/services/visit-guide",
            params={"order_id": "order_iso_a"},
            headers=headers_b,
        ).status_code
        == 404
    )

    assert (
        client.post(
            "/api/v1/chat/stream",
            json={"message": "hi", "user_id": "user_b"},
            headers=headers_a,
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/booking/lock",
            json={"slot_id": "slot|1", "patient_id": "user_b"},
            headers=headers_a,
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/booking/orders",
            json={"lock_id": "lock_1", "patient_id": "user_b"},
            headers=headers_a,
        ).status_code
        == 422
    )


def test_login_username_length_capped(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "a" * 65, "password": "x"},
    )
    assert response.status_code == 422


def test_lockout_uses_username_only_ignoring_forwarded_for(client):
    _create_user(client, "u_xff", "rightpass")
    for index in range(5):
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "u_xff", "password": "badpass"},
            headers={"X-Forwarded-For": f"203.0.113.{index}", "X-Real-IP": f"198.51.100.{index}"},
        )
    assert response.status_code == 423

    response = client.post(
        "/api/v1/auth/login",
        json={"username": "u_xff", "password": "rightpass"},
        headers={"X-Forwarded-For": "192.0.2.55", "X-Real-IP": "192.0.2.55"},
    )
    assert response.status_code == 423


def test_lockout_sliding_window_expires(client, monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(security, "time", SimpleNamespace(time=lambda: clock.now))
    _create_user(client, "u_window", "rightpass")

    for _ in range(4):
        assert _login(client, "u_window", "badpass").status_code == 401
    clock.now += security.LOCKOUT_SECONDS + 1

    for _ in range(4):
        assert _login(client, "u_window", "badpass").status_code == 401
    assert _login(client, "u_window", "badpass").status_code == 423

    clock.now += security.LOCKOUT_SECONDS + 1
    assert _login(client, "u_window", "rightpass").status_code == 200


def test_unknown_user_still_verifies_dummy_hash(client, monkeypatch):
    import app.routes as routes_module

    seen: list[str] = []
    real_verify = routes_module.verify_password

    def spy(password: str, encoded: str) -> bool:
        seen.append(encoded)
        return real_verify(password, encoded)

    monkeypatch.setattr(routes_module, "verify_password", spy)

    for _ in range(4):
        assert _login(client, "no_such_user", "whatever123").status_code == 401
    assert _login(client, "no_such_user", "whatever123").status_code == 423
    assert len(seen) == 5
    assert all(encoded == security.DUMMY_PASSWORD_HASH for encoded in seen)


def test_admin_patch_role_password_and_empty_patch(client):
    _create_user(client, "u_patch", "oldpass1")
    headers = _admin_headers(client)

    empty = client.patch("/api/v1/admin/users/u_patch", json={}, headers=headers)
    assert empty.status_code == 400

    promoted = client.patch("/api/v1/admin/users/u_patch", json={"role": "admin"}, headers=headers)
    assert promoted.status_code == 200
    assert promoted.json()["user"]["role"] == "admin"
    assert "password_hash" not in promoted.json()["user"]

    rotated = client.patch(
        "/api/v1/admin/users/u_patch",
        json={"password": "newpass1"},
        headers=headers,
    )
    assert rotated.status_code == 200
    assert "password_hash" not in rotated.json()["user"]

    assert _login(client, "u_patch", "oldpass1").status_code == 401
    assert _login(client, "u_patch", "newpass1").status_code == 200


def test_last_enabled_admin_cannot_be_disabled_or_demoted(client):
    headers = _admin_headers(client)
    assert client.patch("/api/v1/admin/users/admin", json={"enabled": False}, headers=headers).status_code == 400
    assert client.patch("/api/v1/admin/users/admin", json={"role": "patient"}, headers=headers).status_code == 400

    _create_user(client, "u_second_admin", role="admin")
    disabled = client.patch("/api/v1/admin/users/admin", json={"enabled": False}, headers=headers)
    assert disabled.status_code == 200

    headers2 = _auth(_token(client, "u_second_admin", USER_PASSWORD))
    assert client.patch("/api/v1/admin/users/u_second_admin", json={"role": "patient"}, headers=headers2).status_code == 400
    assert client.patch("/api/v1/admin/users/u_second_admin", json={"enabled": False}, headers=headers2).status_code == 400
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401


def test_load_user_returns_503_when_database_unavailable(client, monkeypatch):
    token = _token(client, "admin", ADMIN_PASSWORD)
    monkeypatch.setattr(client.app.state, "db", SimpleNamespace(pool=None))

    response = client.get("/api/v1/auth/me", headers=_auth(token))
    assert response.status_code == 503
    assert _login(client, "admin", ADMIN_PASSWORD).status_code == 503


class FakeChatAgent:
    async def stream(self, **kwargs):
        yield {
            "seq": 1,
            "type": "final",
            "data": {
                "report": {"department": "全科", "urgency": "normal", "summary": "建议休息。"},
                "summary": "建议休息。",
            },
        }


class FakeGateway:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> ChatResult:
        task = kwargs.get("task", "diagnosis")
        self.calls.append({"task": task})
        if task == "triage":
            content = json.dumps(
                {
                    "intent": "MEDICAL_CONSULT",
                    "department": "全科",
                    "urgency": "normal",
                    "symptoms": "头痛",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )
        else:
            content = "建议充分休息并观察。"
        return ChatResult(content=content, provider="fake", model="fake")


class FakeRetrieval:
    async def search(self, query: str, **kwargs) -> list[dict]:
        return []


def _chat(client: TestClient, message: str, session_id: str, headers: dict[str, str]):
    return client.post(
        "/api/v1/chat/stream",
        json={"message": message, "session_id": session_id},
        headers=headers,
    )


def test_chat_quota(client):
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    client.app.state.agent = FakeChatAgent()
    settings = client.app.state.settings
    original_limit = settings.CHAT_DAILY_LIMIT
    settings.CHAT_DAILY_LIMIT = 2
    dsn = settings.DATABASE_URL
    try:
        assert _chat(client, "头痛", "sess_quota_1", headers).status_code == 200
        assert _chat(client, "头痛", "sess_quota_2", headers).status_code == 200
        used = _sql(
            dsn,
            "SELECT count(*) FROM messages WHERE user_id='patient' AND role='user'",
            fetch=True,
        )
        assert used == 2

        over = _chat(client, "头痛", "sess_quota_3", headers)
        assert over.status_code == 429
        assert "今日问诊次数已达上限" in over.json()["detail"]
        assert (
            _sql(
                dsn,
                "SELECT count(*) FROM messages WHERE session_id='sess_quota_3'",
                fetch=True,
            )
            == 0
        )
        assert (
            _sql(
                dsn,
                "SELECT count(*) FROM messages WHERE user_id='patient' AND role='user'",
                fetch=True,
            )
            == 2
        )

        other = _auth(_token(client, "admin", ADMIN_PASSWORD))
        assert _chat(client, "头痛", "sess_quota_admin", other).status_code == 200
    finally:
        settings.CHAT_DAILY_LIMIT = original_limit


def test_chat_quota_kill_switch_when_zero(client):
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    client.app.state.agent = FakeChatAgent()
    settings = client.app.state.settings
    original_limit = settings.CHAT_DAILY_LIMIT
    settings.CHAT_DAILY_LIMIT = 0
    dsn = settings.DATABASE_URL
    _sql(
        dsn,
        """
        INSERT INTO messages (session_id, user_id, role, content)
        SELECT 'sess_kill', 'patient', 'user', 'm'
        FROM generate_series(1, 65)
        """,
    )
    try:
        assert _chat(client, "头痛", "sess_kill_1", headers).status_code == 200
        assert _chat(client, "头痛", "sess_kill_2", headers).status_code == 200
    finally:
        settings.CHAT_DAILY_LIMIT = original_limit


def test_chat_quota_ignores_previous_day_messages(client):
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    client.app.state.agent = FakeChatAgent()
    settings = client.app.state.settings
    original_limit = settings.CHAT_DAILY_LIMIT
    settings.CHAT_DAILY_LIMIT = 1
    dsn = settings.DATABASE_URL
    _sql(
        dsn,
        """
        INSERT INTO messages (session_id, user_id, role, content, created_at)
        VALUES ('sess_old', 'patient', 'user', 'old', now() - interval '2 days')
        """,
    )
    try:
        assert _chat(client, "头痛", "sess_today_1", headers).status_code == 200
        over = _chat(client, "头痛", "sess_today_2", headers)
        assert over.status_code == 429
        assert "今日问诊次数已达上限" in over.json()["detail"]
    finally:
        settings.CHAT_DAILY_LIMIT = original_limit


def test_model_calls_persist_user_id(client):
    headers = _auth(_token(client, "patient", PATIENT_PASSWORD))
    gateway = FakeGateway()
    client.app.state.agent.gateway = gateway
    client.app.state.agent.retrieval = FakeRetrieval()
    client.app.state.gateway = gateway
    client.app.state.retrieval = client.app.state.agent.retrieval

    response = _chat(client, "头痛两天了", "sess_model_calls", headers)
    assert response.status_code == 200
    assert "event: final" in response.text
    assert {call["task"] for call in gateway.calls} == {"triage", "diagnosis"}

    async def fetch_rows():
        conn = await asyncpg.connect(client.app.state.settings.DATABASE_URL)
        try:
            return await conn.fetch("SELECT task, user_id FROM model_calls ORDER BY id")
        finally:
            await conn.close()

    rows = asyncio.run(fetch_rows())
    assert rows
    assert {row["task"] for row in rows} == {"triage", "diagnosis"}
    assert {row["user_id"] for row in rows} == {"patient"}

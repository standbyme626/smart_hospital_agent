import asyncio

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.api.deps import create_token
from app.main import app

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


def _create_user(client: TestClient, username: str, password: str = USER_PASSWORD) -> dict:
    response = client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": password, "role": "patient"},
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


def test_bootstrap_idempotent_and_never_recreates_deleted_user():
    with TestClient(app) as first:
        settings = first.app.state.settings
        settings.AUTH_ENABLED = True
        headers = _admin_headers(first)
        users = first.get("/api/v1/admin/users", headers=headers).json()["users"]
        assert {item["username"] for item in users} == {"admin", "patient"}
        dsn = settings.DATABASE_URL
        hash_before = _sql(dsn, "SELECT password_hash FROM users WHERE id='admin'", fetch=True)

    with TestClient(app) as second:
        second.app.state.settings.AUTH_ENABLED = True
        assert _login(second, "admin", ADMIN_PASSWORD).status_code == 200
        assert _sql(dsn, "SELECT count(*) FROM users", fetch=True) == 2
        assert _sql(dsn, "SELECT password_hash FROM users WHERE id='admin'", fetch=True) == hash_before

    _sql(dsn, "DELETE FROM users WHERE id='admin'")

    with TestClient(app) as third:
        third.app.state.settings.AUTH_ENABLED = True
        assert _login(third, "admin", ADMIN_PASSWORD).status_code == 401
        assert _login(third, "patient", PATIENT_PASSWORD).status_code == 200


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

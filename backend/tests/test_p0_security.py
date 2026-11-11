import asyncio
import hashlib
import hmac
import io
import json
import zipfile
from pathlib import Path

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import SCHEMA_VERSION, Database
from app.main import app
from app.security import pseudonym_tag

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"
USER_PASSWORD = "secret123"

NEW_TABLES = (
    "his_patient_map",
    "his_encounter_map",
    "his_prescription_map",
    "sync_outbox",
    "reconciliation_finding",
    "his_sync_cursor",
    "human_tasks",
    "consent_record",
    "notifications",
    "schema_migrations",
)

NEW_COLUMNS = (
    ("users", "token_version"),
    ("users", "consent_version"),
    ("users", "consent_at"),
    ("family_members", "id_card_hash"),
    ("family_members", "id_card_last4"),
    ("family_members", "his_link_status"),
    ("bookings", "his_encounter_id"),
    ("bookings", "legacy_demo"),
    ("bookings", "last_synced_at"),
    ("review_requests", "sla_due_at"),
    ("review_requests", "assignee"),
    ("medications", "his_prescription_no"),
    ("reports", "critical_flags"),
    ("health_events", "deep_link"),
    ("sessions", "report_state"),
    ("messages", "member_context"),
)


class FakeMCP:
    def __init__(self) -> None:
        self.paid: set[str] = set()

    async def call_tool(self, name, arguments=None, idempotency_key=None):
        args = dict(arguments or {})
        if name == "pay_order":
            self.paid.add(args["order_id"])
            return {
                "is_error": False,
                "text": "",
                "data": {
                    "order_id": args["order_id"],
                    "status": "booked",
                    "payment_status": "paid",
                    "amount": 50,
                },
            }
        return {"is_error": False, "text": "", "data": {}}


@pytest.fixture()
def client(tmp_path):
    with TestClient(app) as test_client:
        state = test_client.app.state
        state.settings.AUTH_ENABLED = True
        state.mcp = FakeMCP()
        original_uploads = state.settings.UPLOADS_DIR
        state.settings.UPLOADS_DIR = str(tmp_path / "uploads")
        try:
            yield test_client
        finally:
            state.settings.AUTH_ENABLED = False
            state.settings.UPLOADS_DIR = original_uploads


def _dsn(client: TestClient) -> str:
    return client.app.state.settings.DATABASE_URL


def _sql(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.execute(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _fetch(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.fetch(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _fetchval(client: TestClient, query: str, *args):
    async def run():
        conn = await asyncpg.connect(_dsn(client))
        try:
            return await conn.fetchval(query, *args)
        finally:
            await conn.close()

    return asyncio.run(run())


def _login(client: TestClient, username: str, password: str):
    return client.post("/api/v1/auth/login", json={"username": username, "password": password})


def _token(client: TestClient, username: str, password: str) -> str:
    response = _login(client, username, password)
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _admin(client: TestClient) -> dict[str, str]:
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


def _create_user(client: TestClient, username: str, password: str = USER_PASSWORD) -> dict[str, str]:
    response = client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": password, "role": "patient"},
        headers=_admin(client),
    )
    assert response.status_code == 201, response.text
    return _auth(_token(client, username, password))


def _jsonb(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def test_schema_init_is_idempotent_and_complete(client):
    settings = client.app.state.settings

    async def init_twice():
        db = Database(settings)
        await db.init_schema()
        await db.init_schema()
        await db.close()

    asyncio.run(init_twice())
    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM schema_migrations WHERE version=$1",
            SCHEMA_VERSION,
        )
        == 1
    )
    for table in NEW_TABLES:
        assert _fetchval(client, "SELECT to_regclass($1) IS NOT NULL", f"public.{table}")
    for table, column in NEW_COLUMNS:
        assert (
            _fetchval(
                client,
                "SELECT count(*) FROM information_schema.columns "
                "WHERE table_name=$1 AND column_name=$2",
                table,
                column,
            )
            == 1
        )

    _sql(
        client,
        "INSERT INTO review_requests (id, kind, ref_id, user_id, status) "
        "VALUES ('rev_kind_ok', 'clinical_review', 'ref_1', 'patient', 'acknowledged')",
    )
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        _sql(
            client,
            "INSERT INTO review_requests (id, kind, ref_id, user_id) "
            "VALUES ('rev_kind_bad', 'unknown_kind', 'ref_2', 'patient')",
        )


def test_ready_endpoint_reports_schema(client):
    ready = client.get("/api/ready")
    assert ready.status_code == 200, ready.text
    body = ready.json()
    assert body["ok"] is True
    assert body["db"] is True
    assert body["schema_version"] == SCHEMA_VERSION

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True


def test_config_rejects_auth_disabled_in_production():
    with pytest.raises(RuntimeError):
        Settings(ENV="production", AUTH_ENABLED=False).validate_security()
    with pytest.raises(RuntimeError):
        Settings(ENV="dev", APP_ENV="production", AUTH_ENABLED=False).validate_security()
    Settings(ENV="production", AUTH_ENABLED=True).validate_security()


def test_pseudonym_is_keyed_hmac_not_plain_sha256(client):
    settings = client.app.state.settings
    original = settings.PSEUDONYM_KEY
    settings.PSEUDONYM_KEY = "unit-test-pseudonym-key"
    try:
        tag = pseudonym_tag(settings, "patient")
        expected = hmac.new(
            b"unit-test-pseudonym-key", b"patient", hashlib.sha256
        ).hexdigest()[:12]
        assert tag == f"deleted-{expected}"
        assert tag != "deleted-" + hashlib.sha256(b"patient").hexdigest()[:12]
        assert pseudonym_tag(settings, "patient") == tag
        assert pseudonym_tag(settings, "other-patient") != tag
    finally:
        settings.PSEUDONYM_KEY = original


def test_account_delete_redacts_audit_analytics_and_model_calls(client):
    user = "p0_delete_me"
    _create_user(client, user)
    settings = client.app.state.settings
    original_key = settings.PSEUDONYM_KEY
    settings.PSEUDONYM_KEY = "p0-delete-key"

    image = Path(settings.UPLOADS_DIR) / "reports" / "rep_p0.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8p0-delete")

    _sql(
        client,
        "INSERT INTO reports (id, user_id, title, image_path) VALUES ($1,$2,$3,$4)",
        "rep_p0",
        user,
        "血常规",
        str(image),
    )
    _sql(
        client,
        "INSERT INTO audit_log (actor, session_id, action, detail) VALUES ($1,$2,$3,$4::jsonb)",
        user,
        user,
        "chat.completed",
        _jsonb(
            {
                "report": {"summary": "胸痛"},
                "triage": {"department": "心血管内科"},
                "symptoms": "胸痛三天",
                "name": "张三",
                "order_id": "order_p0",
            }
        ),
    )
    _sql(
        client,
        "INSERT INTO analytics_events (user_id, session_id, name, props) "
        "VALUES ($1,$2,$3,$4::jsonb)",
        user,
        "sess_p0",
        "consult_start",
        _jsonb({"symptoms": "胸痛", "name": "张三", "member_id": "mem_p0"}),
    )
    _sql(
        client,
        "INSERT INTO model_calls (provider, model, task, user_id, session_id) "
        "VALUES ('fake-p0', 'fake-p0', 'diagnosis', $1, 'sess_p0')",
        user,
    )

    expected_tag = pseudonym_tag(settings, user)
    try:
        deleted = client.request(
            "DELETE",
            "/api/v1/me",
            json={"password": USER_PASSWORD},
            headers=_auth(_token(client, user, USER_PASSWORD)),
        )
    finally:
        settings.PSEUDONYM_KEY = original_key

    assert deleted.status_code == 200, deleted.text
    assert not image.exists()
    assert _fetchval(client, "SELECT count(*) FROM reports WHERE user_id=$1", user) == 0

    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM audit_log "
            "WHERE detail ? 'report' OR detail ? 'triage' OR detail ? 'symptoms'",
        )
        == 0
    )
    actor = _fetchval(
        client, "SELECT actor FROM audit_log WHERE action='chat.completed'"
    )
    assert actor == expected_tag
    assert actor != "deleted-" + hashlib.sha256(user.encode("utf-8")).hexdigest()[:12]

    props = _fetchval(client, "SELECT props FROM analytics_events WHERE name='consult_start'")
    assert json.loads(props) == {}

    model_user = _fetchval(client, "SELECT user_id FROM model_calls WHERE provider='fake-p0'")
    assert model_user is None
    model_session = _fetchval(
        client, "SELECT session_id FROM model_calls WHERE provider='fake-p0'"
    )
    assert model_session == expected_tag


def test_family_delete_redacts_member_analytics_and_audit(client):
    headers = _create_user(client, "p0_family")
    user = "p0_family"
    member = client.post(
        "/api/v1/family",
        json={"name": "家人", "relation": "father"},
        headers=headers,
    ).json()["member"]

    _sql(
        client,
        "INSERT INTO sessions (id, user_id, member_id, title) VALUES ($1,$2,$3,$4)",
        "sess_p0_member",
        user,
        member["id"],
        "家人会话",
    )
    _sql(
        client,
        "INSERT INTO analytics_events (user_id, session_id, name, props) "
        "VALUES ($1,$2,$3,$4::jsonb)",
        user,
        "sess_p0_member",
        "consult_start",
        _jsonb({"member_id": member["id"], "symptoms": "咳嗽", "name": "家人"}),
    )
    _sql(
        client,
        "INSERT INTO audit_log (actor, session_id, action, detail) VALUES ($1,$2,$3,$4::jsonb)",
        user,
        "sess_p0_member",
        "chat.completed",
        _jsonb({"member_id": member["id"], "report": {"summary": "咳嗽"}, "triage": {"x": 1}}),
    )

    response = client.delete(f"/api/v1/family/{member['id']}", headers=headers)
    assert response.status_code == 200, response.text

    props = _fetchval(
        client, "SELECT props FROM analytics_events WHERE session_id='sess_p0_member'"
    )
    props = json.loads(props)
    assert "member_id" not in props
    assert "symptoms" not in props
    assert "name" not in props

    assert _fetchval(client, "SELECT count(*) FROM audit_log WHERE detail ? 'member_id'") == 0
    assert _fetchval(client, "SELECT count(*) FROM audit_log WHERE detail ? 'report'") == 0
    assert _fetchval(client, "SELECT count(*) FROM audit_log WHERE detail ? 'triage'") == 0


def test_admin_actions_are_audited_and_password_reset_revokes_tokens(client):
    headers = _create_user(client, "p0_reset")
    old_token = headers["Authorization"].split(" ", 1)[1]
    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM audit_log WHERE action='admin.user.create' "
            "AND detail->>'target_user'='p0_reset'",
        )
        >= 1
    )

    changed = client.patch(
        "/api/v1/admin/users/p0_reset",
        json={"password": "newpass123"},
        headers=_admin(client),
    )
    assert changed.status_code == 200, changed.text
    assert (
        _fetchval(client, "SELECT count(*) FROM audit_log WHERE action='admin.user.password_reset'")
        >= 1
    )

    revoked = client.get("/api/v1/auth/me", headers=_auth(old_token))
    assert revoked.status_code == 401
    assert revoked.json()["detail"] == "token revoked"

    refreshed = _login(client, "p0_reset", "newpass123")
    assert refreshed.status_code == 200
    assert client.get(
        "/api/v1/auth/me", headers=_auth(refreshed.json()["access_token"])
    ).status_code == 200


def test_audit_read_redacts_medical_detail(client):
    _sql(
        client,
        "INSERT INTO audit_log (actor, session_id, action, detail) VALUES ($1,$2,$3,$4::jsonb)",
        "patient",
        "sess_audit_p0",
        "chat.completed",
        _jsonb(
            {
                "report": {"summary": "全文"},
                "triage": {"department": "全科"},
                "symptoms": "头痛",
                "order_id": "order_p0",
            }
        ),
    )
    response = client.get("/api/v1/admin/audit", headers=_admin(client))
    assert response.status_code == 200, response.text
    item = next(
        entry for entry in response.json()["audit"] if entry["action"] == "chat.completed"
    )
    detail = item["detail"]
    assert "report" not in detail
    assert "triage" not in detail
    assert "symptoms" not in detail
    assert detail["order_id"] == "order_p0"


def test_approve_requires_confirmation_in_production(client):
    _sql(
        client,
        """
        INSERT INTO bookings
            (id, session_id, patient_id, user_id, department, slot_id, order_id,
             status, payment_status, amount)
        VALUES ('bk_p0', 'bk_p0', 'patient', 'patient', '心血管内科', 'slot|1',
                'order_p0', 'created', 'unpaid', 50)
        """,
    )
    _sql(
        client,
        """
        INSERT INTO review_requests (id, kind, ref_id, user_id, status, payload)
        VALUES ('rev_p0', 'payment', 'bk_p0', 'patient', 'pending', $1::jsonb)
        """,
        _jsonb({"order_id": "order_p0", "amount": 50}),
    )

    settings = client.app.state.settings
    original_env = settings.ENV
    original_app_env = settings.APP_ENV
    settings.ENV = "production"
    settings.APP_ENV = ""
    try:
        admin = _admin(client)
        missing = client.post("/api/v1/admin/reviews/rev_p0/approve", headers=admin)
        assert missing.status_code == 400, missing.text

        approved = client.post(
            "/api/v1/admin/reviews/rev_p0/approve",
            json={"confirm": True},
            headers=admin,
        )
        assert approved.status_code == 200, approved.text
    finally:
        settings.ENV = original_env
        settings.APP_ENV = original_app_env

    assert client.app.state.mcp.paid == {"order_p0"}
    audit = _fetch(
        client,
        "SELECT actor, detail FROM audit_log WHERE action='payment.approve_confirmed'",
    )
    assert audit
    detail = json.loads(audit[0]["detail"]) if isinstance(audit[0]["detail"], str) else audit[0]["detail"]
    assert audit[0]["actor"] == "admin"
    assert detail["order_id"] == "order_p0"
    assert detail["amount"] == 50
    assert detail["confirmed"] is True


def test_export_includes_only_own_audit_analytics_and_model_calls(client):
    headers_a = _create_user(client, "p0_export_a")
    _create_user(client, "p0_export_b")

    for user in ("p0_export_a", "p0_export_b"):
        _sql(
            client,
            "INSERT INTO audit_log (actor, session_id, action, detail) "
            "VALUES ($1,$2,'chat.completed','{}'::jsonb)",
            user,
            user,
        )
        _sql(
            client,
            "INSERT INTO analytics_events (user_id, name, props) VALUES ($1,$2,'{}'::jsonb)",
            user,
            f"evt_{user}",
        )
        _sql(
            client,
            "INSERT INTO model_calls (provider, model, task, user_id) VALUES ($1,'m','t',$2)",
            f"prov_{user}",
            user,
        )

    response = client.get("/api/v1/me/export", headers=headers_a)
    assert response.status_code == 200, response.text
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    data = json.loads(archive.read("data.json"))

    # 登录本身也会留痕（auth.login），所以导出里至少包含本次操作那一条；
    # 这里断言"全部属于本人"且"注入的那条在"，而不是固定条数。
    assert {item["actor"] for item in data["audit"]} == {"p0_export_a"}
    assert any(item["action"] == "chat.completed" for item in data["audit"])
    assert [item["name"] for item in data["analytics"]] == ["evt_p0_export_a"]
    assert [item["provider"] for item in data["model_calls"]] == ["prov_p0_export_a"]


def test_consent_is_recorded_for_user(client):
    token = _token(client, "patient", PATIENT_PASSWORD)
    response = client.post(
        "/api/v1/consent",
        headers=_auth(token),
        json={"consent_type": "privacy_terms", "version": "v2026-11"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["version"] == "v2026-11"
    assert (
        _fetchval(
            client,
            "SELECT consent_version FROM users WHERE id='patient'",
        )
        == "v2026-11"
    )
    assert (
        _fetchval(
            client,
            "SELECT count(*) FROM consent_record WHERE user_id='patient' AND version='v2026-11'",
        )
        >= 1
    )

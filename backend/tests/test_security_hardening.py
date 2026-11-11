"""安全加固：管理端 IP 白名单、分档限流、登录审计、审计抽查工具。

原则验证：
- 白名单留空 = 不改变现有行为；
- 白名单配置后 fail-closed（来源解析不出来就拒绝）；
- 分档限流为 0 时回退全局限流，测试环境不受影响；
- 登录无论成功失败都留痕，且审计里不出现密码。
"""
from __future__ import annotations

import asyncio
import json
import os

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app, reset_rate_buckets
from app.security_hardening import deny_all_networks, ip_allowed, parse_networks, rate_tier, tier_limit

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"


@pytest.fixture()
def client():
    reset_rate_buckets()
    with TestClient(app) as test_client:
        yield test_client
    reset_rate_buckets()


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
def admin(client: TestClient) -> dict[str, str]:
    get_settings().AUTH_ENABLED = True
    return _auth(_token(client, "admin", ADMIN_PASSWORD))


# ------------------------------------------------------------------ 纯函数
def test_parse_networks_and_ip_allowed() -> None:
    networks = parse_networks("10.0.0.0/8, 192.168.1.10")
    assert ip_allowed("10.5.5.5", networks)
    assert ip_allowed("192.168.1.10", networks)
    assert not ip_allowed("8.8.8.8", networks)
    assert not ip_allowed("not-an-ip", networks), "解析不出 IP 必须按不放行处理"
    assert ip_allowed("1.2.3.4", []), "留空表示不限制"
    assert deny_all_networks()


def test_rate_tier_mapping() -> None:
    assert rate_tier("/api/v1/chat/stream")[0] == "chat"
    assert rate_tier("/api/v1/admin/kb")[0] == "admin"
    assert rate_tier("/api/v1/ops/analytics")[0] == "admin"
    assert rate_tier("/api/v1/reports")[0] == "general"


def test_tier_limit_falls_back_to_global() -> None:
    settings = get_settings()
    original = settings.RATE_LIMIT_ADMIN_PER_MINUTE
    try:
        settings.RATE_LIMIT_ADMIN_PER_MINUTE = 0
        assert tier_limit(settings, "admin") == settings.RATE_LIMIT_PER_MINUTE
        settings.RATE_LIMIT_ADMIN_PER_MINUTE = 15
        assert tier_limit(settings, "admin") == 15
    finally:
        settings.RATE_LIMIT_ADMIN_PER_MINUTE = original


# ------------------------------------------------------------------ 白名单
def test_allowlist_disabled_keeps_admin_open(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    settings.ADMIN_IP_ALLOWLIST = ""
    assert client.get("/api/v1/admin/kb", headers=admin).status_code == 200


def test_allowlist_denies_unknown_client(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    settings.ADMIN_IP_ALLOWLIST = "10.0.0.0/8"
    try:
        denied = client.get("/api/v1/admin/kb", headers=admin)
        assert denied.status_code == 403
        assert "白名单" in denied.json()["detail"]

        # 非管理端路径不受影响
        patient = _auth(_token(client, "patient", PATIENT_PASSWORD))
        assert client.get("/api/v1/home/summary", headers=patient).status_code == 200
    finally:
        settings.ADMIN_IP_ALLOWLIST = ""


def test_allowlist_trusts_forwarded_ip(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    settings.ADMIN_IP_ALLOWLIST = "10.0.0.0/8"
    settings.TRUST_PROXY_HEADERS = True
    try:
        allowed = client.get("/api/v1/admin/kb", headers={**admin, "X-Real-IP": "10.1.2.3"})
        assert allowed.status_code == 200, allowed.text
    finally:
        settings.ADMIN_IP_ALLOWLIST = ""


def test_allowlist_ignores_forwarded_ip_when_untrusted(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    settings.ADMIN_IP_ALLOWLIST = "10.0.0.0/8"
    settings.TRUST_PROXY_HEADERS = False
    try:
        denied = client.get("/api/v1/admin/kb", headers={**admin, "X-Real-IP": "10.1.2.3"})
        assert denied.status_code == 403, "不信任代理头时不能靠伪造头绕过白名单"
    finally:
        settings.ADMIN_IP_ALLOWLIST = ""
        settings.TRUST_PROXY_HEADERS = True


def test_invalid_allowlist_fails_closed(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    settings.ADMIN_IP_ALLOWLIST = "这不是IP"
    try:
        assert client.get("/api/v1/admin/kb", headers=admin).status_code == 403
    finally:
        settings.ADMIN_IP_ALLOWLIST = ""


# ------------------------------------------------------------------ 分档限流
def test_chat_tier_rate_limit(client: TestClient) -> None:
    settings = get_settings()
    original = settings.RATE_LIMIT_CHAT_PER_MINUTE
    settings.RATE_LIMIT_CHAT_PER_MINUTE = 2
    reset_rate_buckets()
    try:
        codes = [
            client.post("/api/v1/chat/stream", json={}).status_code
            for _ in range(3)
        ]
        assert codes[-1] == 429, codes
        payload = client.post("/api/v1/chat/stream", json={}).json()
        assert payload["tier"] == "chat"
        assert payload["limit_per_minute"] == 2
    finally:
        settings.RATE_LIMIT_CHAT_PER_MINUTE = original
        reset_rate_buckets()


def test_admin_tier_rate_limit_independent(client: TestClient, admin: dict[str, str]) -> None:
    settings = get_settings()
    original = settings.RATE_LIMIT_ADMIN_PER_MINUTE
    settings.RATE_LIMIT_ADMIN_PER_MINUTE = 1
    reset_rate_buckets()
    try:
        assert client.get("/api/v1/admin/kb", headers=admin).status_code == 200
        blocked = client.get("/api/v1/admin/kb", headers=admin)
        assert blocked.status_code == 429
        assert blocked.json()["tier"] == "admin"
        # chat 档不受 admin 档影响
        assert client.post("/api/v1/chat/stream", json={}).status_code != 429
    finally:
        settings.RATE_LIMIT_ADMIN_PER_MINUTE = original
        reset_rate_buckets()


# ------------------------------------------------------------------ 登录审计
def test_login_is_audited_without_password(client: TestClient) -> None:
    get_settings().AUTH_ENABLED = True
    _token(client, "admin", ADMIN_PASSWORD)
    client.post("/api/v1/auth/login", json={"username": "admin", "password": "wrong-password"})

    rows = _sql(
        "SELECT actor, action, detail FROM audit_log WHERE action='auth.login' ORDER BY id DESC LIMIT 5"
    )
    assert rows, "登录必须留痕"
    details = [json.loads(row["detail"]) if isinstance(row["detail"], str) else row["detail"] for row in rows]
    outcomes = {item["outcome"] for item in details}
    assert "success" in outcomes and "failed" in outcomes
    dumped = json.dumps(details, ensure_ascii=False)
    assert "wrong-password" not in dumped and ADMIN_PASSWORD not in dumped, "审计里绝不能出现密码"
    assert all(item.get("ip") for item in details if item.get("outcome") == "success")

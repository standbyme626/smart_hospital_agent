"""Wave G 多租户：跨租户隔离、租户切换门禁、配置与知识库按租户隔离。

隔离实现：每条查询由数据层设置 app.tenant_id，表上启用行级安全（RLS）强制过滤；
脚本/迁移未设置租户时 fail-open（保持单租户兼容）。
"""
from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

ADMIN_PASSWORD = "admin123456"
PATIENT_PASSWORD = "patient123456"


@pytest.fixture()
def client():
    get_settings().AUTH_ENABLED = True
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


def _auth(token: str, tenant: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if tenant:
        headers["X-Tenant"] = tenant
    return headers


@pytest.fixture()
def admin(client: TestClient) -> str:
    return _token(client, "admin", ADMIN_PASSWORD)


@pytest.fixture()
def patient(client: TestClient) -> str:
    return _token(client, "patient", PATIENT_PASSWORD)


@pytest.fixture()
def tenant_b(client: TestClient, admin: str) -> dict[str, str]:
    """建一个独立院区 + 该院区患者账号，用完清理。"""
    code = "tb_" + uuid.uuid4().hex[:6]
    created = client.post(
        "/api/v1/admin/tenants",
        json={"code": code, "name": "隔离测试院区", "isolation": "logical"},
        headers=_auth(admin),
    )
    assert created.status_code == 200, created.text
    tenant = created.json()["tenant"]
    username = "ub_" + uuid.uuid4().hex[:6]
    user_resp = client.post(
        "/api/v1/admin/users",
        json={"username": username, "password": "tenantb123456", "role": "patient", "tenant_id": tenant["id"]},
        headers=_auth(admin),
    )
    assert user_resp.status_code == 201, user_resp.text
    token = _token(client, username, "tenantb123456")
    yield {"id": tenant["id"], "code": code, "username": username, "token": token}
    # 清理（用 SQL 直连：脚本未设置租户，RLS fail-open）
    _sql("DELETE FROM family_members WHERE user_id=$1", username)
    _sql("DELETE FROM users WHERE id=$1", username)
    _sql("DELETE FROM tenants WHERE id=$1", tenant["id"])


# ------------------------------------------------------------------ 租户切换门禁
def test_tenant_header_unknown_404(client: TestClient, admin: str) -> None:
    response = client.get("/api/v1/admin/kb", headers=_auth(admin, "no-such-tenant"))
    assert response.status_code == 404


def test_suspended_tenant_403(client: TestClient, admin: str, tenant_b: dict[str, str]) -> None:
    client.patch(
        f"/api/v1/admin/tenants/{tenant_b['id']}",
        json={"status": "suspended"},
        headers=_auth(admin),
    )
    response = client.get("/api/v1/admin/kb", headers=_auth(admin, tenant_b["code"]))
    assert response.status_code == 403


def test_patient_cannot_switch_tenant(client: TestClient, patient: str, tenant_b: dict[str, str]) -> None:
    """患者带 X-Tenant 头不生效：仍按自己账号所属租户访问，看不到别的院区。"""
    own = client.get("/api/v1/family", headers=_auth(patient)).json()["members"]
    other = client.get("/api/v1/family", headers=_auth(patient, tenant_b["code"])).json()["members"]
    assert [m["id"] for m in own] == [m["id"] for m in other]


# ------------------------------------------------------------------ 跨租户 404
def test_family_member_cross_tenant_404(
    client: TestClient, admin: str, tenant_b: dict[str, str]
) -> None:
    created = client.post(
        "/api/v1/family",
        json={"name": "A院患者" + uuid.uuid4().hex[:4], "relation": "self", "is_self": True},
        headers=_auth(tenant_b["token"]),
    )
    assert created.status_code == 200, created.text
    member_id = created.json()["member"]["id"]

    # 同院区：A 能看到自己建的成员
    listed = client.get("/api/v1/family", headers=_auth(tenant_b["token"])).json()["members"]
    assert any(item["id"] == member_id for item in listed)

    # 默认院区的管理员：看不到 B 院区的成员（列表为空且单查 404）
    default_list = client.get("/api/v1/family", headers=_auth(_token(client, "patient", PATIENT_PASSWORD))).json()["members"]
    assert all(item["id"] != member_id for item in default_list)
    physician_lookup = client.get(
        f"/api/v1/clinician/patients/{member_id}", headers=_auth(admin)
    )
    assert physician_lookup.status_code == 404


def test_knowledge_base_is_tenant_scoped(client: TestClient, admin: str, tenant_b: dict[str, str]) -> None:
    default_kb = client.get("/api/v1/admin/kb", headers=_auth(admin)).json()
    tenant_kb = client.get("/api/v1/admin/kb", headers=_auth(admin, tenant_b["code"])).json()
    assert default_kb["documents"] >= 0
    assert tenant_kb["documents"] == 0, "新院区不应看到默认院区的知识库语料"


def test_config_isolation(client: TestClient, admin: str, tenant_b: dict[str, str]) -> None:
    key = "iso_" + uuid.uuid4().hex[:6]
    saved = client.put(
        f"/api/v1/admin/config/rule/{key}",
        json={"value": {"severity": "high", "keywords": ["B院规则"]}},
        headers=_auth(admin, tenant_b["code"]),
    )
    assert saved.status_code == 200, saved.text

    tenant_list = client.get("/api/v1/admin/config/rule", headers=_auth(admin, tenant_b["code"])).json()
    default_list = client.get("/api/v1/admin/config/rule", headers=_auth(admin)).json()
    assert any(item["key"] == key for item in tenant_list["items"])
    assert all(item["key"] != key for item in default_list["items"]), "默认院区不应看到 B 院区的配置"


def test_audit_is_tenant_scoped(client: TestClient, admin: str, tenant_b: dict[str, str]) -> None:
    # 在 B 院区做一次写操作（建院区审计属于默认租户，这里用配置草稿产生 B 的审计）
    key = "audit_" + uuid.uuid4().hex[:6]
    client.put(
        f"/api/v1/admin/config/rule/{key}",
        json={"value": {"severity": "low", "keywords": ["审计隔离"]}},
        headers=_auth(admin, tenant_b["code"]),
    )
    rows = _sql(
        "SELECT tenant_id, count(*) AS c FROM audit_log WHERE detail->>'key' = $1 GROUP BY 1",
        key,
    )
    assert rows, "写入必须留痕"
    assert {row["tenant_id"] for row in rows} == {tenant_b["id"]}, "审计记录必须带租户"

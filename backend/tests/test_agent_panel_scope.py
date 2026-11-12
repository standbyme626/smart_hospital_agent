"""右栏 Agent 面板：按角色过滤 + 按账号/租户聚合。"""
from __future__ import annotations

import asyncio
import os
import uuid

import asyncpg
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


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
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_panel_scopes_calls_by_actor(client: TestClient) -> None:
    marker = uuid.uuid4().hex[:8]
    _sql(
        "INSERT INTO model_calls (provider, model, task, latency_ms, user_id) VALUES "
        "('test', 'fake', $1, 12, 'patient'), ('test', 'fake', $2, 34, 'admin')",
        "patient_task_" + marker,
        "admin_task_" + marker,
    )
    patient = _token(client, "patient", "patient123456")
    admin = _token(client, "admin", "admin123456")

    body = client.get("/api/v1/agent/panel", headers={"Authorization": f"Bearer {patient}"}).json()
    assert body["scope"] == "self"
    names = {row["name"] for row in body["recent"]}
    assert "patient_task_" + marker in names
    assert "admin_task_" + marker not in names

    tenant = client.get(
        "/api/v1/agent/panel", params={"end": "platform"}, headers={"Authorization": f"Bearer {admin}"}
    ).json()
    assert tenant["scope"] == "tenant"
    tenant_names = {row["name"] for row in tenant["recent"]}
    assert {"patient_task_" + marker, "admin_task_" + marker} <= tenant_names

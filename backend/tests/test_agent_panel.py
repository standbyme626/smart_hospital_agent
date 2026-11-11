"""右栏 Agent 面板：按角色返回能力/最近调用/边界。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app


@pytest.fixture()
def client():
    get_settings().AUTH_ENABLED = True
    with TestClient(app) as test_client:
        yield test_client


def _token(client: TestClient, username: str, password: str) -> str:
    resp = client.post("/api/v1/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


def test_patient_panel_scope(client: TestClient) -> None:
    token = _token(client, "patient", "patient123456")
    resp = client.get("/api/v1/agent/panel", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["end"] == "patient"
    names = {item["name"] for item in body["capabilities"]}
    assert "chat.triage" in names
    assert any("诊断" in line for line in body["boundaries"])
    assert body["starter"]


def test_admin_panel_scope(client: TestClient) -> None:
    token = _token(client, "admin", "admin123456")
    resp = client.get("/api/v1/agent/panel", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["end"] == "platform"
    names = {item["name"] for item in body["capabilities"]}
    assert "reindex.tsv" in names
    assert body["summary"]["calls_7d"] >= 0
    assert body["recent"] == [] or "latency_ms" in body["recent"][0]


def test_admin_panel_follows_end_switch(client: TestClient) -> None:
    token = _token(client, "admin", "admin123456")
    headers = {"Authorization": f"Bearer {token}"}
    for end, capability in (
        ("patient", "chat.triage"),
        ("clinician", "patient.360"),
        ("ops", "reconcile.run"),
        ("platform", "reindex.tsv"),
    ):
        resp = client.get("/api/v1/agent/panel", params={"end": end}, headers=headers)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["end"] == end
        assert capability in {item["name"] for item in body["capabilities"]}


def test_patient_cannot_borrow_admin_panel(client: TestClient) -> None:
    token = _token(client, "patient", "patient123456")
    resp = client.get(
        "/api/v1/agent/panel",
        params={"end": "platform"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    assert resp.json()["end"] == "patient"

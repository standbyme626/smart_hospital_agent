import time
from datetime import datetime

from fastapi.testclient import TestClient

from app.main import app


class FakeAgent:
    async def stream(self, **kwargs):
        yield {"seq": 1, "type": "status", "data": {"stage": "triage"}}
        yield {
            "seq": 2,
            "type": "final",
            "data": {
                "report": {
                    "department": "心血管内科",
                    "urgency": "normal",
                    "summary": "建议心内科就诊。",
                    "confidence": 0.8,
                },
                "citations": [],
            },
        }


def test_health_and_departments():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["app"] == "smart-hospital-agent"
        departments = client.get("/api/v1/departments")
        assert departments.status_code == 200
        assert len(departments.json()["departments"]) == 21


def test_drug_check_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/drug/check",
            json={"text": "华法林和阿司匹林可以一起吃吗"},
        )
        assert response.status_code == 200
        assert response.json()["highest_severity"] == "major"


def test_chat_sse_with_fake_agent():
    with TestClient(app) as client:
        client.app.state.agent = FakeAgent()
        response = client.post(
            "/api/v1/chat/stream",
            json={"message": "最近心悸，该挂什么科？", "session_id": "test-session"},
        )
        assert response.status_code == 200
        assert "event: final" in response.text
        assert "data: [DONE]" in response.text


def test_login_returns_admin_token():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "admin123456"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["role"] == "admin"
        assert body["access_token"]


def test_api_health():
    class NoExternalCalls:
        def __getattr__(self, name):
            raise AssertionError(f"/api/health must not touch external dependency: {name}")

    with TestClient(app) as client:
        originals = {
            "gateway": client.app.state.gateway,
            "mcp": client.app.state.mcp,
            "db": client.app.state.db,
            "retrieval": client.app.state.retrieval,
        }
        for name in originals:
            setattr(client.app.state, name, NoExternalCalls())
        try:
            start = time.perf_counter()
            response = client.get("/api/health")
            elapsed = time.perf_counter() - start
        finally:
            for name, original in originals.items():
                setattr(client.app.state, name, original)

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["version"] == client.app.state.settings.APP_VERSION
        assert datetime.fromisoformat(body["time"]).tzinfo is not None
        assert elapsed < 0.1

import json
import os

from fastapi.testclient import TestClient

from app.main import app
from tests.fake_gateway import REPORT_MARKER, FakeGateway

PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c63f8cfc0f01f00050001ff2f4b4d0d0000000049454e44ae426082"
)


def _install(client: TestClient) -> FakeGateway:
    fake = FakeGateway()
    state = client.app.state
    state.gateway = fake
    state.agent.gateway = fake
    state.retrieval.gateway = fake
    return fake


def _events(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        event_type = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not event_type or not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            continue
        events.append((event_type, json.loads(raw)))
    return events


def test_chat_stream_with_fake_gateway_submits_report():
    with TestClient(app) as client:
        fake = _install(client)
        response = client.post("/api/v1/chat/stream", json={"message": "头痛两天了，晚上更明显"})
        assert response.status_code == 200, response.text
        events = _events(response.text)
        final = next(data for event_type, data in events if event_type == "final")
        assert final["report"]["department"] == "心血管内科"
        assert REPORT_MARKER in final["report"]["summary"]
        assert final["report"]["followup_question"]

        triage_calls = fake.calls_for("triage")
        diagnosis_calls = fake.calls_for("diagnosis")
        assert triage_calls and triage_calls[0]["response_format"] == {"type": "json_object"}
        assert diagnosis_calls and diagnosis_calls[0]["tools"]

        session_id = next(data["session_id"] for event_type, data in events if event_type == "session")
        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        assert [item["role"] for item in detail["messages"]] == ["user", "assistant"]
        assert detail["latest_final"]["report"]["summary"] == final["report"]["summary"]


def test_report_interpret_with_fake_vision_gateway(tmp_path):
    with TestClient(app) as client:
        fake = _install(client)
        settings = client.app.state.settings
        original = settings.UPLOADS_DIR
        settings.UPLOADS_DIR = str(tmp_path / "uploads")
        try:
            response = client.post(
                "/api/v1/reports/interpret",
                files={"file": ("report.png", PNG_BYTES, "image/png")},
            )
        finally:
            settings.UPLOADS_DIR = original
        assert response.status_code == 200, response.text
        report = response.json()["report"]
        assert REPORT_MARKER in report["summary"]
        assert report["abnormal_values"][0]["name"] == "白细胞计数"
        assert report["questions"]
        assert report["has_image"] is True
        assert any("image_url" in json.dumps(call["messages"], ensure_ascii=False) for call in fake.calls)


def test_report_interpret_survives_missing_upload_dir(tmp_path):
    with TestClient(app) as client:
        _install(client)
        settings = client.app.state.settings
        original = settings.UPLOADS_DIR
        settings.UPLOADS_DIR = str(tmp_path / "missing" / "uploads")
        try:
            response = client.post(
                "/api/v1/reports/interpret",
                files={"file": ("report.png", PNG_BYTES, "image/png")},
            )
        finally:
            settings.UPLOADS_DIR = original
        assert response.status_code == 200, response.text
        assert os.path.isdir(tmp_path / "missing" / "uploads" / "reports")

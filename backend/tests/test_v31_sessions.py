import json

from fastapi.testclient import TestClient

from app.conversation import trim_history
from app.main import app
from app.model_gateway import ChatResult

PATIENT_PASSWORD = "patient123456"
ADMIN_PASSWORD = "admin123456"


class RecordingGateway:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> ChatResult:
        task = kwargs.get("task", "diagnosis")
        self.calls.append({"task": task, "messages": [dict(item) for item in messages]})
        if task == "triage":
            content = json.dumps(
                {
                    "intent": "MEDICAL_CONSULT",
                    "department": "全科",
                    "urgency": "normal",
                    "symptoms": "头痛",
                    "reason": "",
                    "confidence": 0.7,
                },
                ensure_ascii=False,
            )
        else:
            content = "建议充分休息并观察，必要时到全科门诊就诊。"
        return ChatResult(content=content, provider="fake", model="fake")

    def calls_for(self, task: str) -> list[dict]:
        return [call for call in self.calls if call["task"] == task]


class FakeRetrieval:
    async def search(self, query: str, **kwargs) -> list[dict]:
        return []


def _install_fakes(client: TestClient) -> RecordingGateway:
    gateway = RecordingGateway()
    client.app.state.agent.gateway = gateway
    client.app.state.agent.retrieval = FakeRetrieval()
    client.app.state.gateway = gateway
    client.app.state.retrieval = client.app.state.agent.retrieval
    return gateway


def _stream(client: TestClient, message: str, session_id: str, headers: dict | None = None):
    return client.post(
        "/api/v1/chat/stream",
        json={"message": message, "session_id": session_id},
        headers=headers or {},
    )


def test_conversation_detail_returns_session_messages_and_latest_final():
    session_id = "sess-history"
    with TestClient(app) as client:
        _install_fakes(client)
        first = _stream(client, "头痛两天了，晚上更明显", session_id)
        assert first.status_code == 200, first.text
        assert "event: final" in first.text

        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        assert detail["session"]["id"] == session_id
        assert detail["session"]["member_id"]
        assert detail["session"]["last_message_at"]
        assert [item["role"] for item in detail["messages"]] == ["user", "assistant"]
        assert "头痛两天了" in detail["messages"][0]["content"]
        summary = detail["latest_final"]["report"]["summary"]
        assert summary
        assert detail["messages"][1]["content"] == summary
        assert detail["latest_final"]["triage"]["department"]
        assert "safety" in detail["latest_final"]


def test_second_turn_includes_previous_user_and_assistant_history():
    session_id = "sess-followup"
    with TestClient(app) as client:
        gateway = _install_fakes(client)
        assert "event: final" in _stream(client, "头痛两天了，晚上更明显", session_id).text
        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        first_user = detail["messages"][0]["content"]
        first_assistant = detail["messages"][1]["content"]

        gateway.calls.clear()
        second = _stream(client, "还伴有恶心，应该挂什么科", session_id)
        assert "event: final" in second.text

        diagnosis = gateway.calls_for("diagnosis")
        assert len(diagnosis) == 1
        messages = diagnosis[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": first_user}
        assert messages[2] == {"role": "assistant", "content": first_assistant}
        assert messages[-1]["role"] == "user"
        assert "还伴有恶心" in messages[-1]["content"]

        triage = gateway.calls_for("triage")
        assert len(triage) == 1
        triage_messages = triage[0]["messages"]
        assert triage_messages[-1]["role"] == "user"
        assert "还伴有恶心" in triage_messages[-1]["content"]
        context = "\n".join(item["content"] for item in triage_messages[:-1])
        assert "头痛两天了" in context
        assert first_assistant in context

        detail = client.get(f"/api/v1/sessions/{session_id}").json()
        assert [item["role"] for item in detail["messages"]] == ["user", "assistant", "user", "assistant"]
        assert detail["messages"][2]["content"].endswith("还伴有恶心，应该挂什么科")


def test_session_detail_is_owner_scoped():
    session_id = "sess-owner"
    with TestClient(app) as client:
        client.app.state.settings.AUTH_ENABLED = True
        _install_fakes(client)
        try:
            patient = client.post(
                "/api/v1/auth/login",
                json={"username": "patient", "password": PATIENT_PASSWORD},
            ).json()["access_token"]
            admin = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": ADMIN_PASSWORD},
            ).json()["access_token"]
            patient_headers = {"Authorization": f"Bearer {patient}"}
            admin_headers = {"Authorization": f"Bearer {admin}"}

            assert _stream(client, "头痛", session_id, patient_headers).status_code == 200

            assert client.get(f"/api/v1/sessions/{session_id}", headers=admin_headers).status_code == 404
            assert client.get(f"/api/v1/sessions/{session_id}/events", headers=admin_headers).status_code == 404
            assert _stream(client, "继续头痛", session_id, admin_headers).status_code == 404
            assert client.get(f"/api/v1/sessions/{session_id}", headers=patient_headers).status_code == 200
        finally:
            client.app.state.settings.AUTH_ENABLED = False


def test_trim_history_drops_oldest_first():
    messages = [
        {"role": "user", "content": "a" * 6000},
        {"role": "assistant", "content": "b" * 6000},
        {"role": "user", "content": "c" * 6000},
    ]
    trimmed = trim_history(messages, max_chars=12000)
    assert [item["content"][0] for item in trimmed] == ["b", "c"]

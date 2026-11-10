from __future__ import annotations

import json

from fastapi.testclient import TestClient

from his_adapter.config import AdapterConfig
from his_adapter.mcp_server import TOOLS, create_app
from his_adapter.service import AdapterService

PATIENT = {
    "name": "混合模式患者",
    "id_card": "310101198801019876",
    "gender": "female",
    "birth_date": "1988-01-01",
    "phone": "13900000002",
}


async def test_mock_mode_full_flow(monkeypatch):
    monkeypatch.setenv("HIS_ADAPTER_MODE", "mock")
    monkeypatch.delenv("ADAPTER_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIS_PAYMENT_MODE", "cashier")
    config = AdapterConfig.from_env()
    assert config.mode == "mock"

    service = AdapterService(config)
    await service.startup()
    try:
        created = await service.call("patient.create", {"member_id": "m1", **PATIENT})
        assert created["ok"] is True, created
        assert created["data"]["created"] is True
        repeated = await service.call("patient.create", {"member_id": "m1", **PATIENT})
        assert repeated["data"]["created"] is False

        appointment = await service.call(
            "appointment.create",
            {
                "member_id": "m1",
                "org_id": "101",
                "practitioner_id": "201",
                "service_id": "301",
                "date": "2031-02-01",
            },
        )
        assert appointment["ok"] is True, appointment
        appointment_id = appointment["data"]["appointment_id"]
        assert appointment["data"]["payment_mode"] == "cashier"

        task = await service.call("payment.request", {"appointment_id": appointment_id})
        assert task["ok"] is True
        assert task["data"]["status"] == "requested"
        assert task["data"]["task_id"]
        assert "应付" in task["data"]["instructions"]

        registration = await service.call(
            "registration.create", {"appointment_id": appointment_id}
        )
        assert registration["ok"] is True, registration
        assert registration["data"]["status"] == "PENDING_CASHIER"
        assert registration["data"]["next_action"] == "payment.request"
        assert registration["data"]["encounter_id"] is None

        pending = await service.call("payment.status", {"appointment_id": appointment_id})
        assert pending["data"]["payment_status"] == "unpaid"

        paid = await service.simulate_payment(appointment_id)
        assert paid["payment_status"] == "paid"
        assert paid["encounter_id"]

        status = await service.call("payment.status", {"appointment_id": appointment_id})
        assert status["data"]["payment_status"] == "paid"
        assert status["data"]["invoice_no"]

        encounter = await service.call(
            "encounter.get", {"appointment_id": appointment_id}
        )
        assert encounter["ok"] is True, encounter
        assert encounter["data"]["encounter_id"] == paid["encounter_id"]

        prescriptions = await service.call(
            "prescription.list",
            {"encounter_id": paid["encounter_id"], "include_details": True},
        )
        assert prescriptions["ok"] is True
        assert len(prescriptions["data"]["prescriptions"]) == 1
        assert prescriptions["data"]["prescriptions"][0]["item_count"] == 2

        visit = await service.call("visit.status", {"appointment_id": appointment_id})
        assert visit["ok"] is True
        assert visit["data"]["state"] == "registered"
        assert visit["data"]["department"] == "心血管内科"

        reconcile = await service.call("his.reconcile", {"scope": "all"})
        assert reconcile["ok"] is True
        assert reconcile["data"]["summary"]["checked"] == 1

        cancelled = await service.call(
            "appointment.cancel",
            {"appointment_id": appointment_id, "reason": "test", "refund": True},
        )
        assert cancelled["ok"] is True, cancelled
        assert cancelled["data"]["status"] == "refunded"
        assert cancelled["data"]["refund_id"]

        refunded = await service.call("payment.status", {"appointment_id": appointment_id})
        assert refunded["data"]["payment_status"] == "refunded"
    finally:
        await service.close()


def _rpc(client: TestClient, method: str, params: dict | None = None, scope: str | None = None) -> dict:
    headers = {}
    if scope:
        headers["X-Adapter-Scope"] = scope
    response = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_mcp_jsonrpc_surface(monkeypatch):
    monkeypatch.setenv("HIS_ADAPTER_MODE", "mock")
    monkeypatch.delenv("ADAPTER_DATABASE_URL", raising=False)
    config = AdapterConfig.from_env()
    app = create_app(config)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["mode"] == "mock"
        assert health.json()["tools"] == 13

        initialized = _rpc(client, "initialize", {})
        assert initialized["result"]["protocolVersion"] == "2024-11-05"

        listed = _rpc(client, "tools/list", {})
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert names == {tool["name"] for tool in TOOLS}
        assert len(names) == 13

        created = _rpc(
            client,
            "tools/call",
            {"name": "patient.create", "arguments": {"member_id": "mcp1", **PATIENT}},
            scope="service_write",
        )
        payload = json.loads(created["result"]["content"][0]["text"])
        assert payload["ok"] is True, payload
        assert created["result"]["isError"] is False

        read = _rpc(
            client,
            "tools/call",
            {"name": "schedule.list", "arguments": {"department": "心血管内科"}},
            scope="agent_read",
        )
        payload = json.loads(read["result"]["content"][0]["text"])
        assert payload["ok"] is True
        assert payload["data"]["availability_mode"] == "catalog_only"

        denied = _rpc(
            client,
            "tools/call",
            {"name": "appointment.cancel", "arguments": {"appointment_id": "x"}},
            scope="agent_read",
        )
        payload = json.loads(denied["result"]["content"][0]["text"])
        assert payload["ok"] is False
        assert payload["error"]["code"] == "ADAPTER_SCOPE_DENIED"
        assert denied["result"]["isError"] is True

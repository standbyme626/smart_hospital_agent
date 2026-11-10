from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from his_adapter.config import AdapterConfig
from his_adapter.his_rest import OpenHISGateway
from his_adapter.service import AdapterService
from his_adapter.store import Store
from tests.fake_openhis import FakeOpenHISState, create_fake_openhis

PATIENT = {
    "name": "测试患者",
    "id_card": "110101199001011234",
    "gender": "male",
    "birth_date": "1990-01-01",
    "phone": "13800000001",
}


async def _harness(
    *,
    payment_mode: str = "direct",
    tenant_id: str = "1",
    timeout: float = 2.0,
    state: FakeOpenHISState | None = None,
) -> SimpleNamespace:
    fake, fake_state = create_fake_openhis(state)
    config = AdapterConfig(
        mode="openhis",
        openhis_base_url="http://fake-openhis",
        openhis_username=fake_state.username,
        openhis_password=fake_state.password,
        openhis_tenant_id=tenant_id,
        timeout_seconds=timeout,
        payment_mode=payment_mode,
    )
    store = Store()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake))
    gateway = OpenHISGateway(config, client=client)
    service = AdapterService(config, store=store, gateway=gateway)
    await service.startup()
    return SimpleNamespace(
        service=service,
        gateway=gateway,
        store=store,
        state=fake_state,
        config=config,
    )


async def _enroll(service: AdapterService, member_id: str = "m1", patient: dict | None = None) -> dict:
    payload = {"member_id": member_id, **(patient or PATIENT)}
    result = await service.call("patient.create", payload)
    assert result["ok"], result
    return result["data"]


@pytest.fixture()
async def harness():
    env = await _harness()
    yield env
    await env.service.close()


async def test_patient_create_and_resolve(harness):
    created = await _enroll(harness.service)
    assert created["created"] is True
    assert created["his_patient_id"]
    assert len(harness.state.patients) == 1

    resolved = await harness.service.call(
        "patient.resolve",
        {"member_id": "m1", "name": PATIENT["name"], "id_card": PATIENT["id_card"]},
    )
    assert resolved["ok"] is True
    assert resolved["data"]["found"] is True
    assert resolved["data"]["match_method"] == "id_card"
    assert resolved["data"]["his_patient_id"] == created["his_patient_id"]
    assert resolved["meta"]["source"] == "openhis"


async def test_patient_resolve_candidates_when_name_only(harness):
    await _enroll(harness.service)
    result = await harness.service.call(
        "patient.resolve", {"member_id": "m1", "name": PATIENT["name"]}
    )
    assert result["ok"] is True
    assert result["data"]["found"] is False
    assert result["data"]["candidates"][0]["id_card_last4"] == "1234"
    assert "id_card" not in result["data"]["candidates"][0]


async def test_duplicate_patient_resolves_existing(harness):
    first = await _enroll(harness.service, member_id="m1")
    result = await harness.service.call(
        "patient.create", {"member_id": "m2", **PATIENT}
    )
    assert result["ok"] is True
    assert result["data"]["created"] is False
    assert result["data"]["match_method"] == "id_card"
    assert result["data"]["his_patient_id"] == first["his_patient_id"]


async def test_schedule_list_is_catalog_only(harness):
    result = await harness.service.call("schedule.list", {"department": "心血管内科"})
    assert result["ok"] is True
    data = result["data"]
    assert data["availability_mode"] == "catalog_only"
    assert data["slots"] == []
    assert {item["org_id"] for item in data["departments"]} == {"101"}
    assert {item["practitioner_id"] for item in data["doctors"]} == {"201", "202"}
    assert data["services"][0]["price"] > 0


async def test_direct_registration_flow(harness):
    created = await _enroll(harness.service)
    appointment = await harness.service.call(
        "appointment.create",
        {
            "member_id": "m1",
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": "2031-01-01",
        },
    )
    assert appointment["ok"] is True, appointment
    quote = appointment["data"]
    assert quote["status"] == "quoted"
    assert quote["payment_mode"] == "direct"
    assert quote["currency"] == "CNY"
    assert quote["amount"] == 50.0

    registration = await harness.service.call(
        "registration.create", {"appointment_id": quote["appointment_id"]}
    )
    assert registration["ok"] is True, registration
    data = registration["data"]
    assert data["status"] == "registered"
    assert data["payment_status"] == "paid"
    assert data["encounter_id"]
    assert harness.state.patients[created["his_patient_id"]]

    payment = await harness.service.call(
        "payment.status", {"appointment_id": quote["appointment_id"]}
    )
    assert payment["ok"] is True
    assert payment["data"]["payment_status"] == "paid"
    assert payment["data"]["invoice_no"]

    encounter = await harness.service.call(
        "encounter.get", {"appointment_id": quote["appointment_id"]}
    )
    assert encounter["ok"] is True
    assert encounter["data"]["encounter_id"] == data["encounter_id"]
    assert encounter["data"]["visit_no"]

    prescriptions = await harness.service.call(
        "prescription.list", {"encounter_id": data["encounter_id"], "include_details": True}
    )
    assert prescriptions["ok"] is True
    assert len(prescriptions["data"]["prescriptions"]) == 1
    assert prescriptions["data"]["prescriptions"][0]["items"][0]["name"]

    visit = await harness.service.call(
        "visit.status", {"appointment_id": quote["appointment_id"]}
    )
    assert visit["ok"] is True
    assert visit["data"]["state"] == "registered"
    assert visit["data"]["next_action"] == "visit"


async def test_duplicate_registration_returns_stable_error(harness):
    await _enroll(harness.service)
    first = await harness.service.call(
        "appointment.create",
        {
            "member_id": "m1",
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": "2031-01-02",
        },
    )
    assert (await harness.service.call(
        "registration.create", {"appointment_id": first["data"]["appointment_id"]}
    ))["ok"]
    second = await harness.service.call(
        "appointment.create",
        {
            "member_id": "m1",
            "org_id": "101",
            "practitioner_id": "202",
            "service_id": "301",
            "date": "2031-01-02",
        },
    )
    result = await harness.service.call(
        "registration.create", {"appointment_id": second["data"]["appointment_id"]}
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "HIS_DUPLICATE_REGISTRATION"


async def test_401_triggers_relogin(harness):
    await _enroll(harness.service)
    first_logins = harness.state.login_count
    harness.state.expire_tokens()
    result = await harness.service.call(
        "patient.resolve", {"member_id": "m1", "name": PATIENT["name"], "id_card": PATIENT["id_card"]}
    )
    assert result["ok"] is True, result
    assert harness.state.login_count == first_logins + 1


async def test_business_error_on_http_200(harness):
    harness.state.fail_next = "business500"
    result = await harness.service.call(
        "patient.resolve", {"member_id": "m1", "name": PATIENT["name"], "id_card": PATIENT["id_card"]}
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "HIS_VALIDATION"
    assert result["error"]["retryable"] is False


async def test_http_500_maps_to_unavailable(harness):
    harness.state.fail_next = "http500"
    result = await harness.service.call(
        "patient.resolve", {"member_id": "m1", "name": PATIENT["name"], "id_card": PATIENT["id_card"]}
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "HIS_UNAVAILABLE"
    assert result["error"]["retryable"] is True


async def test_timeout_maps_to_his_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    config = AdapterConfig(
        mode="openhis",
        openhis_base_url="http://fake-openhis",
        openhis_username="his_agent",
        openhis_password="his_secret",
        openhis_tenant_id="1",
        timeout_seconds=0.1,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = OpenHISGateway(config, client=client)
    service = AdapterService(config, store=Store(), gateway=gateway)
    await service.startup()
    try:
        result = await service.call(
            "patient.resolve", {"member_id": "m1", "name": "测试患者", "id_card": PATIENT["id_card"]}
        )
    finally:
        await service.close()
    assert result["ok"] is False
    assert result["error"]["code"] == "HIS_TIMEOUT"
    assert result["error"]["retryable"] is True


async def test_tenant_mismatch():
    env = await _harness(tenant_id="999")
    try:
        result = await env.service.call(
            "patient.resolve", {"member_id": "m1", "name": PATIENT["name"], "id_card": PATIENT["id_card"]}
        )
    finally:
        await env.service.close()
    assert result["ok"] is False
    assert result["error"]["code"] == "HIS_TENANT_MISMATCH"


async def test_scope_enforcement(harness):
    denied = await harness.service.call(
        "appointment.create",
        {
            "member_id": "m1",
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": "2031-01-01",
        },
        scope="agent_read",
    )
    assert denied["ok"] is False
    assert denied["error"]["code"] == "ADAPTER_SCOPE_DENIED"
    read = await harness.service.call(
        "schedule.list", {"org_id": "101"}, scope="agent_read"
    )
    assert read["ok"] is True


async def test_idempotency_replay_and_conflict(harness):
    await _enroll(harness.service)
    payload = {
        "member_id": "m1",
        "org_id": "101",
        "practitioner_id": "201",
        "service_id": "301",
        "date": "2031-01-03",
        "idempotency_key": "idem-quote-1",
    }
    first = await harness.service.call("appointment.create", payload)
    second = await harness.service.call("appointment.create", payload)
    assert first["ok"] and second["ok"]
    assert first["data"]["appointment_id"] == second["data"]["appointment_id"]
    assert first["data"]["quote_id"] == second["data"]["quote_id"]
    conflict = await harness.service.call(
        "appointment.create", {**payload, "date": "2031-01-04"}
    )
    assert conflict["ok"] is False
    assert conflict["error"]["code"] == "ADAPTER_IDEMPOTENCY_CONFLICT"


async def test_mapping_missing(harness):
    result = await harness.service.call(
        "appointment.create",
        {
            "member_id": "not-enrolled",
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": "2031-01-01",
        },
    )
    assert result["ok"] is False
    assert result["error"]["code"] == "ADAPTER_MAPPING_MISSING"


async def test_paid_cancel_refunds(harness):
    await _enroll(harness.service)
    appointment = await harness.service.call(
        "appointment.create",
        {
            "member_id": "m1",
            "org_id": "101",
            "practitioner_id": "201",
            "service_id": "301",
            "date": "2031-01-05",
        },
    )
    appointment_id = appointment["data"]["appointment_id"]
    await harness.service.call("registration.create", {"appointment_id": appointment_id})
    result = await harness.service.call(
        "appointment.cancel",
        {"appointment_id": appointment_id, "reason": "schedule_conflict", "refund": True},
    )
    assert result["ok"] is True, result
    assert result["data"]["status"] == "refunded"
    assert result["data"]["refund_id"]
    payment = await harness.service.call("payment.status", {"appointment_id": appointment_id})
    assert payment["data"]["payment_status"] == "refunded"

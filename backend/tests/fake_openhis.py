from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ENCOUNTER_STATUS_TEXT = {
    1: "待诊",
    2: "在诊",
    4: "诊毕",
    5: "完成",
    6: "退号",
}

ORG_LIST = [
    {"id": "101", "name": "心血管内科"},
    {"id": "102", "name": "全科门诊"},
]

PRACTITIONERS = [
    {"id": "201", "name": "张医生", "orgId": "101"},
    {"id": "202", "name": "李医生", "orgId": "101"},
    {"id": "203", "name": "王医生", "orgId": "102"},
]

SERVICES = [
    {
        "id": "301",
        "definitionId": "301",
        "name": "心血管内科普通门诊",
        "offeredOrgId": "101",
        "price": 50,
    },
    {
        "id": "302",
        "definitionId": "302",
        "name": "心血管内科专家门诊",
        "offeredOrgId": "101",
        "price": 80,
    },
    {
        "id": "303",
        "definitionId": "303",
        "name": "全科门诊",
        "offeredOrgId": "102",
        "price": 30,
    },
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _ok(data: Any = None, msg: str = "操作成功") -> dict[str, Any]:
    return {"code": 200, "msg": msg, "data": data}


def _fail(msg: str, code: int = 500) -> dict[str, Any]:
    return {"code": code, "msg": msg, "data": None}


def _page(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {"records": records, "total": len(records)}


class FakeOpenHISState:
    def __init__(self) -> None:
        self.username = "his_agent"
        self.password = "his_secret"
        self.tenant_id = 1
        self.tokens: dict[str, str] = {}
        self.login_count = 0
        self.fail_next: str | None = None
        self.patients: dict[str, dict[str, Any]] = {}
        self.quotes: dict[str, dict[str, Any]] = {}
        self.encounters: dict[str, dict[str, Any]] = {}
        self.payments: dict[str, dict[str, Any]] = {}
        self.prescriptions: dict[str, dict[str, Any]] = {}
        self._seq = 1000

    def next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    def issue_token(self) -> str:
        token = uuid.uuid4().hex
        self.tokens[token] = self.username
        return token

    def expire_tokens(self) -> None:
        self.tokens.clear()

    def add_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        patient_id = self.next_id("pat_")
        row = {
            "id": patient_id,
            "name": patient.get("name"),
            "idCard": patient.get("idCard"),
            "phone": patient.get("phone"),
            "birthDate": patient.get("birthDate"),
            "genderEnum_enumText": "男" if patient.get("genderEnum") == 0 else "女",
        }
        self.patients[patient_id] = row
        return row


async def _failure(state: FakeOpenHISState) -> JSONResponse | None:
    mode = state.fail_next
    if not mode:
        return None
    state.fail_next = None
    if mode == "timeout":
        await asyncio.sleep(0.6)
        return None
    if mode == "http500":
        return JSONResponse(status_code=500, content={"code": 500, "msg": "upstream exploded"})
    if mode == "business500":
        return JSONResponse(status_code=200, content=_fail("上游业务失败"))
    return None


def _unauthorized() -> JSONResponse:
    return JSONResponse(status_code=401, content={"code": 401, "msg": "认证失败，无法访问系统资源"})


def _authed(state: FakeOpenHISState, request: Request) -> bool:
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip()
    return token in state.tokens


def _encounter_row(state: FakeOpenHISState, encounter: dict[str, Any]) -> dict[str, Any]:
    payment = state.payments.get(encounter["id"], {})
    return {
        "id": encounter["id"],
        "patientId": encounter["patientId"],
        "organizationId": encounter["organizationId"],
        "practitionerId": encounter["practitionerId"],
        "statusEnum": encounter["statusEnum"],
        "statusEnum_enumText": ENCOUNTER_STATUS_TEXT.get(encounter["statusEnum"], "未知"),
        "startTime": encounter["startTime"],
        "createTime": encounter["createTime"],
        "busNo": encounter["busNo"],
        "paymentStatus": payment.get("paymentStatus"),
    }


def create_fake_openhis(state: FakeOpenHISState | None = None) -> tuple[FastAPI, FakeOpenHISState]:
    fake_state = state or FakeOpenHISState()
    app = FastAPI(title="Fake OpenHIS", version="test")

    @app.get("/system/tenant/user-bind/{username}")
    async def user_bind(username: str) -> dict[str, Any]:
        if username != fake_state.username:
            return _ok([])
        return _ok(
            [
                {
                    "id": fake_state.tenant_id,
                    "tenantCode": "demo",
                    "tenantName": "示范医院",
                }
            ]
        )

    @app.post("/login")
    async def login(request: Request) -> dict[str, Any]:
        body = await request.json()
        if body.get("username") != fake_state.username or body.get("password") != fake_state.password:
            return _fail("用户名或密码错误")
        if int(body.get("tenantId") or 0) != fake_state.tenant_id:
            return _fail("租户不存在或未绑定")
        fake_state.login_count += 1
        return {"code": 200, "msg": "操作成功", "token": fake_state.issue_token()}

    @app.get("/patient-manage/information/patient-information-page")
    async def patient_page(request: Request, searchKey: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = [
            row
            for row in fake_state.patients.values()
            if not searchKey
            or searchKey in str(row.get("idCard"))
            or searchKey in str(row.get("name"))
            or searchKey in str(row.get("phone"))
        ]
        return _ok(_page(rows))

    @app.post("/patient-manage/information/patient-information")
    async def patient_create(request: Request) -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        body = await request.json()
        id_card = body.get("idCard")
        if not id_card:
            return _fail("身份证号不能为空")
        if any(row.get("idCard") == id_card for row in fake_state.patients.values()):
            return _fail("身份证号已存在")
        row = fake_state.add_patient(body)
        return _ok({"id": row["id"]})

    @app.get("/charge-manage/register/org-list")
    async def org_list(request: Request) -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        return _ok(ORG_LIST)

    @app.get("/charge-manage/register/practitioner-metadata")
    async def practitioner_metadata(request: Request, orgId: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = [row for row in PRACTITIONERS if not orgId or row["orgId"] == str(orgId)]
        return _ok(_page(rows))

    @app.get("/charge-manage/register/healthcare-metadata")
    async def healthcare_metadata(request: Request, organizationId: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = [
            row for row in SERVICES if not organizationId or row["offeredOrgId"] == str(organizationId)
        ]
        return _ok(_page(rows))

    @app.post("/payment/payment/reg-pre-pay")
    async def reg_pre_pay(request: Request) -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        body = await request.json()
        form = body.get("encounterFormData") or {}
        patient_id = str(form.get("patientId") or "")
        if patient_id not in fake_state.patients:
            return _fail("未查询到患者信息")
        quote_id = fake_state.next_id("QB")
        bus_no = fake_state.next_id("REG")
        amount = float((body.get("chargeItemFormData") or {}).get("totalPrice") or 0)
        fake_state.quotes[quote_id] = {
            "quoteId": quote_id,
            "busNo": bus_no,
            "patientId": patient_id,
            "serviceId": str(form.get("serviceTypeId") or ""),
            "organizationId": str(form.get("organizationId") or ""),
            "practitionerId": str(
                (body.get("encounterParticipantFormData") or {}).get("practitionerId") or ""
            ),
            "date": str((body.get("chargeItemFormData") or {}).get("occurrenceTime") or "")[:10],
            "amount": amount,
            "form": body,
            "encounterId": None,
        }
        return _ok({"chrgBchno": quote_id, "busNo": bus_no, "medfeeSumamt": amount})

    @app.post("/payment/payment/reg-pay")
    async def reg_pay(request: Request) -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        body = await request.json()
        quote = fake_state.quotes.get(str(body.get("chrgBchno") or ""))
        if quote is None:
            return _fail("未查询到预结算信息")
        if quote.get("encounterId"):
            return _fail("当日已挂号此服务")
        for encounter in fake_state.encounters.values():
            if (
                encounter["patientId"] == quote["patientId"]
                and encounter["serviceId"] == quote["serviceId"]
                and encounter["date"] == quote["date"]
                and encounter["statusEnum"] != 6
            ):
                return _fail("当日已挂号此服务")
        encounter_id = fake_state.next_id("enc_")
        payment_id = fake_state.next_id("pay_")
        invoice_no = fake_state.next_id("INV")
        fake_state.encounters[encounter_id] = {
            "id": encounter_id,
            "patientId": quote["patientId"],
            "organizationId": quote["organizationId"],
            "practitionerId": quote["practitionerId"],
            "serviceId": quote["serviceId"],
            "date": quote["date"],
            "statusEnum": 1,
            "startTime": _now(),
            "createTime": _now(),
            "busNo": quote["busNo"],
        }
        fake_state.payments[encounter_id] = {
            "id": payment_id,
            "encounterId": encounter_id,
            "statusEnum": 1,
            "displayAmount": quote["amount"],
            "billDate": _now(),
            "invoiceNo": invoice_no,
        }
        quote["encounterId"] = encounter_id
        fake_state.prescriptions[encounter_id] = {
            "prescriptionNo": fake_state.next_id("RX"),
            "items": [
                {
                    "adviceName": "阿司匹林肠溶片",
                    "quantity": 30,
                    "dose": 0.1,
                    "doseUnitCode_dictText": "g",
                    "rateCode_dictText": "每日一次",
                    "methodCode_dictText": "口服",
                },
                {
                    "adviceName": "阿托伐他汀钙片",
                    "quantity": 7,
                    "dose": 20,
                    "doseUnitCode_dictText": "mg",
                    "rateCode_dictText": "每晚一次",
                    "methodCode_dictText": "口服",
                },
            ],
        }
        return _ok(
            {
                "id": payment_id,
                "encounterId": encounter_id,
                "invoiceNo": invoice_no,
                "displayAmount": quote["amount"],
                "billDate": _now(),
            }
        )

    @app.get("/charge-manage/register/current-day-encounter")
    async def current_day_encounter(request: Request, searchKey: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = []
        for encounter in fake_state.encounters.values():
            patient = fake_state.patients.get(encounter["patientId"], {})
            if searchKey and searchKey not in (
                encounter["id"],
                encounter["patientId"],
                encounter["busNo"],
                str(patient.get("name") or ""),
            ):
                continue
            rows.append(_encounter_row(fake_state, encounter))
        return _ok(_page(rows))

    @app.get("/doctor-station/main/prescription-page-info")
    async def prescription_page(request: Request, encounterId: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = []
        for enc_id, prescription in fake_state.prescriptions.items():
            if encounterId and enc_id != str(encounterId):
                continue
            encounter = fake_state.encounters.get(enc_id, {})
            rows.append(
                {
                    "prescriptionNo": prescription["prescriptionNo"],
                    "encounterId": enc_id,
                    "patientId": encounter.get("patientId"),
                    "requestTime": _now(),
                    "statusEnum_enumText": "已开立",
                    "practitionerName": "张医生",
                    "prescriptionInfoDetailList": prescription["items"],
                }
            )
        return _ok(_page(rows))

    @app.get("/doctor-station/main/prescription-detail-info")
    async def prescription_detail(request: Request, prescriptionNo: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        for prescription in fake_state.prescriptions.values():
            if prescription["prescriptionNo"] == prescriptionNo:
                return _ok(prescription["items"])
        return _ok([])

    @app.get("/payment/payment/page")
    async def payment_page(request: Request, searchKey: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        rows = [
            row
            for row in fake_state.payments.values()
            if not searchKey or searchKey == row["encounterId"] or searchKey == row["id"]
        ]
        return _ok(_page(rows))

    @app.put("/charge-manage/register/cancel")
    async def cancel(request: Request, encounterId: str = "") -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        encounter = fake_state.encounters.get(str(encounterId))
        if encounter is None:
            return _fail("未查询到就诊信息")
        payment = fake_state.payments.get(str(encounterId))
        if payment and payment["statusEnum"] == 1:
            return _fail("已支付，请办理退费")
        encounter["statusEnum"] = 6
        return _ok(None)

    @app.put("/charge-manage/register/return")
    async def return_register(request: Request) -> Any:
        failed = await _failure(fake_state)
        if failed:
            return failed
        if not _authed(fake_state, request):
            return _unauthorized()
        body = await request.json()
        encounter_id = str(body.get("encounterId") or "")
        encounter = fake_state.encounters.get(encounter_id)
        if encounter is None:
            return _fail("未查询到就诊信息")
        payment = fake_state.payments.get(encounter_id)
        if not payment:
            return _fail("未查询到支付信息")
        encounter["statusEnum"] = 6
        payment["statusEnum"] = 3
        return _ok({"refundId": fake_state.next_id("RF")})

    return app, fake_state


app, state = create_fake_openhis()

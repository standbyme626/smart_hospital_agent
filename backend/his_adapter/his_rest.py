from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx

from .auth import TokenCache
from .config import AdapterConfig
from .errors import AdapterError, normalize_business, normalize_http
from .readonly import ReadOnlyPool


def _records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("records", "rows", "list", "content"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _first(data: Any, *keys: str) -> Any:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if value not in (None, ""):
                return value
    return None


def _amount(value: Any) -> float:
    try:
        return round(float(value or 0), 2)
    except (TypeError, ValueError):
        return 0.0


def _num(value: Any) -> Any:
    text = str(value)
    return int(text) if text.isdigit() else value


def _iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


GENDER_CODES = {"male": 0, "男": 0, "female": 1, "女": 1, "unknown": 2, "未知": 2}
ENCOUNTER_STATUS_TEXT = {
    1: "待诊",
    2: "在诊",
    3: "暂离",
    4: "诊毕",
    5: "完成",
    6: "退号",
    7: "已中断",
    8: "错误",
    9: "未知",
}


class OpenHISGateway:
    name = "openhis"

    def __init__(
        self,
        config: AdapterConfig,
        client: httpx.AsyncClient | None = None,
        readonly: ReadOnlyPool | None = None,
    ):
        self.config = config
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self.auth = TokenCache(
            config.openhis_base_url,
            config.openhis_username,
            config.openhis_password,
            config.openhis_tenant_id,
            self.client,
            config.timeout_seconds,
        )
        self.readonly = readonly if readonly is not None else ReadOnlyPool(config.readonly_dsn)

    async def close(self) -> None:
        await self.client.aclose()
        await self.readonly.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        context: str | None = None,
        auth: bool = True,
        retry: bool = True,
    ) -> Any:
        url = f"{self.config.openhis_base_url.rstrip('/')}{path}"
        headers: dict[str, str] = {}
        if auth:
            headers["Authorization"] = f"Bearer {await self.auth.get()}"
        try:
            resp = await self.client.request(
                method, url, params=params, json=json, headers=headers, timeout=self.config.timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise AdapterError("HIS_TIMEOUT", f"{method} {path} timed out", retryable=True) from exc
        except httpx.TransportError as exc:
            raise AdapterError("HIS_UNAVAILABLE", f"{method} {path} failed: {exc}", retryable=True) from exc
        if resp.status_code == 401 and auth and retry:
            await self.auth.refresh()
            return await self._request(
                method, path, params=params, json=json, context=context, auth=auth, retry=False
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise normalize_http(resp.status_code, resp.text) from exc
        if resp.status_code >= 400:
            raise normalize_http(resp.status_code, body)
        if not isinstance(body, dict):
            raise normalize_http(resp.status_code, body)
        code = body.get("code")
        if code is not None and str(code) not in ("0", "200"):
            raise normalize_business(body, context)
        return body.get("data") if "data" in body else body

    async def search_patients(
        self,
        *,
        id_card: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        birth_date: str | None = None,
    ) -> list[dict[str, Any]]:
        search_key = id_card or phone or name or ""
        try:
            data = await self._request(
                "GET",
                "/charge-manage/register/patient-metadata",
                params={"searchKey": search_key, "pageNo": 1, "pageSize": 20},
                context="patient",
            )
        except AdapterError as exc:
            if exc.code != "HIS_NOT_FOUND":
                raise
            data = await self._request(
                "GET",
                "/patient-manage/information/patient-information-page",
                params={"searchKey": search_key, "pageNo": 1, "pageSize": 20},
                context="patient",
            )
        patients = []
        for row in _records(data):
            patient_id = _first(row, "id", "patientId")
            if patient_id is None:
                continue
            patients.append(
                {
                    "his_patient_id": str(patient_id),
                    "his_bus_no": _first(row, "busNo", "patientNo"),
                    "name": _first(row, "name", "patientName"),
                    "id_card": _first(row, "idCard", "idNo"),
                    "phone": _first(row, "phone", "phoneNumber", "telcom"),
                    "birth_date": _iso(_first(row, "birthDate", "birthday")),
                    "gender": _first(row, "genderEnum_enumText", "gender"),
                }
            )
        return patients

    async def create_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        gender = str(patient.get("gender") or "unknown").lower()
        payload = {
            "name": patient.get("name"),
            "idCard": patient.get("id_card"),
            "genderEnum": GENDER_CODES.get(gender, 2),
            "birthDate": patient.get("birth_date"),
            "phone": patient.get("phone"),
            "activeFlag": 1,
        }
        data = await self._request(
            "POST",
            "/patient-manage/information/patient-information",
            json=payload,
            context="patient",
        )
        patient_id = _first(data, "id", "patientId") if isinstance(data, dict) else data
        if patient_id in (None, ""):
            matches = await self.search_patients(id_card=patient.get("id_card"))
            if not matches:
                raise AdapterError("HIS_VALIDATION", "patient created but HIS returned no patient id")
            return matches[0]
        return {
            "his_patient_id": str(patient_id),
            "his_bus_no": _first(data, "busNo", "patientNo") if isinstance(data, dict) else None,
            "name": patient.get("name"),
            "id_card": patient.get("id_card"),
            "phone": patient.get("phone"),
            "birth_date": patient.get("birth_date"),
            "gender": patient.get("gender"),
        }

    async def catalog(
        self,
        *,
        department: str | None = None,
        org_id: str | None = None,
        practitioner_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
    ) -> dict[str, Any]:
        raw_orgs = _records(
            await self._request("GET", "/charge-manage/register/org-list", context="catalog")
        )
        departments = [
            {"org_id": str(_first(row, "id", "orgId")), "name": _first(row, "name", "orgName")}
            for row in raw_orgs
            if _first(row, "id", "orgId") is not None
        ]
        if department:
            departments = [
                item
                for item in departments
                if department == item["org_id"]
                or department in str(item["name"] or "")
                or str(item["name"] or "") in department
            ]
        if org_id:
            departments = [item for item in departments if item["org_id"] == str(org_id)]
        doctors: list[dict[str, Any]] = []
        services: list[dict[str, Any]] = []
        for item in departments:
            org = item["org_id"]
            practitioner_data = await self._request(
                "GET",
                "/charge-manage/register/practitioner-metadata",
                params={"orgId": org, "pageNo": 1, "pageSize": 100},
                context="catalog",
            )
            for row in _records(practitioner_data):
                doctor_id = _first(row, "id", "practitionerId")
                if doctor_id is None:
                    continue
                doctors.append(
                    {
                        "practitioner_id": str(doctor_id),
                        "name": _first(row, "name", "practitionerName"),
                        "org_id": org,
                    }
                )
            service_data = await self._request(
                "GET",
                "/charge-manage/register/healthcare-metadata",
                params={"organizationId": org, "pageNo": 1, "pageSize": 100},
                context="catalog",
            )
            for row in _records(service_data):
                item_id = _first(row, "id", "serviceId")
                if item_id is None:
                    continue
                services.append(
                    {
                        "service_id": str(item_id),
                        "definition_id": str(_first(row, "definitionId") or item_id),
                        "name": _first(row, "name", "chargeName", "serviceName"),
                        "org_id": org,
                        "price": _amount(_first(row, "price", "activityPrice")),
                    }
                )
        if practitioner_id:
            doctors = [item for item in doctors if item["practitioner_id"] == str(practitioner_id)]
        if service_id:
            services = [item for item in services if item["service_id"] == str(service_id)]
        return {
            "departments": departments,
            "doctors": doctors,
            "services": services,
            "date": day,
        }

    async def create_quote(self, request: dict[str, Any]) -> dict[str, Any]:
        patient_id = _num(request["patient_id"])
        org_id = _num(request["org_id"])
        practitioner_id = _num(request["practitioner_id"])
        service_id = _num(request["service_id"])
        definition_id = _num(request.get("definition_id") or service_id)
        amount = _amount(request.get("amount"))
        occurred = datetime.now(UTC).isoformat()
        payload = {
            "encounterFormData": {
                "patientId": patient_id,
                "statusEnum": 1,
                "classEnum": 2,
                "ybClassEnum": 11,
                "priorityEnum": 3,
                "typeEnum": 1,
                "serviceTypeId": service_id,
                "subjectStatusEnum": 2,
                "organizationId": org_id,
            },
            "encounterLocationFormData": {"locationId": org_id, "statusEnum": 1, "formEnum": 1},
            "encounterParticipantFormData": {"typeCode": "12", "practitionerId": practitioner_id},
            "accountFormData": {
                "patientId": patient_id,
                "statusEnum": 1,
                "billingStatusEnum": 1,
                "typeCode": "04",
                "contractNo": "0000",
            },
            "chargeItemFormData": {
                "patientId": patient_id,
                "statusEnum": 1,
                "contextEnum": 4,
                "definitionId": definition_id,
                "serviceId": service_id,
                "serviceTable": "adm_healthcare_service",
                "totalPrice": amount,
                "occurrenceTime": occurred,
            },
        }
        data = await self._request(
            "POST", "/payment/payment/reg-pre-pay", json=payload, context="registration"
        )
        quote_id = _first(data, "chrgBchno", "busNo", "id") if isinstance(data, dict) else None
        if not quote_id:
            raise AdapterError("HIS_VALIDATION", "reg-pre-pay did not return a quote id")
        return {
            "quote_id": str(quote_id),
            "chrg_bchno": str(_first(data, "chrgBchno") or quote_id),
            "bus_no": _first(data, "busNo"),
            "amount": _amount(_first(data, "medfeeSumamt", "displayAmount") or amount),
            "patient_id": str(patient_id),
            "org_id": str(org_id),
            "practitioner_id": str(practitioner_id),
            "service_id": str(service_id),
            "date": request.get("date"),
            "form": payload,
        }

    async def pay_registration(
        self, quote: dict[str, Any], payment_details: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        amount = _amount(quote.get("amount"))
        details = payment_details or [
            {"payEnum": 220400, "amount": amount, "returnAmount": 0, "chargeAmount": amount}
        ]
        payload = {
            "outpatientRegistrationAddParam": quote.get("form") or {},
            "chrgBchno": quote.get("chrg_bchno") or quote.get("quote_id"),
            "busNo": quote.get("bus_no"),
            "paymentDetails": details,
        }
        data = await self._request(
            "POST", "/payment/payment/reg-pay", json=payload, context="registration"
        )
        encounter_id = None
        if isinstance(data, dict):
            encounter_id = _first(data, "encounterId", "encounter_id", "admEncounterId")
            payment = data.get("payment") if isinstance(data.get("payment"), dict) else {}
            encounter_id = encounter_id or _first(payment, "encounterId", "encounter_id")
        if not encounter_id:
            records = await self.current_day_encounter(patient_id=quote.get("patient_id"))
            match = next(
                (row for row in records if quote.get("bus_no") and row.get("bus_no") == quote.get("bus_no")),
                records[0] if records else None,
            )
            if match:
                encounter_id = match["encounter_id"]
        if not encounter_id:
            match = await self.find_encounter(
                patient_id=str(quote.get("patient_id") or ""),
                org_id=str(quote.get("org_id") or "") or None,
                service_id=str(quote.get("service_id") or "") or None,
                day=None,
            )
            if match:
                encounter_id = match["encounter_id"]
        if not encounter_id:
            raise AdapterError("HIS_VALIDATION", "reg-pay succeeded but encounter id is unknown")
        return {
            "encounter_id": str(encounter_id),
            "payment_id": str(_first(data, "id", "paymentId") or "") or None,
            "invoice_no": _first(data, "invoiceNo", "billNo"),
            "amount": _amount(_first(data, "displayAmount", "medfeeSumamt") or amount),
            "paid_at": _iso(_first(data, "billDate", "setlTime")) or datetime.now(UTC).isoformat(),
        }

    async def current_day_encounter(
        self, *, encounter_id: str | None = None, patient_id: str | None = None
    ) -> list[dict[str, Any]]:
        search_key = encounter_id or patient_id or ""
        data = await self._request(
            "GET",
            "/charge-manage/register/current-day-encounter",
            params={"searchKey": search_key, "pageNo": 1, "pageSize": 20},
            context="encounter",
        )
        encounters = []
        for row in _records(data):
            item_id = _first(row, "id", "encounterId")
            if item_id is None:
                continue
            status = _first(row, "statusEnum", "status")
            try:
                status_value = int(status)
            except (TypeError, ValueError):
                status_value = None
            encounters.append(
                {
                    "encounter_id": str(item_id),
                    "patient_id": str(_first(row, "patientId", "patient_id") or ""),
                    "org_id": str(_first(row, "organizationId", "orgId") or ""),
                    "practitioner_id": str(_first(row, "practitionerId", "doctorId") or ""),
                    "status": status_value,
                    "status_text": _first(row, "statusEnum_enumText", "statusText")
                    or ENCOUNTER_STATUS_TEXT.get(status_value, "未知"),
                    "start_time": _iso(_first(row, "startTime", "start_time")),
                    "created_at": _iso(_first(row, "createTime", "createdAt", "created_at")),
                    "bus_no": _first(row, "busNo", "bus_no"),
                    "payment_status": _first(row, "paymentStatus", "payment_status"),
                }
            )
        if encounter_id:
            encounters = [row for row in encounters if row["encounter_id"] == str(encounter_id)]
        return encounters

    async def find_encounter(
        self,
        *,
        patient_id: str,
        org_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
        created_after: str | None = None,
    ) -> dict[str, Any] | None:
        rows = await self.readonly.find_encounter(
            patient_id,
            org_id=org_id,
            service_id=service_id,
            day=day,
            created_after=created_after,
        )
        if rows:
            return rows[0]
        records = await self.current_day_encounter(patient_id=patient_id)
        for row in records:
            if org_id and str(row.get("org_id")) != str(org_id):
                continue
            if day and str(row.get("created_at") or "")[:10] != str(day)[:10]:
                continue
            if created_after and str(row.get("created_at") or "") < str(created_after):
                continue
            return row
        return None

    async def get_encounter(self, encounter_id: str) -> dict[str, Any]:
        cached = await self.readonly.encounter(str(encounter_id))
        if cached:
            return cached
        encounters = await self.current_day_encounter(encounter_id=encounter_id)
        if not encounters:
            raise AdapterError("HIS_NOT_FOUND", f"encounter {encounter_id} not found")
        return encounters[0]

    async def payment_status(self, encounter_id: str) -> dict[str, Any]:
        cached = await self.readonly.payment(str(encounter_id))
        if cached:
            return cached
        data = await self._request(
            "GET",
            "/payment/payment/page",
            params={"searchKey": str(encounter_id), "pageNo": 1, "pageSize": 20},
            context="payment",
        )
        rows = _records(data)
        match = next(
            (
                row
                for row in rows
                if str(_first(row, "encounterId", "encounter_id") or "") == str(encounter_id)
            ),
            rows[0] if rows else None,
        )
        if not match:
            return {
                "payment_status": "unpaid",
                "amount": None,
                "paid_at": None,
                "payment_id": None,
                "invoice_no": None,
            }
        status = _first(match, "statusEnum", "status", "paymentStatus", "payment_status")
        status_text = str(status or "").lower()
        invoice_no = _first(match, "invoiceNo", "billNo")
        paid_at = _iso(_first(match, "billDate", "payTime", "paidAt"))
        if status_text in {"2", "3", "4", "5"} or status_text in {"refunded", "refund", "cancel"} or "退" in str(status or ""):
            normalized = "refunded"
        elif invoice_no or paid_at or status_text in {"paid", "1", "success"}:
            normalized = "paid"
        elif status_text in {"failed", "9", "error"}:
            normalized = "failed"
        else:
            normalized = "unknown"
        return {
            "payment_status": normalized,
            "amount": _amount(_first(match, "displayAmount", "tenderedAmount", "amount")),
            "paid_at": paid_at,
            "payment_id": _first(match, "id", "paymentId"),
            "invoice_no": invoice_no,
        }

    async def list_prescriptions(
        self, encounter_id: str, include_details: bool = False
    ) -> list[dict[str, Any]]:
        cached = await self.readonly.prescriptions(encounter_id, include_details)
        if cached is not None:
            return cached
        data = await self._request(
            "GET",
            "/doctor-station/main/prescription-page-info",
            params={"encounterId": encounter_id, "pageNo": 1, "pageSize": 50},
            context="prescription",
        )
        prescriptions = []
        for row in _records(data):
            no = _first(row, "prescriptionNo", "prescription_no")
            if not no:
                continue
            items = row.get("prescriptionInfoDetailList") or row.get("items") or []
            item = {
                "his_prescription_no": str(no),
                "encounter_id": str(_first(row, "encounterId") or encounter_id),
                "patient_id": str(_first(row, "patientId") or ""),
                "status": _first(row, "statusEnum_enumText", "status"),
                "issued_at": _iso(_first(row, "requestTime", "issueTime")),
                "doctor": _first(row, "practitionerName"),
                "item_count": len(items),
            }
            if include_details:
                item["items"] = [
                    {
                        "name": _first(detail, "adviceName", "name"),
                        "quantity": _first(detail, "quantity"),
                        "dose": _first(detail, "dose"),
                        "unit": _first(detail, "doseUnitCode_dictText", "unitCode_dictText"),
                        "frequency": _first(detail, "rateCode_dictText"),
                        "route": _first(detail, "methodCode_dictText"),
                    }
                    for detail in items
                    if isinstance(detail, dict)
                ]
            prescriptions.append(item)
        return prescriptions

    async def cancel_registration(
        self, encounter_id: str, reason: str = "user_cancelled"
    ) -> dict[str, Any]:
        await self._request(
            "PUT",
            "/charge-manage/register/cancel",
            params={"encounterId": _num(encounter_id)},
            context="registration",
        )
        return {"status": "cancelled", "encounter_id": str(encounter_id), "refund_id": None}

    async def return_registration(
        self,
        encounter_id: str,
        reason: str = "user_cancelled",
        refund: bool = True,
        patient_id: str | None = None,
        amount: float | None = None,
        payment_id: str | None = None,
    ) -> dict[str, Any]:
        refund_amount = _amount(amount)
        payload = {
            "encounterId": _num(encounter_id),
            "id": _num(payment_id) if payment_id else None,
            "patientId": _num(patient_id) if patient_id else None,
            "reason": reason,
            "tenderedAmount": refund_amount,
            "returnedAmount": 0,
            "displayAmount": refund_amount,
            "paymentDetails": [
                {
                    "payEnum": 220400,
                    "amount": refund_amount,
                    "returnAmount": 0,
                    "chargeAmount": refund_amount,
                }
            ],
            "chargeItemIds": [],
        }
        data = await self._request(
            "PUT", "/charge-manage/register/return", json=payload, context="registration"
        )
        refund_id = _first(data, "refundId", "id") if isinstance(data, dict) else None
        return {
            "status": "refunded" if refund else "refund_pending",
            "encounter_id": str(encounter_id),
            "refund_id": str(refund_id) if refund_id else None,
        }


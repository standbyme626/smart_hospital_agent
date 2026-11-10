from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .errors import AdapterError

DEPARTMENTS = [
    {"org_id": "101", "name": "心血管内科"},
    {"org_id": "102", "name": "全科门诊"},
    {"org_id": "103", "name": "内分泌科"},
]

DOCTORS = [
    {"practitioner_id": "201", "name": "张医生", "org_id": "101"},
    {"practitioner_id": "202", "name": "李医生", "org_id": "101"},
    {"practitioner_id": "203", "name": "王医生", "org_id": "102"},
    {"practitioner_id": "204", "name": "赵医生", "org_id": "103"},
]

SERVICES = [
    {
        "service_id": "301",
        "definition_id": "301",
        "name": "心血管内科普通门诊",
        "org_id": "101",
        "price": 50.0,
    },
    {
        "service_id": "302",
        "definition_id": "302",
        "name": "心血管内科专家门诊",
        "org_id": "101",
        "price": 80.0,
    },
    {
        "service_id": "303",
        "definition_id": "303",
        "name": "全科门诊",
        "org_id": "102",
        "price": 30.0,
    },
    {
        "service_id": "304",
        "definition_id": "304",
        "name": "内分泌科普通门诊",
        "org_id": "103",
        "price": 40.0,
    },
]


def _now() -> datetime:
    return datetime.now(UTC)


class MockHIS:
    name = "mock"

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.patients: dict[str, dict[str, Any]] = {}
        self.quotes: dict[str, dict[str, Any]] = {}
        self.encounters: dict[str, dict[str, Any]] = {}
        self.payments: dict[str, dict[str, Any]] = {}
        self.prescriptions: dict[str, list[dict[str, Any]]] = {}
        self._seq = 1000

    def _next(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}{self._seq}"

    async def close(self) -> None:
        return None

    async def search_patients(
        self,
        *,
        id_card: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        birth_date: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = []
        for patient in self.patients.values():
            if id_card and patient.get("id_card") != id_card:
                continue
            if name and patient.get("name") != name:
                continue
            if phone and patient.get("phone") != phone:
                continue
            if birth_date and patient.get("birth_date") != birth_date:
                continue
            matches.append(dict(patient))
        return matches

    async def create_patient(self, patient: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            for existing in self.patients.values():
                if existing.get("id_card") == patient.get("id_card"):
                    raise AdapterError(
                        "HIS_DUPLICATE_PATIENT", "patient with the same id_card already exists"
                    )
            patient_id = self._next("pat_")
            row = {
                "his_patient_id": patient_id,
                "his_bus_no": self._next("PB"),
                "name": patient.get("name"),
                "id_card": patient.get("id_card"),
                "phone": patient.get("phone"),
                "birth_date": patient.get("birth_date"),
                "gender": patient.get("gender"),
            }
            self.patients[patient_id] = dict(row)
            return row

    async def catalog(
        self,
        *,
        department: str | None = None,
        org_id: str | None = None,
        practitioner_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
    ) -> dict[str, Any]:
        departments = [dict(item) for item in DEPARTMENTS]
        if department:
            departments = [
                item
                for item in departments
                if department == item["org_id"]
                or department in item["name"]
                or item["name"] in department
            ]
        if org_id:
            departments = [item for item in departments if item["org_id"] == str(org_id)]
        org_ids = {item["org_id"] for item in departments}
        doctors = [dict(item) for item in DOCTORS if item["org_id"] in org_ids]
        services = [dict(item) for item in SERVICES if item["org_id"] in org_ids]
        if practitioner_id:
            doctors = [item for item in doctors if item["practitioner_id"] == str(practitioner_id)]
        if service_id:
            services = [item for item in services if item["service_id"] == str(service_id)]
        return {"departments": departments, "doctors": doctors, "services": services, "date": day}

    async def create_quote(self, request: dict[str, Any]) -> dict[str, Any]:
        patient_id = str(request.get("patient_id"))
        if patient_id not in self.patients:
            raise AdapterError("HIS_NOT_FOUND", f"patient {patient_id} not found")
        service = next(
            (item for item in SERVICES if item["service_id"] == str(request.get("service_id"))),
            None,
        )
        if service is None:
            raise AdapterError("HIS_VALIDATION", "unknown service")
        async with self._lock:
            quote_id = self._next("QB")
            quote = {
                "quote_id": quote_id,
                "chrg_bchno": quote_id,
                "bus_no": self._next("REG"),
                "amount": round(float(request.get("amount") or service["price"]), 2),
                "patient_id": patient_id,
                "org_id": str(request.get("org_id")),
                "practitioner_id": str(request.get("practitioner_id")),
                "service_id": str(request.get("service_id")),
                "date": request.get("date") or date.today().isoformat(),
                "expires_at": (_now() + timedelta(minutes=15)).isoformat(),
                "encounter_id": None,
                "form": dict(request),
            }
            self.quotes[quote_id] = quote
        return dict(quote)

    async def pay_registration(
        self, quote: dict[str, Any], payment_details: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        quote_id = str(quote.get("quote_id"))
        async with self._lock:
            stored = self.quotes.get(quote_id)
            if stored is None:
                raise AdapterError("HIS_NOT_FOUND", f"quote {quote_id} not found")
            if stored.get("encounter_id"):
                raise AdapterError(
                    "HIS_DUPLICATE_REGISTRATION", "quote already registered", retryable=False
                )
            key = (stored["patient_id"], stored["service_id"], stored["date"])
            for encounter in self.encounters.values():
                if (
                    encounter["patient_id"],
                    encounter["service_id"],
                    encounter["appointment_date"],
                ) == key and encounter["status"] != "CANCELLED":
                    raise AdapterError(
                        "HIS_DUPLICATE_REGISTRATION",
                        "same patient already registered for this service today",
                        retryable=False,
                    )
            encounter_id = self._next("enc_")
            payment_id = self._next("pay_")
            invoice_no = self._next("INV")
            doctor = next(
                (item for item in DOCTORS if item["practitioner_id"] == stored["practitioner_id"]),
                None,
            )
            department = next(
                (item for item in DEPARTMENTS if item["org_id"] == stored["org_id"]), None
            )
            encounter = {
                "encounter_id": encounter_id,
                "patient_id": stored["patient_id"],
                "org_id": stored["org_id"],
                "practitioner_id": stored["practitioner_id"],
                "service_id": stored["service_id"],
                "status": 1,
                "status_text": "待诊",
                "his_status": "PLANNED",
                "start_time": _now().isoformat(),
                "created_at": _now().isoformat(),
                "bus_no": stored["bus_no"],
                "appointment_date": stored["date"],
                "doctor_name": doctor["name"] if doctor else None,
                "department": department["name"] if department else None,
            }
            stored["encounter_id"] = encounter_id
            self.encounters[encounter_id] = encounter
            self.payments[encounter_id] = {
                "payment_status": "paid",
                "amount": stored["amount"],
                "paid_at": _now().isoformat(),
                "payment_id": payment_id,
                "invoice_no": invoice_no,
            }
            self.prescriptions.setdefault(encounter_id, self._seed_prescriptions(encounter_id))
            return {
                "encounter_id": encounter_id,
                "payment_id": payment_id,
                "invoice_no": invoice_no,
                "amount": stored["amount"],
                "paid_at": self.payments[encounter_id]["paid_at"],
            }

    @staticmethod
    def _seed_prescriptions(encounter_id: str) -> list[dict[str, Any]]:
        return [
            {
                "his_prescription_no": f"RX{encounter_id[-4:]}",
                "encounter_id": encounter_id,
                "patient_id": "",
                "status": "已开立",
                "issued_at": _now().isoformat(),
                "doctor": None,
                "items": [
                    {
                        "name": "阿司匹林肠溶片",
                        "quantity": 30,
                        "dose": 0.1,
                        "unit": "g",
                        "frequency": "每日一次",
                        "route": "口服",
                    },
                    {
                        "name": "阿托伐他汀钙片",
                        "quantity": 7,
                        "dose": 20,
                        "unit": "mg",
                        "frequency": "每晚一次",
                        "route": "口服",
                    },
                ],
            }
        ]

    async def find_encounter(
        self,
        *,
        patient_id: str,
        org_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
        created_after: str | None = None,
    ) -> dict[str, Any] | None:
        for encounter in self.encounters.values():
            if str(encounter.get("patient_id")) != str(patient_id):
                continue
            if org_id and str(encounter.get("org_id")) != str(org_id):
                continue
            if day and str(encounter.get("appointment_date") or "")[:10] != str(day)[:10]:
                continue
            return dict(encounter)
        return None

    async def get_encounter(self, encounter_id: str) -> dict[str, Any]:
        encounter = self.encounters.get(str(encounter_id))
        if encounter is None:
            raise AdapterError("HIS_NOT_FOUND", f"encounter {encounter_id} not found")
        return dict(encounter)

    async def payment_status(self, encounter_id: str) -> dict[str, Any]:
        payment = self.payments.get(str(encounter_id))
        if payment is None:
            encounter = self.encounters.get(str(encounter_id))
            amount = None
            if encounter:
                quote = next(
                    (
                        item
                        for item in self.quotes.values()
                        if item.get("encounter_id") == str(encounter_id)
                    ),
                    None,
                )
                amount = quote["amount"] if quote else None
            return {
                "payment_status": "unpaid",
                "amount": amount,
                "paid_at": None,
                "payment_id": None,
                "invoice_no": None,
            }
        return dict(payment)

    async def list_prescriptions(
        self, encounter_id: str, include_details: bool = False
    ) -> list[dict[str, Any]]:
        rows = self.prescriptions.get(str(encounter_id), [])
        result = []
        for row in rows:
            item = {key: value for key, value in row.items() if key != "items"}
            item["item_count"] = len(row.get("items") or [])
            if include_details:
                item["items"] = [dict(entry) for entry in row.get("items") or []]
            result.append(item)
        return result

    async def cancel_registration(
        self, encounter_id: str, reason: str = "user_cancelled"
    ) -> dict[str, Any]:
        encounter = self.encounters.get(str(encounter_id))
        if encounter is None:
            raise AdapterError("HIS_NOT_FOUND", f"encounter {encounter_id} not found")
        payment = self.payments.get(str(encounter_id))
        if payment and payment["payment_status"] == "paid":
            raise AdapterError("HIS_CONFLICT", "paid registration must be returned, not cancelled")
        if encounter["status"] == "CANCELLED":
            raise AdapterError("HIS_CONFLICT", "registration already cancelled")
        encounter["status"] = "CANCELLED"
        encounter["status_text"] = "退号"
        encounter["his_status"] = "CANCELLED"
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
        encounter = self.encounters.get(str(encounter_id))
        if encounter is None:
            raise AdapterError("HIS_NOT_FOUND", f"encounter {encounter_id} not found")
        payment = self.payments.get(str(encounter_id))
        if not payment or payment["payment_status"] != "paid":
            raise AdapterError("HIS_PAYMENT_REQUIRED", "registration has no paid record to refund")
        encounter["status"] = "CANCELLED"
        encounter["status_text"] = "退号"
        encounter["his_status"] = "CANCELLED"
        payment["payment_status"] = "refunded" if refund else "refund_pending"
        payment["refund_id"] = self._next("RF")
        return {
            "status": "refunded" if refund else "refund_pending",
            "encounter_id": str(encounter_id),
            "refund_id": payment["refund_id"],
        }

    async def simulate_payment(self, encounter_id: str) -> dict[str, Any]:
        async with self._lock:
            encounter = self.encounters.get(str(encounter_id))
            if encounter is None:
                raise AdapterError("HIS_NOT_FOUND", f"encounter {encounter_id} not found")
            payment = self.payments.get(str(encounter_id))
            if payment and payment["payment_status"] == "paid":
                return dict(payment)
            payment = {
                "payment_status": "paid",
                "amount": 0.0,
                "paid_at": _now().isoformat(),
                "payment_id": uuid.uuid4().hex[:12],
                "invoice_no": self._next("INV"),
            }
            self.payments[str(encounter_id)] = payment
            self.prescriptions.setdefault(str(encounter_id), self._seed_prescriptions(str(encounter_id)))
            return dict(payment)

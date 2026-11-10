from __future__ import annotations

import hashlib
import hmac
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from time import perf_counter
from typing import Any
from zoneinfo import ZoneInfo

from .config import AdapterConfig
from .errors import (
    ADAPTER_IDEMPOTENCY_CONFLICT,
    ADAPTER_MAPPING_MISSING,
    ADAPTER_OUTBOX_PENDING,
    ADAPTER_SCOPE_DENIED,
    HIS_CONFLICT,
    HIS_NOT_FOUND,
    HIS_UNSUPPORTED,
    HIS_VALIDATION,
    AdapterError,
)
from .gateway import Gateway, build_gateway
from .store import Store

logger = logging.getLogger("his-adapter.service")

READ_TOOLS = frozenset(
    {
        "patient.resolve",
        "schedule.list",
        "payment.status",
        "encounter.get",
        "prescription.list",
        "visit.status",
    }
)
WRITE_TOOLS = frozenset(
    {
        "patient.create",
        "appointment.create",
        "registration.create",
        "payment.request",
        "appointment.cancel",
        "his.reconcile",
        "payment.simulate",
    }
)
ALL_TOOLS = READ_TOOLS | WRITE_TOOLS
SCOPES = frozenset({"agent_read", "service_write"})

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _now() -> datetime:
    return datetime.now(UTC)


def _str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def _iso_value(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


class AdapterService:
    def __init__(
        self,
        config: AdapterConfig,
        store: Store | None = None,
        gateway: Gateway | None = None,
    ):
        self.config = config
        self.store = store or Store(config.database_url)
        self.gateway = gateway or build_gateway(config)
        self.handlers: dict[str, Handler] = {
            "patient.resolve": self.patient_resolve,
            "patient.create": self.patient_create,
            "schedule.list": self.schedule_list,
            "appointment.create": self.appointment_create,
            "registration.create": self.registration_create,
            "payment.request": self.payment_request,
            "payment.status": self.payment_status,
            "encounter.get": self.encounter_get,
            "prescription.list": self.prescription_list,
            "visit.status": self.visit_status,
            "appointment.cancel": self.appointment_cancel,
            "his.reconcile": self.his_reconcile,
            "payment.simulate": self.payment_simulate,
        }

    async def startup(self) -> None:
        await self.store.init_schema()

    async def close(self) -> None:
        await self.gateway.close()
        await self.store.close()

    async def call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        scope: str = "service_write",
    ) -> dict[str, Any]:
        started = perf_counter()
        args = dict(arguments or {})
        stale = False
        try:
            if scope not in SCOPES:
                raise AdapterError(ADAPTER_SCOPE_DENIED, f"unknown adapter scope {scope!r}")
            if scope == "agent_read" and tool not in READ_TOOLS:
                raise AdapterError(
                    ADAPTER_SCOPE_DENIED, f"scope agent_read may not call {tool!r}"
                )
            handler = self.handlers.get(tool)
            if handler is None:
                raise AdapterError(HIS_UNSUPPORTED, f"unknown tool {tool!r}")
            data = await handler(args)
            stale = bool(data.get("stale")) if isinstance(data, dict) else False
            return self._envelope(started, data=data, stale=stale)
        except AdapterError as exc:
            return self._envelope(started, error=exc.as_dict(), stale=stale)
        except Exception as exc:
            error = AdapterError("HIS_UNAVAILABLE", f"adapter internal error: {exc}", retryable=False)
            return self._envelope(started, error=error.as_dict(), stale=stale)

    def _envelope(
        self,
        started: float,
        *,
        data: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        stale: bool = False,
    ) -> dict[str, Any]:
        source = self.gateway.name
        if isinstance(data, dict) and data.get("source"):
            source = str(data["source"])
        return {
            "ok": error is None,
            "data": data,
            "error": error,
            "meta": {
                "source": source,
                "stale": bool(stale),
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            },
        }

    @staticmethod
    def _require(args: dict[str, Any], *names: str) -> None:
        missing = [name for name in names if _str(args.get(name)) is None]
        if missing:
            raise AdapterError(HIS_VALIDATION, f"missing required arguments: {', '.join(missing)}")

    async def _idempotent(
        self,
        *,
        aggregate_type: str,
        aggregate_id: str,
        operation: str,
        payload: dict[str, Any],
        key: str | None,
        action: Callable[[], Awaitable[dict[str, Any]]],
    ) -> dict[str, Any]:
        if not key:
            return await action()
        begun = await self.store.outbox_begin(
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            operation=operation,
            payload=payload,
            idempotency_key=key,
        )
        state = begun["state"]
        if state == "replay":
            return dict(begun["response"])
        if state == "pending":
            raise AdapterError(
                ADAPTER_OUTBOX_PENDING,
                f"operation {operation!r} with the same idempotency_key is still pending",
                retryable=True,
            )
        if state == "conflict":
            logger.warning(
                "idempotency conflict key=%s new=%s old=%s",
                key,
                payload,
                (begun.get("row") or {}).get("payload"),
            )
            raise AdapterError(
                ADAPTER_IDEMPOTENCY_CONFLICT,
                "idempotency_key was reused with different arguments",
            )
        outbox_id = begun["id"]
        try:
            result = await action()
        except AdapterError as exc:
            await self.store.outbox_complete(
                outbox_id, "pending" if exc.retryable else "dead", error=exc.message
            )
            raise
        except Exception as exc:
            await self.store.outbox_complete(outbox_id, "pending", error=str(exc))
            raise AdapterError("HIS_UNAVAILABLE", f"{operation} failed: {exc}", retryable=True) from exc
        await self.store.outbox_complete(outbox_id, "done", response=result)
        return result

    async def _projection_for(self, args: dict[str, Any]) -> dict[str, Any] | None:
        appointment_id = _str(args.get("appointment_id"))
        if appointment_id:
            return await self.store.get_projection(appointment_id)
        encounter_id = _str(args.get("encounter_id"))
        if encounter_id:
            return await self.store.find_projection_by_encounter(encounter_id)
        return None

    async def _encounter_id_for(self, args: dict[str, Any]) -> str | None:
        encounter_id = _str(args.get("encounter_id"))
        if encounter_id:
            return encounter_id
        projection = await self._projection_for(args)
        if projection:
            existing = _str(projection.get("his_encounter_id"))
            if existing:
                return existing
            found = await self.gateway.find_encounter(
                patient_id=str(projection.get("his_patient_id") or ""),
                org_id=_str(projection.get("his_org_id")),
                service_id=_str(projection.get("his_service_id")),
                day=_str(projection.get("appointment_date")),
                created_after=_str(projection.get("created_at")),
            )
            if not found:
                found = await self.gateway.find_encounter(
                    patient_id=str(projection.get("his_patient_id") or ""),
                    org_id=_str(projection.get("his_org_id")),
                    service_id=_str(projection.get("his_service_id")),
                    day=None,
                    created_after=_str(projection.get("created_at")),
                )
            if found:
                await self.store.update_projection(
                    projection["id"],
                    his_encounter_id=str(found["encounter_id"]),
                    his_status=str(found.get("status")),
                    last_synced_at=_now().isoformat(),
                )
                await self.store.upsert_encounter_map(
                    projection["id"],
                    {
                        "his_encounter_id": str(found["encounter_id"]),
                        "his_patient_id": projection.get("his_patient_id"),
                        "his_org_id": projection.get("his_org_id"),
                        "his_practitioner_id": projection.get("his_practitioner_id"),
                        "his_service_id": projection.get("his_service_id"),
                        "quote_id": projection.get("quote_id"),
                        "his_status": str(found.get("status")),
                        "last_synced_at": _now().isoformat(),
                    },
                )
                return str(found["encounter_id"])
        return None

    @staticmethod
    def _candidate(patient: dict[str, Any]) -> dict[str, Any]:
        id_card = _str(patient.get("id_card"))
        phone = _str(patient.get("phone"))
        return {
            "his_patient_id": str(patient.get("his_patient_id")),
            "his_bus_no": patient.get("his_bus_no"),
            "name": patient.get("name"),
            "id_card_last4": id_card[-4:] if id_card else None,
            "phone_last4": phone[-4:] if phone else None,
            "birth_date": patient.get("birth_date"),
            "gender": patient.get("gender"),
        }

    @staticmethod
    def _match(
        candidates: list[dict[str, Any]],
        *,
        name: str | None,
        id_card: str | None,
        phone: str | None,
        birth_date: str | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if id_card:
            for item in candidates:
                if item.get("id_card") == id_card:
                    return item, "id_card"
        if name and birth_date:
            for item in candidates:
                if item.get("name") == name and item.get("birth_date") == birth_date:
                    return item, "name_birth_date"
        if name and phone:
            for item in candidates:
                if item.get("name") == name and item.get("phone") == phone:
                    return item, "name_phone"
        return None, None

    async def patient_resolve(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "member_id", "name")
        member_id = str(args["member_id"])
        user_id = _str(args.get("user_id")) or ""
        name = _str(args.get("name"))
        id_card = _str(args.get("id_card"))
        phone = _str(args.get("phone"))
        birth_date = _str(args.get("birth_date"))
        cached = await self.store.get_patient_map(member_id, user_id)
        try:
            candidates = await self.gateway.search_patients(
                id_card=id_card, name=name, phone=phone, birth_date=birth_date
            )
        except AdapterError as exc:
            if cached and exc.retryable:
                return self._found_payload(
                    cached, cached.get("match_method"), source="cache", stale=True
                )
            raise
        match, method = self._match(
            candidates, name=name, id_card=id_card, phone=phone, birth_date=birth_date
        )
        if match:
            mapping = await self._save_patient_map(member_id, user_id, match, method or "matched")
            return self._found_payload(mapping, method, source=self.gateway.name, stale=False)
        return {
            "found": False,
            "his_patient_id": None,
            "his_bus_no": None,
            "match_method": None,
            "candidates": [self._candidate(item) for item in candidates[:10]],
            "source": self.gateway.name,
            "stale": False,
        }

    def _found_payload(
        self, mapping: dict[str, Any], method: str | None, *, source: str, stale: bool
    ) -> dict[str, Any]:
        return {
            "found": True,
            "his_patient_id": str(mapping.get("his_patient_id")),
            "his_bus_no": mapping.get("his_bus_no"),
            "match_method": method,
            "candidates": [],
            "source": source,
            "stale": stale,
        }

    async def _save_patient_map(
        self,
        member_id: str,
        user_id: str,
        patient: dict[str, Any],
        method: str,
    ) -> dict[str, Any]:
        id_card = _str(patient.get("id_card")) or ""
        return await self.store.upsert_patient_map(
            {
                "user_id": user_id,
                "member_id": member_id,
                "his_tenant_id": self.config.openhis_tenant_id or None,
                "his_patient_id": str(patient["his_patient_id"]),
                "his_bus_no": patient.get("his_bus_no"),
                "id_card_hash": (
                    hmac.new(
                        (self.config.id_card_key or "dev-id-card-key").encode(),
                        id_card.encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    if id_card
                    else None
                ),
                "id_card_last4": id_card[-4:] if id_card else None,
                "match_method": method,
            }
        )

    async def patient_create(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "member_id", "name", "id_card")
        member_id = str(args["member_id"])
        user_id = _str(args.get("user_id")) or ""
        existing = await self.store.get_patient_map(member_id, user_id)
        if existing:
            return {
                "his_patient_id": str(existing["his_patient_id"]),
                "his_bus_no": existing.get("his_bus_no"),
                "created": False,
                "match_method": existing.get("match_method") or "mapping",
            }
        id_card = str(args["id_card"])
        patient = {
            "name": str(args["name"]),
            "id_card": id_card,
            "gender": str(args.get("gender") or ""),
            "birth_date": str(args.get("birth_date") or ""),
            "phone": str(args.get("phone") or ""),
        }

        async def action() -> dict[str, Any]:
            matches = await self.gateway.search_patients(id_card=id_card)
            existing = next(
                (item for item in matches if str(item.get("id_card")) == id_card), None
            )
            if existing:
                mapping = await self._save_patient_map(member_id, user_id, existing, "id_card")
                return {
                    "his_patient_id": str(mapping["his_patient_id"]),
                    "his_bus_no": mapping.get("his_bus_no"),
                    "created": False,
                    "match_method": "id_card",
                }
            created = await self.gateway.create_patient(patient)
            mapping = await self._save_patient_map(member_id, user_id, created, "created")
            return {
                "his_patient_id": str(mapping["his_patient_id"]),
                "his_bus_no": mapping.get("his_bus_no"),
                "created": True,
                "match_method": "created",
            }

        payload = {
            "member_id": member_id,
            "id_card_hash": hmac.new(
                (self.config.id_card_key or "dev-id-card-key").encode(),
                id_card.encode(),
                hashlib.sha256,
            ).hexdigest(),
            "name": patient["name"],
            "birth_date": patient["birth_date"],
        }
        return await self._idempotent(
            aggregate_type="patient",
            aggregate_id=member_id,
            operation="patient.create",
            payload=payload,
            key=_str(args.get("idempotency_key")),
            action=action,
        )

    async def schedule_list(self, args: dict[str, Any]) -> dict[str, Any]:
        day = _today()
        catalog = await self.gateway.catalog(
            department=_str(args.get("department")),
            org_id=_str(args.get("org_id")),
            practitioner_id=_str(args.get("practitioner_id")),
            service_id=_str(args.get("service_id")),
            day=day,
        )
        return {
            "departments": catalog.get("departments") or [],
            "doctors": catalog.get("doctors") or [],
            "services": catalog.get("services") or [],
            "date": day,
            "availability_mode": "catalog_only",
            "slots": [],
            "source": self.gateway.name,
            "stale": False,
        }

    async def appointment_create(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "member_id", "org_id", "practitioner_id", "service_id", "date")
        member_id = str(args["member_id"])
        user_id = _str(args.get("user_id")) or ""
        mapping = await self.store.get_patient_map(member_id, user_id)
        if not mapping:
            raise AdapterError(
                ADAPTER_MAPPING_MISSING, f"member {member_id!r} has no HIS patient mapping"
            )
        org_id = str(args["org_id"])
        practitioner_id = str(args["practitioner_id"])
        service_id = str(args["service_id"])
        day = str(args["date"])
        catalog = await self.gateway.catalog(
            org_id=org_id, practitioner_id=practitioner_id, service_id=service_id, day=day
        )
        service = next(
            (item for item in catalog.get("services") or [] if str(item.get("service_id")) == service_id),
            None,
        )
        if service is None:
            raise AdapterError(HIS_VALIDATION, f"unknown service_id {service_id!r} for org {org_id!r}")
        doctor = next(
            (
                item
                for item in catalog.get("doctors") or []
                if str(item.get("practitioner_id")) == practitioner_id
            ),
            None,
        )
        department = next(
            (item for item in catalog.get("departments") or [] if str(item.get("org_id")) == org_id),
            None,
        )
        amount = float(service.get("price") or 0)
        appointment_id = f"appt_{uuid.uuid4().hex}"
        key = _str(args.get("idempotency_key"))
        quote_request = {
            "patient_id": mapping["his_patient_id"],
            "org_id": org_id,
            "practitioner_id": practitioner_id,
            "service_id": service_id,
            "definition_id": service.get("definition_id") or service_id,
            "amount": amount,
            "date": day,
        }

        async def action() -> dict[str, Any]:
            quote = await self.gateway.create_quote(quote_request)
            expires_at = quote.get("expires_at") or (
                _now() + timedelta(seconds=self.config.quote_ttl_seconds)
            ).isoformat()
            projection = await self.store.create_projection(
                {
                    "id": appointment_id,
                    "user_id": user_id,
                    "member_id": member_id,
                    "his_patient_id": str(mapping["his_patient_id"]),
                    "his_org_id": org_id,
                    "his_practitioner_id": practitioner_id,
                    "his_service_id": service_id,
                    "quote_id": quote["quote_id"],
                    "amount": amount,
                    "status": "quoted",
                    "payment_status": "unpaid",
                    "appointment_date": day,
                    "department": department.get("name") if department else None,
                    "doctor_name": doctor.get("name") if doctor else None,
                    "payload": {"quote": quote},
                    "idempotency_key": key,
                    "stale": False,
                    "legacy_demo": False,
                }
            )
            await self.store.upsert_encounter_map(
                appointment_id,
                {
                    "his_patient_id": str(mapping["his_patient_id"]),
                    "his_org_id": org_id,
                    "his_practitioner_id": practitioner_id,
                    "his_service_id": service_id,
                    "quote_id": quote["quote_id"],
                    "payment_status": "unpaid",
                },
            )
            return {
                "appointment_id": projection["id"],
                "quote_id": quote["quote_id"],
                "amount": round(float(projection.get("amount") or amount), 2),
                "currency": "CNY",
                "status": "quoted",
                "expires_at": expires_at,
                "payment_mode": self.config.payment_mode,
            }

        return await self._idempotent(
            aggregate_type="appointment",
            aggregate_id=appointment_id,
            operation="appointment.create",
            payload={
                "member_id": member_id,
                "org_id": org_id,
                "practitioner_id": practitioner_id,
                "service_id": service_id,
                "date": day,
            },
            key=key,
            action=action,
        )

    async def registration_create(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "appointment_id")
        payment_details = args.get("payment_details")
        if payment_details is not None and not isinstance(payment_details, list):
            raise AdapterError(HIS_VALIDATION, "payment_details must be a list")
        appointment_id = str(args["appointment_id"])
        projection = await self.store.get_projection(appointment_id)
        if not projection:
            raise AdapterError(HIS_NOT_FOUND, f"appointment {appointment_id!r} not found")
        if projection.get("status") == "registered":
            return {
                "encounter_id": projection.get("his_encounter_id"),
                "status": "registered",
                "payment_status": projection.get("payment_status") or "paid",
                "quote_id": projection.get("quote_id"),
                "next_action": "visit",
            }
        quote = (projection.get("payload") or {}).get("quote")
        key = _str(args.get("idempotency_key"))

        async def action() -> dict[str, Any]:
            if self.config.payment_mode == "direct":
                if not quote:
                    raise AdapterError(ADAPTER_MAPPING_MISSING, "stored quote is missing")
                payment = await self.gateway.pay_registration(quote, payment_details)
                synced_at = _now().isoformat()
                await self.store.update_projection(
                    appointment_id,
                    his_encounter_id=payment["encounter_id"],
                    payment_id=payment.get("payment_id"),
                    invoice_id=payment.get("invoice_no"),
                    status="registered",
                    payment_status="paid",
                    his_status="PLANNED",
                    last_synced_at=synced_at,
                    stale=False,
                )
                await self.store.upsert_encounter_map(
                    appointment_id,
                    {
                        "his_encounter_id": payment["encounter_id"],
                        "his_patient_id": projection.get("his_patient_id"),
                        "his_org_id": projection.get("his_org_id"),
                        "his_practitioner_id": projection.get("his_practitioner_id"),
                        "his_service_id": projection.get("his_service_id"),
                        "quote_id": projection.get("quote_id"),
                        "payment_id": payment.get("payment_id"),
                        "invoice_id": payment.get("invoice_no"),
                        "his_status": "PLANNED",
                        "payment_status": "paid",
                        "last_synced_at": synced_at,
                    },
                )
                return {
                    "encounter_id": payment["encounter_id"],
                    "status": "registered",
                    "payment_status": "paid",
                    "quote_id": projection.get("quote_id"),
                    "next_action": "visit",
                }
            await self.store.update_projection(
                appointment_id, status="PENDING_CASHIER", stale=False
            )
            return {
                "encounter_id": projection.get("his_encounter_id"),
                "status": "PENDING_CASHIER",
                "payment_status": projection.get("payment_status") or "unpaid",
                "quote_id": projection.get("quote_id"),
                "next_action": "payment.request",
            }

        return await self._idempotent(
            aggregate_type="registration",
            aggregate_id=appointment_id,
            operation="registration.create",
            payload={"appointment_id": appointment_id, "payment_mode": self.config.payment_mode},
            key=key,
            action=action,
        )

    async def payment_request(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "appointment_id")
        appointment_id = str(args["appointment_id"])
        projection = await self.store.get_projection(appointment_id)
        if not projection:
            raise AdapterError(HIS_NOT_FOUND, f"appointment {appointment_id!r} not found")
        if projection.get("payment_status") == "paid":
            return {
                "task_id": None,
                "status": "paid",
                "instructions": "挂号费已支付，无需重复缴费。",
                "sla_due_at": None,
            }
        amount = round(float(projection.get("amount") or 0), 2)
        instructions = (
            f"挂号单号 {projection.get('quote_id')}，应付 {amount:.2f} 元。"
            "请到收费窗口或自助机完成支付，支付后状态会自动同步。"
        )
        key = _str(args.get("idempotency_key"))

        async def action() -> dict[str, Any]:
            existing = await self.store.find_open_task("registration_payment", appointment_id)
            if existing:
                task = existing
            else:
                task = await self.store.create_human_task(
                    {
                        "kind": "registration_payment",
                        "ref_id": appointment_id,
                        "user_id": projection.get("user_id") or "",
                        "member_id": projection.get("member_id") or "",
                        "status": "requested",
                        "payload": {
                            "appointment_id": appointment_id,
                            "quote_id": projection.get("quote_id"),
                            "amount": amount,
                            "payment_mode": self.config.payment_mode,
                            "instructions": instructions,
                        },
                        "assignee": None,
                        "sla_due_at": _now()
                        + timedelta(minutes=self.config.task_sla_minutes),
                    }
                )
            return {
                "task_id": task["id"],
                "status": "requested",
                "instructions": instructions,
                "sla_due_at": _iso_value(task.get("sla_due_at")),
            }

        return await self._idempotent(
            aggregate_type="payment_task",
            aggregate_id=appointment_id,
            operation="payment.request",
            payload={"appointment_id": appointment_id, "amount": amount},
            key=key,
            action=action,
        )

    async def payment_status(self, args: dict[str, Any]) -> dict[str, Any]:
        if not _str(args.get("appointment_id")) and not _str(args.get("encounter_id")):
            raise AdapterError(HIS_VALIDATION, "appointment_id or encounter_id is required")
        projection = await self._projection_for(args)
        encounter_id = await self._encounter_id_for(args)
        if not encounter_id:
            if not projection:
                raise AdapterError(HIS_NOT_FOUND, "no appointment or encounter mapping found")
            return {
                "payment_status": projection.get("payment_status") or "unpaid",
                "amount": float(projection.get("amount") or 0),
                "paid_at": None,
                "payment_id": None,
                "invoice_no": None,
                "source": self.gateway.name,
                "stale": bool(projection.get("stale")),
            }
        payment = await self.gateway.payment_status(encounter_id)
        if projection:
            await self.store.update_projection(
                projection["id"],
                payment_status=payment.get("payment_status") or "unknown",
                payment_id=payment.get("payment_id"),
                invoice_id=payment.get("invoice_no"),
                last_synced_at=_now().isoformat(),
            )
        return {
            "payment_status": payment.get("payment_status") or "unknown",
            "amount": payment.get("amount"),
            "paid_at": payment.get("paid_at"),
            "payment_id": payment.get("payment_id"),
            "invoice_no": payment.get("invoice_no"),
            "source": self.gateway.name,
            "stale": False,
        }

    async def encounter_get(self, args: dict[str, Any]) -> dict[str, Any]:
        encounter_id = await self._encounter_id_for(args)
        if not encounter_id:
            raise AdapterError(HIS_NOT_FOUND, "no encounter mapping found for this appointment")
        projection = await self._projection_for(args)
        try:
            encounter = await self.gateway.get_encounter(encounter_id)
            stale = False
        except AdapterError as exc:
            if not projection or not exc.retryable:
                raise
            encounter = {
                "encounter_id": encounter_id,
                "patient_id": projection.get("his_patient_id"),
                "org_id": projection.get("his_org_id"),
                "practitioner_id": projection.get("his_practitioner_id"),
                "status": projection.get("his_status"),
                "status_text": "未知",
                "start_time": None,
                "created_at": None,
                "bus_no": None,
                "payment_status": projection.get("payment_status"),
            }
            stale = True
        if projection:
            await self.store.update_projection(
                projection["id"],
                his_encounter_id=encounter_id,
                his_status=str(encounter.get("status")),
                last_synced_at=_now().isoformat(),
                stale=stale,
            )
        payment_status = encounter.get("payment_status")
        if not payment_status and projection:
            payment_status = projection.get("payment_status")
        return {
            "encounter_id": str(encounter["encounter_id"]),
            "patient_id": encounter.get("patient_id"),
            "org_id": encounter.get("org_id"),
            "practitioner_id": encounter.get("practitioner_id"),
            "status": encounter.get("status"),
            "status_text": encounter.get("status_text"),
            "start_time": encounter.get("start_time"),
            "created_at": encounter.get("created_at"),
            "payment_status": payment_status,
            "visit_no": encounter.get("bus_no"),
            "source": self.gateway.name,
            "stale": stale,
        }

    async def prescription_list(self, args: dict[str, Any]) -> dict[str, Any]:
        if not _str(args.get("encounter_id")) and not _str(args.get("appointment_id")):
            raise AdapterError(HIS_VALIDATION, "encounter_id or appointment_id is required")
        encounter_id = await self._encounter_id_for(args)
        if not encounter_id:
            raise AdapterError(HIS_NOT_FOUND, "no encounter mapping found for this appointment")
        include_details = _bool(args.get("include_details"))
        prescriptions = await self.gateway.list_prescriptions(encounter_id, include_details)
        for item in prescriptions:
            await self.store.upsert_prescription_map(
                {
                    "his_prescription_no": item["his_prescription_no"],
                    "his_encounter_id": encounter_id,
                    "his_patient_id": item.get("patient_id"),
                    "source_table": self.gateway.name,
                    "status": item.get("status"),
                    "issued_at": item.get("issued_at"),
                }
            )
        return {"prescriptions": prescriptions, "source": self.gateway.name, "stale": False}

    async def visit_status(self, args: dict[str, Any]) -> dict[str, Any]:
        if not _str(args.get("appointment_id")) and not _str(args.get("encounter_id")):
            raise AdapterError(HIS_VALIDATION, "appointment_id or encounter_id is required")
        projection = await self._projection_for(args)
        encounter_id = await self._encounter_id_for(args)
        encounter = None
        if encounter_id:
            try:
                encounter = await self.gateway.get_encounter(encounter_id)
            except AdapterError as exc:
                if not exc.retryable:
                    raise
                encounter = None
        payment_status = "unknown"
        if encounter_id:
            try:
                payment_status = (
                    await self.gateway.payment_status(encounter_id)
                ).get("payment_status") or "unknown"
            except AdapterError as exc:
                if not exc.retryable:
                    raise
                payment_status = "unknown"
        elif projection:
            payment_status = projection.get("payment_status") or "unpaid"
        local_status = str(projection.get("status") or "") if projection else ""
        encounter_status = str(encounter.get("status")) if encounter else ""
        if local_status == "cancelled" or encounter_status in ("6", "CANCELLED"):
            state = "cancelled"
        elif local_status == "refunded" or payment_status == "refunded":
            state = "refunded"
        elif encounter_status in ("4", "5"):
            state = "completed"
        elif payment_status == "paid":
            state = "registered"
        elif local_status == "PENDING_CASHIER":
            state = "awaiting_payment"
        elif local_status == "quoted":
            state = "appointment_quoted"
        else:
            state = "unknown"
        guide = {
            "registered": "请提前 20 分钟到院，携带身份证/医保卡到自助机取号后候诊。",
            "awaiting_payment": "请先完成挂号费支付，支付后即可取号就诊。",
            "appointment_quoted": "预约已报价，请确认后创建挂号并完成支付。",
            "completed": "本次就诊已完成，如需复诊请重新预约。",
            "cancelled": "预约已取消。",
            "refunded": "挂号费已退回，请留意到账时间。",
        }.get(state, "请稍后重试或联系人工客服。")
        next_action = {
            "registered": "visit",
            "awaiting_payment": "payment.request",
            "appointment_quoted": "registration.create",
            "completed": "followup",
            "cancelled": "rebook",
            "refunded": "rebook",
        }.get(state, "contact_support")
        return {
            "state": state,
            "encounter_id": encounter_id,
            "payment_status": payment_status,
            "appointment_date": (encounter or {}).get("appointment_date")
            or (projection or {}).get("appointment_date"),
            "doctor": (encounter or {}).get("doctor_name") or (projection or {}).get("doctor_name"),
            "department": (encounter or {}).get("department")
            or (projection or {}).get("department"),
            "guide": guide,
            "next_action": next_action,
            "source": self.gateway.name,
            "stale": bool(projection.get("stale")) if projection else False,
        }

    async def appointment_cancel(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "appointment_id")
        appointment_id = str(args["appointment_id"])
        projection = await self.store.get_projection(appointment_id)
        if not projection:
            raise AdapterError(HIS_NOT_FOUND, f"appointment {appointment_id!r} not found")
        if projection.get("status") in ("cancelled", "refunded"):
            return {
                "status": "refunded" if projection.get("status") == "refunded" else "cancelled",
                "encounter_id": projection.get("his_encounter_id"),
                "refund_id": None,
            }
        reason = _str(args.get("reason")) or "user_cancelled"
        refund = _bool(args.get("refund", True))
        encounter_id = _str(projection.get("his_encounter_id"))
        key = _str(args.get("idempotency_key"))

        async def action() -> dict[str, Any]:
            status = "cancelled"
            refund_id = None
            if encounter_id:
                try:
                    payment = await self.gateway.payment_status(encounter_id)
                    if payment.get("payment_status") == "paid":
                        result = await self.gateway.return_registration(
                            encounter_id,
                            reason=reason,
                            refund=refund,
                            patient_id=projection.get("his_patient_id"),
                            amount=projection.get("amount"),
                            payment_id=payment.get("payment_id"),
                        )
                        status = result.get("status") or "refund_pending"
                        refund_id = result.get("refund_id")
                    else:
                        await self.gateway.cancel_registration(encounter_id, reason=reason)
                except AdapterError as exc:
                    if exc.code == HIS_CONFLICT:
                        return {
                            "status": "denied",
                            "encounter_id": encounter_id,
                            "refund_id": None,
                        }
                    raise
            await self.store.update_projection(
                appointment_id,
                status="cancelled" if status != "refunded" else "refunded",
                payment_status="refunded" if status == "refunded" else projection.get("payment_status"),
                last_synced_at=_now().isoformat(),
            )
            return {"status": status, "encounter_id": encounter_id, "refund_id": refund_id}

        return await self._idempotent(
            aggregate_type="appointment",
            aggregate_id=appointment_id,
            operation="appointment.cancel",
            payload={"appointment_id": appointment_id, "reason": reason, "refund": refund},
            key=key,
            action=action,
        )

    async def his_reconcile(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = _str(args.get("scope")) or "all"
        since = _str(args.get("since"))
        projections = await self.store.list_projections()
        if since:
            projections = [
                row for row in projections if str(row.get("appointment_date") or "") >= since
            ]
        if scope not in ("all", "appointment", "payment"):
            raise AdapterError(HIS_VALIDATION, f"unknown reconcile scope {scope!r}")
        findings: list[dict[str, Any]] = []
        checked = 0
        for projection in projections:
            encounter_id = _str(projection.get("his_encounter_id"))
            if not encounter_id:
                continue
            checked += 1
            try:
                payment = await self.gateway.payment_status(encounter_id)
            except AdapterError:
                finding = await self.store.add_finding(
                    kind="his_unavailable",
                    severity="medium",
                    local_ref=projection["id"],
                    his_ref=encounter_id,
                    detail="payment status could not be read from HIS",
                )
                findings.append(finding)
                continue
            his_status = payment.get("payment_status") or "unknown"
            local_status = projection.get("payment_status") or "unknown"
            if his_status != "unknown" and his_status != local_status:
                finding = await self.store.add_finding(
                    kind="payment_mismatch",
                    severity="high",
                    local_ref=projection["id"],
                    his_ref=encounter_id,
                    detail=f"local={local_status} his={his_status}",
                )
                findings.append(finding)
        for finding in findings:
            finding["detected_at"] = _iso_value(finding.get("detected_at"))
            finding["resolved_at"] = _iso_value(finding.get("resolved_at"))
        summary = {
            "checked": checked,
            "findings": len(findings),
            "open_findings": len(await self.store.list_findings(status="open")),
            "scope": scope,
        }
        return {"findings": findings, "summary": summary}

    async def payment_simulate(self, args: dict[str, Any]) -> dict[str, Any]:
        self._require(args, "appointment_id")
        return await self.simulate_payment(str(args["appointment_id"]))

    async def sync_active_projections(self) -> dict[str, Any]:
        projections = await self.store.list_projections()
        checked = 0
        updated = 0
        for projection in projections:
            status = str(projection.get("status") or "")
            if status in ("cancelled", "refunded", "completed"):
                continue
            checked += 1
            encounter_id = _str(projection.get("his_encounter_id"))
            if not encounter_id:
                try:
                    found = await self.gateway.find_encounter(
                        patient_id=str(projection.get("his_patient_id") or ""),
                        org_id=_str(projection.get("his_org_id")),
                        service_id=_str(projection.get("his_service_id")),
                        day=_str(projection.get("appointment_date")),
                        created_after=_str(projection.get("created_at")),
                    )
                    if not found:
                        found = await self.gateway.find_encounter(
                            patient_id=str(projection.get("his_patient_id") or ""),
                            org_id=_str(projection.get("his_org_id")),
                            service_id=_str(projection.get("his_service_id")),
                            day=None,
                            created_after=_str(projection.get("created_at")),
                        )
                except AdapterError:
                    continue
                if not found:
                    continue
                encounter_id = str(found["encounter_id"])
                await self.store.update_projection(
                    projection["id"],
                    his_encounter_id=encounter_id,
                    his_status=str(found.get("status")),
                    last_synced_at=_now().isoformat(),
                )
                await self.store.upsert_encounter_map(
                    projection["id"],
                    {
                        "his_encounter_id": encounter_id,
                        "his_patient_id": projection.get("his_patient_id"),
                        "his_org_id": projection.get("his_org_id"),
                        "his_practitioner_id": projection.get("his_practitioner_id"),
                        "his_service_id": projection.get("his_service_id"),
                        "quote_id": projection.get("quote_id"),
                        "his_status": str(found.get("status")),
                        "last_synced_at": _now().isoformat(),
                    },
                )
                updated += 1
            try:
                payment = await self.gateway.payment_status(encounter_id)
                encounter = await self.gateway.get_encounter(encounter_id)
            except AdapterError:
                continue
            payment_status = payment.get("payment_status") or "unknown"
            state = "registered" if payment_status == "paid" else projection.get("status")
            if encounter.get("status") in (4, 5):
                state = "completed"
            elif encounter.get("status") == 6:
                state = "cancelled"
            await self.store.update_projection(
                projection["id"],
                payment_status=payment_status,
                payment_id=payment.get("payment_id"),
                invoice_id=payment.get("invoice_no"),
                his_status=str(encounter.get("status")),
                status=state,
                last_synced_at=_now().isoformat(),
            )
            updated += 1
        return {"checked": checked, "updated": updated}

    async def simulate_payment(self, appointment_id: str) -> dict[str, Any]:
        if self.config.mode != "mock":
            raise AdapterError(HIS_UNSUPPORTED, "simulate_payment is only available in mock mode")
        projection = await self.store.get_projection(appointment_id)
        if not projection:
            raise AdapterError(HIS_NOT_FOUND, f"appointment {appointment_id!r} not found")
        encounter_id = _str(projection.get("his_encounter_id"))
        payment: dict[str, Any]
        if encounter_id:
            payment = await self.gateway.simulate_payment(encounter_id)
        else:
            quote = (projection.get("payload") or {}).get("quote")
            if not quote:
                raise AdapterError(ADAPTER_MAPPING_MISSING, "stored quote is missing")
            paid = await self.gateway.pay_registration(quote)
            encounter_id = paid["encounter_id"]
            payment = {
                "payment_status": "paid",
                "amount": paid.get("amount"),
                "paid_at": paid.get("paid_at"),
                "payment_id": paid.get("payment_id"),
                "invoice_no": paid.get("invoice_no"),
            }
        await self.store.update_projection(
            appointment_id,
            his_encounter_id=encounter_id,
            status="registered",
            payment_status="paid",
            payment_id=payment.get("payment_id"),
            invoice_id=payment.get("invoice_no"),
            last_synced_at=_now().isoformat(),
        )
        return {
            "encounter_id": encounter_id,
            "payment_status": payment.get("payment_status"),
            "amount": payment.get("amount"),
            "paid_at": payment.get("paid_at"),
            "payment_id": payment.get("payment_id"),
            "invoice_no": payment.get("invoice_no"),
        }

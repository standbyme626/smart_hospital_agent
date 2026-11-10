from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import HTTPException

from .config import Settings
from .db import Database
from .security import id_card_key

logger = logging.getLogger(__name__)

STATUS_MAP = {
    "quoted": "quoted",
    "appointment_quoted": "quoted",
    "awaiting_payment": "awaiting_payment",
    "PENDING_CASHIER": "awaiting_payment",
    "registered": "registered",
    "checked_in": "registered",
    "in_progress": "registered",
    "completed": "completed",
    "cancelled": "cancelled",
    "refunded": "refunded",
    "no_show": "no_show",
}


class BookingService:
    """Application boundary for OpenHIS-backed appointments.

    The adapter owns HIS writes/reads; this service only keeps the app-side
    projection and never marks a payment as paid by itself.
    """

    def __init__(self, settings: Settings, db: Database, mcp: Any):
        self.settings = settings
        self.db = db
        self.mcp = mcp

    # --------------------------------------------------------------- transport
    async def _call(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        scope: str = "service_write",
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            tool, arguments or {}, idempotency_key=idempotency_key, scope=scope
        )
        envelope = result.get("data")
        if result.get("is_error") or not isinstance(envelope, dict):
            detail = ""
            if isinstance(envelope, dict) and envelope.get("error"):
                detail = str(envelope["error"].get("message") or envelope["error"])
            else:
                detail = str(result.get("text") or "hospital system unavailable")
            raise HTTPException(status_code=502, detail=detail[:300])
        if envelope.get("ok") is False:
            error = envelope.get("error") or {}
            code = str(error.get("code") or "HIS_UNAVAILABLE")
            message = str(error.get("message") or "hospital system error")
            status = 409 if code in {
                "HIS_CONFLICT",
                "HIS_DUPLICATE_PATIENT",
                "HIS_DUPLICATE_REGISTRATION",
                "ADAPTER_IDEMPOTENCY_CONFLICT",
                "ADAPTER_MAPPING_MISSING",
            } else 404 if code == "HIS_NOT_FOUND" else 502
            raise HTTPException(status_code=status, detail=f"{code}: {message}"[:300])
        data = envelope.get("data")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------ member
    async def _member(self, user_id: str, member_id: str | None) -> dict[str, Any]:
        if member_id:
            row = await self.db.fetchrow(
                "SELECT * FROM family_members WHERE id=$1 AND user_id=$2", member_id, user_id
            )
            if not row:
                raise HTTPException(status_code=404, detail="member not found")
            return dict(row)
        row = await self.db.fetchrow(
            "SELECT * FROM family_members WHERE user_id=$1 AND is_self=true ORDER BY created_at LIMIT 1",
            user_id,
        )
        if row:
            return dict(row)
        raise HTTPException(status_code=409, detail="请先完善本人家庭档案")

    async def _id_card(self, member: dict[str, Any]) -> str | None:
        if not member.get("id_card_ciphertext"):
            return None
        return await self.db.fetchval(
            "SELECT pgp_sym_decrypt(decode($1, 'base64'), $2)::text",
            member["id_card_ciphertext"],
            id_card_key(self.settings),
        )

    async def _ensure_patient(self, user_id: str, member: dict[str, Any]) -> dict[str, Any]:
        id_card = await self._id_card(member)
        if not id_card:
            raise HTTPException(
                status_code=409,
                detail="挂号需要就诊人的身份证号，请先在家庭档案中补充",
            )
        name = str(member.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=409, detail="就诊人姓名不能为空")
        common = {
            "member_id": member["id"],
            "user_id": user_id,
            "name": name,
            "id_card": id_card,
            "phone": member.get("phone"),
            "birth_date": member.get("birth_date"),
            "gender": member.get("gender"),
        }
        resolved = await self._call("patient.resolve", common, scope="service_write")
        if resolved.get("found"):
            return resolved
        id_card_key = hashlib.sha256(id_card.encode("utf-8")).hexdigest()[:16]
        created = await self._call(
            "patient.create",
            {
                **common,
                "idempotency_key": f"member:{user_id}:{member['id']}:{id_card_key}",
            },
            scope="service_write",
        )
        return {"found": True, **created}

    # ----------------------------------------------------------------- catalog
    async def catalog(
        self,
        *,
        department: str | None = None,
        org_id: str | None = None,
        date: str | None = None,
    ) -> dict[str, Any]:
        data = await self._call(
            "schedule.list",
            {"department": department, "org_id": org_id, "date": date},
            scope="agent_read",
        )
        return data

    # -------------------------------------------------------------- appointment
    async def create_appointment(
        self,
        *,
        user_id: str,
        member_id: str | None,
        org_id: str,
        practitioner_id: str,
        service_id: str,
        date: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        member = await self._member(user_id, member_id)
        patient = await self._ensure_patient(user_id, member)

        today = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        if date != today:
            raise HTTPException(
                status_code=400,
                detail="OpenHIS V1.3 仅支持当日挂号，请选择今天",
            )
        his_patient_id = str(patient.get("his_patient_id") or "")
        if not his_patient_id:
            raise HTTPException(status_code=502, detail="医院患者映射失败")

        catalog = await self.catalog(date=date, org_id=org_id)
        departments = {str(item.get("org_id")): item for item in catalog.get("departments") or []}
        doctors = {
            str(item.get("practitioner_id")): item for item in catalog.get("doctors") or []
        }
        services = {str(item.get("service_id")): item for item in catalog.get("services") or []}
        department = departments.get(str(org_id), {})
        doctor = doctors.get(str(practitioner_id), {})
        service = services.get(str(service_id))
        if service is None:
            raise HTTPException(status_code=409, detail="该科室没有可用的挂号服务")

        key = idempotency_key or f"appointment:{user_id}:{member['id']}:{org_id}:{practitioner_id}:{service_id}:{date}"
        data = await self._call(
            "appointment.create",
            {
                "member_id": member["id"],
                "user_id": user_id,
                "org_id": org_id,
                "practitioner_id": practitioner_id,
                "service_id": service_id,
                "date": date,
                "idempotency_key": key,
            },
            scope="service_write",
            idempotency_key=key,
        )
        appointment_id = str(data.get("appointment_id") or "")
        if not appointment_id:
            raise HTTPException(status_code=502, detail="医院未返回预约号")
        amount = float(data.get("amount") or service.get("price") or 0)
        await self.db.execute(
            """
            INSERT INTO bookings
                (id, session_id, patient_id, user_id, department, slot_id, order_id, status,
                 payment_status, amount, idempotency_key, member_id, hospital_patient_id,
                 his_patient_id, quote_id, his_org_id, his_practitioner_id, his_service_id,
                 detail, legacy_demo)
            VALUES ($1, NULL, $2, $3, $4, NULL, NULL, 'quoted', 'unpaid', $5, $6, $7, $8,
                    $8, $9, $10, $11, $12, $13::jsonb, false)
            ON CONFLICT (id) DO NOTHING
            """,
            appointment_id,
            his_patient_id,
            user_id,
            str(department.get("name") or service.get("name") or ""),
            amount,
            key,
            member["id"],
            his_patient_id,
            str(data.get("quote_id") or ""),
            str(org_id),
            str(practitioner_id),
            str(service_id),
            json.dumps(
                {
                    "appointment": data,
                    "date": date,
                    "department": department,
                    "doctor": doctor,
                    "service": service,
                },
                ensure_ascii=False,
            ),
        )
        await self.db.execute(
            """
            UPDATE family_members
            SET hospital_patient_id=$2, his_link_status='linked', updated_at=now()
            WHERE id=$1 AND user_id=$3
            """,
            member["id"],
            his_patient_id,
            user_id,
        )
        return {
            "appointment": await self._appointment_view(appointment_id, user_id),
            "instructions": (
                f"已生成挂号报价，应付 {amount:.2f} 元。"
                "请提交支付申请后到收费窗口/自助机完成缴费。"
            ),
            "payment_mode": data.get("payment_mode") or "cashier",
        }

    async def _appointment_row(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        row = await self.db.fetchrow(
            "SELECT * FROM bookings WHERE id=$1 AND user_id=$2", appointment_id, user_id
        )
        if not row:
            raise HTTPException(status_code=404, detail="appointment not found")
        return dict(row)

    async def _appointment_view(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        row = await self._appointment_row(appointment_id, user_id)
        detail = row.get("detail")
        if isinstance(detail, str):
            try:
                detail = json.loads(detail)
            except json.JSONDecodeError:
                detail = {}
        detail = detail if isinstance(detail, dict) else {}
        return {
            "id": row["id"],
            "appointment_id": row["id"],
            "user_id": row["user_id"],
            "member_id": row.get("member_id"),
            "his_patient_id": row.get("his_patient_id"),
            "quote_id": row.get("quote_id"),
            "amount": float(row.get("amount") or 0),
            "status": row.get("status"),
            "payment_status": row.get("payment_status"),
            "his_status": row.get("his_status"),
            "department": row.get("department"),
            "org_id": row.get("his_org_id"),
            "practitioner_id": row.get("his_practitioner_id"),
            "service_id": row.get("his_service_id"),
            "appointment_date": detail.get("date"),
            "doctor_name": (detail.get("doctor") or {}).get("name"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "legacy_demo": bool(row.get("legacy_demo")),
            "stale": bool(row.get("stale")),
        }

    async def list_appointments(
        self, user_id: str, member_id: str | None = None
    ) -> list[dict[str, Any]]:
        if member_id:
            await self._member(user_id, member_id)
        rows = await self.db.fetch(
            """
            SELECT id FROM bookings
            WHERE user_id=$1 AND ($2::text IS NULL OR member_id=$2)
              AND legacy_demo=false
            ORDER BY created_at DESC, id DESC
            LIMIT 50
            """,
            user_id,
            member_id,
        )
        return [await self._appointment_view(row["id"], user_id) for row in rows]

    async def _sync_appointment(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        row = await self._appointment_row(appointment_id, user_id)
        visit = await self._call(
            "visit.status",
            {"appointment_id": appointment_id},
            scope="agent_read",
        )
        payment = await self._call(
            "payment.status",
            {"appointment_id": appointment_id},
            scope="agent_read",
        )
        state = str(visit.get("state") or "unknown")
        mapped = STATUS_MAP.get(state, row.get("status") or "unknown")
        payment_status = str(payment.get("payment_status") or row.get("payment_status") or "unknown")
        encounter_id = visit.get("encounter_id")
        await self.db.execute(
            """
            UPDATE bookings
            SET status=$2, payment_status=$3, his_encounter_id=COALESCE($4, his_encounter_id),
                his_status=$5, last_synced_at=now(), stale=$6, updated_at=now()
            WHERE id=$1
            """,
            appointment_id,
            mapped,
            payment_status,
            str(encounter_id) if encounter_id else None,
            str(visit.get("state") or ""),
            bool(visit.get("stale") or payment.get("stale")),
        )
        if mapped in {"registered", "completed"}:
            await self._ensure_booking_event(appointment_id, user_id, mapped, visit)
        return {
            "appointment": await self._appointment_view(appointment_id, user_id),
            "visit": visit,
            "payment": payment,
        }

    async def get_appointment(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        await self._appointment_row(appointment_id, user_id)
        try:
            return await self._sync_appointment(appointment_id, user_id)
        except HTTPException:
            view = await self._appointment_view(appointment_id, user_id)
            return {
                "appointment": view,
                "visit": {
                    "state": view["status"],
                    "payment_status": view["payment_status"],
                    "stale": True,
                    "guide": "医院系统暂不可用，以下状态为上次同步结果。",
                },
                "payment": {
                    "payment_status": view["payment_status"],
                    "amount": view["amount"],
                    "stale": True,
                },
            }

    async def request_payment(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        await self._appointment_row(appointment_id, user_id)
        key = f"payment:{appointment_id}"
        task = await self._call(
            "payment.request",
            {"appointment_id": appointment_id, "idempotency_key": key},
            scope="service_write",
            idempotency_key=key,
        )
        await self.db.execute(
            """
            UPDATE bookings
            SET status=CASE WHEN payment_status='paid' THEN status ELSE 'awaiting_payment' END,
                updated_at=now()
            WHERE id=$1 AND user_id=$2
            """,
            appointment_id,
            user_id,
        )
        return {"task": task, "appointment": await self._appointment_view(appointment_id, user_id)}

    async def cancel_appointment(
        self, appointment_id: str, user_id: str, reason: str = "user_cancelled", refund: bool = True
    ) -> dict[str, Any]:
        await self._appointment_row(appointment_id, user_id)
        key = f"cancel:{appointment_id}"
        data = await self._call(
            "appointment.cancel",
            {
                "appointment_id": appointment_id,
                "reason": reason,
                "refund": refund,
                "idempotency_key": key,
            },
            scope="service_write",
            idempotency_key=key,
        )
        status = str(data.get("status") or "cancelled")
        mapped = "refunded" if status == "refunded" else "cancelled"
        await self.db.execute(
            "UPDATE bookings SET status=$2, updated_at=now() WHERE id=$1",
            appointment_id,
            mapped,
        )
        return {"cancel": data, "appointment": await self._appointment_view(appointment_id, user_id)}

    async def prescriptions(self, appointment_id: str, user_id: str) -> dict[str, Any]:
        row = await self._appointment_row(appointment_id, user_id)
        encounter_id = row.get("his_encounter_id")
        if not encounter_id:
            raise HTTPException(status_code=409, detail="尚未生成就诊记录，暂无处方")
        data = await self._call(
            "prescription.list",
            {"encounter_id": str(encounter_id), "include_details": True},
            scope="agent_read",
        )
        return {"prescriptions": data.get("prescriptions") or [], "stale": data.get("stale")}

    async def simulate_payment(self, appointment_id: str, user_id: str | None = None) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "payment.simulate",
            {"appointment_id": appointment_id},
            scope="service_write",
        )
        envelope = result.get("data")
        error = (envelope or {}).get("error") or {}
        code = str(error.get("code") or "")
        if result.get("is_error") or not isinstance(envelope, dict) or envelope.get("ok") is False:
            if code == "HIS_UNSUPPORTED":
                # Real OpenHIS has no mock simulate; use the configured direct
                # settlement path (reg-pay) for test/pilot completion.
                return await self._call(
                    "registration.create",
                    {
                        "appointment_id": appointment_id,
                        "idempotency_key": f"settle:{appointment_id}",
                    },
                    scope="service_write",
                    idempotency_key=f"settle:{appointment_id}",
                )
            raise HTTPException(
                status_code=502,
                detail=f"{code or 'HIS_UNAVAILABLE'}: {error.get('message', 'simulate failed')}",
            )
        return envelope.get("data") or {}

    async def list_payment_tasks(self, status: str | None = None) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """
            SELECT * FROM human_task
            WHERE ($1::text IS NULL OR status=$1)
            ORDER BY created_at DESC LIMIT 100
            """,
            status,
        )
        tasks: list[dict[str, Any]] = []
        for row in rows:
            task = dict(row)
            if isinstance(task.get("payload"), str):
                try:
                    task["payload"] = json.loads(task["payload"])
                except json.JSONDecodeError:
                    task["payload"] = {}
            tasks.append(task)
        return tasks

    async def ack_payment_task(self, task_id: str, status: str = "completed") -> dict[str, Any]:
        if status not in {"acknowledged", "completed", "cancelled"}:
            raise HTTPException(status_code=422, detail="invalid task status")
        row = await self.db.fetchrow(
            "UPDATE human_task SET status=$2, updated_at=now() WHERE id=$1 RETURNING *",
            task_id,
            status,
        )
        if not row:
            raise HTTPException(status_code=404, detail="task not found")
        return dict(row)

    async def _ensure_booking_event(
        self, appointment_id: str, user_id: str, state: str, visit: dict[str, Any]
    ) -> None:
        exists = await self.db.fetchval(
            "SELECT 1 FROM health_events WHERE user_id=$1 AND kind='booking' AND detail->>'appointment_id'=$2 LIMIT 1",
            user_id,
            appointment_id,
        )
        if exists:
            return
        view = await self._appointment_view(appointment_id, user_id)
        title = f"挂号 · {view.get('department') or '门诊'}"
        await self.db.execute(
            """
            INSERT INTO health_events
                (user_id, member_id, kind, title, detail, source, his_ref_type, his_ref_id, deep_link)
            VALUES ($1,$2,'booking',$3,$4::jsonb,'his','encounter',$5,$6)
            """,
            user_id,
            view.get("member_id"),
            title,
            json.dumps(
                {
                    "appointment_id": appointment_id,
                    "order_id": appointment_id,
                    "encounter_id": visit.get("encounter_id"),
                    "state": state,
                    "payment_status": view.get("payment_status"),
                },
                ensure_ascii=False,
            ),
            str(visit.get("encounter_id") or ""),
            f"/services/?appointment={appointment_id}",
        )

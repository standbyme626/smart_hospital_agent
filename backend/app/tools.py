from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .db import Database
from .mcp_client import HospitalMCPClient
from .policy import DEPARTMENT_NAMES, DRUG_CHECKER, merge_drug_results
from .retrieval import RetrievalService

ToolHandler = Callable[[dict[str, Any], "ToolContext"], Awaitable[dict[str, Any]]]


@dataclass
class ToolContext:
    session_id: str
    request_id: str
    user_id: str = "anonymous"
    member_id: str | None = None
    user_text: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    drug_result: dict[str, Any] | None = None
    drug_results: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] | None = None
    booking: dict[str, Any] | None = None
    emitted_citations: set[str] = field(default_factory=set)


class ToolRegistry:
    def __init__(
        self,
        retrieval: RetrievalService,
        mcp: HospitalMCPClient,
        db: Database,
    ):
        self.retrieval = retrieval
        self.mcp = mcp
        self.db = db
        self.handlers: dict[str, ToolHandler] = {
            "retrieve_evidence": self.retrieve_evidence,
            "check_drug_interaction": self.check_drug_interaction,
            "lookup_guideline": self.lookup_guideline,
            "list_departments": self.list_departments,
            "list_schedules": self.list_schedules,
            "lock_slot": self.lock_slot,
            "create_order": self.create_order,
            "get_payment_status": self.get_payment_status,
            "cancel_order": self.cancel_order,
            "encounter_info": self.encounter_info,
            "prescription_list": self.prescription_list,
            "visit_status": self.visit_status,
            "payment_status": self.payment_status,
            "submit_report": self.submit_report,
        }

    def definitions(self) -> list[dict[str, Any]]:
        return [
            _tool("retrieve_evidence", "检索医学证据并返回证据 ID、标题与摘要。", {"query": "string", "department": "string"}, required=["query"]),
            _tool("check_drug_interaction", "从文本中抽取药物并检查药物相互作用与妊娠禁忌。", {"text": "string"}, required=["text"]),
            _tool("lookup_guideline", "检索指南/共识类证据。", {"topic": "string"}, required=["topic"]),
            _tool("list_departments", "列出医院科室。", {}, required=[]),
            _tool("list_schedules", "查询某科室可挂号的服务（不含时间号源）。", {"department": "string", "date": "string"}, required=["department"]),
            _tool("encounter_info", "查询就诊记录状态。", {"appointment_id": "string", "encounter_id": "string"}, required=[]),
            _tool("prescription_list", "查询就诊处方（只读）。", {"encounter_id": "string", "appointment_id": "string"}, required=[]),
            _tool("visit_status", "查询预约/就诊状态与下一步。", {"appointment_id": "string", "encounter_id": "string"}, required=[]),
            _tool(
                "submit_report",
                "提交最终结构化报告。",
                {
                    "department": "string",
                    "urgency": "string",
                    "summary": "string",
                    "advice": "array",
                    "medications": "array",
                    "citations": "array",
                    "confidence": "number",
                    "followup_question": "string",
                },
                required=["department", "urgency", "summary", "confidence"],
            ),
        ]

    async def execute(self, name: str, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        handler = self.handlers.get(name)
        if not handler:
            return {"ok": False, "error": f"unknown tool: {name}"}
        try:
            return await handler(arguments, context)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------- retrieval
    async def retrieve_evidence(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        query = str(args.get("query") or context.user_text).strip()
        department = args.get("department") or None
        results = await self.retrieval.search(query, department=department)
        for item in results:
            if item["id"] not in {citation["id"] for citation in context.citations}:
                context.citations.append(item)
        return {
            "ok": True,
            "count": len(results),
            "filtered": self.retrieval.last_filtered,
            "dept_unavailable": self.retrieval.last_dept_unavailable,
            "evidence": results,
        }

    async def lookup_guideline(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        topic = str(args.get("topic") or context.user_text).strip()
        results = await self.retrieval.search(f"{topic} 指南 共识", top_k=3)
        for item in results:
            if item["id"] not in {citation["id"] for citation in context.citations}:
                context.citations.append(item)
        return {"ok": True, "count": len(results), "guidelines": results}

    async def check_drug_interaction(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        text = str(args.get("text") or context.user_text)
        if context.member_id:
            rows = await self.db.fetch(
                """
                SELECT name FROM medications
                WHERE user_id=$1 AND active=true AND ($2::text IS NULL OR member_id=$2)
                """,
                context.user_id,
                context.member_id,
            )
            names = [str(row["name"]) for row in rows if row["name"]]
            if names:
                text = f"{text} 在用药物：{'、'.join(names)}"
        result = DRUG_CHECKER.check_text(text)
        context.drug_results.append(result)
        context.drug_result = merge_drug_results(context.drug_results)
        return {"ok": True, **context.drug_result}

    # -------------------------------------------------------------- hospital
    async def list_departments(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool("schedule.list", {}, scope="agent_read")
        data = self._adapter_result(result)
        if not data.get("ok"):
            return {"ok": True, "departments": DEPARTMENT_NAMES, "degraded": True}
        return {
            "ok": True,
            "departments": [item.get("name") for item in data.get("departments") or []],
            "availability_mode": data.get("availability_mode"),
        }

    async def list_schedules(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "schedule.list",
            {"department": args.get("department"), "date": args.get("date")},
            scope="agent_read",
        )
        data = self._adapter_result(result)
        if not data.get("ok"):
            return data
        return {"ok": True, "data": data}

    async def _patient_ref(self, context: ToolContext) -> tuple[str, str | None]:
        if context.member_id:
            member = await self.db.fetchrow(
                "SELECT id, hospital_patient_id FROM family_members WHERE id=$1 AND user_id=$2",
                context.member_id,
                context.user_id,
            )
            if member:
                hospital_patient_id = member["hospital_patient_id"] or f"HP-{member['id']}"
                if not member["hospital_patient_id"]:
                    await self.db.execute(
                        "UPDATE family_members SET hospital_patient_id=$2 WHERE id=$1",
                        member["id"],
                        hospital_patient_id,
                    )
                return hospital_patient_id, member["id"]
        return context.user_id, None

    async def lock_slot(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        patient_id, _ = await self._patient_ref(context)
        result = await self.mcp.call_tool(
            "lock_slot",
            {"slot_id": args.get("slot_id"), "patient_id": patient_id},
        )
        if result["is_error"]:
            return {"ok": False, "error": result["text"]}
        return {"ok": True, "data": result["data"]}

    async def create_order(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        idempotency_key = f"chat:{context.session_id}:{args.get('lock_id')}"
        patient_id, member_id = await self._patient_ref(context)
        hospital_patient_id = patient_id if member_id else None
        result = await self.mcp.call_tool(
            "create_order",
            {"lock_id": args.get("lock_id"), "patient_id": patient_id},
            idempotency_key=idempotency_key,
        )
        if result["is_error"]:
            return {"ok": False, "error": result["text"]}
        order = result["data"] or {}
        context.booking = order
        await self.db.execute(
            """
            INSERT INTO bookings
                (id, session_id, patient_id, user_id, department, slot_id, order_id, status,
                 payment_status, amount, idempotency_key, member_id, hospital_patient_id, detail)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
            ON CONFLICT (id) DO NOTHING
            """,
            order.get("order_id"),
            context.session_id,
            context.user_id,
            context.user_id,
            (order.get("slot_id") or "").split("|")[0],
            order.get("slot_id"),
            order.get("order_id"),
            order.get("status", "created"),
            order.get("payment_status", "unpaid"),
            order.get("amount", 50),
            idempotency_key,
            member_id,
            hospital_patient_id,
            json.dumps(order, ensure_ascii=False),
        )
        return {"ok": True, "data": order}

    async def get_payment_status(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool("get_payment_status", {"order_id": args.get("order_id")})
        if result["is_error"]:
            return {"ok": False, "error": result["text"]}
        return {"ok": True, "data": result["data"]}

    async def cancel_order(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool("cancel_order", {"order_id": args.get("order_id")})
        if result["is_error"]:
            return {"ok": False, "error": result["text"]}
        return {"ok": True, "data": result["data"]}

    # ------------------------------------------------------------- read HIT
    async def encounter_info(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "encounter.get",
            {"appointment_id": args.get("appointment_id"), "encounter_id": args.get("encounter_id")},
            scope="agent_read",
        )
        return self._adapter_result(result)

    async def prescription_list(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "prescription.list",
            {
                "encounter_id": args.get("encounter_id"),
                "appointment_id": args.get("appointment_id"),
                "include_details": True,
            },
            scope="agent_read",
        )
        return self._adapter_result(result)

    async def visit_status(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "visit.status",
            {"appointment_id": args.get("appointment_id"), "encounter_id": args.get("encounter_id")},
            scope="agent_read",
        )
        return self._adapter_result(result)

    async def payment_status(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        result = await self.mcp.call_tool(
            "payment.status",
            {"appointment_id": args.get("appointment_id"), "encounter_id": args.get("encounter_id")},
            scope="agent_read",
        )
        return self._adapter_result(result)

    @staticmethod
    def _adapter_result(result: dict[str, Any]) -> dict[str, Any]:
        envelope = result.get("data")
        if result.get("is_error") or not isinstance(envelope, dict):
            return {"ok": False, "error": str(result.get("text") or "hospital system unavailable")}
        if envelope.get("ok") is False:
            error = envelope.get("error") or {}
            return {"ok": False, "error": error.get("message") or "hospital system error"}
        return {"ok": True, **(envelope.get("data") or {})}

    # ---------------------------------------------------------------- report
    async def submit_report(self, args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        report = {
            "department": str(args.get("department") or "General Practice"),
            "urgency": str(args.get("urgency") or "normal"),
            "summary": str(args.get("summary") or ""),
            "advice": list(args.get("advice") or []),
            "medications": list(args.get("medications") or []),
            "citations": list(args.get("citations") or []),
            "confidence": float(args.get("confidence") or 0.0),
            "followup_question": str(args.get("followup_question") or ""),
        }
        context.report = report
        return {"ok": True, "accepted": True, "report": report}


def _tool(name: str, description: str, properties: dict[str, str], required: list[str] | None = None) -> dict[str, Any]:
    json_types = {"string": "string", "number": "number", "array": "array"}
    schema_props: dict[str, Any] = {}
    for prop, kind in properties.items():
        if kind == "array":
            schema_props[prop] = {"type": "array", "items": {"type": "string"}}
        else:
            schema_props[prop] = {"type": json_types.get(kind, "string")}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": schema_props,
                "required": required or list(properties),
            },
        },
    }

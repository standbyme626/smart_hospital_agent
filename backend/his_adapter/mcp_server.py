from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any

from fastapi import FastAPI, Request

from .config import AdapterConfig
from .errors import ADAPTER_SCOPE_DENIED
from .service import READ_TOOLS, SCOPES, AdapterService

TOOLS: list[dict[str, Any]] = [
    {
        "name": "patient.resolve",
        "description": "Resolve a family member against the HIS patient index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string"},
                "name": {"type": "string"},
                "id_card": {"type": "string"},
                "phone": {"type": "string"},
                "birth_date": {"type": "string"},
                "gender": {"type": "string"},
            },
            "required": ["member_id", "name"],
        },
    },
    {
        "name": "patient.create",
        "description": "Create a patient in HIS and persist the local mapping.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string"},
                "name": {"type": "string"},
                "id_card": {"type": "string"},
                "gender": {"type": "string"},
                "birth_date": {"type": "string"},
                "phone": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["member_id", "name", "id_card", "gender", "birth_date", "phone"],
        },
    },
    {
        "name": "schedule.list",
        "description": "List HIS departments, doctors and registration services.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "org_id": {"type": "string"},
                "practitioner_id": {"type": "string"},
                "date": {"type": "string"},
                "service_id": {"type": "string"},
            },
        },
    },
    {
        "name": "appointment.create",
        "description": "Create a quoted registration (reg-pre-pay) for a mapped patient.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "member_id": {"type": "string"},
                "org_id": {"type": "string"},
                "practitioner_id": {"type": "string"},
                "service_id": {"type": "string"},
                "date": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["member_id", "org_id", "practitioner_id", "service_id", "date"],
        },
    },
    {
        "name": "registration.create",
        "description": "Create the registration encounter, either cashier or direct pay mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "payment_details": {
                    "type": "array",
                    "items": {"type": "object"},
                },
                "confirm": {"type": "boolean"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["appointment_id"],
        },
    },
    {
        "name": "payment.request",
        "description": "Create a human payment task for an appointment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "amount": {"type": "number"},
                "mode": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["appointment_id"],
        },
    },
    {
        "name": "payment.status",
        "description": "Read the HIS payment status for an appointment or encounter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "encounter_id": {"type": "string"},
            },
        },
    },
    {
        "name": "encounter.get",
        "description": "Read the HIS encounter for an appointment or encounter id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "encounter_id": {"type": "string"},
                "appointment_id": {"type": "string"},
            },
        },
    },
    {
        "name": "prescription.list",
        "description": "List prescriptions for an encounter.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "encounter_id": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "required": ["encounter_id"],
        },
    },
    {
        "name": "visit.status",
        "description": "Aggregate appointment, payment and encounter state into a visit status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "encounter_id": {"type": "string"},
            },
        },
    },
    {
        "name": "appointment.cancel",
        "description": "Cancel or refund an appointment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appointment_id": {"type": "string"},
                "reason": {"type": "string"},
                "refund": {"type": "boolean"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["appointment_id"],
        },
    },
    {
        "name": "his.reconcile",
        "description": "Compare local appointment projections with HIS payment state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string"},
                "since": {"type": "string"},
            },
            "required": ["scope"],
        },
    },
    {
        "name": "payment.simulate",
        "description": "Mock-mode only: simulate a cashier payment for a quoted appointment.",
        "inputSchema": {
            "type": "object",
            "properties": {"appointment_id": {"type": "string"}},
            "required": ["appointment_id"],
        },
    },
]

TOOL_NAMES = {tool["name"] for tool in TOOLS}


def _local(request: Request, config: AdapterConfig) -> bool:
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1", "localhost"):
        return True
    return config.mode == "mock" and host == "testclient"


def _scope_error(started: float, scope: str) -> dict[str, Any]:
    return {
        "ok": False,
        "data": None,
        "error": {
            "code": ADAPTER_SCOPE_DENIED,
            "message": f"scope {scope!r} is not allowed from this caller",
            "retryable": False,
        },
        "meta": {"source": "adapter", "stale": False, "latency_ms": round((perf_counter() - started) * 1000, 2)},
    }


def create_app(config: AdapterConfig | None = None) -> FastAPI:
    adapter_config = config or AdapterConfig.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        service = AdapterService(adapter_config)
        await service.startup()
        application.state.service = service

        async def poll_loop() -> None:
            while True:
                await asyncio.sleep(max(5, adapter_config.poll_interval))
                try:
                    await service.sync_active_projections()
                except Exception:
                    continue

        poller = asyncio.create_task(poll_loop())
        try:
            yield
        finally:
            poller.cancel()
            try:
                await poller
            except asyncio.CancelledError:
                pass
            await service.close()

    application = FastAPI(title="HIS Adapter", version="3.1.0", lifespan=lifespan)

    @application.post("/mcp")
    async def mcp(request: Request) -> dict[str, Any]:
        body = await request.json()
        request_id = body.get("id")
        method = body.get("method")
        params = body.get("params") or {}
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "his-adapter", "version": "3.1.0"},
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
        if method != "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"method not found: {method}"},
            }
        started = perf_counter()
        name = params.get("name") or ""
        scope = request.headers.get("x-adapter-scope") or "service_write"
        if scope not in SCOPES:
            result = _scope_error(started, scope)
        elif scope == "agent_read" and name not in READ_TOOLS:
            result = _scope_error(started, scope)
        elif scope == "service_write" and not _local(request, adapter_config):
            result = _scope_error(started, scope)
        else:
            service: AdapterService = application.state.service
            result = await service.call(name, params.get("arguments") or {}, scope)
        text = json.dumps(result, ensure_ascii=False, default=str)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "isError": not bool(result.get("ok")),
            },
        }

    @application.get("/health")
    async def health() -> dict[str, Any]:
        service: AdapterService | None = getattr(application.state, "service", None)
        return {
            "ok": service is not None,
            "mode": adapter_config.mode,
            "payment_mode": adapter_config.payment_mode,
            "store": service.store.backend if service else "uninitialized",
            "tools": len(TOOL_NAMES),
        }

    return application


app = create_app()

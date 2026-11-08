from __future__ import annotations

import json
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Request

BASE = Path(__file__).resolve().parents[1]
load_dotenv(BASE.parent / ".env")
SPECIALISTS = json.loads((BASE / "app" / "data" / "specialists.json").read_text(encoding="utf-8"))["specialists"]
DEPARTMENTS = [item.get("name_cn", key) for key, item in SPECIALISTS.items()]

MCP_SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS mcp_locks (
        lock_id TEXT PRIMARY KEY,
        slot_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_orders (
        order_id TEXT PRIMARY KEY,
        slot_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        amount NUMERIC(10, 2) NOT NULL,
        status TEXT NOT NULL,
        payment_status TEXT NOT NULL,
        idempotency_key TEXT UNIQUE,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_booked (
        slot_id TEXT PRIMARY KEY,
        order_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
]


logger = logging.getLogger("mcp-mock")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        conn = await _connect()
        try:
            await conn.execute("DELETE FROM mcp_locks WHERE expires_at < now()")
        finally:
            await conn.close()
        logger.info("mcp mock database ready")
    except Exception as exc:  # allow the process to boot for diagnostics
        logger.error("mcp mock database init failed (DATABASE_URL missing or unreachable): %s", exc)
    yield


app = FastAPI(title="Hospital MCP Mock", version="3.1.0", lifespan=lifespan)

_schema_ready = False


def _dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required for the hospital MCP mock")
    return dsn


async def _connect() -> asyncpg.Connection:
    global _schema_ready
    conn = await asyncpg.connect(_dsn())
    if not _schema_ready:
        for statement in MCP_SCHEMA:
            await conn.execute(statement)
        _schema_ready = True
    return conn


@asynccontextmanager
async def _db():
    conn = await _connect()
    try:
        yield conn
    finally:
        await conn.close()


def _slots(department: str, booked: set[str]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    today = date.today()
    for offset in range(1, 8):
        day = (today + timedelta(days=offset)).isoformat()
        for doctor in ("张医生", "李医生", "王医生"):
            for clock in ("09:00", "10:30", "14:00", "15:30"):
                slot_id = f"{department}|{day}|{clock}|{doctor}"
                slots.append(
                    {
                        "slot_id": slot_id,
                        "department": department,
                        "date": day,
                        "time": clock,
                        "doctor": doctor,
                        "fee": 50,
                        "available": slot_id not in booked,
                    }
                )
    return slots


def _text(data: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False)}], "isError": False}


def _error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps({"error": message}, ensure_ascii=False)}], "isError": True}


def _order_json(row: asyncpg.Record) -> dict[str, Any]:
    payload = row["payload"] or {}
    if isinstance(payload, str):
        payload = json.loads(payload)
    data = dict(payload)
    data.update(
        {
            "order_id": row["order_id"],
            "slot_id": row["slot_id"],
            "patient_id": row["patient_id"],
            "amount": float(row["amount"]),
            "status": row["status"],
            "payment_status": row["payment_status"],
        }
    )
    if row["idempotency_key"]:
        data["idempotency_key"] = row["idempotency_key"]
    return data


TOOLS = [
    {
        "name": "list_departments",
        "description": "List hospital departments.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_schedules",
        "description": "List available appointment slots for a department.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "department": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD, optional"},
            },
            "required": ["department"],
        },
    },
    {
        "name": "lock_slot",
        "description": "Lock an appointment slot for 10 minutes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slot_id": {"type": "string"},
                "patient_id": {"type": "string"},
            },
            "required": ["slot_id"],
        },
    },
    {
        "name": "create_order",
        "description": "Create a registration order from a locked slot.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lock_id": {"type": "string"},
                "patient_id": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["lock_id"],
        },
    },
    {
        "name": "pay_order",
        "description": "Pay an order. Requires a human confirmation token.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "confirm_token": {"type": "string"},
                "method": {"type": "string"},
            },
            "required": ["order_id", "confirm_token"],
        },
    },
    {
        "name": "list_doctors",
        "description": "List doctors for a department.",
        "inputSchema": {
            "type": "object",
            "properties": {"department": {"type": "string"}},
            "required": ["department"],
        },
    },
    {
        "name": "get_visit_guide",
        "description": "Get visit guide for a paid order.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "get_payment_status",
        "description": "Get payment status for an order.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an unpaid order.",
        "inputSchema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
]


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "list_departments":
        return _text({"departments": DEPARTMENTS})
    if name == "list_schedules":
        department = args.get("department") or ""
        if department not in DEPARTMENTS:
            return _error(f"unknown department: {department}")
        async with _db() as conn:
            booked = {row["slot_id"] for row in await conn.fetch("SELECT slot_id FROM mcp_booked")}
        slots = _slots(department, booked)
        day = args.get("date")
        if day:
            slots = [slot for slot in slots if slot["date"] == day]
        return _text({"department": department, "slots": slots[:40]})
    if name == "lock_slot":
        slot_id = args.get("slot_id") or ""
        if not slot_id or "|" not in slot_id:
            return _error("invalid slot_id")
        async with _db() as conn:
            if await conn.fetchval("SELECT 1 FROM mcp_booked WHERE slot_id=$1", slot_id):
                return _error("slot already booked")
            await conn.execute("DELETE FROM mcp_locks WHERE expires_at < now()")
            lock_id = "lock_" + secrets.token_hex(6)
            await conn.execute(
                "INSERT INTO mcp_locks (lock_id, slot_id, patient_id, expires_at) "
                "VALUES ($1, $2, $3, now() + interval '10 minutes')",
                lock_id,
                slot_id,
                args.get("patient_id") or "anonymous",
            )
        return _text({"lock_id": lock_id, "slot_id": slot_id, "expires_in": 600})
    if name == "create_order":
        lock_id = args.get("lock_id") or ""
        key = args.get("idempotency_key") or None
        async with _db() as conn:
            async with conn.transaction():
                if key:
                    row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE idempotency_key=$1", key)
                    if row:
                        return _text(_order_json(row))
                lock = await conn.fetchrow(
                    "SELECT * FROM mcp_locks WHERE lock_id=$1 AND expires_at > now()",
                    lock_id,
                )
                if not lock:
                    return _error("lock not found or expired")
                if await conn.fetchval("SELECT 1 FROM mcp_booked WHERE slot_id=$1", lock["slot_id"]):
                    return _error("slot already booked")
                order_id = "order_" + secrets.token_hex(6)
                payload = {"created_at": time.time(), "lock_id": lock_id}
                row = await conn.fetchrow(
                    """
                    INSERT INTO mcp_orders
                        (order_id, slot_id, patient_id, amount, status, payment_status, idempotency_key, payload)
                    VALUES ($1, $2, $3, 50, 'created', 'unpaid', $4, $5::jsonb)
                    ON CONFLICT (idempotency_key) DO NOTHING
                    RETURNING *
                    """,
                    order_id,
                    lock["slot_id"],
                    args.get("patient_id") or lock["patient_id"],
                    key,
                    json.dumps(payload, ensure_ascii=False),
                )
                if row is None and key:
                    row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE idempotency_key=$1", key)
                if row is None:
                    return _error("order create conflict, retry")
                await conn.execute("DELETE FROM mcp_locks WHERE lock_id=$1", lock_id)
        return _text(_order_json(row))
    if name == "pay_order":
        order_id = args.get("order_id") or ""
        async with _db() as conn:
            row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE order_id=$1", order_id)
            if not row:
                return _error("order not found")
            if row["status"] == "cancelled":
                return _error("cancelled order cannot be paid")
            if row["payment_status"] == "paid":
                return _text(_order_json(row))
            expected = os.getenv("HITL_CONFIRM_TOKEN", "confirm-demo")
            if args.get("confirm_token") != expected:
                return _error("requires_confirmation: human confirmation token missing or invalid")
            payload = row["payload"] or {}
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload.update(
                {
                    "paid_at": time.time(),
                    "method": args.get("method") or "mock",
                }
            )
            async with conn.transaction():
                row = await conn.fetchrow(
                    """
                    UPDATE mcp_orders
                    SET status='booked', payment_status='paid', payload=$2::jsonb, updated_at=now()
                    WHERE order_id=$1
                    RETURNING *
                    """,
                    order_id,
                    json.dumps(payload, ensure_ascii=False),
                )
                await conn.execute(
                    """
                    INSERT INTO mcp_booked (slot_id, order_id) VALUES ($1, $2)
                    ON CONFLICT (slot_id) DO UPDATE SET order_id=EXCLUDED.order_id
                    """,
                    row["slot_id"],
                    order_id,
                )
        return _text(_order_json(row))
    if name == "list_doctors":
        department = args.get("department") or ""
        if department not in DEPARTMENTS:
            return _error(f"unknown department: {department}")
        doctors = [
            {
                "doctor_id": f"doc_{department}_{index}",
                "name": doctor_name,
                "title": title,
                "hospital": "示范医院",
                "department": department,
                "rating": rating,
                "intro": intro,
                "fee": fee,
            }
            for index, (doctor_name, title, rating, intro, fee) in enumerate(
                [
                    ("张医生", "主任医师", 4.9, "擅长常见病与慢性病管理，注重循证与沟通。", 50),
                    ("李医生", "副主任医师", 4.8, "擅长症状鉴别与随访管理，耐心细致。", 40),
                    ("王医生", "主治医师", 4.7, "擅长急慢性疾病初诊与健康宣教。", 30),
                ],
                start=1,
            )
        ]
        return _text({"department": department, "doctors": doctors})
    if name == "get_visit_guide":
        async with _db() as conn:
            row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE order_id=$1", args.get("order_id") or "")
        if not row:
            return _error("order not found")
        if row["payment_status"] != "paid":
            return _error("order not paid")
        return _text(
            {
                "order": _order_json(row),
                "hospital": "示范医院",
                "address": "示例路 88 号",
                "floor": "门诊楼 3 层",
                "checkin": "请提前 20 分钟到自助机取号，携带身份证/医保卡。",
                "items": ["身份证/医保卡", "既往病历", "正在服用的药物清单"],
                "support": "如需帮助，可在问诊页继续提问。",
            }
        )
    if name == "get_payment_status":
        async with _db() as conn:
            row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE order_id=$1", args.get("order_id") or "")
        if not row:
            return _error("order not found")
        return _text(_order_json(row))
    if name == "cancel_order":
        order_id = args.get("order_id") or ""
        async with _db() as conn:
            row = await conn.fetchrow("SELECT * FROM mcp_orders WHERE order_id=$1", order_id)
            if not row:
                return _error("order not found")
            if row["payment_status"] == "paid":
                return _error("paid order cannot be cancelled in mock")
            row = await conn.fetchrow(
                "UPDATE mcp_orders SET status='cancelled', updated_at=now() WHERE order_id=$1 RETURNING *",
                order_id,
            )
            await conn.execute("DELETE FROM mcp_booked WHERE order_id=$1", order_id)
            await conn.execute(
                "DELETE FROM mcp_locks WHERE slot_id=$1 AND patient_id=$2",
                row["slot_id"],
                row["patient_id"],
            )
        return _text(_order_json(row))
    return _error(f"unknown tool: {name}")


@app.post("/mcp")
async def mcp(request: Request) -> dict[str, Any]:
    body = await request.json()
    request_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "hospital-mcp-mock", "version": "3.1.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        try:
            result = await _call(params.get("name") or "", params.get("arguments") or {})
        except Exception as exc:
            result = _error(f"database error: {exc}")
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        async with _db() as conn:
            orders = await conn.fetchval("SELECT count(*) FROM mcp_orders")
            locks = await conn.fetchval("SELECT count(*) FROM mcp_locks")
    except Exception as exc:
        return {"ok": False, "departments": len(DEPARTMENTS), "error": str(exc)}
    return {"ok": True, "departments": len(DEPARTMENTS), "orders": orders or 0, "locks": locks or 0}

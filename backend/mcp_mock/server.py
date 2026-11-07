from __future__ import annotations

import json
import os
import secrets
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request

BASE = Path(__file__).resolve().parents[1]
SPECIALISTS = json.loads((BASE / "app" / "data" / "specialists.json").read_text(encoding="utf-8"))["specialists"]
DEPARTMENTS = [item.get("name_cn", key) for key, item in SPECIALISTS.items()]

app = FastAPI(title="Hospital MCP Mock", version="2.0.0")

locks: dict[str, dict[str, Any]] = {}
orders: dict[str, dict[str, Any]] = {}
booked: set[str] = set()


def _slots(department: str) -> list[dict[str, Any]]:
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


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "list_departments":
        return _text({"departments": DEPARTMENTS})
    if name == "list_schedules":
        department = args.get("department") or ""
        if department not in DEPARTMENTS:
            return _error(f"unknown department: {department}")
        slots = _slots(department)
        day = args.get("date")
        if day:
            slots = [slot for slot in slots if slot["date"] == day]
        return _text({"department": department, "slots": slots[:40]})
    if name == "lock_slot":
        slot_id = args.get("slot_id") or ""
        if not slot_id or "|" not in slot_id:
            return _error("invalid slot_id")
        if slot_id in booked:
            return _error("slot already booked")
        lock_id = "lock_" + secrets.token_hex(6)
        locks[lock_id] = {
            "lock_id": lock_id,
            "slot_id": slot_id,
            "patient_id": args.get("patient_id") or "anonymous",
            "expires_at": time.time() + 600,
        }
        return _text({"lock_id": lock_id, "slot_id": slot_id, "expires_in": 600})
    if name == "create_order":
        lock_id = args.get("lock_id") or ""
        lock = locks.get(lock_id)
        if not lock:
            return _error("lock not found or expired")
        if lock["expires_at"] < time.time():
            return _error("lock expired")
        order_id = "order_" + secrets.token_hex(6)
        orders[order_id] = {
            "order_id": order_id,
            "slot_id": lock["slot_id"],
            "patient_id": args.get("patient_id") or lock["patient_id"],
            "amount": 50,
            "status": "created",
            "payment_status": "unpaid",
            "created_at": time.time(),
        }
        return _text(orders[order_id])
    if name == "pay_order":
        order_id = args.get("order_id") or ""
        order = orders.get(order_id)
        if not order:
            return _error("order not found")
        expected = os.getenv("HITL_CONFIRM_TOKEN", "confirm-demo")
        if args.get("confirm_token") != expected:
            return _error("requires_confirmation: human confirmation token missing or invalid")
        order["payment_status"] = "paid"
        order["status"] = "booked"
        order["paid_at"] = time.time()
        order["method"] = args.get("method") or "mock"
        booked.add(order["slot_id"])
        return _text(order)
    if name == "get_payment_status":
        order = orders.get(args.get("order_id") or "")
        if not order:
            return _error("order not found")
        return _text(order)
    if name == "cancel_order":
        order = orders.get(args.get("order_id") or "")
        if not order:
            return _error("order not found")
        if order["payment_status"] == "paid":
            return _error("paid order cannot be cancelled in mock")
        order["status"] = "cancelled"
        return _text(order)
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
            "serverInfo": {"name": "hospital-mcp-mock", "version": "2.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        result = _call(params.get("name") or "", params.get("arguments") or {})
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "departments": len(DEPARTMENTS), "orders": len(orders), "locks": len(locks)}

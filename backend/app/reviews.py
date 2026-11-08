from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import HTTPException

from .db import Database

PAYMENT_KIND = "payment"
STALE_PROCESSING_SECONDS = 120


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def review_json(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["payload"] = _decode(data.get("payload")) or {}
    return data


def booking_json(row: Any) -> dict[str, Any]:
    data = dict(row)
    if "detail" in data:
        data["detail"] = _decode(data["detail"])
    return data


async def _booking(conn: Any, booking_id: str) -> dict[str, Any] | None:
    row = await conn.fetchrow("SELECT * FROM bookings WHERE id=$1", booking_id)
    return booking_json(row) if row else None


async def _audit(conn: Any, actor: str, action: str, detail: dict[str, Any]) -> None:
    await conn.execute(
        """
        INSERT INTO audit_log (request_id, session_id, actor, action, detail)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        None,
        None,
        actor,
        action,
        json.dumps(detail, ensure_ascii=False, default=str),
    )


async def _ensure_followup(conn: Any, booking: Any) -> None:
    user_id = booking["user_id"] or booking["patient_id"]
    if not user_id:
        return
    exists = await conn.fetchval(
        "SELECT id FROM followups WHERE user_id=$1 AND booking_id=$2 LIMIT 1",
        user_id,
        booking["id"],
    )
    if exists:
        return
    await conn.execute(
        """
        INSERT INTO followups (id, user_id, member_id, booking_id, title, due_date, status, notes)
        VALUES ($1,$2,$3,$4,$5,CURRENT_DATE + INTERVAL '7 days','pending',$6)
        """,
        "fu_" + uuid.uuid4().hex[:10],
        user_id,
        booking["member_id"],
        booking["id"],
        f"诊后随访 · {booking['department']}",
        "就诊后 7 天，记录症状变化与用药情况。",
    )


def _server_token(mcp: Any, override: str | None) -> str:
    if override:
        return override
    token = getattr(getattr(mcp, "settings", None), "HITL_CONFIRM_TOKEN", None)
    return str(token or os.getenv("HITL_CONFIRM_TOKEN", "confirm-demo"))


async def create_payment_review(
    db: Database,
    *,
    user_id: str,
    member_id: str | None,
    booking_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    async with db.transaction() as conn:
        booking = await conn.fetchrow(
            "SELECT * FROM bookings WHERE id=$1 AND user_id=$2 FOR UPDATE",
            booking_id,
            user_id,
        )
        if not booking:
            raise HTTPException(status_code=404, detail="order not found")
        if booking["payment_status"] == "paid" or booking["status"] in {"paid", "booked"}:
            raise HTTPException(status_code=409, detail="order already paid")
        if booking["status"] == "cancelled":
            raise HTTPException(status_code=409, detail="order cancelled")
        existing = await conn.fetchrow(
            """
            SELECT * FROM review_requests
            WHERE kind=$1 AND ref_id=$2 AND status IN ('pending', 'processing')
            ORDER BY created_at DESC LIMIT 1
            """,
            PAYMENT_KIND,
            booking_id,
        )
        if existing:
            return {"review": review_json(existing), "order": booking_json(booking), "duplicate": True}
        review_id = "rev_" + uuid.uuid4().hex[:12]
        body = dict(payload or {})
        body.setdefault("order_id", booking["order_id"] or booking_id)
        body.setdefault("department", booking["department"])
        body.setdefault("slot_id", booking["slot_id"])
        body.setdefault("amount", float(booking["amount"] or 0))
        review = await conn.fetchrow(
            """
            INSERT INTO review_requests (id, kind, ref_id, user_id, member_id, status, payload)
            VALUES ($1,$2,$3,$4,$5,'pending',$6::jsonb)
            RETURNING *
            """,
            review_id,
            PAYMENT_KIND,
            booking_id,
            user_id,
            member_id,
            json.dumps(body, ensure_ascii=False, default=str),
        )
        booking = await conn.fetchrow(
            """
            UPDATE bookings
            SET status='pending_review', review_id=$2, member_id=COALESCE(member_id, $3), updated_at=now()
            WHERE id=$1
            RETURNING *
            """,
            booking_id,
            review_id,
            member_id,
        )
    return {"review": review_json(review), "order": booking_json(booking), "duplicate": False}


async def approve_review(
    db: Database,
    mcp: Any,
    *,
    review_id: str,
    admin_id: str,
    confirm_token: str | None = None,
) -> dict[str, Any]:
    async with db.transaction() as conn:
        review = await conn.fetchrow("SELECT * FROM review_requests WHERE id=$1 FOR UPDATE", review_id)
        if not review:
            raise HTTPException(status_code=404, detail="review not found")
        if review["status"] == "approved":
            return {
                "review": review_json(review),
                "order": await _booking(conn, review["ref_id"]),
                "mcp_called": False,
            }
        stale_before = datetime.now(UTC) - timedelta(seconds=STALE_PROCESSING_SECONDS)
        if review["status"] == "processing" and review["updated_at"] and review["updated_at"] > stale_before:
            return {
                "review": review_json(review),
                "order": await _booking(conn, review["ref_id"]),
                "mcp_called": False,
            }
        if review["status"] not in {"pending", "processing"}:
            raise HTTPException(status_code=409, detail=f"review is {review['status']}")
        review = await conn.fetchrow(
            "UPDATE review_requests SET status='processing', updated_at=now() WHERE id=$1 RETURNING *",
            review_id,
        )
        booking = await _booking(conn, review["ref_id"])
        if not booking:
            raise HTTPException(status_code=409, detail="booking not found for review")

    order_id = booking.get("order_id") or booking["id"]
    token = _server_token(mcp, confirm_token)
    try:
        result = await mcp.call_tool(
            "pay_order",
            {"order_id": order_id, "confirm_token": token, "method": "review"},
        )
    except Exception as exc:
        result = {"is_error": True, "text": str(exc), "data": None}
    if result.get("is_error"):
        note = str(result.get("text") or "payment failed")[:500]
        async with db.transaction() as conn:
            await conn.execute(
                "UPDATE review_requests SET status='failed', decision_note=$2, updated_at=now() WHERE id=$1",
                review_id,
                note,
            )
            await conn.execute(
                "UPDATE bookings SET status='created', updated_at=now() WHERE id=$1 AND status='pending_review'",
                review["ref_id"],
            )
            await _audit(
                conn,
                admin_id,
                "payment.approve_failed",
                {"review_id": review_id, "order_id": order_id, "error": note},
            )
        raise HTTPException(status_code=502, detail=f"payment failed: {note}")

    paid = result.get("data") or {}
    async with db.transaction() as conn:
        current = await conn.fetchrow("SELECT * FROM review_requests WHERE id=$1 FOR UPDATE", review_id)
        if current["status"] == "approved":
            return {
                "review": review_json(current),
                "order": await _booking(conn, current["ref_id"]),
                "mcp_called": False,
            }
        review = await conn.fetchrow(
            """
            UPDATE review_requests SET status='approved', decided_by=$2, decided_at=now(), updated_at=now()
            WHERE id=$1 RETURNING *
            """,
            review_id,
            admin_id,
        )
        booking = await conn.fetchrow(
            """
            UPDATE bookings SET status=$2, payment_status=$3, updated_at=now()
            WHERE id=$1 RETURNING *
            """,
            review["ref_id"],
            paid.get("status") or "booked",
            paid.get("payment_status") or "paid",
        )
        if booking:
            await _ensure_followup(conn, booking)
        await _audit(
            conn,
            admin_id,
            "payment.approve",
            {
                "review_id": review_id,
                "order_id": order_id,
                "amount": float(booking["amount"] or 0) if booking else None,
            },
        )
    return {
        "review": review_json(review),
        "order": booking_json(booking) if booking else None,
        "mcp_called": True,
    }


async def reject_review(
    db: Database,
    *,
    review_id: str,
    admin_id: str,
    note: str,
) -> dict[str, Any]:
    async with db.transaction() as conn:
        review = await conn.fetchrow("SELECT * FROM review_requests WHERE id=$1 FOR UPDATE", review_id)
        if not review:
            raise HTTPException(status_code=404, detail="review not found")
        if review["status"] == "rejected":
            return {"review": review_json(review), "order": await _booking(conn, review["ref_id"])}
        if review["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"review is {review['status']}")
        review = await conn.fetchrow(
            """
            UPDATE review_requests
            SET status='rejected', decision_note=$2, decided_by=$3, decided_at=now(), updated_at=now()
            WHERE id=$1 RETURNING *
            """,
            review_id,
            note,
            admin_id,
        )
        booking = await conn.fetchrow(
            "UPDATE bookings SET status='created', updated_at=now() WHERE id=$1 AND status='pending_review' RETURNING *",
            review["ref_id"],
        )
        if booking is None:
            booking = await conn.fetchrow("SELECT * FROM bookings WHERE id=$1", review["ref_id"])
        payload = _decode(review["payload"]) or {}
        audit_order_id = payload.get("order_id") or (booking["order_id"] if booking else None) or review["ref_id"]
        await _audit(
            conn,
            admin_id,
            "payment.reject",
            {"review_id": review_id, "order_id": audit_order_id, "note": note},
        )
    return {
        "review": review_json(review),
        "order": booking_json(booking) if booking else None,
    }

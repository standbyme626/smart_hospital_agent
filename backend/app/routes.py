from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import uuid
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .api.deps import create_token, get_current_user, require_admin
from .conversation import append_message, load_history
from .policy import DEPARTMENT_NAMES, DRUG_CHECKER
from .reports import (
    has_report_image,
    remove_report_image,
    report_image_media_type,
    resolve_report_image,
)
from .reviews import approve_review, create_payment_review, reject_review, review_json
from .schemas import (
    AccountDeleteRequest,
    AdminUserCreateRequest,
    AdminUserUpdateRequest,
    AnalyticsEventRequest,
    ChatRequest,
    DrugCheckRequest,
    EvalRequest,
    FamilyMemberRequest,
    FollowupRequest,
    LockRequest,
    LoginRequest,
    MedCheckRequest,
    MedicationRequest,
    OrderRequest,
    PayRequest,
    ReviewRejectRequest,
)
from .security import (
    DUMMY_PASSWORD_HASH,
    clear_failures,
    hash_password,
    is_locked,
    login_key,
    record_failure,
    verify_password,
)
from .session_store import jsonable

router = APIRouter()

logger = logging.getLogger(__name__)


def _state(request: Request) -> Any:
    return request.app.state


def _public_user(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        key: value
        for key, value in data.items()
        if key in {"id", "username", "role", "enabled", "created_at", "updated_at"}
    }


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(jsonable(event.get("data", {})), ensure_ascii=False, default=str)
    return f"id: {event.get('seq', 0)}\nevent: {event.get('type', 'message')}\ndata: {data}\n\n"


async def _mcp(state: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await state.mcp.call_tool(name, arguments)
    if result["is_error"]:
        raise HTTPException(status_code=502, detail=result["text"])
    return result["data"] or {}




def _row_json(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("abnormal_values", "questions", "raw", "detail", "reminder_times", "props"):
        if isinstance(data.get(key), str):
            try:
                data[key] = json.loads(data[key])
            except Exception:
                pass
    return data


def _report_json(settings: Any, row: Any) -> dict[str, Any]:
    data = _row_json(row)
    image_path = data.pop("image_path", None)
    data["has_image"] = has_report_image(settings, image_path)
    return data


async def _ensure_self_member(db: Any, user_id: str) -> dict[str, Any]:
    row = await db.fetchrow(
        "SELECT * FROM family_members WHERE user_id=$1 AND is_self=true ORDER BY created_at LIMIT 1",
        user_id,
    )
    if row:
        return dict(row)
    member_id = "mem_" + uuid.uuid4().hex[:10]
    row = await db.fetchrow(
        """
        INSERT INTO family_members (id, user_id, name, relation, is_self, hospital_patient_id)
        VALUES ($1, $2, '本人', 'self', true, $3)
        RETURNING *
        """,
        member_id,
        user_id,
        f"HP-{member_id}",
    )
    return dict(row) if row else {"id": member_id, "user_id": user_id, "name": "本人", "relation": "self", "is_self": True}


async def _owned_member(db: Any, member_id: str, user_id: str) -> dict[str, Any]:
    row = await db.fetchrow(
        "SELECT * FROM family_members WHERE id=$1 AND user_id=$2",
        member_id,
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="member not found")
    return dict(row)


async def _owned_session(db: Any, session_id: str, user_id: str) -> dict[str, Any]:
    row = await db.fetchrow(
        "SELECT * FROM sessions WHERE id=$1 AND user_id=$2",
        session_id,
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="session not found")
    return dict(row)


async def _owned_report(db: Any, report_id: str, user_id: str) -> dict[str, Any]:
    row = await db.fetchrow(
        "SELECT * FROM reports WHERE id=$1 AND user_id=$2",
        report_id,
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="report not found")
    return dict(row)


async def _owned_order(db: Any, order_id: str, user_id: str) -> dict[str, Any]:
    row = await db.fetchrow(
        "SELECT * FROM bookings WHERE id=$1 AND user_id=$2",
        order_id,
        user_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="order not found")
    return dict(row)


def _order_view(row: Any) -> dict[str, Any]:
    data = dict(row)
    detail = data.get("detail")
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except json.JSONDecodeError:
            detail = {}
    view = dict(detail) if isinstance(detail, dict) else {}
    view.update(
        {
            "order_id": data.get("order_id") or data.get("id"),
            "slot_id": data.get("slot_id"),
            "patient_id": data.get("patient_id"),
            "amount": float(data["amount"]) if data.get("amount") is not None else 0,
            "status": data.get("status"),
            "payment_status": data.get("payment_status"),
        }
    )
    return view


async def _member_row(db: Any, member_id: str | None, user_id: str) -> dict[str, Any]:
    if member_id:
        return await _owned_member(db, member_id, user_id)
    return await _ensure_self_member(db, user_id)


async def _add_event(db: Any, user_id: str, member_id: str | None, kind: str, title: str, detail: dict[str, Any]) -> None:
    await db.execute(
        """
        INSERT INTO health_events (user_id, member_id, kind, title, detail)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        user_id,
        member_id,
        kind,
        title,
        json.dumps(detail, ensure_ascii=False, default=str),
    )


def _parse_json_text(content: str) -> dict[str, Any] | None:
    text = (content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None

# --------------------------------------------------------------------- auth
@router.post("/api/v1/auth/login")
async def login(payload: LoginRequest, request: Request) -> dict[str, Any]:
    state = _state(request)
    if not state.db or not getattr(state.db, "pool", None):
        raise HTTPException(status_code=503, detail="database unavailable")
    key = login_key(payload.username)
    if is_locked(key):
        raise HTTPException(status_code=423, detail="account locked, try again later")
    try:
        row = await state.db.fetchrow(
            "SELECT id, username, role, enabled, password_hash FROM users WHERE username=$1",
            payload.username,
        )
    except (OSError, asyncpg.PostgresError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    encoded = row["password_hash"] if row else DUMMY_PASSWORD_HASH
    valid = await asyncio.to_thread(verify_password, payload.password, encoded)
    if not row or not valid:
        if record_failure(key):
            raise HTTPException(status_code=423, detail="account locked, try again later")
        raise HTTPException(status_code=401, detail="invalid credentials")
    if not row["enabled"]:
        raise HTTPException(status_code=403, detail="user disabled")
    clear_failures(key)
    return {
        "access_token": create_token(state.settings, row["id"], row["role"]),
        "token_type": "bearer",
        "role": row["role"],
        "username": row["username"],
    }


@router.get("/api/v1/auth/me")
async def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {
        "user": user.get("sub"),
        "username": user.get("username"),
        "role": user.get("role"),
        "enabled": user.get("enabled", True),
    }


# -------------------------------------------------------------------- chat
@router.post("/api/v1/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
):
    state = _state(request)
    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    user_id = str(user["sub"])
    if payload.session_id:
        existing = await state.db.fetchrow(
            "SELECT user_id FROM sessions WHERE id=$1",
            session_id,
        )
        if existing and existing["user_id"] != user_id:
            raise HTTPException(status_code=404, detail="session not found")
    daily_limit = int(state.settings.CHAT_DAILY_LIMIT)
    if daily_limit > 0:
        used_today = await state.db.fetchval(
            """
            SELECT count(*) FROM messages
            WHERE user_id=$1 AND role='user' AND created_at >= date_trunc('day', now())
            """,
            user_id,
        )
        if int(used_today or 0) >= daily_limit:
            raise HTTPException(
                status_code=429,
                detail=f"daily chat limit reached: {daily_limit} messages per day",
            )
    try:
        member = await _member_row(state.db, payload.member_id, user_id)
    except Exception:
        member = {"id": None, "name": "本人", "relation": "self"}
    member_bits = [f"就诊人：{member.get('name', '本人')}（{member.get('relation', 'self')}）"]
    if member.get("gender"):
        member_bits.append(f"性别：{member['gender']}")
    if member.get("birth_date"):
        member_bits.append(f"出生日期：{member['birth_date']}")
    if member.get("allergies"):
        member_bits.append(f"过敏史：{member['allergies']}")
    if member.get("conditions"):
        member_bits.append(f"既往史：{member['conditions']}")
    member_context = "；".join(member_bits)
    message = f"{member_context}\n\n用户输入：{payload.message}"

    async def generator():
        yield _sse(
            {
                "seq": 0,
                "type": "session",
                "data": {"session_id": session_id, "request_id": request_id},
            }
        )
        final_data: dict[str, Any] | None = None
        try:
            await state.sessions.ensure(
                session_id,
                user_id,
                title=payload.message[:40],
                member_id=member.get("id"),
            )
            history = await load_history(state.db, session_id)
            await append_message(
                state.db,
                session_id=session_id,
                user_id=user_id,
                member_id=member.get("id"),
                role="user",
                content=message,
            )
            async for event in state.agent.stream(
                session_id=session_id,
                message=message,
                user_id=user_id,
                request_id=request_id,
                title=payload.message,
                member_id=member.get("id"),
                history=history,
            ):
                if event.get("type") == "final":
                    final_data = event.get("data")
                yield _sse(event)
            if final_data is not None:
                report = final_data.get("report") or {}
                content = (
                    str(report.get("summary") or "").strip()
                    or str(final_data.get("summary") or "").strip()
                    or str(final_data.get("content") or "").strip()
                    or "已完成"
                )
                try:
                    await append_message(
                        state.db,
                        session_id=session_id,
                        user_id=user_id,
                        member_id=member.get("id"),
                        role="assistant",
                        content=content,
                    )
                    if report:
                        await _add_event(
                        state.db,
                        user_id,
                        member.get("id"),
                        "consult",
                        f"问诊 · {report.get('department', '分诊')}",
                            {
                                "summary": str(report.get("summary", ""))[:300],
                                "urgency": report.get("urgency"),
                                "citations": report.get("citations", []),
                            },
                        )
                except Exception:
                    pass
        except Exception as exc:
            yield _sse(
                {
                    "seq": 0,
                    "type": "error",
                    "data": {"message": "服务暂时不可用，请稍后重试", "detail": str(exc)[:200]},
                }
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/v1/sessions")
async def list_sessions(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    sessions = await _state(request).sessions.list_sessions(
        str(user["sub"]), member_id=member_id
    )
    return {"sessions": sessions}


@router.get("/api/v1/sessions/{session_id}")
async def session_detail(
    session_id: str,
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    session = await _owned_session(db, session_id, str(user["sub"]))
    if member_id and session.get("member_id") != member_id:
        raise HTTPException(status_code=404, detail="session not found")
    rows = await db.fetch(
        """
        SELECT id, role, content, created_at FROM (
            SELECT id, role, content, created_at
            FROM messages
            WHERE session_id=$1
            ORDER BY id DESC
            LIMIT 200
        ) recent
        ORDER BY id ASC
        """,
        session_id,
    )
    latest = await db.fetchrow(
        "SELECT data FROM events WHERE session_id=$1 AND type='final' ORDER BY seq DESC LIMIT 1",
        session_id,
    )
    latest_final = latest["data"] if latest else None
    if isinstance(latest_final, str):
        try:
            latest_final = json.loads(latest_final)
        except json.JSONDecodeError:
            latest_final = None
    return {
        "session": jsonable(session),
        "messages": jsonable([dict(row) for row in rows]),
        "latest_final": jsonable(latest_final),
    }


@router.get("/api/v1/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    after: int = Query(0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    await _owned_session(_state(request).db, session_id, str(user["sub"]))
    events = await _state(request).sessions.events(session_id, after=after)
    return {"session_id": session_id, "events": events}


# --------------------------------------------------------------------- rag
@router.get("/api/v1/rag/status")
async def rag_status(request: Request) -> dict[str, Any]:
    state = _state(request)
    stats = await state.retrieval.stats()
    return {
        "documents": stats["documents"],
        "departments": stats["departments"],
        "models": await state.gateway.health(),
        "mcp": await state.mcp.health(),
    }


@router.get("/api/v1/rag/documents/{doc_key}")
async def rag_document(
    doc_key: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        "SELECT doc_key, title, content, department, source, metadata FROM documents WHERE doc_key=$1",
        doc_key,
    )
    if not row:
        raise HTTPException(status_code=404, detail="document not found")
    data = dict(row)
    metadata = data.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    data["metadata"] = metadata or {}
    return data


@router.get("/api/v1/departments")
async def departments() -> dict[str, Any]:
    return {"departments": DEPARTMENT_NAMES}


@router.post("/api/v1/drug/check")
async def drug_check(
    payload: DrugCheckRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return DRUG_CHECKER.check_text(payload.text)


# ----------------------------------------------------------------- booking
@router.get("/api/v1/booking/departments")
async def booking_departments(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    data = await _mcp(_state(request), "list_departments", {})
    return data


@router.get("/api/v1/booking/schedules")
async def booking_schedules(
    request: Request,
    department: str = Query(...),
    date: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    data = await _mcp(_state(request), "list_schedules", {"department": department, "date": date})
    return data


@router.post("/api/v1/booking/lock")
async def booking_lock(
    payload: LockRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    if payload.member_id:
        await _owned_member(state.db, payload.member_id, str(user["sub"]))
    data = await _mcp(
        state,
        "lock_slot",
        {"slot_id": payload.slot_id, "patient_id": str(user["sub"])},
    )
    expires_at = datetime.now(UTC) + timedelta(seconds=int(data.get("expires_in") or 600))
    return {"lock": {**data, "expires_at": expires_at.isoformat()}}


@router.get("/api/v1/booking/orders")
async def list_orders(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    if member_id:
        await _owned_member(db, member_id, user_id)
    rows = await db.fetch(
        """
        SELECT id, order_id, department, slot_id, member_id, hospital_patient_id,
               status, payment_status, amount, created_at, updated_at
        FROM bookings
        WHERE user_id=$1 AND ($2::text IS NULL OR member_id=$2)
        ORDER BY created_at DESC, id DESC
        LIMIT 50
        """,
        user_id,
        member_id,
    )
    return {"orders": [jsonable(dict(row)) for row in rows]}


@router.post("/api/v1/booking/orders")
async def booking_order(
    payload: OrderRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    member = await _member_row(state.db, payload.member_id, user_id)
    member_id = member["id"]
    hospital_patient_id = member.get("hospital_patient_id") or f"HP-{member_id}"
    if not member.get("hospital_patient_id"):
        await state.db.execute(
            "UPDATE family_members SET hospital_patient_id=$2 WHERE id=$1",
            member_id,
            hospital_patient_id,
        )
    idempotency_key = f"order:{user_id}:{payload.lock_id}"
    result = await state.mcp.call_tool(
        "create_order",
        {"lock_id": payload.lock_id, "patient_id": user_id},
        idempotency_key=idempotency_key,
    )
    if result["is_error"]:
        raise HTTPException(status_code=502, detail=result["text"])
    data = result["data"] or {}
    await state.db.execute(
        """
        INSERT INTO bookings
            (id, session_id, patient_id, user_id, department, slot_id, order_id, status,
             payment_status, amount, idempotency_key, member_id, hospital_patient_id, detail)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        data.get("order_id"),
        user_id,
        user_id,
        user_id,
        (data.get("slot_id") or "").split("|")[0],
        data.get("slot_id"),
        data.get("order_id"),
        data.get("status", "created"),
        data.get("payment_status", "unpaid"),
        data.get("amount", 50),
        idempotency_key,
        member_id,
        hospital_patient_id,
        json.dumps(data, ensure_ascii=False),
    )
    return {"order": data}


@router.post("/api/v1/booking/orders/{order_id}/request-payment")
async def booking_request_payment(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    booking = await _owned_order(state.db, order_id, user_id)
    result = await create_payment_review(
        state.db,
        user_id=user_id,
        member_id=booking.get("member_id"),
        booking_id=order_id,
        payload={"department": booking.get("department"), "slot_id": booking.get("slot_id")},
    )
    await state.sessions.audit(
        request_id=request.headers.get("X-Request-ID") or uuid.uuid4().hex,
        session_id=user_id,
        actor=user_id,
        action="payment.request",
        detail={
            "review_id": result["review"]["id"],
            "order_id": booking.get("order_id") or order_id,
            "duplicate": result.get("duplicate", False),
        },
    )
    return result


@router.post("/api/v1/booking/orders/{order_id}/pay")
async def booking_pay(
    order_id: str,
    request: Request,
    payload: PayRequest | None = None,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    await _owned_order(state.db, order_id, str(user["sub"]))
    raise HTTPException(
        status_code=403,
        detail="patient payment is disabled; request admin review via /request-payment",
    )


@router.get("/api/v1/booking/orders/{order_id}")
async def booking_status(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    await _owned_order(state.db, order_id, str(user["sub"]))
    data = await _mcp(state, "get_payment_status", {"order_id": order_id})
    review = await state.db.fetchrow(
        "SELECT * FROM review_requests WHERE kind='payment' AND ref_id=$1 ORDER BY created_at DESC LIMIT 1",
        order_id,
    )
    return {"order": data, "review": review_json(review) if review else None}


@router.post("/api/v1/booking/orders/{order_id}/cancel")
async def booking_cancel(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    async with state.db.transaction() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM bookings WHERE id=$1 AND user_id=$2 FOR UPDATE",
            order_id,
            user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="order not found")
        booking = dict(row)
        if booking.get("payment_status") == "paid" or booking.get("status") in {"paid", "booked"}:
            raise HTTPException(status_code=409, detail="paid order cannot be cancelled")
        if booking.get("status") == "cancelled":
            return {"order": _order_view(booking)}
        if booking.get("status") not in {"created", "pending_review"}:
            raise HTTPException(status_code=409, detail="order cannot be cancelled")
        result = await state.mcp.call_tool("cancel_order", {"order_id": order_id})
        if result.get("is_error"):
            logger.warning(
                "mcp cancel failed for order %s: %s", order_id, result.get("text")
            )
            raise HTTPException(status_code=502, detail="取消订单失败，请稍后重试")
        updated = await conn.fetchrow(
            """
            UPDATE bookings SET status='cancelled', updated_at=now()
            WHERE id=$1 AND user_id=$2
            RETURNING *
            """,
            order_id,
            user_id,
        )
        await conn.execute(
            """
            UPDATE review_requests
            SET status='rejected',
                decision_note=COALESCE(decision_note, '订单已取消'),
                decided_at=COALESCE(decided_at, now()),
                updated_at=now()
            WHERE kind='payment' AND ref_id=$1 AND status IN ('pending', 'processing')
            """,
            order_id,
        )
    await state.sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=user_id,
        actor=user_id,
        action="order.cancel",
        detail={"order_id": order_id},
    )
    return {"order": _order_view(updated)}


# ------------------------------------------------------------------- admin
@router.get("/api/v1/admin/users")
async def list_users(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        "SELECT id, username, role, enabled, created_at, updated_at FROM users ORDER BY created_at, username"
    )
    return {"users": [_public_user(row) for row in rows]}


@router.post("/api/v1/admin/users", status_code=201)
async def create_user(
    payload: AdminUserCreateRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = _state(request).db
    row = await db.fetchrow(
        """
        INSERT INTO users (id, username, password_hash, role, enabled)
        VALUES ($1, $1, $2, $3, $4)
        ON CONFLICT (id) DO NOTHING
        RETURNING id, username, role, enabled, created_at, updated_at
        """,
        payload.username,
        await asyncio.to_thread(hash_password, payload.password),
        payload.role,
        payload.enabled,
    )
    if not row:
        raise HTTPException(status_code=409, detail="user already exists")
    return {"user": _public_user(row)}


@router.patch("/api/v1/admin/users/{user_id}")
async def update_user(
    user_id: str,
    payload: AdminUserUpdateRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = _state(request).db
    if payload.enabled is None and payload.role is None and payload.password is None:
        raise HTTPException(status_code=400, detail="empty update")
    new_hash = (
        await asyncio.to_thread(hash_password, payload.password)
        if payload.password is not None
        else None
    )
    async with db.transaction() as conn:
        target = await conn.fetchrow(
            "SELECT id, role, enabled FROM users WHERE id=$1",
            user_id,
        )
        if not target:
            raise HTTPException(status_code=404, detail="user not found")
        removing_admin = bool(
            target["role"] == "admin"
            and target["enabled"]
            and (
                payload.enabled is False
                or (payload.role is not None and payload.role != "admin")
            )
        )
        if removing_admin:
            remaining = await conn.fetchval(
                "SELECT count(*) FROM users WHERE role='admin' AND enabled=true AND id<>$1",
                user_id,
            )
            if remaining == 0:
                raise HTTPException(
                    status_code=400,
                    detail="cannot disable or demote the last enabled admin",
                )
        sets = ["updated_at = now()"]
        args: list[Any] = [user_id]
        if payload.enabled is not None:
            args.append(payload.enabled)
            sets.append(f"enabled = ${len(args)}")
        if payload.role is not None:
            args.append(payload.role)
            sets.append(f"role = ${len(args)}")
        if new_hash is not None:
            args.append(new_hash)
            sets.append(f"password_hash = ${len(args)}")
        row = await conn.fetchrow(
            f"UPDATE users SET {', '.join(sets)} WHERE id = $1 "
            "RETURNING id, username, role, enabled, created_at, updated_at",
            *args,
        )
    return {"user": _public_user(row)}


@router.get("/api/v1/admin/reviews")
async def admin_reviews(
    request: Request,
    status: str | None = Query(None),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT r.*, b.department, b.slot_id, b.amount, b.payment_status,
               b.status AS booking_status, b.order_id,
               u.username, fm.name AS member_name
        FROM review_requests r
        LEFT JOIN bookings b ON b.id = r.ref_id
        LEFT JOIN users u ON u.id = r.user_id
        LEFT JOIN family_members fm ON fm.id = r.member_id
        WHERE ($1::text IS NULL OR r.status = $1)
        ORDER BY r.created_at DESC
        LIMIT 100
        """,
        status,
    )
    reviews: list[dict[str, Any]] = []
    for row in rows:
        item = review_json(row)
        if item.get("amount") is not None:
            item["amount"] = float(item["amount"])
        reviews.append(item)
    return {"reviews": reviews}


@router.post("/api/v1/admin/reviews/{review_id}/approve")
async def admin_approve_review(
    review_id: str,
    request: Request,
    response: Response,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    result = await approve_review(
        state.db,
        state.mcp,
        review_id=review_id,
        admin_id=str(user["sub"]),
        confirm_token=state.settings.HITL_CONFIRM_TOKEN,
    )
    if not result.get("mcp_called") and result["review"]["status"] == "processing":
        response.status_code = 202
    return result


@router.post("/api/v1/admin/reviews/{review_id}/reject")
async def admin_reject_review(
    review_id: str,
    payload: ReviewRejectRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    result = await reject_review(
        _state(request).db,
        review_id=review_id,
        admin_id=str(user["sub"]),
        note=payload.note,
    )
    return result


@router.get("/api/v1/admin/model-calls")
async def model_calls(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT provider, model, task, count(*) AS calls, sum(prompt_tokens) AS prompt_tokens,
               sum(completion_tokens) AS completion_tokens, sum(cost) AS cost,
               round(avg(latency_ms)) AS avg_latency_ms
        FROM model_calls
        GROUP BY provider, model, task
        ORDER BY calls DESC
        LIMIT 50
        """
    )
    return {"calls": [dict(row) for row in rows]}


@router.get("/api/v1/admin/audit")
async def audit_log(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        "SELECT request_id, session_id, actor, action, detail, created_at FROM audit_log ORDER BY id DESC LIMIT 100"
    )
    return {"audit": [dict(row) for row in rows]}


# -------------------------------------------------------------------- eval
@router.post("/api/v1/eval/run")
async def run_eval(
    payload: EvalRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    metrics = await state.eval_runner.run(limit=payload.limit)
    await state.db.execute(
        "INSERT INTO eval_runs (name, metrics) VALUES ($1, $2::jsonb)",
        payload.name,
        json.dumps(metrics, ensure_ascii=False),
    )
    return {"name": payload.name, "metrics": metrics}


@router.get("/api/v1/eval/results")
async def eval_results(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        "SELECT id, name, metrics, created_at FROM eval_runs ORDER BY id DESC LIMIT 20"
    )
    return {"results": [dict(row) for row in rows]}


# ------------------------------------------------------------------- family
@router.get("/api/v1/family")
async def list_family(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    await _ensure_self_member(db, user_id)
    rows = await db.fetch(
        "SELECT * FROM family_members WHERE user_id=$1 ORDER BY is_self DESC, created_at ASC",
        user_id,
    )
    return {"members": [dict(row) for row in rows]}


@router.post("/api/v1/family")
async def create_family_member(
    payload: FamilyMemberRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member_id = "mem_" + uuid.uuid4().hex[:10]
    if payload.is_self:
        await db.execute("UPDATE family_members SET is_self=false WHERE user_id=$1", user_id)
    row = await db.fetchrow(
        """
        INSERT INTO family_members
            (id, user_id, name, relation, gender, birth_date, phone, allergies, conditions,
             is_self, hospital_patient_id)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
        RETURNING *
        """,
        member_id,
        user_id,
        payload.name,
        payload.relation,
        payload.gender,
        payload.birth_date,
        payload.phone,
        payload.allergies,
        payload.conditions,
        payload.is_self,
        f"HP-{member_id}",
    )
    return {"member": dict(row) if row else None}


@router.patch("/api/v1/family/{member_id}")
async def update_family_member(
    member_id: str,
    payload: FamilyMemberRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    existing = await db.fetchrow(
        "SELECT * FROM family_members WHERE id=$1 AND user_id=$2", member_id, user_id
    )
    if not existing:
        raise HTTPException(status_code=404, detail="member not found")
    if payload.is_self:
        await db.execute("UPDATE family_members SET is_self=false WHERE user_id=$1", user_id)
    row = await db.fetchrow(
        """
        UPDATE family_members SET
            name=$3, relation=$4, gender=$5, birth_date=$6, phone=$7,
            allergies=$8, conditions=$9, is_self=$10, updated_at=now()
        WHERE id=$1 AND user_id=$2
        RETURNING *
        """,
        member_id,
        user_id,
        payload.name,
        payload.relation,
        payload.gender,
        payload.birth_date,
        payload.phone,
        payload.allergies,
        payload.conditions,
        payload.is_self,
    )
    return {"member": dict(row) if row else None}


@router.delete("/api/v1/family/{member_id}")
async def delete_family_member(
    member_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    db = state.db
    user_id = str(user["sub"])
    row = await db.fetchrow(
        "SELECT is_self FROM family_members WHERE id=$1 AND user_id=$2", member_id, user_id
    )
    if not row:
        raise HTTPException(status_code=404, detail="member not found")
    if row["is_self"]:
        raise HTTPException(status_code=400, detail="cannot delete self member")
    report_rows = await db.fetch(
        "SELECT id, image_path FROM reports WHERE user_id=$1 AND member_id=$2",
        user_id,
        member_id,
    )
    report_ids = [item["id"] for item in report_rows]
    session_ids = [
        item["id"]
        for item in await db.fetch(
            "SELECT id FROM sessions WHERE user_id=$1 AND member_id=$2",
            user_id,
            member_id,
        )
    ]
    try:
        async with db.transaction() as conn:
            if session_ids:
                await conn.execute(
                    "DELETE FROM events WHERE session_id = ANY($1::text[])", session_ids
                )
            await conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = ANY($3::text[])
                   OR (user_id=$1 AND member_id=$2)
                """,
                user_id,
                member_id,
                session_ids,
            )
            await conn.execute(
                "DELETE FROM sessions WHERE user_id=$1 AND member_id=$2", user_id, member_id
            )
            await conn.execute(
                "DELETE FROM followups WHERE user_id=$1 AND member_id=$2", user_id, member_id
            )
            await conn.execute(
                "DELETE FROM medications WHERE user_id=$1 AND member_id=$2", user_id, member_id
            )
            await conn.execute(
                "DELETE FROM reports WHERE user_id=$1 AND member_id=$2", user_id, member_id
            )
            await conn.execute(
                "DELETE FROM health_events WHERE user_id=$1 AND member_id=$2", user_id, member_id
            )
            if report_ids:
                await conn.execute(
                    "DELETE FROM health_events WHERE user_id=$1 AND detail->>'report_id' = ANY($2::text[])",
                    user_id,
                    report_ids,
                )
            await conn.execute(
                "DELETE FROM family_members WHERE id=$1 AND user_id=$2", member_id, user_id
            )
    except Exception as exc:
        logger.exception("family delete failed for member %s", member_id)
        raise HTTPException(status_code=500, detail="failed to delete member data") from exc
    failed_images = [
        str(item["image_path"])
        for item in report_rows
        if not remove_report_image(state.settings, item["image_path"])
    ]
    if failed_images:
        logger.warning(
            "family delete for member %s left report images on disk: %s",
            member_id,
            ", ".join(failed_images),
        )
    await state.sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=user_id,
        actor=user_id,
        action="family.delete",
        detail={"member_id": member_id},
    )
    return {"ok": True}


# ---------------------------------------------------------------- medication
@router.get("/api/v1/medications")
async def list_medications(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    if member_id:
        await _owned_member(db, member_id, user_id)
        rows = await db.fetch(
            "SELECT * FROM medications WHERE user_id=$1 AND member_id=$2 ORDER BY active DESC, created_at DESC",
            user_id,
            member_id,
        )
    else:
        rows = await db.fetch(
            "SELECT * FROM medications WHERE user_id=$1 ORDER BY active DESC, created_at DESC",
            user_id,
        )
    return {"medications": [_row_json(row) for row in rows]}


@router.post("/api/v1/medications")
async def create_medication(
    payload: MedicationRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, payload.member_id, user_id)
    med_id = "med_" + uuid.uuid4().hex[:10]
    row = await db.fetchrow(
        """
        INSERT INTO medications (id, user_id, member_id, name, dose, frequency, reminder_times, notes)
        VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8)
        RETURNING *
        """,
        med_id,
        user_id,
        member["id"],
        payload.name,
        payload.dose,
        payload.frequency,
        json.dumps(payload.reminder_times, ensure_ascii=False),
        payload.notes,
    )
    await _add_event(
        db,
        user_id,
        member["id"],
        "medication",
        f"新增用药 · {payload.name}",
        {"name": payload.name, "dose": payload.dose, "frequency": payload.frequency},
    )
    return {"medication": _row_json(row) if row else None}


@router.patch("/api/v1/medications/{med_id}")
async def update_medication(
    med_id: str,
    request: Request,
    active: bool = Query(...),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    row = await db.fetchrow(
        "UPDATE medications SET active=$3 WHERE id=$1 AND user_id=$2 RETURNING *",
        med_id,
        user_id,
        active,
    )
    if not row:
        raise HTTPException(status_code=404, detail="medication not found")
    return {"medication": _row_json(row)}


@router.delete("/api/v1/medications/{med_id}")
async def delete_medication(
    med_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    deleted = await db.execute("DELETE FROM medications WHERE id=$1 AND user_id=$2", med_id, user_id)
    if deleted == "DELETE 0":
        raise HTTPException(status_code=404, detail="medication not found")
    return {"ok": True}


@router.post("/api/v1/medications/check")
async def check_medications(
    payload: MedCheckRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, payload.member_id, user_id)
    rows = await db.fetch(
        "SELECT name, dose FROM medications WHERE user_id=$1 AND member_id=$2 AND active=true",
        user_id,
        member["id"],
    )
    names = "、".join([row["name"] for row in rows] or [])
    text = f"{names} {payload.text}".strip()
    result = DRUG_CHECKER.check_text(text)
    return {"member_id": member["id"], "current_medications": [dict(row) for row in rows], **result}


# -------------------------------------------------------------------- report
REPORT_SYSTEM = """你是一位严谨的医学检验/检查报告解读助手。只输出 JSON，不要输出 Markdown 代码块：
{
  "summary": "用普通人能懂的话总结报告，<=200字，不诊断、不开药",
  "abnormal_values": [
    {"name": "指标名", "value": "结果", "reference": "参考范围", "status": "high|low|normal", "explanation": "这条指标意味着什么，<=60字"}
  ],
  "questions": ["建议问医生的3个问题"],
  "disclaimer": "本解读不能替代医生诊断"
}
如果图片不是报告或无法识别，summary 说明无法识别，abnormal_values 为空。"""

SKIN_SYSTEM = """你是皮肤健康观察助手。只输出 JSON，不要输出 Markdown：
{
  "summary": "客观描述照片中可见的皮肤表现（部位、颜色、形态、范围），用普通人能懂的话，<=200字，不做诊断",
  "abnormal_values": [],
  "questions": ["建议问医生的3个问题"],
  "disclaimer": "照片观察不能替代皮肤科医生面诊"
}
不要推荐处方药。若出现发热、疼痛明显、快速扩散、呼吸困难、眼周/口唇肿胀等，建议尽快就医。"""


@router.post("/api/v1/reports/interpret")
async def interpret_report(
    request: Request,
    file: UploadFile = File(...),
    member_id: str = Form(""),
    title: str = Form(""),
    kind: str = Form("report"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    settings = state.settings
    user_id = str(user["sub"])
    member = await _member_row(state.db, member_id or None, user_id)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"file larger than {settings.MAX_UPLOAD_MB}MB")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    mime = file.content_type or "image/jpeg"
    upload_dir = Path(settings.UPLOADS_DIR) / "reports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    report_id = "rep_" + uuid.uuid4().hex[:12]
    image_path = upload_dir / f"{report_id}{suffix}"
    committed = False
    try:
        image_path.write_bytes(data)
        try:
            b64 = base64.b64encode(data).decode("ascii")
            is_skin = kind == "skin"
            result = await state.gateway.chat(
                [
                    {"role": "system", "content": SKIN_SYSTEM if is_skin else REPORT_SYSTEM},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "请观察这张皮肤照片，按系统要求的 JSON 输出。" if is_skin else "请解读这份检查/检验报告，按系统要求的 JSON 输出。",
                            },
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        ],
                    },
                ],
                task="diagnosis",
                temperature=0.1,
                max_tokens=2200,
                reasoning_effort="high",
                session_id=f"report-{report_id}",
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="报告解读服务暂不可用，请稍后再试") from exc
        parsed = _parse_json_text(result.content) or {}
        summary = str(parsed.get("summary") or result.content or "报告已上传，但模型未返回可读摘要。")
        abnormal = parsed.get("abnormal_values") or []
        questions = parsed.get("questions") or []
        report_title = title.strip() or (("皮肤照片" if is_skin else file.filename) or "检查报告")
        async with state.db.transaction() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO reports (id, user_id, member_id, title, image_path, summary, abnormal_values, questions, raw)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9::jsonb)
                RETURNING *
                """,
                report_id,
                user_id,
                member["id"],
                report_title,
                str(image_path),
                summary,
                json.dumps(abnormal, ensure_ascii=False),
                json.dumps(questions, ensure_ascii=False),
                json.dumps({"kind": kind, "model": result.model, "provider": result.provider, "content": result.content[:2000]}, ensure_ascii=False),
            )
            await _add_event(
                conn,
                user_id,
                member["id"],
                "report",
                f"报告解读 · {report_title}",
                {"report_id": report_id, "summary": summary[:160], "abnormal_count": len(abnormal)},
            )
        committed = True
    finally:
        if not committed:
            try:
                image_path.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "report": _report_json(settings, row) if row else None,
        "model": {"provider": result.provider, "model": result.model},
    }


@router.get("/api/v1/reports")
async def list_reports(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    db = state.db
    user_id = str(user["sub"])
    if member_id:
        await _owned_member(db, member_id, user_id)
        rows = await db.fetch(
            "SELECT id, title, summary, abnormal_values, questions, image_path, created_at FROM reports WHERE user_id=$1 AND member_id=$2 ORDER BY created_at DESC LIMIT 30",
            user_id,
            member_id,
        )
    else:
        rows = await db.fetch(
            "SELECT id, title, summary, abnormal_values, questions, image_path, created_at FROM reports WHERE user_id=$1 ORDER BY created_at DESC LIMIT 30",
            user_id,
        )
    return {"reports": [_report_json(state.settings, row) for row in rows]}


@router.get("/api/v1/reports/{report_id}")
async def get_report(
    report_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    row = await _owned_report(state.db, report_id, str(user["sub"]))
    return {"report": _report_json(state.settings, row)}


@router.get("/api/v1/reports/{report_id}/image")
async def get_report_image(
    report_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    state = _state(request)
    row = await _owned_report(state.db, report_id, str(user["sub"]))
    path = resolve_report_image(state.settings, row.get("image_path"))
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(
        path,
        media_type=report_image_media_type(path),
        headers={"Cache-Control": "private"},
    )


@router.delete("/api/v1/reports/{report_id}")
async def delete_report(
    report_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    row = await _owned_report(state.db, report_id, user_id)
    async with state.db.transaction() as conn:
        await conn.execute("DELETE FROM reports WHERE id=$1 AND user_id=$2", report_id, user_id)
        await conn.execute(
            "DELETE FROM health_events WHERE user_id=$1 AND kind='report' AND detail->>'report_id'=$2",
            user_id,
            report_id,
        )
    file_removed = remove_report_image(state.settings, row.get("image_path"))
    await state.sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=user_id,
        actor=user_id,
        action="report.delete",
        detail={"report_id": report_id},
    )
    return {"ok": True, "file_removed": file_removed}


# -------------------------------------------------------- account export/delete
@router.get("/api/v1/me/export")
async def export_me(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> Response:
    state = _state(request)
    db = state.db
    user_id = str(user["sub"])
    user_row = await db.fetchrow(
        "SELECT id, username, role, enabled, created_at, updated_at FROM users WHERE id=$1",
        user_id,
    )
    if not user_row:
        raise HTTPException(status_code=404, detail="user not found")
    report_rows = await db.fetch("SELECT * FROM reports WHERE user_id=$1 ORDER BY created_at", user_id)
    collections = {
        "family": await db.fetch("SELECT * FROM family_members WHERE user_id=$1 ORDER BY created_at", user_id),
        "medications": await db.fetch("SELECT * FROM medications WHERE user_id=$1 ORDER BY created_at", user_id),
        "reports": report_rows,
        "followups": await db.fetch("SELECT * FROM followups WHERE user_id=$1 ORDER BY created_at", user_id),
        "health_events": await db.fetch("SELECT * FROM health_events WHERE user_id=$1 ORDER BY occurred_at", user_id),
        "bookings": await db.fetch("SELECT * FROM bookings WHERE user_id=$1 ORDER BY created_at", user_id),
        "sessions": await db.fetch("SELECT * FROM sessions WHERE user_id=$1 ORDER BY created_at", user_id),
        "messages": await db.fetch("SELECT * FROM messages WHERE user_id=$1 ORDER BY id", user_id),
        "review_requests": await db.fetch("SELECT * FROM review_requests WHERE user_id=$1 ORDER BY created_at", user_id),
    }
    data = jsonable(
        {
            "exported_at": datetime.now(UTC).isoformat(),
            "user": _row_json(user_row),
            **{key: [_row_json(row) for row in rows] for key, rows in collections.items()},
        }
    )
    for item in data["reports"]:
        item.pop("image_path", None)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("data.json", json.dumps(data, ensure_ascii=False, indent=2))
        for row in report_rows:
            path = resolve_report_image(state.settings, row["image_path"])
            if path is not None and path.is_file():
                bundle.write(path, f"uploads/reports/{path.name}")
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="smart-hospital-data.zip"',
            "Cache-Control": "private",
        },
    )


@router.delete("/api/v1/me")
async def delete_me(
    payload: AccountDeleteRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    db = state.db
    user_id = str(user["sub"])
    row = await db.fetchrow(
        "SELECT id, role, enabled, password_hash FROM users WHERE id=$1",
        user_id,
    )
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    if not await asyncio.to_thread(verify_password, payload.password, row["password_hash"]):
        raise HTTPException(status_code=403, detail="password incorrect")
    if row["role"] == "admin" and row["enabled"]:
        remaining = await db.fetchval(
            "SELECT count(*) FROM users WHERE role='admin' AND enabled=true AND id<>$1",
            user_id,
        )
        if remaining == 0:
            raise HTTPException(status_code=400, detail="cannot delete the last enabled admin")
    tag = "deleted-" + hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]
    async with db.transaction() as conn:
        report_rows = await conn.fetch("SELECT image_path FROM reports WHERE user_id=$1", user_id)
        session_ids = [
            item["id"] for item in await conn.fetch("SELECT id FROM sessions WHERE user_id=$1", user_id)
        ]
        if session_ids:
            await conn.execute("DELETE FROM events WHERE session_id = ANY($1::text[])", session_ids)
            await conn.execute(
                "UPDATE model_calls SET session_id=$2 WHERE session_id = ANY($1::text[])",
                session_ids,
                tag,
            )
        await conn.execute(
            "DELETE FROM messages WHERE user_id=$1 OR session_id = ANY($2::text[])",
            user_id,
            session_ids,
        )
        await conn.execute("DELETE FROM sessions WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM health_events WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM followups WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM medications WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM reports WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM bookings WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM review_requests WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM family_members WHERE user_id=$1", user_id)
        await conn.execute("DELETE FROM users WHERE id=$1", user_id)
        await conn.execute(
            "UPDATE audit_log SET actor=$2, session_id=$2 WHERE actor=$1 OR session_id=$1",
            user_id,
            tag,
        )
        await conn.execute("UPDATE analytics_events SET user_id=$2 WHERE user_id=$1", user_id, tag)
        await conn.execute(
            """
            INSERT INTO audit_log (request_id, session_id, actor, action, detail)
            VALUES ($1, $2, 'anonymous', 'account.delete', $3::jsonb)
            """,
            getattr(request.state, "request_id", uuid.uuid4().hex),
            tag,
            json.dumps({"deleted_user": tag}, ensure_ascii=False),
        )
    files_removed = all([remove_report_image(state.settings, item["image_path"]) for item in report_rows])
    return {"ok": True, "file_removed": files_removed}


# ----------------------------------------------------------------- followup
@router.get("/api/v1/followups")
async def list_followups(
    request: Request,
    status: str | None = Query(None),
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    if member_id:
        await _owned_member(db, member_id, user_id)
    if status:
        rows = await db.fetch(
            """
            SELECT * FROM followups
            WHERE user_id=$1 AND status=$2 AND ($3::text IS NULL OR member_id=$3)
            ORDER BY due_date NULLS LAST, created_at DESC LIMIT 50
            """,
            user_id,
            status,
            member_id,
        )
    else:
        rows = await db.fetch(
            """
            SELECT * FROM followups
            WHERE user_id=$1 AND ($2::text IS NULL OR member_id=$2)
            ORDER BY status, due_date NULLS LAST, created_at DESC LIMIT 50
            """,
            user_id,
            member_id,
        )
    return {"followups": [dict(row) for row in rows]}


@router.post("/api/v1/followups")
async def create_followup(
    payload: FollowupRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, payload.member_id, user_id)
    followup_id = "fu_" + uuid.uuid4().hex[:10]
    row = await db.fetchrow(
        """
        INSERT INTO followups (id, user_id, member_id, title, due_date, notes)
        VALUES ($1,$2,$3,$4,$5,$6)
        RETURNING *
        """,
        followup_id,
        user_id,
        member["id"],
        payload.title,
        payload.due_date,
        payload.notes,
    )
    await _add_event(db, user_id, member["id"], "followup", f"随访 · {payload.title}", {"followup_id": followup_id})
    return {"followup": dict(row) if row else None}


@router.patch("/api/v1/followups/{followup_id}/complete")
async def complete_followup(
    followup_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        "UPDATE followups SET status='done' WHERE id=$1 AND user_id=$2 RETURNING *",
        followup_id,
        str(user["sub"]),
    )
    if not row:
        raise HTTPException(status_code=404, detail="followup not found")
    return {"followup": dict(row)}


# ------------------------------------------------------------------- health
@router.get("/api/v1/health/timeline")
async def health_timeline(
    request: Request,
    member_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    if member_id:
        await _owned_member(db, member_id, user_id)
        rows = await db.fetch(
            "SELECT id, kind, title, detail, occurred_at FROM health_events WHERE user_id=$1 AND member_id=$2 ORDER BY occurred_at DESC LIMIT $3",
            user_id,
            member_id,
            limit,
        )
    else:
        rows = await db.fetch(
            "SELECT id, kind, title, detail, occurred_at FROM health_events WHERE user_id=$1 ORDER BY occurred_at DESC LIMIT $2",
            user_id,
            limit,
        )
    return {
        "timeline": [
            {
                **_row_json(row),
                "occurred_at": row["occurred_at"].isoformat() if row["occurred_at"] else None,
            }
            for row in rows
        ]
    }


@router.get("/api/v1/home/summary")
async def home_summary(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    member = await _member_row(state.db, member_id, user_id)
    sessions = await state.sessions.list_sessions(user_id, limit=1, member_id=member["id"])
    followups = await state.db.fetch(
        "SELECT id, title, due_date, status FROM followups WHERE user_id=$1 AND member_id=$2 AND status='pending' ORDER BY due_date NULLS LAST LIMIT 5",
        user_id,
        member["id"],
    )
    meds = await state.db.fetch(
        "SELECT id, name, dose, frequency FROM medications WHERE user_id=$1 AND member_id=$2 AND active=true ORDER BY created_at DESC LIMIT 6",
        user_id,
        member["id"],
    )
    reports = await state.db.fetch(
        "SELECT id, title, summary, created_at FROM reports WHERE user_id=$1 AND member_id=$2 ORDER BY created_at DESC LIMIT 3",
        user_id,
        member["id"],
    )
    timeline = await state.db.fetch(
        "SELECT id, kind, title, occurred_at FROM health_events WHERE user_id=$1 AND member_id=$2 ORDER BY occurred_at DESC LIMIT 8",
        user_id,
        member["id"],
    )
    return {
        "member": member,
        "latest_session": sessions[0] if sessions else None,
        "followups": [dict(row) for row in followups],
        "medications": [dict(row) for row in meds],
        "reports": [dict(row) for row in reports],
        "timeline": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "title": row["title"],
                "occurred_at": row["occurred_at"].isoformat() if row["occurred_at"] else None,
            }
            for row in timeline
        ],
    }


# ----------------------------------------------------------------- services
@router.get("/api/v1/services/doctors")
async def list_doctors(
    request: Request,
    department: str = Query(...),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    data = await _mcp(_state(request), "list_doctors", {"department": department})
    return data


@router.get("/api/v1/services/visit-guide")
async def visit_guide(
    request: Request,
    order_id: str = Query(...),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    await _owned_order(state.db, order_id, str(user["sub"]))
    data = await _mcp(state, "get_visit_guide", {"order_id": order_id})
    return data


# ---------------------------------------------------------------- analytics
@router.post("/api/v1/analytics/event")
async def analytics_event(
    payload: AnalyticsEventRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    await _state(request).db.execute(
        "INSERT INTO analytics_events (user_id, session_id, name, props) VALUES ($1,$2,$3,$4::jsonb)",
        str(user["sub"]),
        payload.session_id,
        payload.name,
        json.dumps(payload.props, ensure_ascii=False),
    )
    return {"ok": True}


@router.get("/api/v1/admin/analytics")
async def admin_analytics(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT name, count(*) AS count
        FROM analytics_events
        WHERE created_at > now() - interval '30 days'
        GROUP BY name
        ORDER BY count DESC
        LIMIT 50
        """
    )
    return {"events": [dict(row) for row in rows]}


# ------------------------------------------------------------------ health
@router.get("/api/health")
async def api_health(request: Request) -> dict[str, Any]:
    settings = _state(request).settings
    return {
        "ok": True,
        "version": settings.APP_VERSION,
        "time": datetime.now(UTC).isoformat(),
    }


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    try:
        await state.db.fetchval("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    try:
        documents = await state.db.fetchval("SELECT count(*) FROM documents")
    except Exception:
        documents = 0
    return {
        "ok": db_ok,
        "app": state.settings.APP_NAME,
        "version": state.settings.APP_VERSION,
        "db": db_ok,
        "documents": int(documents or 0),
        "models": await state.gateway.health(),
        "mcp": await state.mcp.health(),
    }


@router.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

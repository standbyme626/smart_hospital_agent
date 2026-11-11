from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import uuid
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import asyncpg
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .api.deps import create_token, get_current_user, require_admin, require_clinician, require_ops
from .conversation import append_message, load_history
from .db import SCHEMA_VERSION
from .policy import DEPARTMENT_NAMES, DRUG_CHECKER
from .prompts import build_member_context
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
    ConsentRequest,
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
    id_card_hash,
    id_card_key,
    is_locked,
    login_key,
    pseudonym_tag,
    record_failure,
    redact_detail,
    verify_password,
)
from .session_store import jsonable

router = APIRouter()

logger = logging.getLogger(__name__)


class AdminApproveRequest(BaseModel):
    confirm: bool | None = None
    confirmation: bool | None = None


class AppointmentCreateRequest(BaseModel):
    member_id: str | None = None
    org_id: str
    practitioner_id: str
    service_id: str
    date: str
    idempotency_key: str | None = None


class AppointmentCancelRequest(BaseModel):
    reason: str = "user_cancelled"
    refund: bool = True


class TaskAckRequest(BaseModel):
    status: str = "completed"


def _state(request: Request) -> Any:
    return request.app.state


def _public_user(row: Any) -> dict[str, Any]:
    data = dict(row)
    return {
        key: value
        for key, value in data.items()
        if key in {"id", "username", "role", "enabled", "created_at", "updated_at"}
    }


def _member_public(row: Any) -> dict[str, Any]:
    data = dict(row)
    data.pop("id_card_ciphertext", None)
    data["has_id_card"] = bool(data.get("id_card_last4"))
    return data


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(jsonable(event.get("data", {})), ensure_ascii=False, default=str)
    return f"id: {event.get('seq', 0)}\nevent: {event.get('type', 'message')}\ndata: {data}\n\n"


async def _mcp(state: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """调用医院系统 MCP。

    错误一律**脱敏**后返回：只给错误码 + 中文提示 + 是否可重试，不回显上游原文
    （上游报错可能带 SQL、路径、凭据片段）。
    """
    try:
        result = await state.mcp.call_tool(name, arguments)
    except (TimeoutError, asyncio.TimeoutError):
        raise HTTPException(status_code=504, detail="HIS_TIMEOUT：医院系统响应超时，可重试") from None
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="HIS_UNAVAILABLE：医院系统暂时不可用，可重试") from None
    if result.get("is_error"):
        raise HTTPException(status_code=502, detail="HIS_ERROR：医院系统返回错误，可重试") from None
    return result.get("data") or {}




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
    flags = data.get("critical_flags") or []
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except json.JSONDecodeError:
            flags = []
    data["critical_flags"] = flags if isinstance(flags, list) else []
    if data["critical_flags"]:
        data["urgency"] = "critical"
    else:
        data.setdefault("urgency", "normal")
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


async def _hospital_patient_id(db: Any, member: dict[str, Any], fallback: str) -> str:
    hospital_patient_id = member.get("hospital_patient_id")
    if hospital_patient_id:
        return str(hospital_patient_id)
    member_id = member.get("id")
    if not member_id:
        return fallback
    hospital_patient_id = f"HP-{member_id}"
    await db.execute(
        "UPDATE family_members SET hospital_patient_id=$2 WHERE id=$1",
        member_id,
        hospital_patient_id,
    )
    return hospital_patient_id


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


async def _notify(
    db: Any,
    user_id: str,
    kind: str,
    title: str,
    body: str | None = None,
    link: str | None = None,
) -> None:
    """站内通知。短信/微信通道接入后，由通知中心在此处扇出。"""
    await db.execute(
        """
        INSERT INTO notifications (user_id, kind, title, body, link)
        VALUES ($1, $2, $3, $4, $5)
        """,
        user_id,
        kind,
        title,
        body,
        link,
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
async def _audit_login(db: Any, request: Request, username: str, outcome: str, role: str | None) -> None:
    """登录留痕：只记结果、角色与来源 IP，绝不记密码。"""
    try:
        from .security_hardening import client_ip as _client_ip

        settings = _state(request).settings
        ip = _client_ip(request, bool(getattr(settings, "TRUST_PROXY_HEADERS", True)))
        await db.execute(
            """
            INSERT INTO audit_log (request_id, session_id, actor, action, detail)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            getattr(request.state, "request_id", None),
            None,
            username,
            "auth.login",
            json.dumps({"outcome": outcome, "role": role, "ip": ip}, ensure_ascii=False),
        )
    except Exception:
        return


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
            """
            SELECT id, username, role, enabled, password_hash, token_version
            FROM users WHERE username=$1
            """,
            payload.username,
        )
    except (OSError, asyncpg.PostgresError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    encoded = row["password_hash"] if row else DUMMY_PASSWORD_HASH
    valid = await asyncio.to_thread(verify_password, payload.password, encoded)
    if not row or not valid:
        await _audit_login(state.db, request, payload.username, "failed", None)
        if record_failure(key):
            raise HTTPException(status_code=423, detail="account locked, try again later")
        raise HTTPException(status_code=401, detail="invalid credentials")
    if not row["enabled"]:
        await _audit_login(state.db, request, payload.username, "disabled", row["role"])
        raise HTTPException(status_code=403, detail="user disabled")
    clear_failures(key)
    await _audit_login(state.db, request, payload.username, "success", row["role"])
    return {
        "access_token": create_token(
            state.settings, row["id"], row["role"], row["token_version"]
        ),
        "token_type": "bearer",
        "role": row["role"],
        "username": row["username"],
    }


@router.post("/api/v1/consent")
async def record_consent(
    payload: ConsentRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    user_id = str(user["sub"])
    consent_id = "consent_" + uuid.uuid4().hex[:12]
    await state.db.execute(
        """
        INSERT INTO consent_record
            (id, user_id, subject_member_id, consent_type, version, granted_by, granted_at, evidence)
        VALUES ($1,$2,$3,$4,$5,$2,now(),$6::jsonb)
        """,
        consent_id,
        user_id,
        payload.subject_member_id,
        payload.consent_type,
        payload.version,
        json.dumps({"source": "web", "request_id": getattr(request.state, "request_id", "")}),
    )
    if payload.consent_type in {"privacy_terms", "privacy", "terms"}:
        await state.db.execute(
            "UPDATE users SET consent_version=$2, consent_at=now(), updated_at=now() WHERE id=$1",
            user_id,
            payload.version,
        )
    return {"ok": True, "consent_id": consent_id, "version": payload.version}


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
                detail=f"今日问诊次数已达上限（{daily_limit} 次/天），请明天再试或联系管理员",
            )
    member = await _member_row(state.db, payload.member_id, user_id)
    member_context = build_member_context(member)
    message = payload.message

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
                title=(payload.title or payload.message)[:80],
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
                member_context=member_context,
            )
            async for event in state.agent.stream(
                session_id=session_id,
                message=message,
                user_id=user_id,
                request_id=request_id,
                title=payload.title or payload.message,
                member_id=member.get("id"),
                history=history,
                member_context=member_context,
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
                                "session_id": session_id,
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
    user_id = str(user["sub"])
    member = await _member_row(state.db, payload.member_id, user_id)
    patient_id = await _hospital_patient_id(state.db, member, user_id)
    data = await _mcp(
        state,
        "lock_slot",
        {"slot_id": payload.slot_id, "patient_id": patient_id},
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
    hospital_patient_id = await _hospital_patient_id(state.db, member, user_id)
    idempotency_key = f"order:{user_id}:{payload.lock_id}"
    result = await state.mcp.call_tool(
        "create_order",
        {"lock_id": payload.lock_id, "patient_id": hospital_patient_id},
        idempotency_key=idempotency_key,
    )
    if result.get("is_error"):
        raise HTTPException(status_code=502, detail=result["text"])
    data = result.get("data") or {}

    # 幂等重放：同一 lock 重复下单只保留第一条投影（HIS 侧靠 idempotency_key 去重，
    # 平台侧靠唯一索引 + ON CONFLICT，避免并发/重放时 500）。
    inserted = await state.db.fetchrow(
        """
        INSERT INTO bookings
            (id, session_id, patient_id, user_id, department, slot_id, order_id, status,
             payment_status, amount, idempotency_key, member_id, hospital_patient_id, detail)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
        ON CONFLICT (idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
        RETURNING id
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
    if inserted is None:
        existing = await state.db.fetchrow(
            "SELECT order_id, slot_id, status, payment_status, amount, detail FROM bookings WHERE idempotency_key=$1",
            idempotency_key,
        )
        if existing is not None:
            stored = _row_json(dict(existing)).get("detail") or {}
            if not isinstance(stored, dict):
                stored = {}
            return {
                "order": {
                    **stored,
                    "order_id": existing["order_id"],
                    "slot_id": existing["slot_id"],
                    "status": existing["status"],
                    "payment_status": existing["payment_status"],
                    "amount": float(existing["amount"] or 0),
                },
                "replayed": True,
            }
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


# ------------------------------------------------------- HIS appointments
@router.get("/api/v1/booking/config")
async def booking_config(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return {
        "payment_mode": _state(request).settings.HIS_PAYMENT_MODE,
        "adapter_url": _state(request).settings.HOSPITAL_MCP_URL,
    }


@router.get("/api/v1/booking/catalog")
async def booking_catalog(
    request: Request,
    department: str | None = Query(None),
    org_id: str | None = Query(None),
    date: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.catalog(department=department, org_id=org_id, date=date)


@router.post("/api/v1/booking/appointments")
async def create_appointment(
    payload: AppointmentCreateRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.create_appointment(
        user_id=str(user["sub"]),
        member_id=payload.member_id,
        org_id=payload.org_id,
        practitioner_id=payload.practitioner_id,
        service_id=payload.service_id,
        date=payload.date,
        idempotency_key=payload.idempotency_key,
    )


@router.get("/api/v1/booking/appointments")
async def list_appointments(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    items = await _state(request).booking.list_appointments(str(user["sub"]), member_id)
    return {"appointments": items}


@router.get("/api/v1/booking/appointments/{appointment_id}")
async def get_appointment(
    appointment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.get_appointment(appointment_id, str(user["sub"]))


@router.post("/api/v1/booking/appointments/{appointment_id}/request-payment")
async def request_appointment_payment(
    appointment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.request_payment(appointment_id, str(user["sub"]))


@router.post("/api/v1/booking/appointments/{appointment_id}/cancel")
async def cancel_appointment(
    appointment_id: str,
    payload: AppointmentCancelRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.cancel_appointment(
        appointment_id, str(user["sub"]), reason=payload.reason, refund=payload.refund
    )


@router.get("/api/v1/booking/appointments/{appointment_id}/prescriptions")
async def appointment_prescriptions(
    appointment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    return await _state(request).booking.prescriptions(appointment_id, str(user["sub"]))


@router.get("/api/v1/booking/appointments/{appointment_id}/guide")
async def appointment_guide(
    appointment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    result = await state.booking.get_appointment(appointment_id, str(user["sub"]))
    visit = result.get("visit") or {}
    guide_text = str(visit.get("guide") or "")
    return {
        "guide": {
            "hospital": "医院",
            "address": "",
            "floor": "",
            "checkin": guide_text or "请以医院现场指引为准。",
            "items": ["身份证/医保卡", "既往病历", "正在服用的药物清单"],
            "support": "如需帮助，请在问诊页继续提问或联系医院服务台。",
            "source": "his" if not visit.get("stale") else "projection",
            "stale": bool(visit.get("stale")),
        },
        "appointment": result.get("appointment"),
    }


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
    await _state(request).sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=str(user["sub"]),
        actor=str(user["sub"]),
        action="admin.user.create",
        detail={"target_user": row["id"], "role": row["role"], "enabled": row["enabled"]},
    )
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
        if payload.password is not None or payload.enabled is False:
            sets.append("token_version = token_version + 1")
        row = await conn.fetchrow(
            f"UPDATE users SET {', '.join(sets)} WHERE id = $1 "
            "RETURNING id, username, role, enabled, created_at, updated_at",
            *args,
        )
    changed = []
    if payload.enabled is not None:
        changed.append("enabled")
    if payload.role is not None:
        changed.append("role")
    if payload.password is not None:
        changed.append("password")
    await _state(request).sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=str(user["sub"]),
        actor=str(user["sub"]),
        action="admin.user.password_reset"
        if payload.password is not None
        else "admin.user.update",
        detail={"target_user": user_id, "changed": changed},
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
    payload: AdminApproveRequest | None = None,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    confirmed = bool(payload and (payload.confirm is True or payload.confirmation is True))
    if state.settings.production and not confirmed:
        raise HTTPException(
            status_code=400,
            detail="confirmation required: pass confirm=true",
        )
    result = await approve_review(
        state.db,
        state.mcp,
        review_id=review_id,
        admin_id=str(user["sub"]),
        confirm_token=state.settings.HITL_CONFIRM_TOKEN,
    )
    order = result.get("order") or {}
    await state.sessions.audit(
        request_id=getattr(request.state, "request_id", uuid.uuid4().hex),
        session_id=str(user["sub"]),
        actor=str(user["sub"]),
        action="payment.approve_confirmed",
        detail={
            "review_id": review_id,
            "order_id": order.get("order_id") or order.get("id"),
            "amount": float(order["amount"]) if order.get("amount") is not None else None,
            "confirmed": confirmed,
        },
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
        SELECT provider, model, task, user_id, count(*) AS calls,
               sum(prompt_tokens) AS prompt_tokens,
               sum(completion_tokens) AS completion_tokens, sum(cost) AS cost,
               round(avg(latency_ms)) AS avg_latency_ms
        FROM model_calls
        GROUP BY provider, model, task, user_id
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
    audit: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["detail"] = redact_detail(_row_json(item).get("detail"))
        audit.append(item)
    return {"audit": audit}


# ------------------------------------------------------ admin payment tasks
@router.get("/api/v1/admin/payment-tasks")
async def admin_payment_tasks(
    request: Request,
    status: str | None = Query("requested"),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    tasks = await _state(request).booking.list_payment_tasks(status)
    return {"tasks": [jsonable(task) for task in tasks]}


@router.post("/api/v1/admin/payment-tasks/{task_id}/ack")
async def admin_ack_payment_task(
    task_id: str,
    payload: TaskAckRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    task = await _state(request).booking.ack_payment_task(task_id, payload.status)
    return {"task": jsonable(task)}


@router.post("/api/v1/admin/appointments/{appointment_id}/simulate-payment")
async def admin_simulate_payment(
    appointment_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return {"payment": await _state(request).booking.simulate_payment(appointment_id, str(user["sub"]))}


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
    return {"members": [_member_public(row) for row in rows]}


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
    id_card = (payload.id_card or "").strip() or None
    key = id_card_key(_state(request).settings)

    # 同一账号下同一身份证只保留一份档案（剧本 P-01）：重复提交直接复用，不新增。
    if id_card:
        reused = await db.fetchrow(
            """
            SELECT * FROM family_members
            WHERE user_id=$1 AND id_card_hash=$2
            LIMIT 1
            """,
            user_id,
            id_card_hash(_state(request).settings, id_card),
        )
        if reused is not None:
            return {"member": _member_public(reused), "reused": True}

    row = await db.fetchrow(
        """
        INSERT INTO family_members
            (id, user_id, name, relation, gender, birth_date, phone, allergies, conditions,
             is_self, hospital_patient_id, id_card_ciphertext, id_card_hash, id_card_last4)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,
                CASE WHEN $12::text IS NULL THEN NULL
                     ELSE encode(pgp_sym_encrypt($12::text, $13::text), 'base64') END,
                $14, $15)
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
        id_card,
        key,
        id_card_hash(_state(request).settings, id_card) if id_card else None,
        id_card[-4:] if id_card else None,
    )
    return {"member": _member_public(row) if row else None}


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
    id_card = (payload.id_card or "").strip() or None
    key = id_card_key(_state(request).settings)
    row = await db.fetchrow(
        """
        UPDATE family_members SET
            name=$3, relation=$4, gender=$5, birth_date=$6, phone=$7,
            allergies=$8, conditions=$9, is_self=$10,
            id_card_ciphertext = CASE WHEN $11::text IS NULL THEN id_card_ciphertext
                ELSE encode(pgp_sym_encrypt($11::text, $12::text), 'base64') END,
            id_card_hash = CASE WHEN $11::text IS NULL THEN id_card_hash
                ELSE $13 END,
            id_card_last4 = CASE WHEN $11::text IS NULL THEN id_card_last4
                ELSE $14 END,
            updated_at=now()
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
        id_card,
        key,
        id_card_hash(_state(request).settings, id_card) if id_card else None,
        id_card[-4:] if id_card else None,
    )
    return {"member": _member_public(row) if row else None}


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
                """
                UPDATE review_requests
                SET status='rejected',
                    decision_note=COALESCE(decision_note, '成员已删除'),
                    decided_at=COALESCE(decided_at, now()),
                    updated_at=now()
                WHERE kind='payment'
                  AND status IN ('pending', 'processing')
                  AND (
                      member_id=$2
                      OR ref_id IN (SELECT id FROM bookings WHERE user_id=$1 AND member_id=$2)
                  )
                """,
                user_id,
                member_id,
            )
            await conn.execute(
                """
                UPDATE bookings
                SET status='cancelled', updated_at=now()
                WHERE user_id=$1 AND member_id=$2
                  AND payment_status <> 'paid'
                  AND status NOT IN ('booked', 'paid', 'cancelled')
                """,
                user_id,
                member_id,
            )
            await conn.execute(
                """
                UPDATE analytics_events
                SET props = props - 'member_id' - 'symptoms' - 'name' - 'phone'
                WHERE user_id=$1
                  AND (props->>'member_id'=$2 OR session_id = ANY($3::text[]))
                """,
                user_id,
                member_id,
                session_ids,
            )
            await conn.execute(
                """
                UPDATE audit_log
                SET detail = detail - 'member_id' - 'report' - 'triage' - 'safety'
                    - 'symptoms' - 'name' - 'phone'
                WHERE (actor=$1 OR session_id = ANY($2::text[]))
                  AND (detail->>'member_id'=$3 OR session_id = ANY($2::text[]))
                """,
                user_id,
                session_ids,
                member_id,
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
        detail={"member_tag": pseudonym_tag(state.settings, member_id)},
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
_CRITICAL_RULES = (
    (("血钾", "钾", "potassium", "k+"), 6.0, 2.5),
    (("血钠", "钠", "sodium", "na+"), 160.0, 120.0),
    (("血糖", "葡萄糖", "glucose"), 30.0, 2.5),
    (("血红蛋白", "hb", "hemoglobin"), 200.0, 50.0),
    (("血小板", "platelet", "plt"), 1000.0, 20.0),
    (("肌钙蛋白", "troponin", "tni", "tnt"), 1.0, -1.0),
    (("肌酐", "creatinine"), 700.0, -1.0),
)


def _critical_flags(abnormal: list[dict[str, Any]], parsed: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for item in parsed.get("critical_flags") or []:
        text = str(item).strip()
        if text:
            flags.append(text)
    for item in abnormal:
        name = str(item.get("name") or "").lower()
        value_text = str(item.get("value") or "")
        match = re.search(r"-?\d+(?:\.\d+)?", value_text)
        if not match:
            continue
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        for aliases, high, low in _CRITICAL_RULES:
            if any(alias in name for alias in aliases):
                # Hemoglobin is commonly reported in g/dL (13.5) or g/L (135);
                # use the matching threshold instead of treating 13.5 as g/L.
                if any(alias in name for alias in ("血红蛋白", "hemoglobin", "hb")):
                    if value <= 30:
                        high, low = 20.0, 5.0
                    else:
                        high, low = 200.0, 50.0
                if high >= 0 and value >= high:
                    flags.append(f"{item.get('name')} {value_text} 达到危急值上限")
                elif low >= 0 and value <= low:
                    flags.append(f"{item.get('name')} {value_text} 达到危急值下限")
                break
    # dedupe preserving order
    seen: set[str] = set()
    result: list[str] = []
    for item in flags:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


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
                        "role": "system",
                        "content": (
                            "患者背景（仅用于选择合适的参考范围和表述，不要据此诊断）："
                            + (build_member_context(member) or "未提供")
                        ),
                    },
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
        critical_flags = _critical_flags(abnormal if isinstance(abnormal, list) else [], parsed)
        report_title = title.strip() or (("皮肤照片" if is_skin else file.filename) or "检查报告")
        async with state.db.transaction() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO reports
                    (id, user_id, member_id, title, image_path, summary, abnormal_values,
                     questions, raw, critical_flags, source)
                VALUES ($1,$2,$3,$4,$5,$6,$7::jsonb,$8::jsonb,$9::jsonb,$10::jsonb,'upload')
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
                json.dumps(critical_flags, ensure_ascii=False),
            )
            await _add_event(
                conn,
                user_id,
                member["id"],
                "report",
                f"报告解读 · {report_title}",
                {
                    "report_id": report_id,
                    "summary": summary[:160],
                    "abnormal_count": len(abnormal),
                    "critical_flags": critical_flags,
                    "urgency": "critical" if critical_flags else "normal",
                },
            )
            await _notify(
                conn,
                user_id,
                "report",
                "检查报告已解读" + (" · 危急值" if critical_flags else ""),
                report_title,
                "/reports/",
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
            "SELECT id, title, summary, abnormal_values, questions, image_path, critical_flags, created_at FROM reports WHERE user_id=$1 AND member_id=$2 ORDER BY created_at DESC LIMIT 30",
            user_id,
            member_id,
        )
    else:
        rows = await db.fetch(
            "SELECT id, title, summary, abnormal_values, questions, image_path, critical_flags, created_at FROM reports WHERE user_id=$1 ORDER BY created_at DESC LIMIT 30",
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
        "audit": await db.fetch(
            """
            SELECT request_id, session_id, actor, action, detail, created_at
            FROM audit_log
            WHERE actor=$1 OR session_id IN (SELECT id FROM sessions WHERE user_id=$1)
            ORDER BY id
            """,
            user_id,
        ),
        "analytics": await db.fetch(
            "SELECT session_id, name, props, created_at FROM analytics_events WHERE user_id=$1 ORDER BY id",
            user_id,
        ),
        "model_calls": await db.fetch(
            """
            SELECT provider, model, task, prompt_tokens, completion_tokens, cost,
                   latency_ms, created_at
            FROM model_calls WHERE user_id=$1 ORDER BY id
            """,
            user_id,
        ),
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
    tag = pseudonym_tag(state.settings, user_id)
    async with db.transaction() as conn:
        report_rows = await conn.fetch("SELECT image_path FROM reports WHERE user_id=$1", user_id)
        session_ids = [
            item["id"] for item in await conn.fetch("SELECT id FROM sessions WHERE user_id=$1", user_id)
        ]
        await conn.execute(
            """
            UPDATE model_calls
            SET user_id = NULL, session_id = $2
            WHERE user_id=$1 OR session_id = ANY($3::text[])
            """,
            user_id,
            tag,
            session_ids,
        )
        if session_ids:
            await conn.execute("DELETE FROM events WHERE session_id = ANY($1::text[])", session_ids)
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
            """
            UPDATE audit_log
            SET actor=$2, session_id=$2,
                detail = detail - 'report' - 'triage' - 'safety' - 'symptoms'
                    - 'name' - 'phone' - 'member_context'
            WHERE actor=$1 OR session_id=$1 OR session_id = ANY($3::text[])
            """,
            user_id,
            tag,
            session_ids,
        )
        await conn.execute(
            """
            UPDATE analytics_events
            SET user_id=$2, props='{}'::jsonb
            WHERE user_id=$1 OR session_id = ANY($3::text[])
            """,
            user_id,
            tag,
            session_ids,
        )
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
    await _notify(db, user_id, "followup", "随访已安排", payload.title, "/followups/")
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


# -------------------------------------------------------------- notification
@router.get("/api/v1/notifications")
async def list_notifications(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    rows = await db.fetch(
        """
        SELECT id, kind, title, body, link, read_at, created_at
        FROM notifications
        WHERE user_id=$1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        user_id,
        limit,
    )
    unread = await db.fetchval(
        "SELECT count(*) FROM notifications WHERE user_id=$1 AND read_at IS NULL",
        user_id,
    )
    return {"notifications": [dict(row) for row in rows], "unread": int(unread or 0)}


@router.post("/api/v1/notifications/{notification_id}/read")
async def read_notification(
    notification_id: int,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    await db.execute(
        "UPDATE notifications SET read_at=now() WHERE id=$1 AND user_id=$2 AND read_at IS NULL",
        notification_id,
        str(user["sub"]),
    )
    return {"ok": True}


@router.post("/api/v1/notifications/read-all")
async def read_all_notifications(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    await _state(request).db.execute(
        "UPDATE notifications SET read_at=now() WHERE user_id=$1 AND read_at IS NULL",
        str(user["sub"]),
    )
    return {"ok": True}


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


@router.get("/api/ready")
async def api_ready(request: Request) -> JSONResponse:
    state = _state(request)
    version: str | None = None
    db_ok = False
    try:
        await state.db.fetchval("SELECT 1")
        version = await state.db.fetchval("SELECT max(version) FROM schema_migrations")
        db_ok = version == SCHEMA_VERSION
    except Exception as exc:
        logger.warning("readiness DB check failed: %s", type(exc).__name__)
    body = {
        "ok": db_ok,
        "db": db_ok,
        "schema_version": version,
        "expected_schema_version": SCHEMA_VERSION,
        "adapters": "skipped",
    }
    return JSONResponse(status_code=200 if db_ok else 503, content=body)


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


# ----------------------------------------------------------------- clinician
# 医生端只读（S8a）：先复用本地投影与档案，HIS 写操作留给 S9。
@router.get("/api/v1/clinician/queue")
async def clinician_queue(
    request: Request,
    date: str | None = Query(None),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    day = date or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    rows = await db.fetch(
        """
        SELECT p.id, p.member_id, p.department, p.doctor_name, p.appointment_date,
               p.status, p.payment_status, p.his_encounter_id, p.his_status,
               f.name AS member_name, f.gender, f.birth_date
        FROM appointment_projection p
        LEFT JOIN family_members f ON f.id = p.member_id
        WHERE p.appointment_date = $1
        ORDER BY p.created_at ASC
        """,
        day,
    )
    return {"date": day, "items": [dict(row) for row in rows]}


@router.get("/api/v1/clinician/patients/{member_id}")
async def clinician_patient(
    member_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    member = await db.fetchrow("SELECT * FROM family_members WHERE id=$1", member_id)
    if not member:
        raise HTTPException(status_code=404, detail="member not found")
    reports = await db.fetch(
        """
        SELECT id, title, summary, created_at, critical_flags
        FROM reports WHERE member_id=$1 ORDER BY created_at DESC LIMIT 10
        """,
        member_id,
    )
    medications = await db.fetch(
        """
        SELECT id, name, dose, frequency, active
        FROM medications WHERE member_id=$1 ORDER BY created_at DESC LIMIT 30
        """,
        member_id,
    )
    followups = await db.fetch(
        """
        SELECT id, title, due_date, status
        FROM followups WHERE member_id=$1
        ORDER BY status, due_date NULLS LAST LIMIT 20
        """,
        member_id,
    )
    appointments = await db.fetch(
        """
        SELECT id, department, doctor_name, appointment_date, status, payment_status
        FROM appointment_projection WHERE member_id=$1
        ORDER BY created_at DESC LIMIT 10
        """,
        member_id,
    )
    previsit = await db.fetchrow(
        """
        SELECT id, chief_complaint, duration, medications, allergies, history, questions, created_at
        FROM previsit_forms WHERE member_id=$1 ORDER BY created_at DESC LIMIT 1
        """,
        member_id,
    )
    return {
        "member": {
            "id": member["id"],
            "name": member["name"],
            "relation": member["relation"],
            "gender": member["gender"],
            "birth_date": member["birth_date"],
            "allergies": member["allergies"],
            "conditions": member["conditions"],
        },
        "reports": [dict(row) for row in reports],
        "medications": [dict(row) for row in medications],
        "followups": [dict(row) for row in followups],
        "appointments": [dict(row) for row in appointments],
        "previsit": dict(previsit) if previsit else None,
    }


class ClinicianDraftRequest(BaseModel):
    member_id: str
    appointment_id: str | None = None
    content: str


class ClinicianDraftUpdateRequest(BaseModel):
    content: str


@router.get("/api/v1/clinician/config")
async def clinician_config(
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    settings = _state(request).settings
    return {
        "write_enabled": bool(settings.CLINICIAN_WRITE_ENABLED),
        "note": "写操作需要医生 HIS 账号与签名透传；未启用时只能保存草稿",
    }


@router.get("/api/v1/clinician/drafts")
async def list_clinician_drafts(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    if member_id:
        rows = await db.fetch(
            """
            SELECT id, user_id, member_id, appointment_id, content, status,
                   signed_by, signed_at, created_at, updated_at
            FROM clinician_drafts WHERE member_id=$1
            ORDER BY created_at DESC LIMIT 50
            """,
            member_id,
        )
    else:
        rows = await db.fetch(
            """
            SELECT id, user_id, member_id, appointment_id, content, status,
                   signed_by, signed_at, created_at, updated_at
            FROM clinician_drafts ORDER BY created_at DESC LIMIT 50
            """
        )
    return {"drafts": [dict(row) for row in rows]}


@router.post("/api/v1/clinician/drafts")
async def create_clinician_draft(
    payload: ClinicianDraftRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    member = await db.fetchrow("SELECT id FROM family_members WHERE id=$1", payload.member_id)
    if not member:
        raise HTTPException(status_code=404, detail="member not found")
    draft_id = "cd_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO clinician_drafts (id, user_id, member_id, appointment_id, content)
        VALUES ($1,$2,$3,$4,$5)
        RETURNING id, user_id, member_id, appointment_id, content, status,
                  signed_by, signed_at, created_at, updated_at
        """,
        draft_id,
        str(user["sub"]),
        payload.member_id,
        payload.appointment_id,
        payload.content,
    )
    return {"draft": dict(row) if row else None}


@router.patch("/api/v1/clinician/drafts/{draft_id}")
async def update_clinician_draft(
    draft_id: str,
    payload: ClinicianDraftUpdateRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    row = await db.fetchrow(
        """
        UPDATE clinician_drafts SET content=$2, updated_at=now()
        WHERE id=$1 AND status='draft'
        RETURNING id, user_id, member_id, appointment_id, content, status,
                  signed_by, signed_at, created_at, updated_at
        """,
        draft_id,
        payload.content,
    )
    if not row:
        raise HTTPException(status_code=404, detail="draft not found or already signed")
    return {"draft": dict(row)}


@router.post("/api/v1/clinician/drafts/{draft_id}/sign")
async def sign_clinician_draft(
    draft_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    settings = _state(request).settings
    if not settings.CLINICIAN_WRITE_ENABLED:
        raise HTTPException(
            status_code=409,
            detail="HIS 写入尚未启用：需要医生 HIS 账号与签名透传，当前只能保存草稿",
        )
    db = _state(request).db
    row = await db.fetchrow(
        """
        UPDATE clinician_drafts
        SET status='signed', signed_by=$2, signed_at=now(), updated_at=now()
        WHERE id=$1 AND status='draft'
        RETURNING id, user_id, member_id, appointment_id, content, status,
                  signed_by, signed_at, created_at, updated_at
        """,
        draft_id,
        str(user.get("user") or user.get("sub")),
    )
    if not row:
        raise HTTPException(status_code=404, detail="draft not found or already signed")
    return {"draft": dict(row)}


# ---------------------------------------------------------------------- ops
@router.get("/api/v1/ops/summary")
async def ops_summary(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    pending_reviews = await db.fetchval("SELECT count(*) FROM review_requests WHERE status='pending'")
    open_findings = await db.fetchval("SELECT count(*) FROM reconciliation_finding WHERE status='open'")
    handoff_waiting = await db.fetchval(
        "SELECT count(*) FROM human_task WHERE kind='handoff' AND status IN ('requested','claimed')"
    )
    paid_today = await db.fetchval(
        """
        SELECT count(*) FROM appointment_projection
        WHERE payment_status='paid'
          AND created_at::date = (now() AT TIME ZONE 'Asia/Shanghai')::date
        """
    )
    return {
        "pending_reviews": int(pending_reviews or 0),
        "open_findings": int(open_findings or 0),
        "handoff_waiting": int(handoff_waiting or 0),
        "paid_today": int(paid_today or 0),
    }


@router.get("/api/v1/ops/reconciliation")
async def ops_reconciliation(
    request: Request,
    status: str | None = Query(None),
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, kind, severity, local_ref, his_ref, detail, status, detected_at, resolved_at
        FROM reconciliation_finding
        WHERE ($1::text IS NULL OR status=$1)
        ORDER BY detected_at DESC LIMIT 100
        """,
        status,
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/ops/reconciliation/{finding_id}/resolve")
async def resolve_reconciliation(
    finding_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    row = await db.fetchrow(
        """
        UPDATE reconciliation_finding SET status='resolved', resolved_at=now()
        WHERE id=$1 AND status='open'
        RETURNING id, kind, severity, status, resolved_at
        """,
        finding_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="finding not found or already resolved")
    return {"finding": dict(row)}


@router.get("/api/v1/ops/handoff")
async def ops_handoff(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, kind, ref_id, user_id, member_id, status, payload, assignee, sla_due_at, created_at
        FROM human_task
        WHERE kind='handoff'
        ORDER BY created_at DESC LIMIT 50
        """
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/ops/handoff/{task_id}/claim")
async def claim_handoff(
    task_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    row = await db.fetchrow(
        """
        UPDATE human_task SET status='claimed', assignee=$2, updated_at=now()
        WHERE id=$1 AND kind='handoff' AND status='requested'
        RETURNING id, status, assignee, updated_at
        """,
        task_id,
        str(user.get("user") or user.get("sub")),
    )
    if not row:
        raise HTTPException(status_code=404, detail="handoff task not found or already claimed")
    return {"task": dict(row)}


# ------------------------------------------------------------------ platform
CAPABILITY_MATRIX = [
    {"name": "医保结算 / 电子处方", "owner": "HIS", "note": "yb_* 表与医保接口，平台只读状态与展示"},
    {"name": "发药 / 取药进度", "owner": "HIS", "note": "pharmacy 配药与发药状态"},
    {"name": "患者主索引 / 就诊 / 处方", "owner": "HIS", "note": "HIS 为唯一事实源"},
    {"name": "多租户 / 多院区", "owner": "HIS", "note": "HIS 原生 tenant；平台做路由与隔离"},
    {"name": "三方支付回调", "owner": "HIS", "note": "支付成功以回调驱动，不靠人工"},
    {"name": "药师审方", "owner": "HIS", "note": "平台只读审方结果"},
    {"name": "通知（站内/短信/微信）", "owner": "平台自建", "note": "通知表 + 已读状态；短信/微信留适配位"},
    {"name": "随访与依从", "owner": "平台自建", "note": "随访计划与完成记录"},
    {"name": "移动端 / 无障碍", "owner": "平台自建", "note": "患者端 PWA、字号与对比度"},
    {"name": "运营分析", "owner": "平台自建", "note": "漏斗、留存、成本（S12）"},
    {"name": "商保直付", "owner": "平台自建", "note": "需保司接口（未开始）"},
    {"name": "医生排班 / 停诊", "owner": "待确认", "note": "HIS 只有当日号源，需院方确认归属"},
]


@router.get("/api/v1/admin/platform")
async def admin_platform(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = _state(request).db
    settings = _state(request).settings
    queries = {
        "users": "SELECT count(*) FROM users",
        "sessions": "SELECT count(*) FROM sessions",
        "reports": "SELECT count(*) FROM reports",
        "followups": "SELECT count(*) FROM followups",
        "notifications": "SELECT count(*) FROM notifications",
        "documents": "SELECT count(*) FROM documents",
        "pending_reviews": "SELECT count(*) FROM review_requests WHERE status='pending'",
        "open_findings": "SELECT count(*) FROM reconciliation_finding WHERE status='open'",
    }
    counts: dict[str, int] = {}
    for key, sql in queries.items():
        try:
            counts[key] = int(await db.fetchval(sql) or 0)
        except Exception:
            counts[key] = 0
    return {
        "app": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "env": settings.ENV,
            "schema_version": SCHEMA_VERSION,
        },
        "flags": {
            "auth_enabled": bool(settings.AUTH_ENABLED),
            "clinician_write_enabled": bool(settings.CLINICIAN_WRITE_ENABLED),
            "rate_limit_per_minute": int(settings.RATE_LIMIT_PER_MINUTE),
            "chat_daily_limit": int(settings.CHAT_DAILY_LIMIT),
        },
        "counts": counts,
        "capabilities": CAPABILITY_MATRIX,
    }


@router.get("/api/v1/admin/compliance")
async def admin_compliance(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = _state(request).db
    queries = {
        "users": "SELECT count(*) FROM users",
        "consents": "SELECT count(*) FROM consent_record",
        "audit_events": "SELECT count(*) FROM audit_log",
        "reports": "SELECT count(*) FROM reports",
        "notifications": "SELECT count(*) FROM notifications",
    }
    counts: dict[str, int] = {}
    for key, sql in queries.items():
        try:
            counts[key] = int(await db.fetchval(sql) or 0)
        except Exception:
            counts[key] = 0
    return {
        "retention": {
            "sessions_days": 180,
            "audit_days": 180,
            "notifications_days": 90,
            "reports": "按医疗记录保留规则，账号注销后脱敏保留",
        },
        "redaction": [
            "审计详情脱敏：不落完整报告正文",
            "管理端默认不展示患者明细，回放需额外授权",
            "应用日志不记录模型原文、身份证与手机号",
            "匿名化使用带盐 HMAC（不可反推用户名）",
        ],
        "endpoints": {
            "export": "/api/v1/me/export",
            "delete": "/api/v1/me",
            "consent": "/api/v1/consent",
            "audit": "/api/v1/admin/audit",
        },
        "counts": counts,
    }


@router.get("/api/v1/admin/health-summary")
async def admin_health_summary(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    db = state.db
    settings = state.settings
    checks: list[dict[str, Any]] = []

    db_ok = False
    schema_version: str | None = None
    try:
        schema_version = await db.fetchval("SELECT max(version) FROM schema_migrations")
        db_ok = schema_version == SCHEMA_VERSION
    except Exception:
        db_ok = False
    checks.append(
        {
            "name": "数据库与 Schema",
            "status": "ok" if db_ok else "fail",
            "detail": f"schema={schema_version or 'unknown'} / 期望={SCHEMA_VERSION}",
        }
    )

    try:
        open_findings = int(await db.fetchval("SELECT count(*) FROM reconciliation_finding WHERE status='open'") or 0)
    except Exception:
        open_findings = 0
    checks.append(
        {
            "name": "对账差异",
            "status": "ok" if open_findings == 0 else "warn",
            "detail": f"{open_findings} 条未处理",
        }
    )

    try:
        pending_reviews = int(await db.fetchval("SELECT count(*) FROM review_requests WHERE status='pending'") or 0)
    except Exception:
        pending_reviews = 0
    checks.append(
        {
            "name": "待审队列",
            "status": "ok" if pending_reviews == 0 else "warn",
            "detail": f"{pending_reviews} 条待处理",
        }
    )

    try:
        documents = int(await db.fetchval("SELECT count(*) FROM documents") or 0)
    except Exception:
        documents = 0
    checks.append(
        {
            "name": "知识库",
            "status": "ok" if documents > 0 else "warn",
            "detail": f"{documents} 篇文档",
        }
    )

    checks.append(
        {
            "name": "医生写入门禁",
            "status": "info",
            "detail": "已启用" if settings.CLINICIAN_WRITE_ENABLED else "未启用（仅草稿）",
        }
    )
    checks.append(
        {
            "name": "鉴权",
            "status": "ok" if settings.AUTH_ENABLED else "fail",
            "detail": "已开启" if settings.AUTH_ENABLED else "已关闭（仅限开发）",
        }
    )
    return {
        "checks": checks,
        "counts": {
            "open_findings": open_findings,
            "pending_reviews": pending_reviews,
            "documents": documents,
        },
    }


@router.get("/api/v1/admin/evaluation")
async def admin_evaluation(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        "SELECT id, name, metrics, created_at FROM eval_runs ORDER BY created_at DESC LIMIT 20"
    )
    runs = [dict(row) for row in rows]
    return {"runs": runs, "latest": runs[0] if runs else None, "total": len(runs)}


# ---------------------------------------------------------------- previsit
class PrevisitRequest(BaseModel):
    member_id: str | None = None
    chief_complaint: str
    duration: str | None = None
    medications: str | None = None
    allergies: str | None = None
    history: str | None = None
    questions: str | None = None


@router.post("/api/v1/previsit")
async def create_previsit(
    payload: PrevisitRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, payload.member_id, user_id)
    previsit_id = "pv_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO previsit_forms
            (id, user_id, member_id, chief_complaint, duration, medications,
             allergies, history, questions)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        RETURNING *
        """,
        previsit_id,
        user_id,
        member["id"],
        payload.chief_complaint,
        payload.duration,
        payload.medications,
        payload.allergies,
        payload.history,
        payload.questions,
    )
    await _add_event(
        db,
        user_id,
        member["id"],
        "previsit",
        "就诊准备 · " + payload.chief_complaint[:30],
        {"previsit_id": previsit_id},
    )
    return {"previsit": dict(row) if row else None}


@router.get("/api/v1/previsit")
async def list_previsit(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, member_id, user_id)
    rows = await db.fetch(
        """
        SELECT id, member_id, chief_complaint, duration, medications, allergies,
               history, questions, created_at
        FROM previsit_forms
        WHERE user_id=$1 AND member_id=$2
        ORDER BY created_at DESC LIMIT 5
        """,
        user_id,
        member["id"],
    )
    return {"items": [dict(row) for row in rows]}


# ----------------------------------------------------------- doctor directory
@router.get("/api/v1/doctors")
async def list_doctor_directory(
    request: Request,
    department: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, name, department, title, intro, fee, schedule
        FROM doctor_directory
        WHERE active = true AND ($1::text IS NULL OR department=$1)
        ORDER BY department, name
        """,
        department,
    )
    return {"doctors": [dict(row) for row in rows]}


# ------------------------------------------------------------ wave B: 取药
@router.get("/api/v1/dispensing")
async def list_dispensing(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, member_id, user_id)
    rows = await db.fetch(
        """
        SELECT id, member_id, appointment_id, status, "window", ready_at, created_at
        FROM dispensing_progress
        WHERE user_id=$1 AND member_id=$2
        ORDER BY created_at DESC LIMIT 10
        """,
        user_id,
        member["id"],
    )
    return {"items": [dict(row) for row in rows]}


# ------------------------------------------------------------ wave B: 医保
@router.get("/api/v1/insurance/settlements")
async def list_insurance_settlements(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, member_id, user_id)
    rows = await db.fetch(
        """
        SELECT id, member_id, appointment_id, total, insurance, account, self_pay, status, settled_at
        FROM insurance_settlements
        WHERE user_id=$1 AND member_id=$2
        ORDER BY settled_at DESC LIMIT 20
        """,
        user_id,
        member["id"],
    )
    return {"items": [dict(row) for row in rows], "source": "平台模拟结算（HIS 医保接口待开通）"}


# ------------------------------------------------------------ wave B: 叫号
@router.get("/api/v1/queue/today")
async def queue_today(
    request: Request,
    member_id: str | None = Query(None),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, member_id, user_id)
    mine = await db.fetchrow(
        """
        SELECT id, ticket_no, department, status, called_at, created_at
        FROM queue_tickets
        WHERE user_id=$1 AND member_id=$2
          AND created_at::date = (now() AT TIME ZONE 'Asia/Shanghai')::date
        ORDER BY created_at DESC LIMIT 1
        """,
        user_id,
        member["id"],
    )
    current = None
    if mine:
        current = await db.fetchrow(
            """
            SELECT ticket_no, department, called_at
            FROM queue_tickets
            WHERE department=$1 AND status='called'
              AND created_at::date = (now() AT TIME ZONE 'Asia/Shanghai')::date
            ORDER BY called_at DESC NULLS LAST LIMIT 1
            """,
            mine["department"],
        )
    waiting = 0
    if mine and mine["department"]:
        waiting = int(
            await db.fetchval(
                """
                SELECT count(*) FROM queue_tickets
                WHERE department=$1 AND status='waiting'
                  AND created_at::date = (now() AT TIME ZONE 'Asia/Shanghai')::date
                """,
                mine["department"],
            )
            or 0
        )
    return {
        "mine": dict(mine) if mine else None,
        "current_called": dict(current) if current else None,
        "waiting_ahead": waiting,
    }


# ------------------------------------------------------------ wave B: 客服
class FeedbackRequest(BaseModel):
    category: str
    content: str
    member_id: str | None = None


@router.post("/api/v1/feedback")
async def create_feedback(
    payload: FeedbackRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    category = payload.category.strip() or "其他"
    if category not in {"服务态度", "排队等候", "费用问题", "隐私疑虑", "其他"}:
        raise HTTPException(status_code=422, detail="invalid category")
    if not payload.content.strip():
        raise HTTPException(status_code=422, detail="content required")
    ticket_id = "fb_" + uuid.uuid4().hex[:12]
    member_id = payload.member_id or ""
    row = await db.fetchrow(
        """
        INSERT INTO feedback_tickets (id, user_id, member_id, category, content)
        VALUES ($1,$2,$3,$4,$5) RETURNING *
        """,
        ticket_id,
        user_id,
        member_id,
        category,
        payload.content.strip(),
    )
    await _notify(db, user_id, "system", "反馈已受理", category + " · 24 小时内回复", "/support/")
    return {"ticket": dict(row) if row else None}


@router.get("/api/v1/feedback")
async def list_feedback(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, category, content, status, reply, created_at
        FROM feedback_tickets WHERE user_id=$1 ORDER BY created_at DESC LIMIT 20
        """,
        str(user["sub"]),
    )
    return {"items": [dict(row) for row in rows]}


# ------------------------------------------------------------ wave B: 授权
class ConsentScopeRequest(BaseModel):
    member_id: str
    can_view: bool = True
    can_act: bool = False


@router.get("/api/v1/consent/scopes")
async def list_consent_scopes(
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT s.id, s.member_id, s.can_view, s.can_act, s.version, s.updated_at,
               f.name, f.relation
        FROM consent_scopes s
        LEFT JOIN family_members f ON f.id = s.member_id
        WHERE s.user_id=$1
        ORDER BY s.updated_at DESC
        """,
        str(user["sub"]),
    )
    return {"items": [dict(row) for row in rows]}


@router.put("/api/v1/consent/scopes")
async def upsert_consent_scope(
    payload: ConsentScopeRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    user_id = str(user["sub"])
    member = await _member_row(db, payload.member_id, user_id)
    scope_id = "cs_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO consent_scopes (id, user_id, member_id, can_view, can_act, version)
        VALUES ($1,$2,$3,$4,$5,'v1')
        ON CONFLICT (user_id, member_id) DO UPDATE
        SET can_view=EXCLUDED.can_view, can_act=EXCLUDED.can_act,
            version='v' || (COALESCE(NULLIF(regexp_replace(consent_scopes.version, '\D', '', 'g'), ''), '0')::int + 1),
            updated_at=now()
        RETURNING id, member_id, can_view, can_act, version, updated_at
        """,
        scope_id,
        user_id,
        member["id"],
        payload.can_view,
        payload.can_act,
    )
    return {"scope": dict(row)}


# ------------------------------------------------------------ wave C: 病历
class ClinicalNoteRequest(BaseModel):
    member_id: str
    chief_complaint: str | None = None
    present_illness: str | None = None
    physical_exam: str | None = None
    diagnosis: str | None = None


async def _ensure_member_exists(db: Any, member_id: str) -> None:
    row = await db.fetchrow("SELECT id FROM family_members WHERE id=$1", member_id)
    if not row:
        raise HTTPException(status_code=404, detail="member not found")


@router.get("/api/v1/clinician/notes")
async def list_clinical_notes(
    request: Request,
    member_id: str = Query(...),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, doctor, member_id, chief_complaint, present_illness, physical_exam,
               diagnosis, status, created_at, updated_at
        FROM clinical_notes WHERE member_id=$1 ORDER BY created_at DESC LIMIT 20
        """,
        member_id,
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/clinician/notes")
async def create_clinical_note(
    payload: ClinicalNoteRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    await _ensure_member_exists(db, payload.member_id)
    note_id = "cn_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO clinical_notes
            (id, doctor, member_id, chief_complaint, present_illness, physical_exam, diagnosis)
        VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *
        """,
        note_id,
        str(user.get("user") or user.get("sub")),
        payload.member_id,
        payload.chief_complaint,
        payload.present_illness,
        payload.physical_exam,
        payload.diagnosis,
    )
    return {"note": dict(row) if row else None}


@router.post("/api/v1/clinician/notes/{note_id}/complete")
async def complete_clinical_note(
    note_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        """
        UPDATE clinical_notes SET status='completed', updated_at=now()
        WHERE id=$1 AND status='draft' RETURNING *
        """,
        note_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="note not found or already completed")
    return {"note": dict(row)}


# ------------------------------------------------------------ wave C: 医嘱
class ClinicalOrderRequest(BaseModel):
    member_id: str
    kind: str = "medication"
    name: str
    dose: str | None = None
    usage: str | None = None


@router.get("/api/v1/clinician/orders")
async def list_clinical_orders(
    request: Request,
    member_id: str = Query(...),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    rows = await db.fetch(
        """
        SELECT id, doctor, member_id, kind, name, dose, usage, status,
               review_status, review_note, created_at, signed_at
        FROM clinical_orders WHERE member_id=$1 ORDER BY created_at DESC LIMIT 50
        """,
        member_id,
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/clinician/orders")
async def create_clinical_order(
    payload: ClinicalOrderRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    if payload.kind not in {"medication", "exam", "lab"}:
        raise HTTPException(status_code=422, detail="invalid kind")
    db = _state(request).db
    await _ensure_member_exists(db, payload.member_id)
    order_id = "co_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO clinical_orders (id, doctor, member_id, kind, name, dose, usage)
        VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *
        """,
        order_id,
        str(user.get("user") or user.get("sub")),
        payload.member_id,
        payload.kind,
        payload.name,
        payload.dose,
        payload.usage,
    )
    return {"order": dict(row) if row else None}


@router.post("/api/v1/clinician/orders/{order_id}/sign")
async def sign_clinical_order(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    settings = _state(request).settings
    if not settings.CLINICIAN_WRITE_ENABLED:
        raise HTTPException(status_code=409, detail="HIS 写入尚未启用：医嘱签名需要医生 HIS 账号透传")
    row = await _state(request).db.fetchrow(
        """
        UPDATE clinical_orders SET status='signed', signed_at=now(), review_status='pending'
        WHERE id=$1 AND status='draft' RETURNING *
        """,
        order_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="order not found or already signed")
    return {"order": dict(row)}


class OrderReviewRequest(BaseModel):
    status: str = "approved"
    note: str | None = None


@router.post("/api/v1/clinician/orders/{order_id}/review")
async def review_clinical_order(
    order_id: str,
    payload: OrderReviewRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    if payload.status not in {"approved", "rejected"}:
        raise HTTPException(status_code=422, detail="invalid review status")
    row = await _state(request).db.fetchrow(
        """
        UPDATE clinical_orders SET review_status=$2, review_note=$3
        WHERE id=$1 AND status='signed' RETURNING *
        """,
        order_id,
        payload.status,
        payload.note,
    )
    if not row:
        raise HTTPException(status_code=404, detail="order not found or not signed")
    return {"order": dict(row)}


@router.get("/api/v1/clinician/labs")
async def list_lab_results(
    request: Request,
    member_id: str = Query(...),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, name, result_value, reference_range, abnormal, measured_at
        FROM lab_results WHERE member_id=$1 ORDER BY measured_at DESC LIMIT 50
        """,
        member_id,
    )
    return {"items": [dict(row) for row in rows]}


@router.get("/api/v1/clinician/schedules")
async def my_schedules(
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, doctor, department, weekday, period, status, note
        FROM doctor_schedules WHERE doctor=$1 ORDER BY weekday, period
        """,
        str(user.get("user") or user.get("sub")),
    )
    return {"items": [dict(row) for row in rows]}


class ConsultationRequest(BaseModel):
    member_id: str
    target_department: str
    reason: str | None = None


@router.get("/api/v1/clinician/consultations")
async def list_consultations(
    request: Request,
    member_id: str = Query(...),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, doctor, member_id, target_department, reason, status, reply, created_at
        FROM consultations WHERE member_id=$1 ORDER BY created_at DESC LIMIT 20
        """,
        member_id,
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/clinician/consultations")
async def create_consultation(
    payload: ConsultationRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = _state(request).db
    await _ensure_member_exists(db, payload.member_id)
    consult_id = "cons_" + uuid.uuid4().hex[:10]
    row = await db.fetchrow(
        """
        INSERT INTO consultations (id, doctor, member_id, target_department, reason)
        VALUES ($1,$2,$3,$4,$5) RETURNING *
        """,
        consult_id,
        str(user.get("user") or user.get("sub")),
        payload.member_id,
        payload.target_department,
        payload.reason,
    )
    return {"consultation": dict(row) if row else None}


# ---------------------------------------------------------------- wave D: ops
class ReviewDecisionRequest(BaseModel):
    action: str
    note: str | None = None


@router.get("/api/v1/ops/reviews")
async def ops_reviews(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, kind, ref_id, user_id, member_id, status, payload, decision_note,
               decided_by, decided_at, created_at
        FROM review_requests
        WHERE status IN ('pending', 'processing', 'acknowledged')
        ORDER BY created_at DESC LIMIT 50
        """
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/ops/reviews/{review_id}/decide")
async def decide_review(
    review_id: str,
    payload: ReviewDecisionRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    if payload.action not in {"approve", "reject"}:
        raise HTTPException(status_code=422, detail="action must be approve or reject")
    db = _state(request).db
    actor = str(user.get("user") or user.get("sub"))
    row = await db.fetchrow(
        """
        UPDATE review_requests
        SET status=$2, decision_note=$3, decided_by=$4, decided_at=now()
        WHERE id=$1 AND status IN ('pending', 'processing', 'acknowledged')
        RETURNING id, kind, ref_id, status, decision_note, decided_by, decided_at
        """,
        review_id,
        "approved" if payload.action == "approve" else "rejected",
        payload.note,
        actor,
    )
    if not row:
        raise HTTPException(status_code=404, detail="review not found or already decided")
    await db.execute(
        "INSERT INTO audit_log (actor, action, detail) VALUES ($1,$2,$3)",
        actor,
        "ops.review.decide",
        json.dumps({"review_id": review_id, "action": payload.action}, ensure_ascii=False),
    )
    return {"review": dict(row)}


@router.get("/api/v1/ops/audit")
async def ops_audit(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, request_id, session_id, actor, action, detail, created_at
        FROM audit_log ORDER BY created_at DESC LIMIT $1
        """,
        limit,
    )
    return {"items": [dict(row) for row in rows]}


@router.get("/api/v1/ops/analytics")
async def ops_analytics(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    db = _state(request).db
    events = await db.fetch(
        """
        SELECT name, count(*) AS count FROM analytics_events
        GROUP BY name ORDER BY count DESC LIMIT 20
        """
    )
    totals = {
        "sessions": int(await db.fetchval("SELECT count(*) FROM sessions") or 0),
        "reports": int(await db.fetchval("SELECT count(*) FROM reports") or 0),
        "bookings": int(await db.fetchval("SELECT count(*) FROM bookings") or 0),
        "users": int(await db.fetchval("SELECT count(*) FROM users") or 0),
    }
    return {"events": [dict(row) for row in events], "totals": totals}


@router.get("/api/v1/ops/insurance-recon")
async def ops_insurance_recon(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT s.id, s.member_id, s.appointment_id, s.total AS settlement_total,
               a.amount AS appointment_amount, s.status, s.settled_at
        FROM insurance_settlements s
        LEFT JOIN appointment_projection a ON a.id = s.appointment_id
        ORDER BY s.settled_at DESC LIMIT 100
        """
    )
    items: list[dict[str, Any]] = []
    diff_count = 0
    for row in rows:
        item = dict(row)
        total = float(item["settlement_total"] or 0)
        amount = float(item["appointment_amount"] or 0)
        item["diff"] = round(total - amount, 2)
        if abs(item["diff"]) > 0.01:
            diff_count += 1
        items.append(item)
    return {"items": items, "diff_count": diff_count, "checked": len(items)}


@router.get("/api/v1/ops/crisis")
async def ops_crisis(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, user_id, member_id, kind, detail, status, handled_by, handled_at, created_at
        FROM crisis_events ORDER BY created_at DESC LIMIT 50
        """
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/ops/crisis/{event_id}/resolve")
async def resolve_crisis(
    event_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        """
        UPDATE crisis_events SET status='handled', handled_by=$2, handled_at=now()
        WHERE id=$1 AND status='open'
        RETURNING id, status, handled_by, handled_at
        """,
        event_id,
        str(user.get("user") or user.get("sub")),
    )
    if not row:
        raise HTTPException(status_code=404, detail="crisis event not found or already handled")
    return {"event": dict(row)}


@router.get("/api/v1/ops/notifications")
async def ops_notifications(
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        """
        SELECT id, notification_id, channel, target, status, attempts, error, created_at, updated_at
        FROM notification_deliveries ORDER BY created_at DESC LIMIT 100
        """
    )
    return {"items": [dict(row) for row in rows]}


@router.post("/api/v1/ops/notifications/{delivery_id}/retry")
async def retry_delivery(
    delivery_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_ops),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        """
        UPDATE notification_deliveries
        SET status='sent', attempts=attempts+1, error=NULL, updated_at=now()
        WHERE id=$1 AND status IN ('failed', 'pending')
        RETURNING id, status, attempts, updated_at
        """,
        delivery_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="delivery not found or already sent")
    return {"delivery": dict(row)}

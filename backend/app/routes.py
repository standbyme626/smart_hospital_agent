from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .api.deps import create_token, get_current_user, require_admin
from .policy import DEPARTMENT_NAMES, DRUG_CHECKER
from .schemas import (
    ChatRequest,
    DrugCheckRequest,
    EvalRequest,
    LocalAnalyzeRequest,
    LockRequest,
    LoginRequest,
    OrderRequest,
    PayRequest,
    RagSearchRequest,
    RuleRequest,
)
from .session_store import jsonable

router = APIRouter()


def _state(request: Request) -> Any:
    return request.app.state


def _sse(event: dict[str, Any]) -> str:
    data = json.dumps(jsonable(event.get("data", {})), ensure_ascii=False, default=str)
    return f"id: {event.get('seq', 0)}\nevent: {event.get('type', 'message')}\ndata: {data}\n\n"


async def _mcp(state: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await state.mcp.call_tool(name, arguments)
    if result["is_error"]:
        raise HTTPException(status_code=502, detail=result["text"])
    return result["data"] or {}


# --------------------------------------------------------------------- auth
@router.post("/api/v1/auth/login")
async def login(payload: LoginRequest, request: Request) -> dict[str, Any]:
    settings = _state(request).settings
    if payload.username == settings.ADMIN_USER and payload.password == settings.ADMIN_PASSWORD:
        return {"access_token": create_token(settings, payload.username, "admin"), "token_type": "bearer", "role": "admin"}
    if payload.username == "guest" and payload.password == "guest":
        return {"access_token": create_token(settings, "guest", "patient"), "token_type": "bearer", "role": "patient"}
    raise HTTPException(status_code=401, detail="invalid credentials")


@router.get("/api/v1/auth/me")
async def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": user.get("sub"), "role": user.get("role")}


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
    user_id = payload.user_id or str(user.get("sub") or "anonymous")

    async def generator():
        yield _sse(
            {
                "seq": 0,
                "type": "session",
                "data": {"session_id": session_id, "request_id": request_id},
            }
        )
        try:
            async for event in state.agent.stream(
                session_id=session_id,
                message=payload.message,
                user_id=user_id,
                request_id=request_id,
            ):
                yield _sse(event)
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
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    sessions = await _state(request).sessions.list_sessions(str(user.get("sub") or "anonymous"))
    return {"sessions": sessions}


@router.get("/api/v1/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    after: int = Query(0, ge=0),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    events = await _state(request).sessions.events(session_id, after=after)
    return {"session_id": session_id, "events": events}


# --------------------------------------------------------------------- rag
@router.post("/api/v1/rag/search")
async def rag_search(
    payload: RagSearchRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    results = await _state(request).retrieval.search(
        payload.query,
        top_k=payload.top_k,
        department=payload.department,
    )
    return {"query": payload.query, "count": len(results), "results": results}


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


@router.get("/api/v1/departments")
async def departments() -> dict[str, Any]:
    return {"departments": DEPARTMENT_NAMES}


@router.post("/api/v1/local/analyze")
async def local_analyze(
    payload: LocalAnalyzeRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    result = await state.gateway.chat(
        [
            {"role": "system", "content": payload.instruction},
            {"role": "user", "content": payload.text},
        ],
        task="local",
        temperature=0.3,
        max_tokens=800,
        session_id=f"local-{uuid.uuid4().hex[:8]}",
    )
    return {
        "content": result.content,
        "provider": result.provider,
        "model": result.model,
        "tokens": {"prompt": result.prompt_tokens, "completion": result.completion_tokens},
    }


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
    data = await _mcp(
        _state(request),
        "lock_slot",
        {"slot_id": payload.slot_id, "patient_id": payload.patient_id or str(user.get("sub"))},
    )
    return {"lock": data}


@router.post("/api/v1/booking/orders")
async def booking_order(
    payload: OrderRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    data = await _mcp(
        state,
        "create_order",
        {"lock_id": payload.lock_id, "patient_id": payload.patient_id or str(user.get("sub"))},
    )
    await state.db.execute(
        """
        INSERT INTO bookings (id, session_id, patient_id, department, slot_id, order_id, status, payment_status, amount, detail)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
        ON CONFLICT (id) DO NOTHING
        """,
        data.get("order_id"),
        str(user.get("sub")),
        data.get("patient_id"),
        (data.get("slot_id") or "").split("|")[0],
        data.get("slot_id"),
        data.get("order_id"),
        data.get("status", "created"),
        data.get("payment_status", "unpaid"),
        data.get("amount", 50),
        json.dumps(data, ensure_ascii=False),
    )
    return {"order": data}


@router.post("/api/v1/booking/orders/{order_id}/pay")
async def booking_pay(
    order_id: str,
    payload: PayRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    state = _state(request)
    if payload.confirm_token != state.settings.HITL_CONFIRM_TOKEN:
        raise HTTPException(status_code=403, detail="invalid confirmation token")
    data = await _mcp(
        state,
        "pay_order",
        {"order_id": order_id, "confirm_token": payload.confirm_token, "method": payload.method},
    )
    await state.db.execute(
        "UPDATE bookings SET payment_status=$2, status=$3, updated_at=now() WHERE id=$1",
        order_id,
        data.get("payment_status", "paid"),
        data.get("status", "booked"),
    )
    await state.sessions.audit(
        request_id=request.headers.get("X-Request-ID") or uuid.uuid4().hex,
        session_id=str(user.get("sub")),
        actor=str(user.get("sub")),
        action="booking.pay",
        detail=data,
    )
    return {"order": data}


@router.get("/api/v1/booking/orders/{order_id}")
async def booking_status(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    data = await _mcp(_state(request), "get_payment_status", {"order_id": order_id})
    return {"order": data}


@router.post("/api/v1/booking/orders/{order_id}/cancel")
async def booking_cancel(
    order_id: str,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    data = await _mcp(_state(request), "cancel_order", {"order_id": order_id})
    return {"order": data}


# ------------------------------------------------------------------- admin
@router.get("/api/v1/admin/rules")
async def list_rules(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    rows = await _state(request).db.fetch(
        "SELECT id, kind, key, value, enabled, created_at FROM rules ORDER BY id DESC LIMIT 200"
    )
    return {"rules": [dict(row) for row in rows]}


@router.post("/api/v1/admin/rules")
async def upsert_rule(
    payload: RuleRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    row = await _state(request).db.fetchrow(
        """
        INSERT INTO rules (kind, key, value, enabled)
        VALUES ($1, $2, $3::jsonb, $4)
        ON CONFLICT (kind, key) DO UPDATE SET value=EXCLUDED.value, enabled=EXCLUDED.enabled
        RETURNING id, kind, key, value, enabled
        """,
        payload.kind,
        payload.key,
        json.dumps(payload.value, ensure_ascii=False),
        payload.enabled,
    )
    return {"rule": dict(row) if row else None}


@router.delete("/api/v1/admin/rules/{rule_id}")
async def disable_rule(
    rule_id: int,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    await _state(request).db.execute("UPDATE rules SET enabled=false WHERE id=$1", rule_id)
    return {"ok": True}


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


@router.post("/api/v1/evolution/run")
async def evolution_run(
    payload: EvalRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return await run_eval(payload, request, user)


@router.get("/api/v1/evolution/results")
async def evolution_results(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return await eval_results(request, user)


# ------------------------------------------------------------------ doctor
@router.post("/api/v1/doctor/consult")
async def doctor_consult(
    payload: ChatRequest,
    request: Request,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    if user.get("role") not in {"doctor", "admin"}:
        raise HTTPException(status_code=403, detail="doctor role required")
    state = _state(request)
    session_id = payload.session_id or f"doc_{uuid.uuid4().hex[:12]}"
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    final: dict[str, Any] | None = None
    async for event in state.agent.stream(
        session_id=session_id,
        message=f"医生工作台咨询：{payload.message}",
        user_id=str(user.get("sub")),
        request_id=request_id,
    ):
        if event["type"] == "final":
            final = event["data"]
    if not final:
        raise HTTPException(status_code=502, detail="no report produced")
    return {"session_id": session_id, **final}


# ------------------------------------------------------------------ health
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

"""管理端 Wave E（第二批）API：知识库 / 检索 / 未命中 / 模型 / 回放 / 审计 / 验收 / HIS / 多院区 / 实验。

一律 require_admin；所有数据都是真库真接口，不构造演示数据：
- 知识库与语料读 documents；
- 检索调试与 RAG 链路直接跑 RetrievalService.search；
- 未命中问题由 RetrievalService 在「命中 0 条」时写入；
- 模型网关读 model_calls；会话回放读 sessions/messages/events/audit_log；
- HIS 接入读配置并真实调用 MCP health；
- 多院区与实验灰度落库，变更写 audit_log。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .api.deps import require_admin
from .playbooks import PLAYBOOKS, summary as playbook_summary
from .policy import DEPARTMENT_NAMES
from .schema_admin_ext import (
    EXPERIMENT_STATUSES,
    ISOLATION_MODES,
    TENANT_STATUSES,
    ensure_admin_ext_schema,
)

router = APIRouter()

TRACE_TABLES = ("bookings", "mcp_orders", "dispensing_progress", "queue_tickets", "review_requests", "followups", "experiments")


# ------------------------------------------------------------------ 请求体
class ResolveRequest(BaseModel):
    question: str
    answer: str
    department: str | None = None
    doc_key: str | None = None
    question_key: str | None = None


class RetrievalDebugRequest(BaseModel):
    query: str
    department: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class PipelineRequest(BaseModel):
    query: str
    department: str | None = None


class TenantCreate(BaseModel):
    code: str
    name: str
    isolation: str = "logical"
    note: str | None = None


class TenantUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    isolation: str | None = None
    note: str | None = None


class ExperimentCreate(BaseModel):
    key: str
    name: str
    variants: list[str] = Field(default_factory=lambda: ["A", "B"])
    note: str | None = None


class ExperimentUpdate(BaseModel):
    name: str | None = None
    status: str | None = None
    rollout: int | None = None
    variants: list[str] | None = None
    note: str | None = None


# ------------------------------------------------------------------ 工具
def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _load(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw


def _question_key(question: str) -> str:
    normalized = re.sub(r"\s+", "", question).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]


def _action(detail: Any) -> str:
    return str(detail or "")


def _row_dict(row: Any) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


async def _audit(db: Any, request: Request, actor: str, action: str, detail: dict[str, Any]) -> None:
    await db.execute(
        """
        INSERT INTO audit_log (request_id, session_id, actor, action, detail)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        """,
        getattr(request.state, "request_id", None),
        None,
        actor,
        action,
        json.dumps(detail, ensure_ascii=False),
    )


def _state(request: Request) -> Any:
    return request.app.state


async def _ready(request: Request) -> Any:
    db = _state(request).db
    await ensure_admin_ext_schema(db)
    return db


def _actor(user: dict[str, Any]) -> str:
    return str(user.get("sub") or user.get("username") or "unknown")


async def _count(db: Any, sql: str, *args: Any) -> int:
    try:
        return int(await db.fetchval(sql, *args) or 0)
    except Exception:
        return 0


# ================================================================== 知识库
@router.get("/api/v1/admin/kb")
async def admin_kb(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    total = await _count(db, "SELECT count(*) FROM documents")
    rows = await db.fetch(
        "SELECT department, count(*) AS count FROM documents GROUP BY department ORDER BY count DESC"
    )
    departments = [
        {"department": row["department"] or "未标注", "count": int(row["count"])} for row in rows
    ]
    covered = {(row["department"] or "").strip() for row in rows}
    missing = [name for name in DEPARTMENT_NAMES if name not in covered]
    retrieval_row = await db.fetchrow(
        "SELECT draft, published FROM config_entries WHERE kind='agent_config' AND key='retrieval'"
    )
    retrieval_cfg = _load(retrieval_row["published"]) if retrieval_row and retrieval_row["published"] else (
        _load(retrieval_row["draft"]) if retrieval_row else {}
    )
    unmatched = await _count(db, "SELECT count(*) FROM unmatched_questions")
    resolved = await _count(db, "SELECT count(*) FROM unmatched_resolutions")
    return {
        "documents": total,
        "departments": departments,
        "coverage": {
            "covered": len(DEPARTMENT_NAMES) - len(missing),
            "total": len(DEPARTMENT_NAMES),
            "missing": missing,
        },
        "retrieval_config": retrieval_cfg or {},
        "unmatched": unmatched,
        "resolved": resolved,
    }


@router.get("/api/v1/admin/rag/documents")
async def admin_rag_documents(
    request: Request,
    department: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT id, doc_key, title, department, source, length(content) AS chars, created_at
        FROM documents
        WHERE ($1::text IS NULL OR department = $1)
        ORDER BY id DESC
        LIMIT $2
        """,
        department,
        limit,
    )
    return {
        "items": [
            {
                "id": int(row["id"]),
                "doc_key": row["doc_key"],
                "title": row["title"],
                "department": row["department"],
                "source": row["source"],
                "chars": int(row["chars"] or 0),
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ],
        "total": await _count(db, "SELECT count(*) FROM documents"),
    }


@router.post("/api/v1/admin/retrieval/debug")
async def admin_retrieval_debug(
    payload: RetrievalDebugRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    retrieval = state.retrieval
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query 不能为空")

    started = time.perf_counter()
    hits = await retrieval.search(query, top_k=payload.top_k, department=payload.department)
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {
        "query": query,
        "department": payload.department,
        "latency_ms": elapsed,
        "total": len(hits),
        "filtered": bool(getattr(retrieval, "last_filtered", False)),
        "dept_unavailable": bool(getattr(retrieval, "last_dept_unavailable", False)),
        "settings": {
            "top_k": int(getattr(state.settings, "RETRIEVAL_TOP_K", 0)),
            "rerank_top_n": int(getattr(state.settings, "RERANK_TOP_N", 0)),
        },
        "hits": hits,
    }


@router.post("/api/v1/admin/rag/pipeline-check")
async def admin_pipeline_check(
    payload: PipelineRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    retrieval = state.retrieval
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query 不能为空")

    started = time.perf_counter()
    hits: list[dict[str, Any]] = []
    error: str | None = None
    try:
        hits = await retrieval.search(query, top_k=5, department=payload.department)
    except Exception as exc:  # 链路自检要把失败原因暴露出来，而不是 500
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round((time.perf_counter() - started) * 1000, 1)

    stages = [
        {
            "name": "问题归一化与科室过滤",
            "ok": error is None,
            "detail": f"科室={payload.department or '不限'}，回退={bool(getattr(retrieval, 'last_dept_unavailable', False))}",
        },
        {
            "name": "向量 + 关键词双路检索",
            "ok": error is None,
            "detail": f"命中 {len(hits)} 条，耗时 {elapsed}ms",
        },
        {
            "name": "RRF 融合与重排",
            "ok": bool(hits),
            "detail": "有命中才会进入重排" if not hits else f"重排后保留 {len(hits)} 条",
        },
        {
            "name": "证据可引用（可回溯原文）",
            "ok": bool(hits),
            "detail": "无证据时应回答「未覆盖」，不得编造" if not hits else "可点击查看原文",
        },
    ]
    return {
        "query": query,
        "latency_ms": elapsed,
        "hits": len(hits),
        "error": error,
        "stages": stages,
        "top": hits[:5],
    }


# ================================================================== 未命中问题
@router.get("/api/v1/admin/unmatched")
async def admin_unmatched(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    only_open: bool = Query(True),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT u.question_key, u.question, u.department, u.hits, u.first_seen, u.last_seen,
               r.doc_key AS resolved_doc_key, r.actor AS resolved_by, r.created_at AS resolved_at
        FROM unmatched_questions u
        LEFT JOIN unmatched_resolutions r USING (question_key)
        WHERE ($2::boolean = false OR r.question_key IS NULL)
        ORDER BY u.last_seen DESC
        LIMIT $1
        """,
        limit,
        only_open,
    )
    items = [
        {
            "question_key": row["question_key"],
            "question": row["question"],
            "department": row["department"],
            "hits": int(row["hits"]),
            "first_seen": _iso(row["first_seen"]),
            "last_seen": _iso(row["last_seen"]),
            "resolved": row["resolved_doc_key"] is not None,
            "resolved_doc_key": row["resolved_doc_key"],
            "resolved_by": row["resolved_by"],
            "resolved_at": _iso(row["resolved_at"]),
        }
        for row in rows
    ]
    return {
        "items": items,
        "open_total": await _count(
            db,
            "SELECT count(*) FROM unmatched_questions u LEFT JOIN unmatched_resolutions r USING (question_key) "
            "WHERE r.question_key IS NULL",
        ),
    }


@router.post("/api/v1/admin/unmatched/resolve")
async def admin_unmatched_resolve(
    payload: ResolveRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    state = _state(request)
    db = await _ready(request)
    actor = _actor(user)
    question = payload.question.strip()
    answer = payload.answer.strip()
    if not question or not answer:
        raise HTTPException(status_code=422, detail="问题与答案都不能为空")

    doc_key = (payload.doc_key or "admin-faq-" + _question_key(question)).strip()
    rows = [
        {
            "doc_key": doc_key,
            "title": question[:80],
            "content": answer,
            "department": payload.department,
            "source": "admin-faq",
            "metadata": {"from": "unmatched", "actor": actor},
        }
    ]
    try:
        written = await state.retrieval.ingest(rows)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"写入知识库失败：{exc}") from exc

    # 以池里已有的 question_key 为准（同一问题可能已由检索侧写入）；
    # 都没有时再按文本算一次，保证前后端 join 得上。
    existing = await db.fetchrow(
        "SELECT question_key FROM unmatched_questions WHERE question=$1 LIMIT 1", question
    )
    question_key = payload.question_key or (
        existing["question_key"] if existing else _question_key(question)
    )
    await db.execute(
        """
        INSERT INTO unmatched_resolutions (question_key, doc_key, answer, actor)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (question_key) DO UPDATE
            SET doc_key=$2, answer=$3, actor=$4, created_at=now()
        """,
        question_key,
        doc_key,
        answer,
        actor,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.unmatched.resolve",
        {"question_key": question_key, "doc_key": doc_key, "written": written},
    )
    return {"doc_key": doc_key, "written": written, "question_key": question_key}


# ================================================================== 模型网关
@router.get("/api/v1/admin/model-calls/summary")
async def admin_model_summary(
    request: Request,
    days: int = Query(7, ge=1, le=90),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    window = f"now() - interval '{days} days'"
    rows = await db.fetch(
        f"""
        SELECT provider, model, coalesce(task, '未标注') AS task,
               count(*) AS calls,
               coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
               coalesce(sum(completion_tokens), 0) AS completion_tokens,
               coalesce(sum(cost), 0) AS cost
        FROM model_calls
        WHERE created_at >= {window}
        GROUP BY 1, 2, 3
        ORDER BY calls DESC
        LIMIT 50
        """
    )
    totals = await db.fetchrow(
        f"""
        SELECT count(*) AS calls,
               coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
               coalesce(sum(completion_tokens), 0) AS completion_tokens,
               coalesce(sum(cost), 0) AS cost
        FROM model_calls WHERE created_at >= {window}
        """
    )
    return {
        "days": days,
        "totals": {
            "calls": int(totals["calls"] or 0),
            "prompt_tokens": int(totals["prompt_tokens"] or 0),
            "completion_tokens": int(totals["completion_tokens"] or 0),
            "cost": float(totals["cost"] or 0),
        },
        "items": [
            {
                "provider": row["provider"],
                "model": row["model"],
                "task": row["task"],
                "calls": int(row["calls"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "cost": float(row["cost"] or 0),
            }
            for row in rows
        ],
    }


# ================================================================== 会话回放
@router.get("/api/v1/admin/sessions")
async def admin_sessions(
    request: Request,
    limit: int = Query(30, ge=1, le=100),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT s.id, s.user_id, s.title, s.created_at, s.updated_at,
               (SELECT count(*) FROM messages m WHERE m.session_id = s.id) AS messages,
               (SELECT max(m.created_at) FROM messages m WHERE m.session_id = s.id) AS last_at
        FROM sessions s
        ORDER BY s.updated_at DESC
        LIMIT $1
        """,
        limit,
    )
    return {
        "items": [
            {
                "id": row["id"],
                "user_tag": str(row["user_id"])[:8],
                "title": row["title"] or "未命名会话",
                "messages": int(row["messages"] or 0),
                "created_at": _iso(row["created_at"]),
                "last_at": _iso(row["last_at"] or row["updated_at"]),
            }
            for row in rows
        ]
    }


@router.get("/api/v1/admin/sessions/{session_id}/trace")
async def admin_session_trace(
    session_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    session = await db.fetchrow(
        "SELECT id, user_id, title, created_at, updated_at FROM sessions WHERE id=$1", session_id
    )
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")

    messages = await db.fetch(
        """
        SELECT role, content, created_at FROM messages
        WHERE session_id=$1 ORDER BY id LIMIT 200
        """,
        session_id,
    )
    events = await db.fetch(
        "SELECT * FROM events WHERE session_id=$1 ORDER BY seq LIMIT 200", session_id
    )
    audits = await db.fetch(
        """
        SELECT actor, action, detail, created_at FROM audit_log
        WHERE session_id=$1 ORDER BY id DESC LIMIT 100
        """,
        session_id,
    )
    return {
        "session": {
            "id": session["id"],
            "user_tag": str(session["user_id"])[:8],
            "title": session["title"] or "未命名会话",
            "created_at": _iso(session["created_at"]),
            "updated_at": _iso(session["updated_at"]),
        },
        "messages": [
            {
                "role": row["role"],
                "content": (row["content"] or "")[:800],
                "created_at": _iso(row["created_at"]),
            }
            for row in messages
        ],
        "events": [
            {
                "kind": _row_dict(row).get("type") or _row_dict(row).get("kind") or "事件",
                "data": _load(_row_dict(row).get("data") or _row_dict(row).get("payload") or {}),
                "created_at": _iso(_row_dict(row).get("created_at")),
            }
            for row in events
        ],
        "audit": [
            {
                "actor": row["actor"],
                "action": row["action"],
                "detail": _load(row["detail"]),
                "created_at": _iso(row["created_at"]),
            }
            for row in audits
        ],
    }


# ================================================================== 配置审计
@router.get("/api/v1/admin/audit/aggregate")
async def admin_audit_aggregate(
    request: Request,
    days: int = Query(1, ge=1, le=30),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    window = f"now() - interval '{days} days'"
    by_action = await db.fetch(
        f"""
        SELECT action, count(*) AS count FROM audit_log
        WHERE created_at >= {window} GROUP BY 1 ORDER BY count DESC LIMIT 30
        """
    )
    by_actor = await db.fetch(
        f"""
        SELECT coalesce(actor, '匿名') AS actor, count(*) AS count FROM audit_log
        WHERE created_at >= {window} GROUP BY 1 ORDER BY count DESC LIMIT 20
        """
    )
    recent = await db.fetch(
        f"""
        SELECT request_id, actor, action, created_at FROM audit_log
        WHERE created_at >= {window} ORDER BY id DESC LIMIT 30
        """
    )
    config_changes = await db.fetch(
        "SELECT kind, key, version, action, actor, note, created_at FROM config_history ORDER BY id DESC LIMIT 20"
    )
    return {
        "days": days,
        "total": await _count(db, f"SELECT count(*) FROM audit_log WHERE created_at >= {window}"),
        "by_action": [{"action": row["action"], "count": int(row["count"])} for row in by_action],
        "by_actor": [{"actor": row["actor"], "count": int(row["count"])} for row in by_actor],
        "recent": [
            {
                "request_id": row["request_id"],
                "actor": row["actor"],
                "action": row["action"],
                "created_at": _iso(row["created_at"]),
            }
            for row in recent
        ],
        "config_changes": [
            {
                "kind": row["kind"],
                "key": row["key"],
                "version": int(row["version"]),
                "action": row["action"],
                "actor": row["actor"],
                "note": row["note"],
                "created_at": _iso(row["created_at"]),
            }
            for row in config_changes
        ],
    }


# ================================================================== 验收自测
@router.get("/api/v1/admin/acceptance")
async def admin_acceptance(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    state = _state(request)
    checks: list[dict[str, Any]] = []

    async def add(name: str, ok: bool, detail: str, level: str = "ok") -> None:
        checks.append({"name": name, "ok": ok, "status": level if ok else "fail", "detail": detail})

    try:
        await db.fetchval("SELECT 1")
        await add("数据库连接", True, "SELECT 1 通过")
    except Exception as exc:
        await add("数据库连接", False, f"{type(exc).__name__}: {exc}")

    tables = await _count(db, "SELECT count(*) FROM pg_tables WHERE schemaname='public'")
    await add("业务表数量", tables >= 40, f"public 下 {tables} 张表（阈值 40）", "warn" if tables < 40 else "ok")

    documents = await _count(db, "SELECT count(*) FROM documents")
    await add("知识库条目", documents > 0, f"{documents} 条文档", "ok" if documents else "warn")

    configs = await _count(db, "SELECT count(*) FROM config_entries")
    await add("配置注册表", configs > 0, f"{configs} 个配置项", "ok" if configs else "warn")

    audits = await _count(db, "SELECT count(*) FROM audit_log WHERE created_at >= now() - interval '1 day'")
    await add("近 24 小时审计", audits >= 0, f"{audits} 条审计记录", "ok" if audits else "warn")

    calls = await _count(db, "SELECT count(*) FROM model_calls WHERE created_at >= now() - interval '1 day'")
    await add("近 24 小时模型调用", calls >= 0, f"{calls} 次调用", "ok" if calls else "warn")

    latest_eval = await db.fetchrow("SELECT id, name, metrics, created_at FROM eval_runs ORDER BY id DESC LIMIT 1")
    await add(
        "最近评测批次",
        latest_eval is not None,
        f"#{latest_eval['id']} {latest_eval['name']}" if latest_eval else "还没有评测批次",
        "ok" if latest_eval else "warn",
    )

    auth_enabled = bool(getattr(state.settings, "AUTH_ENABLED", False))
    await add("鉴权开关", auth_enabled, "AUTH_ENABLED=true" if auth_enabled else "AUTH_ENABLED=false（仅本地调试）", "ok" if auth_enabled else "warn")

    playbooks = [
        {
            "pid": pb.pid,
            "name": pb.name,
            "area": pb.area,
            "status": pb.status,
            "endpoint": pb.endpoint,
            "evidence": pb.evidence_test,
            "note": pb.note,
        }
        for pb in PLAYBOOKS
    ]
    counts = playbook_summary()
    return {
        "checks": checks,
        "playbooks": playbooks,
        "playbook_summary": counts,
        "summary": {
            "ok": sum(1 for c in checks if c["status"] == "ok"),
            "warn": sum(1 for c in checks if c["status"] == "warn"),
            "fail": sum(1 for c in checks if c["status"] == "fail"),
            "total": len(checks),
        },
    }


# ================================================================== HIS 接入
@router.get("/api/v1/admin/his")
async def admin_his(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    settings = _state(request).settings
    checks = await db.fetch(
        "SELECT ok, detail, latency_ms, actor, created_at FROM his_checks ORDER BY id DESC LIMIT 10"
    )
    from .routes import CAPABILITY_MATRIX  # 延迟导入，避免模块循环

    return {
        "config": {
            "mcp_url": getattr(settings, "HOSPITAL_MCP_URL", ""),
            "payment_mode": getattr(settings, "HIS_PAYMENT_MODE", ""),
            "clinician_write_enabled": bool(getattr(settings, "CLINICIAN_WRITE_ENABLED", False)),
            "auth_enabled": bool(getattr(settings, "AUTH_ENABLED", False)),
        },
        "capabilities": CAPABILITY_MATRIX,
        "checks": [
            {
                "ok": bool(row["ok"]),
                "detail": row["detail"],
                "latency_ms": row["latency_ms"],
                "actor": row["actor"],
                "created_at": _iso(row["created_at"]),
            }
            for row in checks
        ],
    }


@router.post("/api/v1/admin/his/check")
async def admin_his_check(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    state = _state(request)
    actor = _actor(user)
    started = time.perf_counter()
    ok = False
    detail = ""
    try:
        ok = bool(await state.mcp.health())
        detail = "MCP 健康检查通过" if ok else "MCP 返回不健康"
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    latency = int((time.perf_counter() - started) * 1000)
    await db.execute(
        "INSERT INTO his_checks (ok, detail, latency_ms, actor) VALUES ($1, $2, $3, $4)",
        ok,
        detail,
        latency,
        actor,
    )
    await _audit(db, request, actor, "admin.his.check", {"ok": ok, "latency_ms": latency})
    return {"ok": ok, "detail": detail, "latency_ms": latency}


# ================================================================== 多院区
@router.get("/api/v1/admin/tenants")
async def admin_tenants(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch("SELECT * FROM tenants ORDER BY created_at")
    return {
        "items": [
            {
                "id": row["id"],
                "code": row["code"],
                "name": row["name"],
                "status": row["status"],
                "isolation": row["isolation"],
                "note": row["note"],
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ],
        "statuses": list(TENANT_STATUSES),
        "isolation_modes": list(ISOLATION_MODES),
    }


@router.post("/api/v1/admin/tenants")
async def admin_tenant_create(
    payload: TenantCreate,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    code = payload.code.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{2,32}", code):
        raise HTTPException(status_code=422, detail="code 只允许小写字母、数字、下划线、短横线（2-32 位）")
    if payload.isolation not in ISOLATION_MODES:
        raise HTTPException(status_code=422, detail=f"isolation 只能是 {ISOLATION_MODES}")
    tenant_id = "tn_" + uuid.uuid4().hex[:10]
    try:
        row = await db.fetchrow(
            """
            INSERT INTO tenants (id, code, name, status, isolation, note)
            VALUES ($1, $2, $3, 'active', $4, $5)
            RETURNING *
            """,
            tenant_id,
            code,
            payload.name.strip(),
            payload.isolation,
            payload.note,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"院区编码已存在或不合法：{exc}") from exc
    await _audit(db, request, actor, "admin.tenant.create", {"code": code, "id": tenant_id})
    return {"tenant": _row_dict(row)}


@router.patch("/api/v1/admin/tenants/{tenant_id}")
async def admin_tenant_update(
    tenant_id: str,
    payload: TenantUpdate,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    tenant = await db.fetchrow("SELECT * FROM tenants WHERE id=$1", tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail="院区不存在")
    status = payload.status or tenant["status"]
    if status not in TENANT_STATUSES:
        raise HTTPException(status_code=422, detail=f"status 只能是 {TENANT_STATUSES}")
    isolation = payload.isolation or tenant["isolation"]
    if isolation not in ISOLATION_MODES:
        raise HTTPException(status_code=422, detail=f"isolation 只能是 {ISOLATION_MODES}")
    row = await db.fetchrow(
        """
        UPDATE tenants
        SET name=coalesce($2, name), status=$3, isolation=$4, note=coalesce($5, note), updated_at=now()
        WHERE id=$1
        RETURNING *
        """,
        tenant_id,
        payload.name,
        status,
        isolation,
        payload.note,
    )
    await _audit(db, request, actor, "admin.tenant.update", {"id": tenant_id, "status": status})
    return {"tenant": _row_dict(row)}


# ================================================================== 实验与灰度
@router.get("/api/v1/admin/experiments")
async def admin_experiments(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch("SELECT * FROM experiments ORDER BY created_at DESC")
    assignments = await db.fetch(
        "SELECT experiment_key, variant, count(*) AS count FROM experiment_assignments GROUP BY 1, 2"
    )
    by_key: dict[str, dict[str, int]] = {}
    for row in assignments:
        by_key.setdefault(row["experiment_key"], {})[row["variant"]] = int(row["count"])
    return {
        "items": [
            {
                "id": row["id"],
                "key": row["key"],
                "name": row["name"],
                "status": row["status"],
                "rollout": int(row["rollout"]),
                "variants": _load(row["variants"]) or [],
                "note": row["note"],
                "started_at": _iso(row["started_at"]),
                "created_at": _iso(row["created_at"]),
                "assignments": by_key.get(row["key"], {}),
            }
            for row in rows
        ],
        "statuses": list(EXPERIMENT_STATUSES),
    }


@router.post("/api/v1/admin/experiments")
async def admin_experiment_create(
    payload: ExperimentCreate,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    key = payload.key.strip().lower()
    if not re.fullmatch(r"[a-z0-9_-]{2,40}", key):
        raise HTTPException(status_code=422, detail="key 只允许小写字母、数字、下划线、短横线（2-40 位）")
    variants = [item.strip() for item in payload.variants if item.strip()]
    if len(variants) < 2:
        raise HTTPException(status_code=422, detail="至少要两个实验组")
    exp_id = "exp_" + uuid.uuid4().hex[:10]
    try:
        row = await db.fetchrow(
            """
            INSERT INTO experiments (id, key, name, status, rollout, variants, note)
            VALUES ($1, $2, $3, 'draft', 0, $4::jsonb, $5)
            RETURNING *
            """,
            exp_id,
            key,
            payload.name.strip(),
            json.dumps(variants, ensure_ascii=False),
            payload.note,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"实验 key 已存在或不合法：{exc}") from exc
    await _audit(db, request, actor, "admin.experiment.create", {"key": key, "variants": variants})
    return {"experiment": _row_dict(row)}


@router.patch("/api/v1/admin/experiments/{key}")
async def admin_experiment_update(
    key: str,
    payload: ExperimentUpdate,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    experiment = await db.fetchrow("SELECT * FROM experiments WHERE key=$1", key)
    if experiment is None:
        raise HTTPException(status_code=404, detail="实验不存在")

    status = payload.status or experiment["status"]
    if status not in EXPERIMENT_STATUSES:
        raise HTTPException(status_code=422, detail=f"status 只能是 {EXPERIMENT_STATUSES}")
    rollout = experiment["rollout"] if payload.rollout is None else payload.rollout
    if not 0 <= int(rollout) <= 100:
        raise HTTPException(status_code=422, detail="rollout 必须在 0-100 之间")
    if status == "running" and int(rollout) <= 0:
        raise HTTPException(status_code=422, detail="运行时灰度比例必须大于 0")
    variants = _load(experiment["variants"]) or []
    if payload.variants is not None:
        variants = [item.strip() for item in payload.variants if item.strip()]
        if len(variants) < 2:
            raise HTTPException(status_code=422, detail="至少要两个实验组")

    started_at_sql = "coalesce(started_at, now())" if status == "running" else "started_at"
    row = await db.fetchrow(
        f"""
        UPDATE experiments
        SET name=coalesce($2, name), status=$3, rollout=$4, variants=$5::jsonb,
            note=coalesce($6, note), started_at={started_at_sql}, updated_at=now()
        WHERE key=$1
        RETURNING *
        """,
        key,
        payload.name,
        status,
        int(rollout),
        json.dumps(variants, ensure_ascii=False),
        payload.note,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.experiment.update",
        {"key": key, "status": status, "rollout": int(rollout)},
    )
    return {"experiment": _row_dict(row)}


@router.get("/api/v1/admin/experiments/{key}/assignments")
async def admin_experiment_assignments(
    key: str,
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    totals = await db.fetch(
        "SELECT variant, count(*) AS count FROM experiment_assignments WHERE experiment_key=$1 GROUP BY 1",
        key,
    )
    rows = await db.fetch(
        """
        SELECT subject_id, variant, created_at FROM experiment_assignments
        WHERE experiment_key=$1 ORDER BY id DESC LIMIT $2
        """,
        key,
        limit,
    )
    return {
        "by_variant": [{"variant": row["variant"], "count": int(row["count"])} for row in totals],
        "recent": [
            {"subject_id": row["subject_id"][:8], "variant": row["variant"], "created_at": _iso(row["created_at"])}
            for row in rows
        ],
    }


# ================================================================== 数据流 / 状态规范
@router.get("/api/v1/admin/flow")
async def admin_flow(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    stages = [
        ("知识库文档", await _count(db, "SELECT count(*) FROM documents")),
        ("问诊会话", await _count(db, "SELECT count(*) FROM sessions")),
        ("用户提问", await _count(db, "SELECT count(*) FROM messages WHERE role='user'")),
        ("助手回答", await _count(db, "SELECT count(*) FROM messages WHERE role='assistant'")),
        ("报告解读", await _count(db, "SELECT count(*) FROM reports")),
        ("模型调用", await _count(db, "SELECT count(*) FROM model_calls")),
        ("审计记录", await _count(db, "SELECT count(*) FROM audit_log")),
        ("未命中问题", await _count(db, "SELECT count(*) FROM unmatched_questions")),
    ]
    return {"stages": [{"name": name, "count": count} for name, count in stages]}


@router.get("/api/v1/admin/states")
async def admin_states(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ready(request)
    machines = [
        ("预约 / 订单", "bookings", "status"),
        ("HIS 订单", "mcp_orders", "status"),
        ("支付状态", "mcp_orders", "payment_status"),
        ("取药进度", "dispensing_progress", "status"),
        ("排队叫号", "queue_tickets", "status"),
        ("待审核请求", "review_requests", "status"),
        ("随访", "followups", "status"),
        ("实验灰度", "experiments", "status"),
    ]
    result = []
    for label, table, column in machines:
        try:
            rows = await db.fetch(
                f"SELECT coalesce({column}, '未设置') AS value, count(*) AS count FROM {table} GROUP BY 1 ORDER BY 2 DESC"
            )
            result.append(
                {
                    "label": label,
                    "table": table,
                    "column": column,
                    "values": [{"value": row["value"], "count": int(row["count"])} for row in rows],
                }
            )
        except Exception as exc:
            result.append({"label": label, "table": table, "column": column, "values": [], "error": str(exc)})
    return {"items": result}

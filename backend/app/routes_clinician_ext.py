"""医生端 Wave C 收尾 API：文书打印、诊毕与随访计划、审方记录（只读聚合）。

三条原则与前面一致：
- 全部 require_clinician，写操作落 audit_log；
- 打印正文由**服务端渲染并存库**，页面只负责展示与触发浏览器打印，避免"打印按钮没实质"；
- 审方只做平台侧记录聚合，HIS 审方记录标注「待院方开通」，不伪造。
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from .api.deps import require_clinician
from .schema_clinician_ext import PRINT_TEMPLATES, ensure_clinician_ext_schema

router = APIRouter()


class PrintRequest(BaseModel):
    template: str
    note_id: str | None = None
    member_id: str | None = None


class CloseVisitRequest(BaseModel):
    advice: str | None = None
    followup_days: int = 7


class ReceiveRequest(BaseModel):
    member_id: str
    note: str | None = None


def _state(request: Request) -> Any:
    return request.app.state


def _actor(user: dict[str, Any]) -> str:
    return str(user.get("sub") or user.get("username") or "unknown")


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


async def _ready(request: Request) -> Any:
    db = _state(request).db
    await ensure_clinician_ext_schema(db)
    return db


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


async def _require_note(db: Any, note_id: str) -> Any:
    note = await db.fetchrow("SELECT * FROM clinical_notes WHERE id=$1", note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="病历不存在")
    return note


async def _member_exists(db: Any, member_id: str) -> None:
    exists = await db.fetchval("SELECT 1 FROM family_members WHERE id=$1", member_id)
    if not exists:
        raise HTTPException(status_code=404, detail="就诊人不存在")


# ------------------------------------------------------------------ 打印
@router.get("/api/v1/clinician/print/templates")
async def clinician_print_templates(
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    await _ready(request)
    return {
        "items": [
            {"key": key, "name": meta["name"], "description": meta["description"]}
            for key, meta in PRINT_TEMPLATES.items()
        ]
    }


@router.post("/api/v1/clinician/print")
async def clinician_print_create(
    payload: PrintRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    template = payload.template
    if template not in PRINT_TEMPLATES:
        raise HTTPException(status_code=422, detail=f"未知模板：{template}（可选 {sorted(PRINT_TEMPLATES)}）")

    note = await _require_note(db, payload.note_id) if payload.note_id else None
    member_id = payload.member_id or (note["member_id"] if note else None)
    if not member_id:
        raise HTTPException(status_code=422, detail="需要 note_id 或 member_id")
    await _member_exists(db, member_id)

    member = await db.fetchrow(
        "SELECT id, name, gender, birth_date FROM family_members WHERE id=$1", member_id
    )
    orders = await db.fetch(
        """
        SELECT kind, name, dose, usage, status, signed_at FROM clinical_orders
        WHERE member_id=$1 AND status='signed' ORDER BY created_at
        """,
        member_id,
    )

    header = [
        "大医 · 智能医院平台",
        "模板：" + PRINT_TEMPLATES[template]["name"],
        "就诊人：" + (member["name"] if member else member_id),
        "医生：" + actor,
        "生成时间：" + datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M"),
        "-" * 36,
    ]

    if template == "outpatient_note":
        if note is None:
            raise HTTPException(status_code=422, detail="门诊病历必须指定 note_id")
        body = [
            "主诉：" + (note["chief_complaint"] or "未记录"),
            "现病史：" + (note["present_illness"] or "未记录"),
            "查体：" + (note["physical_exam"] or "未记录"),
            "诊断：" + (note["diagnosis"] or "未记录"),
            "状态：" + ("已诊毕" if note["status"] == "completed" else "草稿（未诊毕）"),
        ]
        if note["status"] != "completed":
            body.append("※ 本件为草稿打印，未经诊毕，不可作为正式病历使用")
    elif template == "prescription":
        meds = [row for row in orders if row["kind"] == "medication"]
        body = ["药品医嘱（仅已签名）："] + (
            [f"{index}. {row['name']} {row['dose'] or ''} {row['usage'] or ''}".strip() for index, row in enumerate(meds, 1)]
            or ["（没有已签名的药品医嘱）"]
        )
    else:
        exams = [row for row in orders if row["kind"] in {"exam", "lab"}]
        body = ["检查 / 检验申请（仅已签名）："] + (
            [
                f"{index}. [{'检查' if row['kind'] == 'exam' else '检验'}] {row['name']} {row['dose'] or ''}".strip()
                for index, row in enumerate(exams, 1)
            ]
            or ["（没有已签名的检查 / 检验医嘱）"]
        )

    content = "\n".join(header + body)
    job_id = "pj_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO print_jobs (id, template, note_id, member_id, doctor, content)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, template, note_id, member_id, doctor, status, created_at
        """,
        job_id,
        template,
        payload.note_id,
        member_id,
        actor,
        content,
    )
    await _audit(
        db,
        request,
        actor,
        "clinician.print.create",
        {"job_id": job_id, "template": template, "member_id": member_id},
    )
    return {"job": {**dict(row), "created_at": _iso(row["created_at"])}, "content": content}


@router.get("/api/v1/clinician/print")
async def clinician_print_list(
    request: Request,
    member_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT id, template, note_id, member_id, doctor, status, created_at, printed_at
        FROM print_jobs
        WHERE ($1::text IS NULL OR member_id=$1)
        ORDER BY created_at DESC LIMIT $2
        """,
        member_id,
        limit,
    )
    return {
        "items": [
            {
                "id": row["id"],
                "template": row["template"],
                "template_name": PRINT_TEMPLATES.get(row["template"], {}).get("name", row["template"]),
                "note_id": row["note_id"],
                "member_id": row["member_id"],
                "doctor": row["doctor"],
                "status": row["status"],
                "created_at": _iso(row["created_at"]),
                "printed_at": _iso(row["printed_at"]),
            }
            for row in rows
        ]
    }


@router.get("/api/v1/clinician/print/{job_id}")
async def clinician_print_detail(
    job_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    row = await db.fetchrow("SELECT * FROM print_jobs WHERE id=$1", job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="打印任务不存在")
    return {
        "job": {
            "id": row["id"],
            "template": row["template"],
            "member_id": row["member_id"],
            "doctor": row["doctor"],
            "status": row["status"],
            "created_at": _iso(row["created_at"]),
            "printed_at": _iso(row["printed_at"]),
        },
        "content": row["content"],
    }


@router.post("/api/v1/clinician/print/{job_id}/mark-printed")
async def clinician_print_mark(
    job_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    row = await db.fetchrow(
        """
        UPDATE print_jobs SET status='printed', printed_at=now()
        WHERE id=$1 RETURNING id, status, printed_at
        """,
        job_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="打印任务不存在")
    await _audit(db, request, actor, "clinician.print.marked", {"job_id": job_id})
    return {"job": {"id": row["id"], "status": row["status"], "printed_at": _iso(row["printed_at"])}}


# ------------------------------------------------------------------ 接诊
@router.post("/api/v1/clinician/receive")
async def clinician_receive(
    payload: ReceiveRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    """接诊：平台侧记录"在诊"状态。

    HIS encounter 写入待院方开通，所以这里 source='platform'、his_encounter_id 为空，
    页面必须标注"平台数据 · HIS 待对接"。
    """
    db = await _ready(request)
    actor = _actor(user)
    await _member_exists(db, payload.member_id)
    existing = await db.fetchrow(
        "SELECT id FROM visit_encounters WHERE member_id=$1 AND status='in_progress'",
        payload.member_id,
    )
    if existing is not None:
        raise HTTPException(status_code=409, detail="该就诊人已在诊，不能重复接诊")

    encounter_id = "enc_" + uuid.uuid4().hex[:12]
    row = await db.fetchrow(
        """
        INSERT INTO visit_encounters (id, member_id, doctor, status, source, note)
        VALUES ($1, $2, $3, 'in_progress', 'platform', $4)
        RETURNING *
        """,
        encounter_id,
        payload.member_id,
        actor,
        payload.note,
    )
    await _audit(
        db,
        request,
        actor,
        "clinician.encounter.received",
        {"encounter_id": encounter_id, "member_id": payload.member_id},
    )
    return {
        "encounter": {
            "id": row["id"],
            "member_id": row["member_id"],
            "doctor": row["doctor"],
            "status": row["status"],
            "source": row["source"],
            "his_encounter_id": row["his_encounter_id"],
            "started_at": _iso(row["started_at"]),
        },
        "his_note": "HIS encounter 写入待院方开通；当前为平台侧接诊状态",
    }


@router.get("/api/v1/clinician/encounters")
async def clinician_encounters(
    request: Request,
    member_id: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT e.*, m.name AS member_name FROM visit_encounters e
        LEFT JOIN family_members m ON m.id = e.member_id
        WHERE ($1::text IS NULL OR e.member_id=$1)
        ORDER BY e.started_at DESC LIMIT $2
        """,
        member_id,
        limit,
    )
    return {
        "items": [
            {
                "id": row["id"],
                "member_id": row["member_id"],
                "member_name": row["member_name"],
                "doctor": row["doctor"],
                "status": row["status"],
                "source": row["source"],
                "his_encounter_id": row["his_encounter_id"],
                "started_at": _iso(row["started_at"]),
                "ended_at": _iso(row["ended_at"]),
            }
            for row in rows
        ]
    }


@router.post("/api/v1/clinician/encounters/{encounter_id}/complete")
async def clinician_encounter_complete(
    encounter_id: str,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    row = await db.fetchrow(
        """
        UPDATE visit_encounters SET status='completed', ended_at=now()
        WHERE id=$1 AND status='in_progress' RETURNING *
        """,
        encounter_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="接诊记录不存在或已结束")
    await _audit(db, request, actor, "clinician.encounter.completed", {"encounter_id": encounter_id})
    return {
        "encounter": {
            "id": row["id"],
            "member_id": row["member_id"],
            "status": row["status"],
            "ended_at": _iso(row["ended_at"]),
        }
    }


# ------------------------------------------------------------------ 诊毕 / 随访
@router.post("/api/v1/clinician/notes/{note_id}/close-visit")
async def clinician_close_visit(
    note_id: str,
    payload: CloseVisitRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    actor = _actor(user)
    if not 1 <= payload.followup_days <= 180:
        raise HTTPException(status_code=422, detail="随访天数必须在 1-180 之间")

    note = await _require_note(db, note_id)
    if note["status"] == "completed":
        raise HTTPException(status_code=409, detail="该病历已诊毕，不能重复诊毕")

    closed = await db.fetchrow(
        """
        UPDATE clinical_notes SET status='completed', updated_at=now()
        WHERE id=$1 AND status<>'completed' RETURNING *
        """,
        note_id,
    )
    if closed is None:
        raise HTTPException(status_code=409, detail="诊毕状态更新失败，请刷新后重试")

    member_id = closed["member_id"]
    await _member_exists(db, member_id)
    owner = await db.fetchval("SELECT user_id FROM family_members WHERE id=$1", member_id) or "unknown"

    followup_id = "fu_" + uuid.uuid4().hex[:12]
    due_date = (datetime.now(UTC) + timedelta(days=payload.followup_days)).date()
    followup = await db.fetchrow(
        """
        INSERT INTO followups (id, user_id, member_id, title, due_date, status, notes)
        VALUES ($1, $2, $3, $4, $5, 'pending', $6)
        RETURNING *
        """,
        followup_id,
        owner,
        member_id,
        "诊后随访（" + (closed["diagnosis"] or "通用") + "）",
        due_date,
        payload.advice,
    )

    closure_id = "vc_" + uuid.uuid4().hex[:12]
    closure = await db.fetchrow(
        """
        INSERT INTO visit_closures (id, note_id, member_id, doctor, followup_id, advice, followup_days)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING *
        """,
        closure_id,
        note_id,
        member_id,
        actor,
        followup_id,
        payload.advice,
        payload.followup_days,
    )
    await _audit(
        db,
        request,
        actor,
        "clinician.visit.closed",
        {"note_id": note_id, "member_id": member_id, "followup_id": followup_id, "days": payload.followup_days},
    )
    return {
        "note": {**dict(closed), "created_at": _iso(closed["created_at"]), "updated_at": _iso(closed["updated_at"])},
        "followup": {
            "id": followup["id"],
            "member_id": followup["member_id"],
            "title": followup["title"],
            "due_date": str(followup["due_date"]),
            "status": followup["status"],
            "notes": followup["notes"],
        },
        "closure": {
            "id": closure["id"],
            "followup_days": int(closure["followup_days"]),
            "created_at": _iso(closure["created_at"]),
        },
    }


@router.get("/api/v1/clinician/followup-plans")
async def clinician_followup_plans(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    rows = await db.fetch(
        """
        SELECT c.id, c.note_id, c.member_id, c.doctor, c.advice, c.followup_days, c.created_at,
               f.id AS followup_id, f.title, f.due_date, f.status,
               m.name AS member_name
        FROM visit_closures c
        LEFT JOIN followups f ON f.id = c.followup_id
        LEFT JOIN family_members m ON m.id = c.member_id
        ORDER BY c.created_at DESC LIMIT $1
        """,
        limit,
    )
    return {
        "items": [
            {
                "id": row["id"],
                "note_id": row["note_id"],
                "member_id": row["member_id"],
                "member_name": row["member_name"],
                "doctor": row["doctor"],
                "advice": row["advice"],
                "followup_days": int(row["followup_days"] or 0),
                "followup_id": row["followup_id"],
                "title": row["title"],
                "due_date": str(row["due_date"]) if row["due_date"] else None,
                "status": row["status"],
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ]
    }


# ------------------------------------------------------------------ 审方（只读聚合）
@router.get("/api/v1/clinician/prescription-reviews")
async def clinician_prescription_reviews(
    request: Request,
    review_status: str | None = Query(None),
    limit: int = Query(30, ge=1, le=100),
    user: dict[str, Any] = Depends(require_clinician),
) -> dict[str, Any]:
    db = await _ready(request)
    if review_status not in {None, "pending", "approved", "rejected"}:
        raise HTTPException(status_code=422, detail="review_status 只能是 pending / approved / rejected")

    rows = await db.fetch(
        """
        SELECT o.id, o.member_id, o.kind, o.name, o.dose, o.usage, o.doctor,
               o.status, o.review_status, o.review_note, o.signed_at,
               m.name AS member_name
        FROM clinical_orders o
        LEFT JOIN family_members m ON m.id = o.member_id
        WHERE o.status='signed' AND ($1::text IS NULL OR o.review_status=$1)
        ORDER BY o.signed_at DESC NULLS LAST
        LIMIT $2
        """,
        review_status,
        limit,
    )
    counts = await db.fetch(
        """
        SELECT review_status, count(*) AS count FROM clinical_orders
        WHERE status='signed' GROUP BY 1
        """
    )
    return {
        "items": [
            {
                "id": row["id"],
                "member_id": row["member_id"],
                "member_name": row["member_name"],
                "kind": row["kind"],
                "name": row["name"],
                "dose": row["dose"],
                "usage": row["usage"],
                "doctor": row["doctor"],
                "review_status": row["review_status"],
                "review_note": row["review_note"],
                "signed_at": _iso(row["signed_at"]),
            }
            for row in rows
        ],
        "summary": {row["review_status"]: int(row["count"]) for row in counts},
        "his_note": "HIS 药师审方记录待院方开通；当前为平台侧审方状态（签名后产生）",
    }

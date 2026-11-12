"""右栏 Agent 面板：能力、最近调用、边界。

数据都是真的：
- 能力来自角色能力矩阵（静态注册，和剧本/验收同一套口径）；
- 最近调用读 model_calls（近 7 天，带延迟）；
- 边界说明是当前实现的诚实边界，不做营销话术。
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from .api.deps import get_current_user
from .routes import _state

router = APIRouter(tags=["agent"])

END_BY_ROLE = {
    "admin": "admin",
    "doctor": "clinician",
    "clinician": "clinician",
    "ops": "ops",
    "patient": "patient",
}

CAPABILITIES: dict[str, list[dict[str, str]]] = {
    "patient": [
        {"name": "chat.triage", "note": "分诊与就医方向"},
        {"name": "report.interpret", "note": "报告解读"},
        {"name": "medication.check", "note": "用药与相互作用"},
        {"name": "appointment.book", "note": "挂号号源"},
        {"name": "knowledge.search", "note": "指南检索"},
    ],
    "clinician": [
        {"name": "patient.360", "note": "跨端患者 360"},
        {"name": "note.draft", "note": "病历草稿"},
        {"name": "order.sign", "note": "医嘱签名（默认关闭）"},
        {"name": "draft.print", "note": "文书打印"},
        {"name": "ddi.check", "note": "相互作用"},
    ],
    "ops": [
        {"name": "reconcile.run", "note": "对账"},
        {"name": "payment.review", "note": "支付异常审核"},
        {"name": "handoff.claim", "note": "人工接管"},
        {"name": "audit.search", "note": "审计检索"},
        {"name": "analytics.summary", "note": "运营分析"},
    ],
    "admin": [
        {"name": "runtime.status", "note": "运行自检"},
        {"name": "rag.inspect", "note": "检索调试"},
        {"name": "config.diff", "note": "配置草稿/发布"},
        {"name": "trace.replay", "note": "会话回放"},
        {"name": "reindex.tsv", "note": "全文索引重建"},
    ],
}

BOUNDARIES: dict[str, list[str]] = {
    "patient": [
        "只读取本人与家庭成员的医疗数据",
        "不做诊断结论，只给分诊与就医方向",
        "所有回答标注证据来源与生成时间",
    ],
    "clinician": [
        "临床写入落到草稿，签名后生效（需 HIS 工号）",
        "不跨患者查询；360 视图只展示本院就诊记录",
        "签名与真实 HIS 写入默认关闭，需院方开通",
    ],
    "ops": [
        "只处理支付、对账、接管与审计，不改临床内容",
        "所有操作留痕，危险操作需二次确认",
        "实时医保结算待院方开通",
    ],
    "admin": [
        "仅运维网段可登录（部署侧控制）",
        "配置发布需审批，可回滚，全程审计",
        "默认看不到患者明细",
    ],
}

STARTER = {
    "patient": "问：这个报告里血脂偏高要注意什么？",
    "clinician": "问：这个患者今天要警惕什么？",
    "ops": "问：今天有哪些支付异常需要处理？",
    "admin": "问：今天运行态有什么告警？",
}


@router.get("/api/v1/agent/panel")
async def agent_panel(
    request: Request,
    end: str | None = Query(None, description="patient | clinician | ops | platform"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    db = _state(request).db
    role = str(user.get("role"))
    role_end = END_BY_ROLE.get(role, "patient")

    requested = end if end in {"patient", "clinician", "ops", "platform"} else None
    if requested is None:
        public_end = "platform" if role_end == "admin" else role_end
    elif role == "admin":
        public_end = requested
    elif requested == ("platform" if role_end == "admin" else role_end):
        public_end = requested
    else:
        public_end = "platform" if role_end == "admin" else role_end

    table_end = "admin" if public_end == "platform" else public_end

    # 平台端看全租户；其余端按“本账号”过滤，避免把别人的调用显示给患者/医生
    scope = "tenant" if public_end == "platform" else "self"
    actor = None if scope == "tenant" else str(user.get("sub"))

    if actor:
        recent_rows = await db.fetch(
            """
            SELECT model, task, latency_ms, created_at
            FROM model_calls
            WHERE created_at >= now() - interval '7 days' AND user_id = $1
            ORDER BY created_at DESC
            LIMIT 6
            """,
            actor,
        )
    else:
        recent_rows = await db.fetch(
            """
            SELECT model, task, latency_ms, created_at
            FROM model_calls
            WHERE created_at >= now() - interval '7 days'
            ORDER BY created_at DESC
            LIMIT 6
            """
        )
    recent = [
        {
            "name": row["task"] or row["model"],
            "model": row["model"],
            "latency_ms": int(row["latency_ms"] or 0),
            "at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in recent_rows
    ]

    if actor:
        counts = await db.fetchrow(
            """
            SELECT count(*) AS calls, coalesce(avg(latency_ms), 0) AS avg_ms
            FROM model_calls
            WHERE created_at >= now() - interval '7 days' AND user_id = $1
            """,
            actor,
        )
    else:
        counts = await db.fetchrow(
            """
            SELECT count(*) AS calls, coalesce(avg(latency_ms), 0) AS avg_ms
            FROM model_calls
            WHERE created_at >= now() - interval '7 days'
            """
        )

    return {
        "end": public_end,
        "role": role,
        "capabilities": CAPABILITIES[table_end],
        "recent": recent,
        "scope": scope,
        "summary": {
            "calls_7d": int(counts["calls"] or 0),
            "avg_latency_ms": int(counts["avg_ms"] or 0),
        },
        "boundaries": BOUNDARIES[table_end],
        "starter": STARTER[table_end],
    }

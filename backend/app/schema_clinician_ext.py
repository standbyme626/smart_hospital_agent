"""医生端 Wave C 收尾：文书打印、诊毕与随访计划、审方记录。

与 schema_admin*.py 同样的取舍：表定义独立成模块，避免与其他波次争用 db.py。
Wave A 的编号化迁移落地后，整体并入 backend/migrations/。
"""
from __future__ import annotations

from typing import Any

CLINICIAN_EXT_SCHEMA: tuple[str, ...] = (
    # 文书打印记录：模板 + 服务端渲染好的正文（可复核，不依赖浏览器）
    """
    CREATE TABLE IF NOT EXISTS print_jobs (
        id TEXT PRIMARY KEY,
        template TEXT NOT NULL,
        note_id TEXT,
        member_id TEXT NOT NULL,
        doctor TEXT,
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'queued',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        printed_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_print_jobs_member ON print_jobs (member_id, created_at DESC)",
    # 诊毕记录：把病历完结与随访计划绑在一起，留痕可查
    """
    CREATE TABLE IF NOT EXISTS visit_closures (
        id TEXT PRIMARY KEY,
        note_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        doctor TEXT,
        followup_id TEXT,
        advice TEXT,
        followup_days INTEGER,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_visit_closures_member ON visit_closures (member_id, created_at DESC)",
    # 接诊状态投影：平台侧记录"谁在什么时候接诊了谁"；HIS encounter 写入待院方开通
    """
    CREATE TABLE IF NOT EXISTS visit_encounters (
        id TEXT PRIMARY KEY,
        member_id TEXT NOT NULL,
        doctor TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'in_progress',
        source TEXT NOT NULL DEFAULT 'platform',
        his_encounter_id TEXT,
        note TEXT,
        started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        ended_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_visit_encounters_member ON visit_encounters (member_id, started_at DESC)",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_visit_encounters_open ON visit_encounters (member_id) WHERE status='in_progress'",
)

PRINT_TEMPLATES: dict[str, dict[str, str]] = {
    "outpatient_note": {
        "name": "门诊病历",
        "description": "主诉 / 现病史 / 查体 / 诊断，用于病历归档与患者留存",
    },
    "prescription": {
        "name": "处方笺",
        "description": "已签名药品医嘱；未签名医嘱不会出现在处方上",
    },
    "exam_request": {
        "name": "检查申请单",
        "description": "已签名的检查 / 检验项，用于院内执行",
    },
}


async def ensure_clinician_ext_schema(db: Any) -> None:
    for statement in CLINICIAN_EXT_SCHEMA:
        await db.execute_system(statement)
    await db.grant_app_role()

"""管理端 Wave E（第二批）：未命中问题 / 多院区 / 实验灰度 / HIS 连通性检查。

与 schema_admin.py 同样的取舍：表定义独立成模块，避免与其他波次争用 db.py；
Wave A 的编号化迁移落地后，这里整体并入 backend/migrations/。
"""
from __future__ import annotations

from typing import Any

ADMIN_EXT_SCHEMA: tuple[str, ...] = (
    # 未命中问题池：检索命中 0 条时由 RetrievalService 记录（真实流量产生）
    """
    CREATE TABLE IF NOT EXISTS unmatched_questions (
        question_key TEXT PRIMARY KEY,
        question TEXT NOT NULL,
        department TEXT,
        hits INTEGER NOT NULL DEFAULT 1,
        first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_seen TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_unmatched_last ON unmatched_questions (last_seen DESC)",
    # 未命中问题的处理记录：转成知识条目后留痕
    """
    CREATE TABLE IF NOT EXISTS unmatched_resolutions (
        question_key TEXT PRIMARY KEY,
        doc_key TEXT NOT NULL,
        answer TEXT NOT NULL,
        actor TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 多院区（平台侧路由与隔离口径；HIS 原生 tenant 由适配器承接）
    """
    CREATE TABLE IF NOT EXISTS tenants (
        id TEXT PRIMARY KEY,
        code TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        isolation TEXT NOT NULL DEFAULT 'logical',
        note TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    # 实验与灰度
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        key TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        rollout INTEGER NOT NULL DEFAULT 0,
        variants JSONB NOT NULL DEFAULT '[]'::jsonb,
        note TEXT,
        started_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_assignments (
        id BIGSERIAL PRIMARY KEY,
        experiment_key TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        variant TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (experiment_key, subject_id)
    )
    """,
    # HIS 连通性检查记录（每次「测试连接」都留痕）
    """
    CREATE TABLE IF NOT EXISTS his_checks (
        id BIGSERIAL PRIMARY KEY,
        ok BOOLEAN NOT NULL,
        detail TEXT,
        latency_ms INTEGER,
        actor TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
)

TENANT_STATUSES = ("active", "suspended", "archived")
ISOLATION_MODES = ("logical", "physical")
EXPERIMENT_STATUSES = ("draft", "running", "paused", "archived")


async def ensure_admin_ext_schema(db: Any) -> None:
    """幂等建表；只灌「部署事实」级别的默认租户，不灌任何业务假数据。"""
    for statement in ADMIN_EXT_SCHEMA:
        await db.execute_system(statement)
    await db.grant_app_role()
    await db.execute_system(
        """
        INSERT INTO tenants (id, code, name, status, isolation, note)
        VALUES ('tn_default', 'default', '本地 OpenHIS 试点院区', 'active', 'logical',
                '默认租户；HIS 为唯一事实源，平台只做路由与隔离')
        ON CONFLICT (id) DO NOTHING
        """
    )

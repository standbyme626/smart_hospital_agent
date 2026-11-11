"""管理端 Wave E：配置注册表（草稿 → 发布 → 回滚 + 审计）。

设计取舍：
- 规则库 / 通知模板 / Prompt / Agent 配置四屏共用一张 config_entries，
  区别在于 kind 与各 kind 的校验规则；这样版本、发布、回滚、审计只需实现一次。
- 表定义放在独立模块，避免与其他波次争用 db.py 的 SCHEMA 列表；
  Wave A 的编号化迁移落地后，这里整体并入 backend/migrations/。
"""
from __future__ import annotations

import json
from typing import Any

# kind -> (中文名, 说明)
KINDS: dict[str, tuple[str, str]] = {
    "rule": ("规则库", "危机识别 / DDI / 危急值等确定性规则，发布后立即生效"),
    "notify_template": ("通知模板", "站内信与未来短信 / 微信共用的模板，支持变量预览"),
    "prompt": ("Prompt 管理", "Agent 各环节提示词与模型参数"),
    "agent_config": ("Agent 配置", "模型路由、检索参数、安全护栏"),
}

ADMIN_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS config_entries (
        id BIGSERIAL PRIMARY KEY,
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        draft JSONB NOT NULL DEFAULT '{}'::jsonb,
        published JSONB,
        version INTEGER NOT NULL DEFAULT 0,
        enabled BOOLEAN NOT NULL DEFAULT true,
        updated_by TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        published_by TEXT,
        published_at TIMESTAMPTZ,
        UNIQUE (kind, key)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_config_entries_kind ON config_entries (kind, key)",
    """
    CREATE TABLE IF NOT EXISTS config_history (
        id BIGSERIAL PRIMARY KEY,
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        version INTEGER NOT NULL,
        value JSONB NOT NULL,
        action TEXT NOT NULL,
        actor TEXT,
        note TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_config_history_entry ON config_history (kind, key, version DESC)",
)


def _notify(body: str) -> dict[str, Any]:
    return {
        "title": "",
        "body": body,
        "channels": ["inbox"],
        "retry": {"max": 3, "backoff_s": 30},
    }


# (kind, key, draft 初值, 备注)
DEFAULT_ENTRIES: tuple[tuple[str, str, dict[str, Any], str], ...] = (
    # ---- 规则库
    (
        "rule",
        "crisis_red_flags",
        {
            "severity": "critical",
            "keywords": ["胸痛伴大汗", "意识不清", "大出血", "呼吸困难加重"],
            "action": "立即拨打 120 或前往急诊",
            "bypass_model": True,
        },
        "红旗症状关键词：命中即阻断式引导，不等模型输出",
    ),
    (
        "rule",
        "ddi_critical",
        {
            "severity": "high",
            "pairs": [["华法林", "阿司匹林"], ["氯吡格雷", "奥美拉唑"]],
            "note": "命中后必须提示咨询医生，不给出自行停药建议",
        },
        "严重药物相互作用对",
    ),
    (
        "rule",
        "critical_lab_values",
        {
            "severity": "critical",
            "items": {"血钾": [2.8, 6.0], "血红蛋白": [60, 200], "肌钙蛋白": [0, 0.4]},
            "note": "超出区间即标记危急值",
        },
        "危急值区间（[下限, 上限]）",
    ),
    # ---- 通知模板
    (
        "notify_template",
        "report_ready",
        {
            **_notify("{{member}}的{{kind}}报告已解读完成，可在「报告」中查看。"),
            "title": "报告解读完成",
        },
        "报告解读完成通知",
    ),
    (
        "notify_template",
        "followup_due",
        {
            **_notify("{{member}}的随访「{{title}}」计划于{{due}}到期，请及时完成。"),
            "title": "随访计划到期",
        },
        "随访到期提醒",
    ),
    (
        "notify_template",
        "payment_pending",
        {
            **_notify("订单{{order}}待缴费{{amount}}元，请在{{due}}前完成。"),
            "title": "待缴费提醒",
        },
        "待缴费提醒",
    ),
    # ---- Prompt
    (
        "prompt",
        "triage",
        {
            "label": "分诊主提示词",
            "content": "你是医院分诊助手。先识别红旗信号，再按科室目录给出分诊建议；不确诊、不开处方。",
            "model": "default",
            "temperature": 0.2,
        },
        "分诊链路提示词",
    ),
    (
        "prompt",
        "report_interpret",
        {
            "label": "报告解读提示词",
            "content": "逐项解读检验指标，异常项标注参考区间；无证据时明确回答『未覆盖』，不得编造引用。",
            "model": "default",
            "temperature": 0.1,
        },
        "报告解读链路提示词",
    ),
    (
        "prompt",
        "clinician_summary",
        {
            "label": "诊前摘要提示词",
            "content": "基于患者 360 与预问诊生成诊前摘要，每条结论必须可回溯到来源字段。",
            "model": "default",
            "temperature": 0.2,
        },
        "医生端诊前摘要提示词",
    ),
    # ---- Agent 配置
    (
        "agent_config",
        "model_routing",
        {
            "primary": "default",
            "fallback": "default",
            "timeout_s": 30,
            "note": "主模型不可用时降级；不允许静默切到付费模型",
        },
        "模型路由",
    ),
    (
        "agent_config",
        "retrieval",
        {"top_k": 6, "min_score": 0.2, "fallback_without_department": True},
        "检索参数",
    ),
    (
        "agent_config",
        "guardrails",
        {"block_diagnosis": True, "require_citation": True, "max_answer_chars": 4000},
        "安全护栏",
    ),
)


async def ensure_admin_schema(db: Any) -> None:
    """幂等建表 + 灌默认条目（已存在则不动，避免覆盖运营改过的配置）。"""
    for statement in ADMIN_SCHEMA:
        await db.execute_system(statement)
    await db.grant_app_role()
    for kind, key, value, _note in DEFAULT_ENTRIES:
        await db.execute_system(
            """
            INSERT INTO config_entries (kind, key, draft)
            VALUES ($1, $2, $3::jsonb)
            ON CONFLICT (tenant_id, kind, key) DO NOTHING
            """,
            kind,
            key,
            json.dumps(value, ensure_ascii=False),
        )

"""管理端 Wave E：配置注册表 API（规则库 / 通知模板 / Prompt / Agent 配置）。

四屏共用一套语义，页面不需要各写一套版本逻辑：
- PUT  保存草稿：只改 draft，线上生效值不动
- 发布  version+1，published=draft，写 config_history + audit_log
- 回滚  以历史版本的值重新发布成新版本号，时间线不丢
- 停用  enabled=false 后 effective 为空，页面显示「未启用」而不是白屏

权限：全部 require_admin（patient / doctor / ops 一律 403）。
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from .api.deps import require_admin
from .schema_admin import KINDS, ensure_admin_schema

router = APIRouter()

KEY_RE = re.compile(r"^[a-z0-9_]{2,40}$")
VAR_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


# ------------------------------------------------------------------ 请求体
class DraftRequest(BaseModel):
    value: dict[str, Any]
    note: str | None = None


class PublishRequest(BaseModel):
    note: str | None = None


class RollbackRequest(BaseModel):
    version: int
    note: str | None = None


class EnabledRequest(BaseModel):
    enabled: bool


class PreviewRequest(BaseModel):
    vars: dict[str, str] = {}


# ------------------------------------------------------------------ 工具
def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _load(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw


def _entry_json(row: Any) -> dict[str, Any]:
    draft = _load(row["draft"]) or {}
    published = _load(row["published"])
    enabled = bool(row["enabled"])
    return {
        "kind": row["kind"],
        "key": row["key"],
        "draft": draft,
        "published": published,
        "version": int(row["version"]),
        "enabled": enabled,
        "effective": published if (enabled and published) else None,
        "dirty": draft != (published or {}),
        "updated_by": row["updated_by"],
        "updated_at": _iso(row["updated_at"]),
        "published_by": row["published_by"],
        "published_at": _iso(row["published_at"]),
    }


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise HTTPException(status_code=404, detail=f"未知的配置类型：{kind}")


def _check_key(key: str) -> None:
    if not KEY_RE.match(key):
        raise HTTPException(status_code=422, detail="key 只允许小写字母、数字与下划线（2-40 位）")


def _validate(kind: str, value: dict[str, Any]) -> None:
    if not isinstance(value, dict) or not value:
        raise HTTPException(status_code=422, detail="配置内容必须是非空 JSON 对象")
    if kind == "rule":
        if not any(field in value for field in ("severity", "keywords", "items", "pairs")):
            raise HTTPException(status_code=422, detail="规则至少需要 severity / keywords / items / pairs 之一")
    elif kind == "notify_template":
        for field in ("title", "body"):
            text = value.get(field)
            if not isinstance(text, str) or not text.strip():
                raise HTTPException(status_code=422, detail=f"通知模板缺少必填字段：{field}")
        channels = value.get("channels")
        if channels is not None and (not isinstance(channels, list) or not channels):
            raise HTTPException(status_code=422, detail="channels 必须是非空数组")
    elif kind == "prompt":
        content = value.get("content")
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(status_code=422, detail="Prompt 缺少 content")


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


async def _fetch_entry(db: Any, kind: str, key: str) -> Any:
    row = await db.fetchrow("SELECT * FROM config_entries WHERE kind=$1 AND key=$2", kind, key)
    if row is None:
        raise HTTPException(status_code=404, detail=f"配置项不存在：{kind}/{key}")
    return row


async def _ensure_ready(request: Request) -> Any:
    db = request.app.state.db
    await ensure_admin_schema(db)  # 幂等；默认条目已存在则跳过
    return db


# ------------------------------------------------------------------ 端点
@router.get("/api/v1/admin/config/kinds")
async def config_kinds(
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ensure_ready(request)
    rows = await db.fetch(
        """
        SELECT kind,
               count(*) AS total,
               count(*) FILTER (WHERE published IS NOT NULL) AS published,
               count(*) FILTER (WHERE NOT enabled) AS disabled
        FROM config_entries
        GROUP BY kind
        """
    )
    stats = {row["kind"]: row for row in rows}
    items = []
    for kind, (name, description) in KINDS.items():
        row = stats.get(kind)
        items.append(
            {
                "kind": kind,
                "name": name,
                "description": description,
                "total": int(row["total"]) if row else 0,
                "published": int(row["published"]) if row else 0,
                "disabled": int(row["disabled"]) if row else 0,
            }
        )
    return {"items": items}


@router.get("/api/v1/admin/config/changes")
async def config_changes(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ensure_ready(request)
    rows = await db.fetch(
        """
        SELECT kind, key, version, action, actor, note, created_at
        FROM config_history
        ORDER BY id DESC
        LIMIT $1
        """,
        limit,
    )
    return {
        "items": [
            {
                "kind": row["kind"],
                "kind_name": KINDS.get(row["kind"], ("未知", ""))[0],
                "key": row["key"],
                "version": int(row["version"]),
                "action": row["action"],
                "actor": row["actor"],
                "note": row["note"],
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ]
    }


@router.get("/api/v1/admin/config/{kind}")
async def config_list(
    kind: str,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    rows = await db.fetch(
        "SELECT * FROM config_entries WHERE kind=$1 ORDER BY key",
        kind,
    )
    name, description = KINDS[kind]
    return {
        "kind": kind,
        "name": name,
        "description": description,
        "items": [_entry_json(row) for row in rows],
    }


@router.get("/api/v1/admin/config/{kind}/{key}")
async def config_get(
    kind: str,
    key: str,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    return {"entry": _entry_json(await _fetch_entry(db, kind, key))}


@router.put("/api/v1/admin/config/{kind}/{key}")
async def config_save_draft(
    kind: str,
    key: str,
    payload: DraftRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    _check_key(key)
    _validate(kind, payload.value)
    db = await _ensure_ready(request)
    actor = str(user.get("sub") or user.get("username") or "unknown")
    row = await db.fetchrow(
        """
        INSERT INTO config_entries (kind, key, draft, updated_by, updated_at)
        VALUES ($1, $2, $3::jsonb, $4, now())
        ON CONFLICT (kind, key) DO UPDATE
            SET draft=$3::jsonb, updated_by=$4, updated_at=now()
        RETURNING *
        """,
        kind,
        key,
        json.dumps(payload.value, ensure_ascii=False),
        actor,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.config.draft",
        {"kind": kind, "key": key, "note": payload.note},
    )
    return {"entry": _entry_json(row)}


@router.post("/api/v1/admin/config/{kind}/{key}/publish")
async def config_publish(
    kind: str,
    key: str,
    payload: PublishRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    entry = await _fetch_entry(db, kind, key)
    actor = str(user.get("sub") or user.get("username") or "unknown")

    draft = _load(entry["draft"]) or {}
    published = _load(entry["published"])
    if published is not None and draft == published:
        raise HTTPException(status_code=409, detail="草稿与已发布内容一致，无需重复发布")
    _validate(kind, draft)

    version = int(entry["version"]) + 1
    row = await db.fetchrow(
        """
        UPDATE config_entries
        SET published=$3::jsonb, version=$4, published_by=$5, published_at=now(),
            updated_by=$5, updated_at=now()
        WHERE kind=$1 AND key=$2
        RETURNING *
        """,
        kind,
        key,
        json.dumps(draft, ensure_ascii=False),
        version,
        actor,
    )
    await db.execute(
        """
        INSERT INTO config_history (kind, key, version, value, action, actor, note)
        VALUES ($1, $2, $3, $4::jsonb, 'publish', $5, $6)
        """,
        kind,
        key,
        version,
        json.dumps(draft, ensure_ascii=False),
        actor,
        payload.note,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.config.publish",
        {"kind": kind, "key": key, "version": version, "note": payload.note},
    )
    return {"entry": _entry_json(row)}


@router.post("/api/v1/admin/config/{kind}/{key}/rollback")
async def config_rollback(
    kind: str,
    key: str,
    payload: RollbackRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    entry = await _fetch_entry(db, kind, key)
    actor = str(user.get("sub") or user.get("username") or "unknown")

    history = await db.fetchrow(
        "SELECT value FROM config_history WHERE kind=$1 AND key=$2 AND version=$3",
        kind,
        key,
        payload.version,
    )
    if history is None:
        raise HTTPException(status_code=404, detail=f"历史版本不存在：v{payload.version}")

    value = _load(history["value"]) or {}
    _validate(kind, value)
    version = int(entry["version"]) + 1
    row = await db.fetchrow(
        """
        UPDATE config_entries
        SET draft=$3::jsonb, published=$3::jsonb, version=$4,
            published_by=$5, published_at=now(), updated_by=$5, updated_at=now()
        WHERE kind=$1 AND key=$2
        RETURNING *
        """,
        kind,
        key,
        json.dumps(value, ensure_ascii=False),
        version,
        actor,
    )
    note = payload.note or f"回滚自 v{payload.version}"
    await db.execute(
        """
        INSERT INTO config_history (kind, key, version, value, action, actor, note)
        VALUES ($1, $2, $3, $4::jsonb, 'rollback', $5, $6)
        """,
        kind,
        key,
        version,
        json.dumps(value, ensure_ascii=False),
        actor,
        note,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.config.rollback",
        {"kind": kind, "key": key, "from_version": payload.version, "version": version},
    )
    return {"entry": _entry_json(row)}


@router.post("/api/v1/admin/config/{kind}/{key}/enabled")
async def config_set_enabled(
    kind: str,
    key: str,
    payload: EnabledRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    await _fetch_entry(db, kind, key)
    actor = str(user.get("sub") or user.get("username") or "unknown")
    row = await db.fetchrow(
        """
        UPDATE config_entries
        SET enabled=$3, updated_by=$4, updated_at=now()
        WHERE kind=$1 AND key=$2
        RETURNING *
        """,
        kind,
        key,
        payload.enabled,
        actor,
    )
    await _audit(
        db,
        request,
        actor,
        "admin.config.enabled",
        {"kind": kind, "key": key, "enabled": payload.enabled},
    )
    return {"entry": _entry_json(row)}


@router.get("/api/v1/admin/config/{kind}/{key}/history")
async def config_history(
    kind: str,
    key: str,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    _check_kind(kind)
    db = await _ensure_ready(request)
    rows = await db.fetch(
        """
        SELECT version, value, action, actor, note, created_at
        FROM config_history
        WHERE kind=$1 AND key=$2
        ORDER BY version DESC, id DESC
        """,
        kind,
        key,
    )
    return {
        "items": [
            {
                "version": int(row["version"]),
                "value": _load(row["value"]) or {},
                "action": row["action"],
                "actor": row["actor"],
                "note": row["note"],
                "created_at": _iso(row["created_at"]),
            }
            for row in rows
        ]
    }


@router.post("/api/v1/admin/config/notify_template/{key}/preview")
async def notify_preview(
    key: str,
    payload: PreviewRequest,
    request: Request,
    user: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    db = await _ensure_ready(request)
    entry = await db.fetchrow(
        "SELECT * FROM config_entries WHERE kind='notify_template' AND key=$1", key
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"通知模板不存在：{key}")

    template = _load(entry["published"]) or _load(entry["draft"]) or {}
    vars = payload.vars or {}
    missing: set[str] = set()

    def render(text: Any) -> str:
        if not isinstance(text, str):
            return ""

        def substitute(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in vars:
                return str(vars[name])
            missing.add(name)
            return match.group(0)

        return VAR_RE.sub(substitute, text)

    return {
        "template_key": key,
        "title": render(template.get("title")),
        "body": render(template.get("body")),
        "channels": template.get("channels", []),
        "retry": template.get("retry", {}),
        "missing_vars": sorted(missing),
        "source": "published" if entry["published"] is not None else "draft",
    }

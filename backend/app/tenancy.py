"""多租户（Wave G）：租户上下文 + 归属解析 + 请求租户读取。

设计：
- 每次请求由 %load_user% 解析出"当前租户"，写入 ContextVar；SQL helper 用 %tenant()% 取。
- 默认租户 %tn_default%：存量数据在迁移时统一回填，行为与单租户一致。
- 管理员可用 %X-Tenant% 头临时切到其它**已启用**租户（跨院区运维）；其它角色只能访问自己账号所属租户。
- 跨租户访问一律按 404 处理（不泄露"存在但无权"）。
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException

DEFAULT_TENANT = "tn_default"
TENANT_HEADER = "x-tenant"

_current: ContextVar[str] = ContextVar("tenant_id", default=DEFAULT_TENANT)
_enforce: ContextVar[bool] = ContextVar("tenant_enforce", default=True)


def set_enforce(value: bool):
    """脚本/数据重置需要全库权限时关掉隔离（应用进程始终保持开启）。"""
    return _enforce.set(bool(value))


def enforcing() -> bool:
    return _enforce.get()


def tenant() -> str:
    """当前请求所属租户（SQL 参数用）。"""
    return _current.get()


def set_tenant(value: str):
    return _current.set(value or DEFAULT_TENANT)


def reset_tenant(token: Any) -> None:
    try:
        _current.reset(token)
    except Exception:
        _current.set(DEFAULT_TENANT)


async def resolve_tenant(db: Any, user: dict[str, Any], header_value: str | None) -> str:
    """解析本次请求的租户。

    - 无头：用账号自己的 tenant_id；
    - 有头且是管理员：按 code 或 id 找到**启用中**的租户；
    - 有头但非管理员：忽略（不能靠带头越权）。
    """
    base = str(user.get("tenant_id") or DEFAULT_TENANT)
    header = (header_value or "").strip()
    if not header or user.get("role") != "admin":
        return base
    row = await db.fetchrow(
        "SELECT id, code, status FROM tenants WHERE (id=$1 OR code=$1) LIMIT 1",
        header,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="租户不存在")
    if row["status"] != "active":
        raise HTTPException(status_code=403, detail="租户已停用")
    return str(row["id"])


def tenant_clause(column: str = "tenant_id") -> str:
    """生成 SQL 片段：默认租户也走过滤，避免"忘了加条件"。"""
    return f"{column}=$TENANT"

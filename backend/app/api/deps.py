from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, Request

from ..config import Settings
from ..security import load_user


def create_token(settings: Settings, subject: str, role: str, token_version: int = 0) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role,
        "token_version": int(token_version),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.JWT_TTL_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    if not settings.AUTH_ENABLED:
        return {
            "sub": "dev",
            "user": "dev",
            "username": "dev",
            "role": "admin",
            "enabled": True,
        }
    return await load_user(request, authorization)


async def require_admin(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="admin role required")
    return user


async def require_ops(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """运营/质控端：试点期允许 admin 与 ops 角色。"""
    if user.get("role") not in {"admin", "ops"}:
        raise HTTPException(status_code=403, detail="ops role required")
    return user


async def require_clinician(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """医生端只读接口：试点期允许 admin 与 doctor 角色访问。

    生产形态应由 HIS 医生账号登录并透传 operator token；这里先做角色门禁。
    """
    if user.get("role") not in {"admin", "doctor"}:
        raise HTTPException(status_code=403, detail="clinician role required")
    return user

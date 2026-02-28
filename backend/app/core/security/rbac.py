from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set

from fastapi import Depends, HTTPException, Request, status

from app.core.config import settings
from app.core.security.auth_audit import record_auth_audit

ROLE_PRIORITY = {
    "viewer": 10,
    "auditor": 20,
    "operator": 30,
    "admin": 40,
}


@dataclass
class Actor:
    user_id: str
    role: str
    source: str
    token_id: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "source": self.source,
            "token_id": self.token_id,
        }


def _role_rank(role: str) -> int:
    return ROLE_PRIORITY.get(str(role or "").strip().lower(), 0)


def _is_role_allowed(actor_role: str, required: Iterable[str]) -> bool:
    actor_rank = _role_rank(actor_role)
    required_ranks = [_role_rank(item) for item in required]
    required_ranks = [rank for rank in required_ranks if rank > 0]
    if not required_ranks:
        return False
    return actor_rank >= min(required_ranks)


def _parse_static_tokens() -> Dict[str, Dict[str, str]]:
    raw = str(getattr(settings, "AUTH_RBAC_STATIC_TOKENS", "") or "")
    token_map: Dict[str, Dict[str, str]] = {}
    for part in raw.split(","):
        item = part.strip()
        if not item or ":" not in item:
            continue
        token, role = item.split(":", 1)
        token = token.strip()
        role = role.strip().lower()
        if not token or role not in ROLE_PRIORITY:
            continue
        token_map[token] = {
            "role": role,
            "user_id": f"token::{role}",
            "token_id": token[-8:] if len(token) > 8 else token,
        }
    return token_map


def _extract_bearer_token(request: Request) -> str:
    auth = str(request.headers.get("Authorization", "") or "").strip()
    if not auth:
        return ""
    if not auth.lower().startswith("bearer "):
        return ""
    return auth[7:].strip()


def resolve_actor(request: Request) -> Actor:
    rbac_enabled = bool(getattr(settings, "AUTH_RBAC_ENABLED", True))
    token_map = _parse_static_tokens()
    token = _extract_bearer_token(request)
    if token and token in token_map:
        info = token_map[token]
        return Actor(
            user_id=str(info.get("user_id") or "token-user"),
            role=str(info.get("role") or "viewer"),
            source="token",
            token_id=str(info.get("token_id") or ""),
        )

    allow_header_role = bool(getattr(settings, "AUTH_RBAC_ALLOW_HEADER_ROLE", False))
    if allow_header_role:
        header_role = str(request.headers.get("X-User-Role", "") or "").strip().lower()
        if header_role in ROLE_PRIORITY:
            header_user = str(request.headers.get("X-User-Id", "") or "").strip() or f"header::{header_role}"
            return Actor(user_id=header_user, role=header_role, source="trusted_header")

    if not rbac_enabled:
        return Actor(user_id="anonymous", role="admin", source="rbac_disabled")

    return Actor(user_id="anonymous", role="viewer", source="anonymous")


async def get_actor(request: Request) -> Dict[str, str]:
    return resolve_actor(request).to_dict()


def require_roles(*roles: str):
    normalized_roles: Set[str] = {str(role or "").strip().lower() for role in roles if str(role or "").strip()}

    async def _dep(request: Request, actor: Dict[str, str] = Depends(get_actor)) -> Dict[str, str]:
        actor_role = str(actor.get("role", "") or "").lower()
        allowed = _is_role_allowed(actor_role, normalized_roles)

        record_auth_audit(
            event="rbac_check",
            request=request,
            actor=actor,
            allowed=allowed,
            reason="allowed" if allowed else "insufficient_role",
            target_roles=sorted(normalized_roles),
            metadata={"rbac_enabled": bool(getattr(settings, "AUTH_RBAC_ENABLED", True))},
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "forbidden",
                    "required_roles": sorted(normalized_roles),
                    "actor_role": actor_role or "unknown",
                },
            )
        return actor

    return _dep

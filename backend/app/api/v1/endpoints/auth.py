from __future__ import annotations

import json
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.security.auth_audit import record_auth_audit
from app.core.security.rbac import get_actor

router = APIRouter()

MOCK_USERS: Dict[str, Dict[str, str]] = {
    "admin": {"user_id": "u-admin", "role": "admin", "token": "dev-admin-token"},
    "ops": {"user_id": "u-ops", "role": "operator", "token": "dev-ops-token"},
    "auditor": {"user_id": "u-auditor", "role": "auditor", "token": "dev-auditor-token"},
}


class LoginRequest(BaseModel):
    username: str = Field(default="ops", min_length=1, max_length=64)
    password: str = Field(default="", max_length=256)


class SSOExchangeRequest(BaseModel):
    audience: str = Field(default="frontend_new")
    claims: Dict[str, Any] = Field(default_factory=dict)


def _load_sso_role_map() -> Dict[str, List[str]]:
    raw = str(getattr(settings, "AUTH_SSO_ROLE_MAP_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}

    out: Dict[str, List[str]] = {}
    if not isinstance(payload, dict):
        return out
    for role, groups in payload.items():
        role_key = str(role or "").strip().lower()
        if not role_key:
            continue
        if isinstance(groups, list):
            out[role_key] = [str(item).strip() for item in groups if str(item).strip()]
    return out


def _resolve_sso_role(groups: List[str], role_map: Dict[str, List[str]]) -> str:
    normalized_groups = {str(item or "").strip() for item in groups if str(item or "").strip()}
    if not normalized_groups:
        return "viewer"
    for role in ("admin", "operator", "auditor"):
        candidates = set(role_map.get(role) or [])
        if normalized_groups.intersection(candidates):
            return role
    return "viewer"


@router.post("/login")
async def login(payload: LoginRequest, request: Request):
    """
    Upgrade2 P2-1 minimal RBAC login mock.
    """
    profile = MOCK_USERS.get(payload.username)
    ok = bool(profile)
    if not ok:
        fallback = MOCK_USERS["ops"]
        profile = {"user_id": fallback["user_id"], "role": "viewer", "token": ""}

    result = {
        "ok": ok,
        "token_type": "bearer",
        "access_token": profile["token"],
        "user": {
            "user_id": profile["user_id"],
            "role": profile["role"],
            "username": payload.username,
        },
    }
    record_auth_audit(
        event="login_mock",
        request=request,
        actor={"user_id": profile["user_id"], "role": profile["role"], "source": "login_mock"},
        allowed=ok,
        reason="user_found" if ok else "unknown_user",
        metadata={"username": payload.username},
    )
    return result


@router.get("/whoami")
async def whoami(actor: Dict[str, str] = Depends(get_actor)):
    return {
        "rbac_enabled": bool(getattr(settings, "AUTH_RBAC_ENABLED", True)),
        "actor": actor,
    }


@router.get("/sso/mapping")
async def sso_mapping():
    """
    Unified mapping contract for frontend_new and Open WebUI.
    """
    role_map = _load_sso_role_map()
    return {
        "sso_enabled": bool(getattr(settings, "AUTH_SSO_ENABLED", False)),
        "issuer": str(getattr(settings, "AUTH_SSO_ISSUER", "")),
        "claims": {
            "sub": str(getattr(settings, "AUTH_SSO_CLAIM_SUB", "sub")),
            "email": str(getattr(settings, "AUTH_SSO_CLAIM_EMAIL", "email")),
            "groups": str(getattr(settings, "AUTH_SSO_CLAIM_GROUPS", "groups")),
        },
        "audiences": {
            "frontend_new": str(getattr(settings, "AUTH_SSO_FRONTEND_AUDIENCE", "frontend_new")),
            "open_webui": str(getattr(settings, "AUTH_SSO_OPENWEBUI_AUDIENCE", "open_webui")),
        },
        "role_mapping": role_map,
        "notes": [
            "frontend_new should exchange upstream claims via /api/v1/auth/sso/exchange",
            "Open WebUI should forward mapped role in trusted gateway header or RBAC token",
        ],
    }


@router.post("/sso/exchange")
async def sso_exchange(payload: SSOExchangeRequest, request: Request):
    role_map = _load_sso_role_map()
    claim_sub = str(getattr(settings, "AUTH_SSO_CLAIM_SUB", "sub"))
    claim_email = str(getattr(settings, "AUTH_SSO_CLAIM_EMAIL", "email"))
    claim_groups = str(getattr(settings, "AUTH_SSO_CLAIM_GROUPS", "groups"))

    claims = dict(payload.claims or {})
    groups_raw = claims.get(claim_groups)
    groups = groups_raw if isinstance(groups_raw, list) else []
    groups_norm = [str(item).strip() for item in groups if str(item).strip()]
    role = _resolve_sso_role(groups_norm, role_map)

    subject = str(claims.get(claim_sub) or "").strip()
    email = str(claims.get(claim_email) or "").strip()
    user_id = subject or email or "sso-anonymous"
    mapped = {
        "user_id": user_id,
        "role": role,
        "audience": payload.audience,
        "groups": groups_norm,
    }

    record_auth_audit(
        event="sso_exchange",
        request=request,
        actor={"user_id": user_id, "role": role, "source": "sso_claims"},
        allowed=True,
        reason="mapped",
        metadata={"audience": payload.audience},
    )
    return mapped

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any

import asyncpg
import jwt
from fastapi import HTTPException, Request

from .tenancy import DEFAULT_TENANT, TENANT_HEADER, resolve_tenant, set_tenant

logger = logging.getLogger(__name__)

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16

LOCKOUT_MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60
_MAX_TRACKED_KEYS = 10_000

BOOTSTRAP_MARKER = "users_bootstrapped"

REDACTED_DETAIL_KEYS = frozenset(
    {
        "report",
        "triage",
        "safety",
        "symptoms",
        "name",
        "phone",
        "content",
        "message",
        "member_context",
    }
)

_login_failures: dict[str, list[float]] = {}
_locked_until: dict[str, float] = {}
_fallback_pseudonym_key: bytes | None = None


def redact_detail(detail: Any) -> dict[str, Any]:
    if not isinstance(detail, dict):
        return {}
    return {key: value for key, value in detail.items() if key not in REDACTED_DETAIL_KEYS}


def pseudonym_tag(settings: Any, value: str) -> str:
    global _fallback_pseudonym_key
    key = str(getattr(settings, "PSEUDONYM_KEY", "") or "")
    if key:
        secret = key.encode("utf-8")
    else:
        if _fallback_pseudonym_key is None:
            _fallback_pseudonym_key = secrets.token_bytes(32)
            logger.warning(
                "PSEUDONYM_KEY is not configured; using a random per-process key "
                "for deletion pseudonyms"
            )
        secret = _fallback_pseudonym_key
    digest = hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"deleted-{digest}"


def id_card_key(settings: Any) -> str:
    return str(
        getattr(settings, "ID_CARD_KEY", "")
        or getattr(settings, "PSEUDONYM_KEY", "")
        or getattr(settings, "JWT_SECRET", "")
        or "dev-id-card-key"
    )


def id_card_hash(settings: Any, id_card: str) -> str:
    return hmac.new(
        id_card_key(settings).encode("utf-8"),
        (id_card or "").encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_hex, hash_hex = encoded.split("$")
        if scheme != "scrypt":
            return False
        expected = bytes.fromhex(hash_hex)
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, expected)


DUMMY_PASSWORD_HASH = hash_password(secrets.token_hex(16))


def login_key(username: str) -> str:
    return username


def _prune(key: str, now: float) -> None:
    until = _locked_until.get(key)
    if until is not None:
        if until <= now:
            _locked_until.pop(key, None)
            _login_failures.pop(key, None)
        return
    failures = _login_failures.get(key)
    if not failures:
        return
    cutoff = now - LOCKOUT_SECONDS
    fresh = [stamp for stamp in failures if stamp > cutoff]
    if fresh:
        _login_failures[key] = fresh
    else:
        _login_failures.pop(key, None)


def _sweep(now: float) -> None:
    for key in list(_login_failures):
        _prune(key, now)
    for key in list(_locked_until):
        _prune(key, now)


def _enforce_bound(now: float) -> None:
    if len(_login_failures) + len(_locked_until) <= _MAX_TRACKED_KEYS:
        return
    _sweep(now)
    overflow = len(_login_failures) - _MAX_TRACKED_KEYS
    if overflow > 0:
        oldest = sorted(_login_failures, key=lambda key: _login_failures[key][-1])[:overflow]
        for key in oldest:
            _login_failures.pop(key, None)


def is_locked(key: str) -> bool:
    now = time.time()
    _prune(key, now)
    return key in _locked_until


def record_failure(key: str) -> bool:
    now = time.time()
    _prune(key, now)
    _enforce_bound(now)
    failures = _login_failures.setdefault(key, [])
    failures.append(now)
    if len(failures) >= LOCKOUT_MAX_FAILURES:
        _locked_until[key] = now + LOCKOUT_SECONDS
        _login_failures.pop(key, None)
        return True
    return False


def clear_failures(key: str) -> None:
    _login_failures.pop(key, None)
    _locked_until.pop(key, None)


def reset_lockout_state() -> None:
    _login_failures.clear()
    _locked_until.clear()


async def bootstrap_users(settings: Any, db: Any) -> None:
    if getattr(db, "pool", None) is None:
        logger.error("bootstrap users skipped: database pool unavailable")
        return
    try:
        async with db.transaction() as conn:
            marker = await conn.fetchval(
                "SELECT value FROM app_meta WHERE key=$1",
                BOOTSTRAP_MARKER,
            )
            if marker:
                return
            count = await conn.fetchval("SELECT count(*) FROM users")
            if count == 0:
                logger.info(
                    "seeding bootstrap users: admin=%s patient=%s",
                    settings.ADMIN_USER,
                    settings.PATIENT_USER,
                )
                accounts = (
                    (settings.ADMIN_USER, settings.ADMIN_PASSWORD, "admin"),
                    (settings.PATIENT_USER, settings.PATIENT_PASSWORD, "patient"),
                )
                for username, password, role in accounts:
                    await conn.execute(
                        """
                        INSERT INTO users (id, username, password_hash, role)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        username,
                        username,
                        hash_password(password),
                        role,
                    )
            else:
                logger.info("users table already has %s rows; skipping bootstrap seed", count)
            await conn.execute(
                "INSERT INTO app_meta (key, value) VALUES ($1, $2) "
                "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
                BOOTSTRAP_MARKER,
                "true",
            )
    except Exception:
        logger.exception("bootstrap users failed: users/app_meta query or seeding error")
        raise


async def load_user(request: Request, authorization: str | None) -> dict[str, Any]:
    settings = request.app.state.settings
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="invalid token payload")
    db = getattr(request.app.state, "db", None)
    if db is None or getattr(db, "pool", None) is None:
        raise HTTPException(status_code=503, detail="database unavailable")
    try:
        row = await db.fetchrow(
            "SELECT id, username, role, enabled, token_version, tenant_id FROM users WHERE id=$1",
            subject,
        )
    except (AssertionError, OSError, asyncpg.PostgresError) as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    if not row["enabled"]:
        raise HTTPException(status_code=401, detail="user disabled")
    if int(payload.get("token_version", 0)) != int(row["token_version"] or 0):
        raise HTTPException(status_code=401, detail="token revoked")
    user = {
        "sub": row["id"],
        "user": row["id"],
        "username": row["username"],
        "role": row["role"],
        "enabled": row["enabled"],
        "tenant_id": row["tenant_id"] or DEFAULT_TENANT,
    }
    # 解析本次请求的租户（管理员可用 X-Tenant 跨院区运维），写进上下文供数据层读取
    tenant_id = await resolve_tenant(db, user, request.headers.get(TENANT_HEADER))
    set_tenant(tenant_id)
    user["tenant_id"] = tenant_id
    return user

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Any

import jwt
from fastapi import HTTPException, Request

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16

LOCKOUT_MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60

_login_failures: dict[str, list[float]] = {}
_locked_until: dict[str, float] = {}


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


def login_key(username: str, client_ip: str) -> str:
    return f"{username}|{client_ip}"


def _purge(key: str) -> None:
    until = _locked_until.get(key)
    if until is not None and until <= time.time():
        _locked_until.pop(key, None)
        _login_failures.pop(key, None)


def is_locked(key: str) -> bool:
    _purge(key)
    return key in _locked_until


def record_failure(key: str) -> bool:
    _purge(key)
    failures = _login_failures.setdefault(key, [])
    failures.append(time.time())
    if len(failures) >= LOCKOUT_MAX_FAILURES:
        _locked_until[key] = time.time() + LOCKOUT_SECONDS
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
    try:
        count = await db.fetchval("SELECT count(*) FROM users")
    except Exception:
        return
    if count:
        return
    accounts = (
        (settings.ADMIN_USER, settings.ADMIN_PASSWORD, "admin"),
        (settings.PATIENT_USER, settings.PATIENT_PASSWORD, "patient"),
    )
    for username, password, role in accounts:
        await db.execute(
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
    row = await request.app.state.db.fetchrow(
        "SELECT id, username, role, enabled FROM users WHERE id=$1",
        subject,
    )
    if not row:
        raise HTTPException(status_code=401, detail="user not found")
    if not row["enabled"]:
        raise HTTPException(status_code=401, detail="user disabled")
    return {
        "sub": row["id"],
        "user": row["id"],
        "username": row["username"],
        "role": row["role"],
        "enabled": row["enabled"],
    }

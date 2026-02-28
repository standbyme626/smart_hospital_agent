from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from fastapi import Request

from app.core.config import settings

logger = structlog.get_logger(__name__)
_audit_lock = threading.RLock()


def _safe_request_id(request: Optional[Request]) -> str:
    if request is None:
        return ""
    state = getattr(request, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "request_id", "") or "").strip()


def record_auth_audit(
    *,
    event: str,
    request: Optional[Request],
    actor: Dict[str, Any],
    allowed: bool,
    reason: str,
    target_roles: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Best-effort auth audit sink.
    Writes one JSON line and mirrors to structlog for centralized shipping.
    """
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "event": str(event or "").strip() or "auth_decision",
        "request_id": _safe_request_id(request),
        "path": str(request.url.path) if request else "",
        "method": str(request.method) if request else "",
        "actor_id": str(actor.get("user_id", "") or ""),
        "actor_role": str(actor.get("role", "") or ""),
        "actor_source": str(actor.get("source", "") or ""),
        "allowed": bool(allowed),
        "reason": str(reason or "").strip(),
        "target_roles": list(target_roles or []),
        "metadata": dict(metadata or {}),
    }

    log_path = Path(str(getattr(settings, "AUTH_RBAC_AUDIT_LOG_PATH", "") or "")).expanduser()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with _audit_lock:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("auth_audit_file_write_failed", error=str(exc), path=str(log_path))

    logger.info("auth_audit", **payload)

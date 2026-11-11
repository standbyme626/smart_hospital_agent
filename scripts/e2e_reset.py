#!/usr/bin/env python3
"""Reset the E2E Postgres database before Playwright runs.

Only databases whose name ends with ``_e2e`` are accepted, so the production
``sha`` database can never be truncated by accident.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

def _default_dsn() -> str:
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL="):
                url = line.split("=", 1)[1].strip()
                return url.rsplit("/", 1)[0] + "/sha_e2e"
    return "postgresql://sha:change-me@127.0.0.1:5434/sha_e2e"


DEFAULT_DSN = _default_dsn()

APP_TABLES = (
    "users",
    "app_meta",
    "sessions",
    "events",
    "messages",
    "audit_log",
    "bookings",
    "review_requests",
    "family_members",
    "medications",
    "reports",
    "followups",
    "health_events",
    "analytics_events",
    "model_calls",
    "eval_runs",
    "mcp_locks",
    "mcp_orders",
    "mcp_booked",
    "appointment_projection",
    "human_task",
    "his_patient_map",
    "his_encounter_map",
    "his_prescription_map",
    "sync_outbox",
    "reconciliation_finding",
    # —— Wave B/C/D/E 新增业务表：不清理会串到下一次 E2E 与截图里 ——
    "notifications",
    "clinical_notes",
    "clinical_orders",
    "previsit_forms",
    "feedback_tickets",
    "visit_encounters",
    "print_jobs",
    "visit_closures",
    "queue_tickets",
    "dispensing_progress",
    "insurance_settlements",
    "consent_scopes",
)


async def _reset(dsn: str) -> None:
    from app.tenancy import set_enforce

    # 数据重置需要全库权限：关掉租户隔离（应用进程不受影响）
    set_enforce(False)
    database = urlsplit(dsn).path.lstrip("/")
    if not database.endswith("_e2e"):
        raise SystemExit(f"refusing to reset non-E2E database: {database!r}")

    from app.config import Settings
    from app.db import Database
    from app.security import BOOTSTRAP_MARKER, hash_password

    settings = Settings(DATABASE_URL=dsn)
    db = Database(settings)
    try:
        await db.init_schema()
        await db.execute(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE")
        accounts = (
            (os.getenv("ADMIN_USER", settings.ADMIN_USER), os.getenv("ADMIN_PASSWORD", "e2e-admin-pass"), "admin"),
            (os.getenv("PATIENT_USER", settings.PATIENT_USER), os.getenv("PATIENT_PASSWORD", "e2e-patient-pass"), "patient"),
        )
        for username, password, role in accounts:
            await db.execute(
                "INSERT INTO users (id, username, password_hash, role) VALUES ($1, $2, $3, $4)",
                username,
                username,
                hash_password(password),
                role,
            )
        await db.execute(
            "INSERT INTO app_meta (key, value) VALUES ($1, $2) "
            "ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value",
            BOOTSTRAP_MARKER,
            "true",
        )
    finally:
        await db.close()


def main() -> None:
    dsn = os.getenv("E2E_DATABASE_URL", DEFAULT_DSN)
    asyncio.run(_reset(dsn))
    print(f"[e2e-reset] reset {urlsplit(dsn).path.lstrip('/')} and seeded admin/patient users")


if __name__ == "__main__":
    main()

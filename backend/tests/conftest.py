import asyncio
import os
import re
import urllib.parse
from pathlib import Path

import asyncpg
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _test_database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL")
    if url:
        database = urllib.parse.urlsplit(url).path
        if not database.endswith("_test"):
            raise RuntimeError(
                f"Refusing to run tests against database {database!r}: "
                "the test database name must end with '_test' "
                "(production 'sha' is forbidden). Set TEST_DATABASE_URL to an isolated *_test database."
            )
        return url
    env_file = REPO_ROOT / ".env"
    match = re.search(r"^DATABASE_URL=(.+)$", env_file.read_text(encoding="utf-8"), re.M)
    if not match:
        raise RuntimeError("DATABASE_URL not found in .env; set TEST_DATABASE_URL")
    parts = urllib.parse.urlsplit(match.group(1).strip().strip('"').strip("'"))
    return parts._replace(path="/sha_test").geturl()


TEST_DATABASE_URL = _test_database_url()

os.environ["AUTH_ENABLED"] = "false"
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ.setdefault("OLLAMA_ENABLED", "false")
os.environ.setdefault("HOSPITAL_MCP_URL", "http://127.0.0.1:5999/mcp")
os.environ.setdefault("OPENCODE_API_KEY", "test-key")
os.environ.setdefault("SILICONFLOW_API_KEY", "")

os.environ.setdefault("ADMIN_PASSWORD", "admin123456")
os.environ.setdefault("PATIENT_PASSWORD", "patient123456")

_TRUNCATE_TABLES = (
    "users, app_meta, sessions, events, messages, audit_log, bookings, review_requests, family_members, "
    "medications, reports, followups, notifications, health_events, analytics_events, model_calls, "
    "mcp_locks, mcp_orders, mcp_booked"
)


async def _truncate() -> None:
    conn = await asyncpg.connect(TEST_DATABASE_URL)
    try:
        rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname='public'")
        existing = {row["tablename"] for row in rows}
        tables = [name.strip() for name in _TRUNCATE_TABLES.split(",") if name.strip() in existing]
        if tables:
            await conn.execute(f"TRUNCATE {', '.join(tables)}")
    finally:
        await conn.close()


@pytest.fixture(autouse=True)
def isolated_state():
    from app.config import get_settings
    from app.security import reset_lockout_state

    get_settings().AUTH_ENABLED = False
    reset_lockout_state()
    try:
        asyncio.run(_truncate())
    except OSError:
        pass
    yield
    get_settings().AUTH_ENABLED = False
    reset_lockout_state()

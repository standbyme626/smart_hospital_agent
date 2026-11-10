from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from .config import Settings

SCHEMA_VERSION = "2026-11-09.p0"

SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pgcrypto",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY,
        doc_key TEXT UNIQUE,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        department TEXT,
        source TEXT,
        embedding vector(4096),
        tsv tsvector,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_documents_tsv ON documents USING GIN (tsv)",
    "CREATE INDEX IF NOT EXISTS idx_documents_department ON documents (department)",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT 'anonymous',
        title TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        seq BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        event_id TEXT NOT NULL,
        type TEXT NOT NULL,
        data JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id, seq)",
    """
    CREATE TABLE IF NOT EXISTS audit_log (
        id BIGSERIAL PRIMARY KEY,
        request_id TEXT,
        session_id TEXT,
        actor TEXT,
        action TEXT NOT NULL,
        detail JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bookings (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        patient_id TEXT,
        department TEXT NOT NULL,
        slot_id TEXT,
        order_id TEXT,
        status TEXT NOT NULL DEFAULT 'created',
        payment_status TEXT NOT NULL DEFAULT 'unpaid',
        amount NUMERIC(10, 2) NOT NULL DEFAULT 0,
        detail JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rules (
        id BIGSERIAL PRIMARY KEY,
        kind TEXT NOT NULL,
        key TEXT NOT NULL,
        value JSONB NOT NULL DEFAULT '{}'::jsonb,
        enabled BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (kind, key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_calls (
        id BIGSERIAL PRIMARY KEY,
        request_id TEXT,
        session_id TEXT,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        task TEXT,
        prompt_tokens INTEGER NOT NULL DEFAULT 0,
        completion_tokens INTEGER NOT NULL DEFAULT 0,
        cost NUMERIC(12, 6) NOT NULL DEFAULT 0,
        latency_ms INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        id BIGSERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS family_members (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        name TEXT NOT NULL,
        relation TEXT NOT NULL DEFAULT 'self',
        gender TEXT,
        birth_date TEXT,
        phone TEXT,
        allergies TEXT,
        conditions TEXT,
        is_self BOOLEAN NOT NULL DEFAULT false,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_family_user ON family_members (user_id)",
    """
    CREATE TABLE IF NOT EXISTS medications (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT,
        name TEXT NOT NULL,
        dose TEXT,
        frequency TEXT,
        reminder_times JSONB NOT NULL DEFAULT '[]'::jsonb,
        active BOOLEAN NOT NULL DEFAULT true,
        notes TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_medications_user ON medications (user_id, member_id)",
    """
    CREATE TABLE IF NOT EXISTS reports (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT,
        title TEXT NOT NULL,
        image_path TEXT,
        summary TEXT NOT NULL DEFAULT '',
        abnormal_values JSONB NOT NULL DEFAULT '[]'::jsonb,
        questions JSONB NOT NULL DEFAULT '[]'::jsonb,
        raw JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reports_user ON reports (user_id, member_id)",
    """
    CREATE TABLE IF NOT EXISTS followups (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT,
        booking_id TEXT,
        title TEXT NOT NULL,
        due_date DATE,
        status TEXT NOT NULL DEFAULT 'pending',
        notes TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_followups_user ON followups (user_id, status)",
    """
    CREATE TABLE IF NOT EXISTS health_events (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        detail JSONB NOT NULL DEFAULT '{}'::jsonb,
        occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_health_events_user ON health_events (user_id, member_id, occurred_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS analytics_events (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT,
        session_id TEXT,
        name TEXT NOT NULL,
        props JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_analytics_name ON analytics_events (name, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS app_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'patient',
        enabled BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS review_requests (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('payment', 'payment_collection', 'clinical_review')),
        ref_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        member_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending'
            CHECK (status IN ('pending', 'acknowledged', 'processing', 'approved',
                              'completed', 'rejected', 'cancelled', 'failed')),
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        decision_note TEXT,
        decided_by TEXT,
        decided_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reviews_status ON review_requests (status, created_at)",
    """
    CREATE TABLE IF NOT EXISTS mcp_locks (
        lock_id TEXT PRIMARY KEY,
        slot_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_orders (
        order_id TEXT PRIMARY KEY,
        slot_id TEXT NOT NULL,
        patient_id TEXT NOT NULL,
        amount NUMERIC(10, 2) NOT NULL,
        status TEXT NOT NULL,
        payment_status TEXT NOT NULL,
        idempotency_key TEXT UNIQUE,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS mcp_booked (
        slot_id TEXT PRIMARY KEY,
        order_id TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT 'anonymous',
        member_id TEXT,
        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id)",
    """
    CREATE TABLE IF NOT EXISTS schema_migrations (
        version TEXT PRIMARY KEY,
        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS his_patient_map (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        member_id TEXT NOT NULL,
        his_tenant_id TEXT,
        his_patient_id TEXT,
        his_bus_no TEXT,
        id_card_hash TEXT,
        id_card_last4 TEXT,
        match_method TEXT,
        verified_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (user_id, member_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_his_patient_map_his ON his_patient_map (his_tenant_id, his_patient_id)",
    """
    CREATE TABLE IF NOT EXISTS his_encounter_map (
        appointment_id TEXT PRIMARY KEY,
        his_encounter_id TEXT,
        his_patient_id TEXT,
        his_org_id TEXT,
        his_practitioner_id TEXT,
        his_service_id TEXT,
        quote_id TEXT,
        payment_id TEXT,
        invoice_id TEXT,
        his_status TEXT,
        payment_status TEXT,
        last_synced_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_his_encounter_map_his ON his_encounter_map (his_encounter_id)",
    """
    CREATE TABLE IF NOT EXISTS his_prescription_map (
        his_prescription_no TEXT PRIMARY KEY,
        his_encounter_id TEXT,
        his_patient_id TEXT,
        source_table TEXT,
        status TEXT,
        issued_at TIMESTAMPTZ,
        synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_his_prescription_map_encounter ON his_prescription_map (his_encounter_id)",
    """
    CREATE TABLE IF NOT EXISTS sync_outbox (
        id TEXT PRIMARY KEY,
        aggregate_type TEXT NOT NULL,
        aggregate_id TEXT,
        operation TEXT NOT NULL,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        idempotency_key TEXT UNIQUE,
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TIMESTAMPTZ,
        last_error TEXT,
        response_payload JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sync_outbox_status ON sync_outbox (status, next_attempt_at)",
    """
    CREATE TABLE IF NOT EXISTS reconciliation_finding (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        severity TEXT NOT NULL,
        local_ref TEXT,
        his_ref TEXT,
        detail TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        resolved_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_reconciliation_finding_status ON reconciliation_finding (status, detected_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS his_sync_cursor (
        stream TEXT PRIMARY KEY,
        last_update_time TIMESTAMPTZ,
        last_id TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS appointment_projection (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        member_id TEXT NOT NULL,
        his_patient_id TEXT NOT NULL,
        his_encounter_id TEXT,
        his_org_id TEXT,
        his_practitioner_id TEXT,
        his_service_id TEXT,
        quote_id TEXT,
        amount NUMERIC(12, 2) NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'quoted',
        payment_status TEXT NOT NULL DEFAULT 'unpaid',
        his_status TEXT,
        appointment_date TEXT,
        department TEXT,
        doctor_name TEXT,
        payment_id TEXT,
        invoice_id TEXT,
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        idempotency_key TEXT UNIQUE,
        last_synced_at TIMESTAMPTZ,
        stale BOOLEAN NOT NULL DEFAULT false,
        legacy_demo BOOLEAN NOT NULL DEFAULT false,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_appointment_projection_encounter ON appointment_projection (his_encounter_id)",
    "CREATE INDEX IF NOT EXISTS idx_appointment_projection_user ON appointment_projection (user_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS human_task (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        ref_id TEXT NOT NULL,
        user_id TEXT NOT NULL DEFAULT '',
        member_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT 'requested',
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        assignee TEXT,
        sla_due_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_human_task_open ON human_task (status, kind, ref_id)",
    """
    CREATE TABLE IF NOT EXISTS human_tasks (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('payment_collection', 'clinical_review')),
        ref_id TEXT NOT NULL,
        user_id TEXT,
        member_id TEXT,
        status TEXT NOT NULL DEFAULT 'pending',
        payload JSONB NOT NULL DEFAULT '{}'::jsonb,
        assignee TEXT,
        sla_due_at TIMESTAMPTZ,
        decision_note TEXT,
        decided_by TEXT,
        decided_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_human_tasks_status ON human_tasks (status, sla_due_at)",
    "CREATE INDEX IF NOT EXISTS idx_human_tasks_ref ON human_tasks (kind, ref_id)",
    """
    CREATE TABLE IF NOT EXISTS consent_record (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        subject_member_id TEXT,
        consent_type TEXT NOT NULL,
        version TEXT NOT NULL,
        granted_by TEXT,
        granted_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ,
        revoked_at TIMESTAMPTZ,
        evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_consent_record_user ON consent_record (user_id, consent_type)",
    """
    CREATE TABLE IF NOT EXISTS notifications (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        body TEXT,
        link TEXT,
        read_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications (user_id, created_at DESC)",
]

MIGRATIONS = [
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS user_id TEXT",
    """
    UPDATE bookings b
    SET user_id = s.user_id
    FROM sessions s
    WHERE b.user_id IS NULL AND b.session_id = s.id
    """,
    "UPDATE bookings SET user_id = session_id WHERE user_id IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_bookings_user ON bookings (user_id)",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS member_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS hospital_patient_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS review_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS idempotency_key TEXT",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_bookings_idem ON bookings (idempotency_key) WHERE idempotency_key IS NOT NULL",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS hospital_patient_id TEXT",
    "UPDATE family_members SET hospital_patient_id = 'HP-' || id WHERE hospital_patient_id IS NULL",
    "ALTER TABLE review_requests ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ",
    "UPDATE review_requests SET updated_at = created_at WHERE updated_at IS NULL",
    "ALTER TABLE review_requests ALTER COLUMN updated_at SET DEFAULT now()",
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS member_id TEXT",
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_message_at TIMESTAMPTZ",
    "ALTER TABLE model_calls ADD COLUMN IF NOT EXISTS user_id TEXT",
    "CREATE INDEX IF NOT EXISTS idx_model_calls_user ON model_calls (user_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_messages_user_day ON messages (user_id, created_at)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS consent_version TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS consent_at TIMESTAMPTZ",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS id_card_ciphertext TEXT",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS id_card_hash TEXT",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS id_card_last4 TEXT",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS consent_id TEXT",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS his_link_status TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_patient_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_encounter_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_org_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_practitioner_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_service_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS quote_id TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS his_status TEXT",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS stale BOOLEAN NOT NULL DEFAULT false",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS legacy_demo BOOLEAN NOT NULL DEFAULT false",
    "CREATE INDEX IF NOT EXISTS idx_bookings_his_encounter ON bookings (his_encounter_id)",
    "ALTER TABLE review_requests DROP CONSTRAINT IF EXISTS review_requests_kind_check",
    "ALTER TABLE review_requests ADD CONSTRAINT review_requests_kind_check "
    "CHECK (kind IN ('payment', 'payment_collection', 'clinical_review'))",
    "ALTER TABLE review_requests DROP CONSTRAINT IF EXISTS review_requests_status_check",
    "ALTER TABLE review_requests ADD CONSTRAINT review_requests_status_check "
    "CHECK (status IN ('pending', 'acknowledged', 'processing', 'approved', "
    "'completed', 'rejected', 'cancelled', 'failed'))",
    "ALTER TABLE review_requests ADD COLUMN IF NOT EXISTS sla_due_at TIMESTAMPTZ",
    "ALTER TABLE review_requests ADD COLUMN IF NOT EXISTS assignee TEXT",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS source TEXT",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS his_medication_request_id TEXT",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS his_prescription_no TEXT",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS last_synced_at TIMESTAMPTZ",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS source TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS his_observation_id TEXT",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS critical_flags JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS consent_id TEXT",
    "ALTER TABLE health_events ADD COLUMN IF NOT EXISTS source TEXT",
    "ALTER TABLE health_events ADD COLUMN IF NOT EXISTS his_ref_type TEXT",
    "ALTER TABLE health_events ADD COLUMN IF NOT EXISTS his_ref_id TEXT",
    "ALTER TABLE health_events ADD COLUMN IF NOT EXISTS deep_link TEXT",
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS report_state JSONB",
    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS member_context JSONB",
]


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                dsn=self.settings.DATABASE_URL,
                min_size=1,
                max_size=10,
                command_timeout=60,
            )

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def init_schema(self) -> None:
        await self.connect()
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            for statement in SCHEMA:
                await conn.execute(statement)
            for statement in MIGRATIONS:
                await conn.execute(statement)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT (version) DO NOTHING",
                SCHEMA_VERSION,
            )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg

from .config import Settings
from .tenancy import enforcing, tenant

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
    """
    CREATE TABLE IF NOT EXISTS clinician_drafts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft',
        signed_by TEXT,
        signed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clinician_drafts_member ON clinician_drafts (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS previsit_forms (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        chief_complaint TEXT NOT NULL,
        duration TEXT,
        medications TEXT,
        allergies TEXT,
        history TEXT,
        questions TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_previsit_member ON previsit_forms (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS doctor_directory (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        title TEXT,
        intro TEXT,
        fee NUMERIC(10, 2) NOT NULL DEFAULT 0,
        active BOOLEAN NOT NULL DEFAULT true,
        schedule TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doctor_directory_dept ON doctor_directory (department, active)",
    """
    CREATE TABLE IF NOT EXISTS dispensing_progress (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        status TEXT NOT NULL DEFAULT 'waiting_dispense',
        "window" TEXT,
        ready_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_dispensing_member ON dispensing_progress (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS insurance_settlements (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        total NUMERIC(10, 2) NOT NULL DEFAULT 0,
        insurance NUMERIC(10, 2) NOT NULL DEFAULT 0,
        account NUMERIC(10, 2) NOT NULL DEFAULT 0,
        self_pay NUMERIC(10, 2) NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'settled',
        settled_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_insurance_member ON insurance_settlements (member_id, settled_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS queue_tickets (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        ticket_no TEXT NOT NULL,
        department TEXT,
        status TEXT NOT NULL DEFAULT 'waiting',
        called_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_queue_tickets_day ON queue_tickets (status, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS feedback_tickets (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT,
        category TEXT NOT NULL,
        content TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'submitted',
        reply TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_tickets (user_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS clinical_notes (
        id TEXT PRIMARY KEY,
        doctor TEXT NOT NULL,
        member_id TEXT NOT NULL,
        appointment_id TEXT,
        chief_complaint TEXT,
        present_illness TEXT,
        physical_exam TEXT,
        diagnosis TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clinical_notes_member ON clinical_notes (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS clinical_orders (
        id TEXT PRIMARY KEY,
        doctor TEXT NOT NULL,
        member_id TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'medication',
        name TEXT NOT NULL,
        dose TEXT,
        usage TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        review_status TEXT NOT NULL DEFAULT 'pending',
        review_note TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        signed_at TIMESTAMPTZ
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clinical_orders_member ON clinical_orders (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS lab_results (
        id TEXT PRIMARY KEY,
        member_id TEXT NOT NULL,
        name TEXT NOT NULL,
        result_value TEXT,
        reference_range TEXT,
        abnormal TEXT,
        measured_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lab_results_member ON lab_results (member_id, measured_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS doctor_schedules (
        id TEXT PRIMARY KEY,
        doctor TEXT NOT NULL,
        department TEXT,
        weekday TEXT NOT NULL,
        period TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'clinic',
        note TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_doctor_schedules_doctor ON doctor_schedules (doctor, weekday)",
    """
    CREATE TABLE IF NOT EXISTS crisis_events (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        member_id TEXT,
        kind TEXT NOT NULL,
        detail TEXT,
        status TEXT NOT NULL DEFAULT 'open',
        handled_by TEXT,
        handled_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_crisis_events_status ON crisis_events (status, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS notification_deliveries (
        id TEXT PRIMARY KEY,
        notification_id BIGINT,
        channel TEXT NOT NULL DEFAULT 'station',
        target TEXT,
        status TEXT NOT NULL DEFAULT 'sent',
        attempts INT NOT NULL DEFAULT 1,
        error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_notification_deliveries_status ON notification_deliveries (status, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS consultations (
        id TEXT PRIMARY KEY,
        doctor TEXT NOT NULL,
        member_id TEXT NOT NULL,
        target_department TEXT NOT NULL,
        reason TEXT,
        status TEXT NOT NULL DEFAULT 'requested',
        reply TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_consultations_member ON consultations (member_id, created_at DESC)",
    """
    CREATE TABLE IF NOT EXISTS consent_scopes (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        member_id TEXT NOT NULL,
        can_view BOOLEAN NOT NULL DEFAULT true,
        can_act BOOLEAN NOT NULL DEFAULT false,
        version TEXT NOT NULL DEFAULT 'v1',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (user_id, member_id)
    )
    """,
]

MIGRATIONS = [
    # ---- Wave G：租户隔离用的非超级用户角色（超级用户会绕过 RLS）----
    """
    DO $$
    BEGIN
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname='app_tenant') THEN
            CREATE ROLE app_tenant NOLOGIN;
        END IF;
    END
    $$;
    """,
    "GRANT USAGE, CREATE ON SCHEMA public TO app_tenant",
    "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public TO app_tenant",
    "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_tenant",
    # 之后由 sha 新建的表/序列自动授权（新表在启动时由 ensure_* 创建）
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO app_tenant",
    "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO app_tenant",
    # ---- Wave G 多租户：先加列，再索引/默认值，最后开行级安全 ----
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE family_members ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE messages ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE reports ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE medications ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE followups ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE notifications ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE bookings ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE clinical_orders ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE config_entries ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE config_entries DROP CONSTRAINT IF EXISTS config_entries_kind_key_key",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_config_entries_tenant_kind_key ON config_entries (tenant_id, kind, key)",
    "ALTER TABLE audit_log ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE review_requests ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE previsit_forms ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE visit_encounters ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE print_jobs ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE visit_closures ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE unmatched_questions ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "ALTER TABLE feedback_tickets ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'tn_default'",
    "CREATE INDEX IF NOT EXISTS idx_family_members_tenant ON family_members (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_bookings_tenant ON bookings (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_config_entries_tenant ON config_entries (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_log_tenant ON audit_log (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_clinical_notes_tenant ON clinical_notes (tenant_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions (tenant_id)",
    "ALTER TABLE users ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE family_members ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE sessions ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE messages ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE reports ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE medications ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE followups ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE notifications ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE bookings ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE clinical_notes ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE clinical_orders ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE documents ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE config_entries ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE audit_log ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE review_requests ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE previsit_forms ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE visit_encounters ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE print_jobs ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE visit_closures ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE unmatched_questions ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE feedback_tickets ALTER COLUMN tenant_id SET DEFAULT coalesce(nullif(current_setting('app.tenant_id', true), ''), 'tn_default')",
    "ALTER TABLE family_members ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE family_members FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON family_members",
    "CREATE POLICY tenant_iso ON family_members USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE sessions ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE sessions FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON sessions",
    "CREATE POLICY tenant_iso ON sessions USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE messages ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE messages FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON messages",
    "CREATE POLICY tenant_iso ON messages USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE reports ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE reports FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON reports",
    "CREATE POLICY tenant_iso ON reports USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE medications ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE medications FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON medications",
    "CREATE POLICY tenant_iso ON medications USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE followups ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE followups FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON followups",
    "CREATE POLICY tenant_iso ON followups USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE notifications ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE notifications FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON notifications",
    "CREATE POLICY tenant_iso ON notifications USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE bookings ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE bookings FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON bookings",
    "CREATE POLICY tenant_iso ON bookings USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE clinical_notes ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE clinical_notes FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON clinical_notes",
    "CREATE POLICY tenant_iso ON clinical_notes USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE clinical_orders ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE clinical_orders FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON clinical_orders",
    "CREATE POLICY tenant_iso ON clinical_orders USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE documents ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE documents FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON documents",
    "CREATE POLICY tenant_iso ON documents USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE config_entries ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE config_entries FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON config_entries",
    "CREATE POLICY tenant_iso ON config_entries USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE audit_log FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON audit_log",
    "CREATE POLICY tenant_iso ON audit_log USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE review_requests ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE review_requests FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON review_requests",
    "CREATE POLICY tenant_iso ON review_requests USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE previsit_forms ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE previsit_forms FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON previsit_forms",
    "CREATE POLICY tenant_iso ON previsit_forms USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE visit_encounters ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE visit_encounters FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON visit_encounters",
    "CREATE POLICY tenant_iso ON visit_encounters USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE print_jobs ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE print_jobs FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON print_jobs",
    "CREATE POLICY tenant_iso ON print_jobs USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE visit_closures ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE visit_closures FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON visit_closures",
    "CREATE POLICY tenant_iso ON visit_closures USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE unmatched_questions ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE unmatched_questions FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON unmatched_questions",
    "CREATE POLICY tenant_iso ON unmatched_questions USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
    "ALTER TABLE feedback_tickets ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE feedback_tickets FORCE ROW LEVEL SECURITY",
    "DROP POLICY IF EXISTS tenant_iso ON feedback_tickets",
    "CREATE POLICY tenant_iso ON feedback_tickets USING (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id)) WITH CHECK (tenant_id = coalesce(nullif(current_setting('app.tenant_id', true), ''), tenant_id))",
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

    async def _scoped(self, conn: asyncpg.Connection) -> Any:
        """每个事务：写入当前租户 + 切到非超级用户角色。

        应用连接用的是超级用户（迁移/运维需要），而超级用户会绕过 RLS，
        所以这里在事务内 SET LOCAL ROLE app_tenant，让行级安全真正生效。
        未设置租户时策略 fail-open（脚本 / 迁移 / 备份不受影响）。
        数据重置脚本会关闭 enforcing，以免被 RLS/权限挡住。
        """
        if not enforcing():
            return
        await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant())
        await conn.execute("SET LOCAL ROLE app_tenant")

    async def grant_app_role(self) -> None:
        """给租户隔离角色补齐权限（新表创建后调用一次即可）。"""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            for statement in (
                "GRANT USAGE, CREATE ON SCHEMA public TO app_tenant",
                "GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public TO app_tenant",
                "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_tenant",
            ):
                try:
                    await conn.execute(statement)
                except Exception:
                    continue

    async def execute_system(self, query: str, *args: Any) -> str:
        """系统级写入：不设租户、不切角色（建表、灌种子、迁移用）。"""
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._scoped(conn)
                return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> asyncpg.Record | None:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._scoped(conn)
                return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._scoped(conn)
                return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        assert self.pool is not None
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await self._scoped(conn)
                return await conn.execute(query, *args)

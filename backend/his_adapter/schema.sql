CREATE TABLE IF NOT EXISTS his_patient_map (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    member_id TEXT NOT NULL,
    his_tenant_id TEXT,
    his_patient_id TEXT NOT NULL,
    his_bus_no TEXT,
    id_card_hash TEXT,
    id_card_last4 TEXT,
    match_method TEXT,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, member_id)
);

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
);

CREATE TABLE IF NOT EXISTS his_prescription_map (
    his_prescription_no TEXT PRIMARY KEY,
    his_encounter_id TEXT NOT NULL,
    his_patient_id TEXT,
    source_table TEXT,
    status TEXT,
    issued_at TIMESTAMPTZ,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
);

CREATE INDEX IF NOT EXISTS sync_outbox_status_idx ON sync_outbox (status, next_attempt_at);

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
);

CREATE INDEX IF NOT EXISTS reconciliation_finding_open_idx
    ON reconciliation_finding (status, kind, local_ref);

CREATE TABLE IF NOT EXISTS his_sync_cursor (
    stream TEXT PRIMARY KEY,
    last_update_time TIMESTAMPTZ,
    last_id TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
);

CREATE INDEX IF NOT EXISTS appointment_projection_encounter_idx
    ON appointment_projection (his_encounter_id);

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
);

CREATE INDEX IF NOT EXISTS human_task_open_idx ON human_task (status, kind, ref_id);

from __future__ import annotations

import asyncio
import copy
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

PROJECTION_FIELDS = {
    "user_id",
    "member_id",
    "his_patient_id",
    "his_encounter_id",
    "his_org_id",
    "his_practitioner_id",
    "his_service_id",
    "quote_id",
    "amount",
    "status",
    "payment_status",
    "his_status",
    "appointment_date",
    "department",
    "doctor_name",
    "payment_id",
    "invoice_id",
    "payload",
    "idempotency_key",
    "last_synced_at",
    "stale",
    "legacy_demo",
}

JSON_COLUMNS = {"payload", "response_payload"}


def _now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _as_dt(value: Any) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return value


def _canon(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True, ensure_ascii=False, default=str)


def _load(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _row(record: Any) -> dict[str, Any]:
    data = dict(record)
    for key in JSON_COLUMNS:
        if key in data:
            data[key] = _load(data[key])
    if "amount" in data and data["amount"] is not None:
        data["amount"] = round(float(data["amount"]), 2)
    return data


class Store:
    def __init__(self, database_url: str = ""):
        self.database_url = (database_url or "").strip()
        self._pool: asyncpg.Pool | None = None
        self._lock = asyncio.Lock()
        self._memory: dict[str, dict[str, dict[str, Any]]] = {
            name: {}
            for name in (
                "his_patient_map",
                "his_encounter_map",
                "his_prescription_map",
                "sync_outbox",
                "reconciliation_finding",
                "his_sync_cursor",
                "appointment_projection",
                "human_task",
            )
        }

    @property
    def backend(self) -> str:
        return "memory" if not self.database_url else "asyncpg"

    async def init_schema(self) -> None:
        if not self.database_url:
            return
        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")
        async with self._pool.acquire() as conn:
            for statement in ddl.split(";"):
                if statement.strip():
                    await conn.execute(statement)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _table(self, name: str) -> dict[str, dict[str, Any]]:
        return self._memory[name]

    @staticmethod
    def _patient_key(member_id: str, user_id: str | None) -> str:
        return f"{user_id or ''}\x1f{member_id}"

    async def get_patient_map(self, member_id: str, user_id: str | None = None) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("his_patient_map").get(self._patient_key(member_id, user_id))
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM his_patient_map WHERE user_id=$1 AND member_id=$2",
                user_id or "",
                member_id,
            )
        return _row(row) if row else None

    async def upsert_patient_map(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data.setdefault("user_id", "")
        data["updated_at"] = _now()
        data.setdefault("created_at", data["updated_at"])
        data.setdefault("verified_at", data["updated_at"])
        if self._pool is None:
            key = self._patient_key(data["member_id"], data["user_id"])
            existing = self._table("his_patient_map").get(key) or {}
            merged = {**existing, **data}
            merged.setdefault("id", len(self._table("his_patient_map")) + 1)
            self._table("his_patient_map")[key] = merged
            return copy.deepcopy(merged)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                INSERT INTO his_patient_map
                    (user_id, member_id, his_tenant_id, his_patient_id, his_bus_no,
                     id_card_hash, id_card_last4, match_method, verified_at, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (user_id, member_id) DO UPDATE SET
                    his_tenant_id=EXCLUDED.his_tenant_id,
                    his_patient_id=EXCLUDED.his_patient_id,
                    his_bus_no=EXCLUDED.his_bus_no,
                    id_card_hash=EXCLUDED.id_card_hash,
                    id_card_last4=EXCLUDED.id_card_last4,
                    match_method=EXCLUDED.match_method,
                    verified_at=EXCLUDED.verified_at,
                    updated_at=EXCLUDED.updated_at
                RETURNING *
                """,
                data["user_id"],
                data["member_id"],
                data.get("his_tenant_id"),
                data["his_patient_id"],
                data.get("his_bus_no"),
                data.get("id_card_hash"),
                data.get("id_card_last4"),
                data.get("match_method"),
                data["verified_at"],
                data["created_at"],
                data["updated_at"],
            )
        return _row(record)

    async def get_encounter_map(self, appointment_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("his_encounter_map").get(appointment_id)
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM his_encounter_map WHERE appointment_id=$1", appointment_id
            )
        return _row(row) if row else None

    async def upsert_encounter_map(self, appointment_id: str, row: dict[str, Any]) -> dict[str, Any]:
        data = {key: value for key, value in row.items() if key in PROJECTION_FIELDS or key in {
            "his_encounter_id", "his_patient_id", "his_org_id", "his_practitioner_id",
            "his_service_id", "quote_id", "payment_id", "invoice_id", "his_status",
            "payment_status", "last_synced_at",
        }}
        data["updated_at"] = _now()
        data.setdefault("created_at", data["updated_at"])
        if self._pool is None:
            existing = self._table("his_encounter_map").get(appointment_id) or {}
            merged = {**existing, **data, "appointment_id": appointment_id}
            self._table("his_encounter_map")[appointment_id] = merged
            return copy.deepcopy(merged)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                INSERT INTO his_encounter_map
                    (appointment_id, his_encounter_id, his_patient_id, his_org_id,
                     his_practitioner_id, his_service_id, quote_id, payment_id, invoice_id,
                     his_status, payment_status, last_synced_at, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (appointment_id) DO UPDATE SET
                    his_encounter_id=EXCLUDED.his_encounter_id,
                    his_patient_id=EXCLUDED.his_patient_id,
                    his_org_id=EXCLUDED.his_org_id,
                    his_practitioner_id=EXCLUDED.his_practitioner_id,
                    his_service_id=EXCLUDED.his_service_id,
                    quote_id=EXCLUDED.quote_id,
                    payment_id=EXCLUDED.payment_id,
                    invoice_id=EXCLUDED.invoice_id,
                    his_status=EXCLUDED.his_status,
                    payment_status=EXCLUDED.payment_status,
                    last_synced_at=EXCLUDED.last_synced_at,
                    updated_at=EXCLUDED.updated_at
                RETURNING *
                """,
                appointment_id,
                data.get("his_encounter_id"),
                data.get("his_patient_id"),
                data.get("his_org_id"),
                data.get("his_practitioner_id"),
                data.get("his_service_id"),
                data.get("quote_id"),
                data.get("payment_id"),
                data.get("invoice_id"),
                data.get("his_status"),
                data.get("payment_status"),
                _as_dt(data.get("last_synced_at")) or data["updated_at"],
                data["created_at"],
                data["updated_at"],
            )
        return _row(record)

    async def upsert_prescription_map(self, row: dict[str, Any]) -> None:
        data = dict(row)
        if self._pool is None:
            self._table("his_prescription_map")[data["his_prescription_no"]] = data
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO his_prescription_map
                    (his_prescription_no, his_encounter_id, his_patient_id, source_table,
                     status, issued_at, synced_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (his_prescription_no) DO UPDATE SET
                    his_encounter_id=EXCLUDED.his_encounter_id,
                    his_patient_id=EXCLUDED.his_patient_id,
                    source_table=EXCLUDED.source_table,
                    status=EXCLUDED.status,
                    issued_at=EXCLUDED.issued_at,
                    synced_at=EXCLUDED.synced_at
                """,
                data["his_prescription_no"],
                data["his_encounter_id"],
                data.get("his_patient_id"),
                data.get("source_table"),
                str(data.get("status")) if data.get("status") is not None else None,
                _as_dt(data.get("issued_at")),
                _now(),
            )

    async def create_projection(self, row: dict[str, Any]) -> dict[str, Any]:
        data = {"id": row["id"]}
        data.update({key: value for key, value in row.items() if key in PROJECTION_FIELDS})
        data.setdefault("user_id", "")
        data.setdefault("amount", 0)
        data.setdefault("status", "quoted")
        data.setdefault("payment_status", "unpaid")
        data.setdefault("payload", {})
        data.setdefault("stale", False)
        data.setdefault("legacy_demo", False)
        now = _now()
        data.setdefault("created_at", now)
        data["updated_at"] = now
        if self._pool is None:
            existing = self._table("appointment_projection").get(data["id"])
            if existing:
                return copy.deepcopy(existing)
            self._table("appointment_projection")[data["id"]] = copy.deepcopy(data)
            return copy.deepcopy(data)
        columns = [
            "id", "user_id", "member_id", "his_patient_id", "his_encounter_id", "his_org_id",
            "his_practitioner_id", "his_service_id", "quote_id", "amount", "status",
            "payment_status", "his_status", "appointment_date", "department", "doctor_name",
            "payment_id", "invoice_id", "payload", "idempotency_key", "last_synced_at",
            "stale", "legacy_demo", "created_at", "updated_at",
        ]
        values = [data.get(name) for name in columns]
        values[columns.index("payload")] = json.dumps(data.get("payload") or {}, ensure_ascii=False)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                f"""
                INSERT INTO appointment_projection ({", ".join(columns)})
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
                        $19::jsonb,$20,$21,$22,$23,$24,$25)
                ON CONFLICT (id) DO NOTHING
                RETURNING *
                """,
                *values,
            )
            if record is None:
                record = await conn.fetchrow(
                    "SELECT * FROM appointment_projection WHERE id=$1", data["id"]
                )
        return _row(record)

    async def get_projection(self, appointment_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("appointment_projection").get(appointment_id)
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM appointment_projection WHERE id=$1", appointment_id
            )
        return _row(row) if row else None

    async def find_projection_by_encounter(self, encounter_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            for row in self._table("appointment_projection").values():
                if row.get("his_encounter_id") == str(encounter_id):
                    return copy.deepcopy(row)
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM appointment_projection WHERE his_encounter_id=$1", str(encounter_id)
            )
        return _row(row) if row else None

    async def update_projection(self, appointment_id: str, **fields: Any) -> dict[str, Any] | None:
        updates = {key: value for key, value in fields.items() if key in PROJECTION_FIELDS}
        if not updates:
            return await self.get_projection(appointment_id)
        updates["updated_at"] = _now()
        if self._pool is None:
            row = self._table("appointment_projection").get(appointment_id)
            if not row:
                return None
            row.update(updates)
            return copy.deepcopy(row)
        assignments = []
        values = []
        for index, (key, value) in enumerate(updates.items(), start=1):
            if key == "payload":
                assignments.append(f"{key}=${index}::jsonb")
                values.append(json.dumps(value or {}, ensure_ascii=False))
            elif key == "last_synced_at":
                assignments.append(f"{key}=${index}")
                values.append(_as_dt(value))
            else:
                assignments.append(f"{key}=${index}")
                values.append(value)
        values.append(appointment_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE appointment_projection SET {', '.join(assignments)} "
                f"WHERE id=${len(values)} RETURNING *",
                *values,
            )
        return _row(row) if row else None

    async def list_projections(self) -> list[dict[str, Any]]:
        if self._pool is None:
            return [copy.deepcopy(row) for row in self._table("appointment_projection").values()]
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM appointment_projection ORDER BY created_at")
        return [_row(row) for row in rows]

    async def outbox_begin(
        self,
        *,
        aggregate_type: str,
        aggregate_id: str,
        operation: str,
        payload: dict[str, Any],
        idempotency_key: str | None,
    ) -> dict[str, Any]:
        now = _now()
        if self._pool is None:
            async with self._lock:
                if idempotency_key:
                    existing = next(
                        (
                            row
                            for row in self._table("sync_outbox").values()
                            if row.get("idempotency_key") == idempotency_key
                        ),
                        None,
                    )
                    if existing:
                        return self._outbox_state(existing, payload)
                row = {
                    "id": _new_id("outbox"),
                    "aggregate_type": aggregate_type,
                    "aggregate_id": aggregate_id,
                    "operation": operation,
                    "payload": copy.deepcopy(payload),
                    "idempotency_key": idempotency_key,
                    "status": "in_progress",
                    "attempts": 0,
                    "response_payload": None,
                    "last_error": None,
                    "created_at": now,
                    "updated_at": now,
                }
                self._table("sync_outbox")[row["id"]] = row
                return {"state": "new", "id": row["id"], "row": copy.deepcopy(row)}
        if idempotency_key:
            async with self._pool.acquire() as conn:
                existing = await conn.fetchrow(
                    "SELECT * FROM sync_outbox WHERE idempotency_key=$1", idempotency_key
                )
            if existing:
                return self._outbox_state(_row(existing), payload)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO sync_outbox
                    (id, aggregate_type, aggregate_id, operation, payload, idempotency_key,
                     status, attempts, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'in_progress', 0, $7, $7)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING *
                """,
                _new_id("outbox"),
                aggregate_type,
                aggregate_id,
                operation,
                json.dumps(payload, ensure_ascii=False, default=str),
                idempotency_key,
                now,
            )
            if row is None:
                row = await conn.fetchrow(
                    "SELECT * FROM sync_outbox WHERE idempotency_key=$1", idempotency_key
                )
                return self._outbox_state(_row(row), payload)
        return {"state": "new", "id": row["id"], "row": _row(row)}

    @staticmethod
    def _outbox_state(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if _canon(row.get("payload")) != _canon(payload):
            return {"state": "conflict", "row": row}
        status = row.get("status")
        if status == "done" and row.get("response_payload") is not None:
            return {"state": "replay", "row": row, "response": row["response_payload"]}
        if status in ("in_progress", "pending"):
            return {"state": "pending", "row": row}
        if status == "dead":
            # A dead letter with identical arguments may be retried; keep the
            # same row/idempotency key so the downstream call remains guarded.
            return {"state": "new", "id": row.get("id"), "row": row}
        return {"state": "conflict", "row": row}

    async def outbox_complete(
        self,
        outbox_id: str,
        status: str,
        *,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = _now()
        if self._pool is None:
            row = self._table("sync_outbox").get(outbox_id)
            if row:
                row.update(
                    {
                        "status": status,
                        "response_payload": response,
                        "last_error": error,
                        "attempts": int(row.get("attempts") or 0) + 1,
                        "updated_at": now,
                    }
                )
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE sync_outbox
                SET status=$2, response_payload=$3::jsonb, last_error=$4,
                    attempts=attempts+1, updated_at=$5
                WHERE id=$1
                """,
                outbox_id,
                status,
                json.dumps(response, ensure_ascii=False, default=str) if response is not None else None,
                error,
                now,
            )

    async def get_outbox(self, outbox_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("sync_outbox").get(outbox_id)
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM sync_outbox WHERE id=$1", outbox_id)
        return _row(row) if row else None

    async def create_human_task(self, row: dict[str, Any]) -> dict[str, Any]:
        data = dict(row)
        data.setdefault("id", _new_id("task"))
        data.setdefault("status", "requested")
        data.setdefault("payload", {})
        now = _now()
        data.setdefault("created_at", now)
        data["updated_at"] = now
        if self._pool is None:
            self._table("human_task")[data["id"]] = copy.deepcopy(data)
            return copy.deepcopy(data)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                INSERT INTO human_task
                    (id, kind, ref_id, user_id, member_id, status, payload, assignee,
                     sla_due_at, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8, $9, $10, $11)
                RETURNING *
                """,
                data["id"],
                data["kind"],
                data["ref_id"],
                data.get("user_id") or "",
                data.get("member_id") or "",
                data["status"],
                json.dumps(data["payload"], ensure_ascii=False),
                data.get("assignee"),
                _as_dt(data.get("sla_due_at")),
                data["created_at"],
                data["updated_at"],
            )
        return _row(record)

    async def get_human_task(self, task_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("human_task").get(task_id)
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM human_task WHERE id=$1", task_id)
        return _row(row) if row else None

    async def find_open_task(self, kind: str, ref_id: str) -> dict[str, Any] | None:
        if self._pool is None:
            for row in self._table("human_task").values():
                if (
                    row.get("kind") == kind
                    and row.get("ref_id") == ref_id
                    and row.get("status") in ("requested", "in_progress")
                ):
                    return copy.deepcopy(row)
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM human_task WHERE kind=$1 AND ref_id=$2 "
                "AND status IN ('requested', 'in_progress') ORDER BY created_at DESC LIMIT 1",
                kind,
                ref_id,
            )
        return _row(row) if row else None

    async def update_human_task(self, task_id: str, **fields: Any) -> dict[str, Any] | None:
        allowed = {"status", "assignee", "payload", "sla_due_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = _now()
        if self._pool is None:
            row = self._table("human_task").get(task_id)
            if not row:
                return None
            row.update(updates)
            return copy.deepcopy(row)
        assignments = []
        values = []
        for index, (key, value) in enumerate(updates.items(), start=1):
            if key == "payload":
                assignments.append(f"{key}=${index}::jsonb")
                values.append(json.dumps(value or {}, ensure_ascii=False))
            else:
                assignments.append(f"{key}=${index}")
                values.append(value)
        values.append(task_id)
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE human_task SET {', '.join(assignments)} "
                f"WHERE id=${len(values)} RETURNING *",
                *values,
            )
        return _row(row) if row else None

    async def add_finding(
        self,
        *,
        kind: str,
        severity: str,
        local_ref: str | None,
        his_ref: str | None = None,
        detail: str | None = None,
    ) -> dict[str, Any]:
        existing = await self.find_open_finding(kind, local_ref)
        if existing:
            return existing
        data = {
            "id": _new_id("find"),
            "kind": kind,
            "severity": severity,
            "local_ref": local_ref,
            "his_ref": his_ref,
            "detail": detail,
            "status": "open",
            "detected_at": _now(),
        }
        if self._pool is None:
            self._table("reconciliation_finding")[data["id"]] = copy.deepcopy(data)
            return copy.deepcopy(data)
        async with self._pool.acquire() as conn:
            record = await conn.fetchrow(
                """
                INSERT INTO reconciliation_finding
                    (id, kind, severity, local_ref, his_ref, detail, status, detected_at)
                VALUES ($1, $2, $3, $4, $5, $6, 'open', $7)
                RETURNING *
                """,
                data["id"],
                kind,
                severity,
                local_ref,
                his_ref,
                detail,
                data["detected_at"],
            )
        return _row(record)

    async def find_open_finding(self, kind: str, local_ref: str | None) -> dict[str, Any] | None:
        if self._pool is None:
            for row in self._table("reconciliation_finding").values():
                if row.get("kind") == kind and row.get("local_ref") == local_ref and row.get("status") == "open":
                    return copy.deepcopy(row)
            return None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM reconciliation_finding WHERE kind=$1 AND local_ref IS NOT DISTINCT FROM $2 "
                "AND status='open' LIMIT 1",
                kind,
                local_ref,
            )
        return _row(row) if row else None

    async def list_findings(self, status: str | None = None) -> list[dict[str, Any]]:
        if self._pool is None:
            rows = list(self._table("reconciliation_finding").values())
            if status:
                rows = [row for row in rows if row.get("status") == status]
            return [copy.deepcopy(row) for row in rows]
        if status:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM reconciliation_finding WHERE status=$1 ORDER BY detected_at", status
                )
        else:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch("SELECT * FROM reconciliation_finding ORDER BY detected_at")
        return [_row(row) for row in rows]

    async def get_cursor(self, stream: str) -> dict[str, Any] | None:
        if self._pool is None:
            row = self._table("his_sync_cursor").get(stream)
            return copy.deepcopy(row) if row else None
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM his_sync_cursor WHERE stream=$1", stream)
        return _row(row) if row else None

    async def set_cursor(
        self, stream: str, last_update_time: str | None = None, last_id: str | None = None
    ) -> None:
        now = _now()
        if self._pool is None:
            self._table("his_sync_cursor")[stream] = {
                "stream": stream,
                "last_update_time": last_update_time,
                "last_id": last_id,
                "updated_at": now,
            }
            return
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO his_sync_cursor (stream, last_update_time, last_id, updated_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (stream) DO UPDATE SET
                    last_update_time=EXCLUDED.last_update_time,
                    last_id=EXCLUDED.last_id,
                    updated_at=EXCLUDED.updated_at
                """,
                stream,
                _as_dt(last_update_time),
                last_id,
                now,
            )

    async def ping(self) -> bool:
        if self._pool is None:
            return True
        async with self._pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True

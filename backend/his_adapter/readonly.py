from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import asyncpg

logger = logging.getLogger("his-adapter.readonly")

STATUS_TEXT = {
    1: "待诊",
    2: "在诊",
    3: "暂离",
    4: "诊毕",
    5: "完成",
    6: "退号",
    7: "已中断",
    8: "错误",
    9: "未知",
}

PAYMENT_STATUS = {
    0: "unknown",
    1: "paid",
    2: "refunded",
    3: "refunded",
    4: "refunded",
    5: "refunded",
    9: "failed",
}


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _amount(value: Any) -> float:
    try:
        return round(float(value or 0), 2)
    except (TypeError, ValueError):
        return 0.0


class ReadOnlyPool:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self._pool: asyncpg.Pool | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.dsn)

    async def _get_pool(self) -> asyncpg.Pool | None:
        if not self.dsn:
            return None
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=3, command_timeout=10)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _check(sql: str) -> str:
        stripped = sql.strip().lstrip("(-").strip().lower()
        if not (stripped.startswith("select") or stripped.startswith("with")):
            raise ValueError("ReadOnlyPool only accepts SELECT statements")
        return sql

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]] | None:
        pool = await self._get_pool()
        if pool is None:
            return None
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(self._check(sql), *args)
            return [dict(row) for row in rows]
        except Exception as exc:
            logger.warning("read-only query failed, falling back to REST: %s", exc)
            return None

    async def encounter(self, encounter_id: str) -> dict[str, Any] | None:
        try:
            identifier = int(encounter_id)
        except (TypeError, ValueError):
            return None
        rows = await self.fetch(
            """
            SELECT id, patient_id, bus_no, status_enum, start_time, create_time, organization_id
            FROM adm_encounter
            WHERE id = $1
            """,
            identifier,
        )
        if not rows:
            return None
        row = rows[0]
        status = row.get("status_enum")
        return {
            "encounter_id": str(row["id"]),
            "patient_id": str(row.get("patient_id") or ""),
            "org_id": str(row.get("organization_id") or ""),
            "practitioner_id": "",
            "status": status,
            "status_text": STATUS_TEXT.get(status, "未知"),
            "start_time": _iso(row.get("start_time")),
            "created_at": _iso(row.get("create_time")),
            "bus_no": row.get("bus_no"),
            "payment_status": None,
        }

    async def find_encounter(
        self,
        patient_id: str,
        org_id: str | None = None,
        service_id: str | None = None,
        day: str | None = None,
        created_after: str | None = None,
    ) -> list[dict[str, Any]] | None:
        try:
            patient = int(patient_id)
        except (TypeError, ValueError):
            return None
        org = int(org_id) if org_id and str(org_id).isdigit() else None
        service = int(service_id) if service_id and str(service_id).isdigit() else None
        since = None
        if created_after:
            try:
                since = datetime.fromisoformat(str(created_after).replace("Z", "+00:00"))
            except ValueError:
                since = None
        rows = await self.fetch(
            """
            SELECT id, patient_id, organization_id, service_type_id, status_enum,
                   start_time, create_time, bus_no
            FROM adm_encounter
            WHERE patient_id = $1
              AND delete_flag = '0'
              AND ($2::bigint IS NULL OR organization_id = $2::bigint)
              AND ($3::bigint IS NULL OR service_type_id = $3::bigint)
              AND ($4::text IS NULL OR COALESCE(create_time, start_time)::date = $4::date)
              AND ($5::timestamptz IS NULL OR COALESCE(create_time, start_time) >= $5::timestamptz)
            ORDER BY id DESC
            LIMIT 5
            """,
            patient,
            org,
            service,
            day,
            since,
        )
        if rows is None:
            return None
        out = []
        for row in rows:
            status = row.get("status_enum")
            out.append(
                {
                    "encounter_id": str(row["id"]),
                    "patient_id": str(row.get("patient_id") or ""),
                    "org_id": str(row.get("organization_id") or ""),
                    "practitioner_id": "",
                    "status": status,
                    "status_text": STATUS_TEXT.get(status, "未知"),
                    "start_time": _iso(row.get("start_time")),
                    "created_at": _iso(row.get("create_time")),
                    "bus_no": row.get("bus_no"),
                    "payment_status": None,
                }
            )
        return out

    async def payment(self, encounter_id: str) -> dict[str, Any] | None:
        try:
            identifier = int(encounter_id)
        except (TypeError, ValueError):
            return None
        rows = await self.fetch(
            """
            SELECT p.id, p.encounter_id, p.display_amount, p.tendered_amount, p.status_enum,
                   p.bill_date, p.invoice_id, i.bill_no
            FROM fin_payment_reconciliation p
            LEFT JOIN adm_invoice i ON i.id = p.invoice_id
            WHERE p.encounter_id = $1
            ORDER BY p.id DESC
            LIMIT 1
            """,
            identifier,
        )
        if rows is None:
            return None
        if not rows:
            return {
                "payment_status": "unpaid",
                "amount": None,
                "paid_at": None,
                "payment_id": None,
                "invoice_no": None,
            }
        row = rows[0]
        return {
            "payment_status": PAYMENT_STATUS.get(row.get("status_enum"), "unknown"),
            "amount": _amount(row.get("display_amount") or row.get("tendered_amount")),
            "paid_at": _iso(row.get("bill_date")),
            "payment_id": str(row.get("id")),
            "invoice_no": row.get("bill_no"),
        }

    async def prescriptions(
        self, encounter_id: str, include_details: bool
    ) -> list[dict[str, Any]] | None:
        try:
            identifier = int(encounter_id)
        except (TypeError, ValueError):
            return None
        rows = await self.fetch(
            """
            SELECT prescription_no, encounter_id, patient_id, issue_time, status_enum,
                   medication_id, med_dosage, quantity
            FROM elep_medication_request
            WHERE encounter_id = $1
            ORDER BY id DESC
            """,
            identifier,
        )
        if rows is None:
            return None
        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            no = str(row.get("prescription_no") or "")
            if not no:
                continue
            item = grouped.setdefault(
                no,
                {
                    "his_prescription_no": no,
                    "encounter_id": str(row.get("encounter_id") or encounter_id),
                    "patient_id": str(row.get("patient_id") or ""),
                    "status": row.get("status_enum"),
                    "issued_at": _iso(row.get("issue_time")),
                    "doctor": None,
                    "item_count": 0,
                },
            )
            item["item_count"] += 1
            if include_details:
                item.setdefault("items", []).append(
                    {
                        "name": row.get("medication_id"),
                        "quantity": row.get("quantity"),
                        "dose": _amount(row.get("med_dosage")),
                        "unit": None,
                        "frequency": None,
                        "route": None,
                    }
                )
        return list(grouped.values())

    INCREMENTAL_SQL = {
        "encounter": """
            SELECT id, patient_id, organization_id, status_enum, start_time, create_time, update_time
            FROM adm_encounter
            WHERE (COALESCE(update_time, create_time), id) > ($1, $2)
            ORDER BY COALESCE(update_time, create_time), id
            LIMIT $3
        """,
        "payment": """
            SELECT id, encounter_id, patient_id, status_enum, display_amount, bill_date, update_time
            FROM fin_payment_reconciliation
            WHERE (COALESCE(update_time, create_time), id) > ($1, $2)
            ORDER BY COALESCE(update_time, create_time), id
            LIMIT $3
        """,
        "prescription": """
            SELECT id, prescription_no, encounter_id, patient_id, status_enum, issue_time, update_time
            FROM elep_medication_request
            WHERE (COALESCE(update_time, issue_time), id) > ($1, $2)
            ORDER BY COALESCE(update_time, issue_time), id
            LIMIT $3
        """,
    }

    async def incremental(
        self,
        stream: str,
        since: datetime | None = None,
        last_id: int | str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]] | None:
        sql = self.INCREMENTAL_SQL.get(stream)
        if sql is None:
            raise ValueError(f"unknown incremental stream {stream!r}")
        anchor = since or datetime.min.replace(tzinfo=UTC)
        try:
            identifier = int(last_id or 0)
        except (TypeError, ValueError):
            identifier = 0
        rows = await self.fetch(sql, anchor, identifier, max(1, min(limit, 1000)))
        if rows is None:
            return None
        return [
            {
                **row,
                "update_time": _iso(row.get("update_time") or row.get("create_time") or row.get("issue_time")),
            }
            for row in rows
        ]

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

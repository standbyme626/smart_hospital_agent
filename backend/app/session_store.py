from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from .db import Database


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


class SessionStore:
    def __init__(self, db: Database):
        self.db = db

    async def ensure(self, session_id: str, user_id: str = "anonymous", title: str | None = None) -> None:
        await self.db.execute(
            """
            INSERT INTO sessions (id, user_id, title)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO UPDATE SET updated_at = now()
            """,
            session_id,
            user_id,
            title,
        )

    async def append(self, session_id: str, event_type: str, data: dict[str, Any]) -> int:
        event_id = uuid.uuid4().hex
        payload = json.dumps(jsonable(data), ensure_ascii=False, default=str)
        seq = await self.db.fetchval(
            """
            INSERT INTO events (session_id, event_id, type, data)
            VALUES ($1, $2, $3, $4::jsonb)
            RETURNING seq
            """,
            session_id,
            event_id,
            event_type,
            payload,
        )
        return int(seq)

    async def events(self, session_id: str, after: int = 0) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """
            SELECT seq, event_id, type, data, created_at
            FROM events
            WHERE session_id = $1 AND seq > $2
            ORDER BY seq ASC
            """,
            session_id,
            after,
        )
        return [
            {
                "seq": row["seq"],
                "event_id": row["event_id"],
                "type": row["type"],
                "data": row["data"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]

    async def list_sessions(self, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = await self.db.fetch(
            """
            SELECT id, user_id, title, created_at, updated_at
            FROM sessions
            WHERE user_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            user_id,
            limit,
        )
        return [dict(row) for row in rows]

    async def audit(
        self,
        *,
        request_id: str,
        session_id: str,
        actor: str,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO audit_log (request_id, session_id, actor, action, detail)
            VALUES ($1, $2, $3, $4, $5::jsonb)
            """,
            request_id,
            session_id,
            actor,
            action,
            json.dumps(jsonable(detail or {}), ensure_ascii=False, default=str),
        )

    async def record_model_call(
        self,
        *,
        request_id: str,
        session_id: str,
        provider: str,
        model: str,
        task: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: int,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO model_calls
                (request_id, session_id, provider, model, task, prompt_tokens, completion_tokens, cost, latency_ms)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            request_id,
            session_id,
            provider,
            model,
            task,
            prompt_tokens,
            completion_tokens,
            cost,
            latency_ms,
        )

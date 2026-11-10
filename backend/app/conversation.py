from __future__ import annotations

import json
from typing import Any


async def append_message(
    db: Any,
    *,
    session_id: str,
    user_id: str,
    member_id: str | None,
    role: str,
    content: str,
    member_context: str | None = None,
) -> None:
    async with db.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO messages
                (session_id, user_id, member_id, role, content, member_context)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            """,
            session_id,
            user_id,
            member_id,
            role,
            content,
            json.dumps({"text": member_context}, ensure_ascii=False) if member_context else None,
        )
        await conn.execute(
            "UPDATE sessions SET last_message_at = now(), updated_at = now() WHERE id = $1",
            session_id,
        )


async def load_history(db: Any, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    rows = await db.fetch(
        """
        SELECT role, content FROM messages
        WHERE session_id = $1
        ORDER BY id DESC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def trim_history(messages: list[dict[str, Any]], max_chars: int = 12000) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    total = 0
    for item in reversed(messages):
        length = len(str(item.get("content") or ""))
        if kept and total + length > max_chars:
            break
        kept.append(item)
        total += length
    return list(reversed(kept))

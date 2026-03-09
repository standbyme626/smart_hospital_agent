from __future__ import annotations

from typing import Any

from app.services.chat_contracts import ChatEvent


def adapt_langgraph_event(event: dict[str, Any]) -> ChatEvent:
    metadata = event.get("metadata", {}) or {}
    node_id = str(metadata.get("langgraph_node") or event.get("name") or "")
    event_name = str(event.get("event") or "")
    payload = event.get("data", {}) or {}

    return ChatEvent(
        event_type=event_name,
        content=str(payload.get("content") or ""),
        phase="",
        node_id=node_id,
        payload=payload if isinstance(payload, dict) else {},
    )

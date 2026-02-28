from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from app.core.config import settings


StreamEventType = Literal[
    "thought",
    "token",
    "status",
    "ping",
    "department_result",
    "doctor_slots",
    "booking_preview",
    "payment_required",
    "booking_confirmed",
    "booking_error",
    "tool_call",
    "tool_output",
    "phase",
    "final",
    "error",
]


class UnifiedStreamEnvelope(BaseModel):
    version: str = Field(default_factory=lambda: settings.UNIFIED_STREAM_SCHEMA_VERSION)
    type: StreamEventType
    content: str = ""
    node: str = ""
    session_id: str = ""
    request_id: str = ""
    seq: int = 0
    stage: str = ""
    ts: float = Field(default_factory=lambda: time.time())
    meta: Dict[str, Any] = Field(default_factory=dict)


def build_stream_payload(
    *,
    event_type: StreamEventType,
    content: str,
    session_id: str = "",
    request_id: str = "",
    seq: int = 0,
    stage: str = "",
    node: str = "",
    meta: Optional[Dict[str, Any]] = None,
    force_unified: bool = False,
) -> Dict[str, Any]:
    """
    Compatibility layer:
    - Legacy mode (default): keeps historical payload keys.
    - Unified mode: returns normalized envelope for chat + doctor/workflow alignment.
    """
    if force_unified or settings.ENABLE_UNIFIED_STREAM_SCHEMA:
        envelope = UnifiedStreamEnvelope(
            type=event_type,
            content=content,
            node=node,
            session_id=session_id,
            request_id=request_id,
            seq=seq,
            stage=stage,
            meta=meta or {},
        )
        return envelope.model_dump()

    payload: Dict[str, Any] = {
        "type": event_type,
        "content": content,
        "ts": time.time(),
    }
    if request_id:
        payload["request_id"] = request_id
    if seq > 0:
        payload["seq"] = seq
    if stage:
        payload["stage"] = stage
    if node:
        payload["node"] = node
    if meta:
        payload.update(meta)
    return payload

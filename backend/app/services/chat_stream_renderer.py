from __future__ import annotations

import json
from typing import Any

from app.core.stream_schema import build_stream_payload


def render_sse_frame(
    *,
    event_type: str,
    content: str,
    session_id: str,
    request_id: str,
    seq: int,
    stage: str = "",
    node: str = "",
    meta: dict[str, Any] | None = None,
) -> str:
    payload = build_stream_payload(
        event_type=event_type,
        content=content,
        session_id=session_id,
        request_id=request_id,
        seq=seq,
        stage=stage,
        node=node,
        meta=meta or {},
    )
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

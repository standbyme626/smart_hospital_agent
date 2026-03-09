from __future__ import annotations

from typing import Any, AsyncGenerator


async def stream_chat_events(
    *,
    message: str,
    session_id: str,
    rag_options: dict[str, Any] | None = None,
    request_id: str | None = None,
    debug_include_nodes: list[str] | None = None,
    rewrite_timeout: float | None = None,
    crisis_fastlane: bool | None = None,
) -> AsyncGenerator[str, None]:
    from app.api.v1.endpoints import chat as chat_endpoint

    async for chunk in chat_endpoint.event_generator(
        message=message,
        session_id=session_id,
        rag_options=rag_options,
        request_id=request_id,
        debug_include_nodes=debug_include_nodes,
        rewrite_timeout=rewrite_timeout,
        crisis_fastlane=crisis_fastlane,
    ):
        yield chunk

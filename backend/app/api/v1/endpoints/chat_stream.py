"""
Phase1 compatibility shell for chat streaming.

The canonical route remains defined in `app.api.v1.endpoints.chat` to avoid
duplicate registration under `/api/v1/chat/stream`.
"""

from fastapi import APIRouter

from app.api.v1.endpoints.chat import event_generator, stream_chat

router = APIRouter()

__all__ = [
    "event_generator",
    "router",
    "stream_chat",
]

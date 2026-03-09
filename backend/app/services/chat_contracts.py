from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass(slots=True)
class ChatRuntimeConfig:
    top_k: int = 3
    rerank_threshold: float | None = None
    rewrite_timeout_s: float = 4.0
    crisis_fastlane: bool = True
    debug_include_nodes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ChatEvent:
    event_type: str
    content: str
    phase: str = ""
    node_id: str = ""
    severity: str = "info"
    terminal: bool = False
    payload: dict[str, Any] = field(default_factory=dict)


class ChatChunk(TypedDict, total=False):
    type: str
    content: str
    session_id: str
    request_id: str
    seq: int
    stage: str
    node: str
    meta: dict[str, Any]

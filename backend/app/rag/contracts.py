from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryContext:
    query: str
    session_id: str | None = None
    request_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RecallHit:
    id: str
    content: str
    score: float
    source: str
    department: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RerankResult:
    hits: list[RecallHit]
    threshold: float | None = None
    used_rerank: bool = False


@dataclass(slots=True)
class PipelineTrace:
    route: str = ""
    steps: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    cache_type: str = "miss"

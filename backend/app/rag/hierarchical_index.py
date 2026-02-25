from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Protocol

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

IndexLevel = Literal["document", "section", "paragraph"]


@dataclass
class HierarchicalHit:
    doc_id: str
    level: IndexLevel
    text: str
    score: float
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class HierarchicalIndexBackend(Protocol):
    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[HierarchicalHit]:
        ...


class NoopHierarchicalIndex:
    """Default placeholder backend: keeps behavior unchanged until real index is ready."""

    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[HierarchicalHit]:
        logger.debug(
            "hierarchical_index_noop",
            query=(query or "")[:80],
            level=level,
            top_k=top_k,
        )
        return []


class HierarchicalIndexGateway:
    def __init__(self, backend: Optional[HierarchicalIndexBackend] = None):
        self.backend: HierarchicalIndexBackend = backend or NoopHierarchicalIndex()

    def set_backend(self, backend: HierarchicalIndexBackend) -> None:
        self.backend = backend

    async def search(
        self,
        query: str,
        level: IndexLevel = "paragraph",
        top_k: int = 3,
        filters: Optional[Dict[str, object]] = None,
        force_enable: bool = False,
    ) -> List[Dict[str, object]]:
        enabled = bool(force_enable or settings.ENABLE_HIERARCHICAL_INDEX)
        if not enabled:
            return []

        normalized_level = (level or "paragraph").lower().strip()
        if normalized_level not in {"document", "section", "paragraph"}:
            normalized_level = "paragraph"

        try:
            hits = await self.backend.search(
                query=query,
                level=normalized_level,  # type: ignore[arg-type]
                top_k=max(1, int(top_k)),
                filters=filters,
            )
            serialized = [h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in hits]
            logger.info(
                "hierarchical_index_search",
                level=normalized_level,
                top_k=top_k,
                hit_count=len(serialized),
            )
            return serialized
        except Exception as exc:
            logger.warning("hierarchical_index_search_failed", error=str(exc))
            return []


hierarchical_index_gateway = HierarchicalIndexGateway()

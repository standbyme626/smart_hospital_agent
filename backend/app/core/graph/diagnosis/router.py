from __future__ import annotations

from typing import Any

from app.core.department_normalization import build_department_result, extract_department_mentions

from .contracts import RouteDecision


def route_case(case_context: dict[str, Any]) -> RouteDecision:
    query = str(case_context.get("query") or case_context.get("message") or "").strip()
    _, canonical = extract_department_mentions(query, top_k=3)
    top1 = canonical[0] if canonical else "Unknown"
    confidence = 0.7 if canonical else 0.0
    return build_department_result(top3=canonical, confidence=confidence, source="route_case_shell")


async def quick_triage_router(state):
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis.quick_triage_router(state)


async def post_retrieval_router(state):
    from app.core.graph.sub_graphs import diagnosis as legacy_diagnosis

    return await legacy_diagnosis.post_retrieval_router(state)


__all__ = [
    "post_retrieval_router",
    "quick_triage_router",
    "route_case",
]

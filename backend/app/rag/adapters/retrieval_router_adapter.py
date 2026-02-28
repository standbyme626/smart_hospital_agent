from __future__ import annotations

from typing import Any, Dict, List, Mapping


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _as_top_k(value: Any, default: int = 3) -> int:
    try:
        parsed = int(value)
    except Exception:
        return max(1, default)
    return max(1, min(10, parsed))


def _as_scope(value: Any, default: str = "paragraph") -> str:
    candidate = _as_text(value).lower() or default
    if candidate not in {"document", "section", "paragraph"}:
        return default
    return candidate


def _as_fusion_method(value: Any, default: str = "weighted_rrf") -> str:
    candidate = _as_text(value).lower() or default
    if candidate not in {"weighted_rrf", "rrf", "concat_merge", "original_only"}:
        return default
    return candidate


def _coerce_variant_payload(variants: Any, query: str) -> List[Dict[str, Any]]:
    if isinstance(variants, list):
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(variants):
            if isinstance(item, dict):
                text = _as_text(item.get("text"))
                if not text:
                    continue
                payload = dict(item)
                payload["text"] = text
                payload.setdefault("type", "original" if idx == 0 else "synonym")
                payload.setdefault("source", "plan")
                payload.setdefault("weight", 1.0 if idx == 0 else 0.8)
                normalized.append(payload)
            elif isinstance(item, str):
                text = _as_text(item)
                if not text:
                    continue
                normalized.append(
                    {
                        "text": text,
                        "type": "original" if idx == 0 else "synonym",
                        "source": "plan",
                        "weight": 1.0 if idx == 0 else 0.8,
                    }
                )
        if normalized:
            return normalized
    fallback = _as_text(query)
    if not fallback:
        return []
    return [{"text": fallback, "type": "original", "source": "fallback", "weight": 1.0}]


class RetrievalRouterAdapter:
    """
    Resolve runtime retrieval knobs from state + retrieval_plan.
    The adapter is feature-flagged and defaults to legacy behavior.
    """

    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = bool(enabled)

    def resolve(
        self,
        *,
        state: Mapping[str, Any],
        fallback_query: str,
        query_source: str,
        default_top_k: int,
        default_index_scope: str,
        default_fusion_method: str,
        default_enable_multi_query: bool,
        default_pure_mode: bool,
        disable_intent_router_when_pure: bool,
    ) -> Dict[str, Any]:
        plan = state.get("retrieval_plan")
        plan = plan if isinstance(plan, dict) else {}
        state_query = _as_text(state.get("retrieval_query"))
        plan_primary = _as_text(plan.get("primary_query"))
        plan_original = _as_text(plan.get("original_query"))
        fallback = _as_text(fallback_query)

        if self.enabled and plan:
            query = plan_primary or plan_original or state_query or fallback
            route_source = "router_adapter"
            resolved_query_source = "retrieval_plan"
            top_k = _as_top_k(plan.get("top_k", default_top_k), default=default_top_k)
            index_scope = _as_scope(plan.get("index_scope", default_index_scope), default=default_index_scope)
            pure_mode = _as_bool(plan.get("pure_mode"), default=default_pure_mode)
            retrieval_query_variants = plan.get("query_variants")
            use_rerank = plan.get("use_rerank")
            if not isinstance(use_rerank, bool):
                use_rerank = None
            rerank_threshold = plan.get("rerank_threshold")
            if not isinstance(rerank_threshold, (int, float)):
                rerank_threshold = None
            else:
                rerank_threshold = max(0.0, min(1.0, float(rerank_threshold)))
        else:
            query = state_query or fallback
            route_source = "legacy_state"
            resolved_query_source = query_source or ("retrieval_query" if state_query else "fallback")
            top_k = _as_top_k(state.get("retrieval_top_k", default_top_k), default=default_top_k)
            index_scope = _as_scope(state.get("retrieval_index_scope", default_index_scope), default=default_index_scope)
            pure_mode = (
                _as_bool(state.get("rag_pure_mode"), default=False)
                if isinstance(state.get("rag_pure_mode"), bool)
                else _as_bool(plan.get("pure_mode"), default=default_pure_mode)
            )
            retrieval_query_variants = state.get("retrieval_query_variants")
            use_rerank = state.get("retrieval_use_rerank")
            if not isinstance(use_rerank, bool):
                use_rerank = None
            rerank_threshold = state.get("retrieval_rerank_threshold")
            if not isinstance(rerank_threshold, (int, float)):
                rerank_threshold = None
            else:
                rerank_threshold = max(0.0, min(1.0, float(rerank_threshold)))

        if not query:
            query = fallback
            resolved_query_source = query_source or "fallback"

        enable_multi_query = _as_bool(
            plan.get("enable_multi_query"),
            default=default_enable_multi_query,
        )
        enable_graph_rag = _as_bool(plan.get("enable_graph_rag"), default=True)
        fusion_method = _as_fusion_method(
            plan.get("fusion_method", default_fusion_method),
            default=default_fusion_method,
        )
        route_mode = _as_text(plan.get("route_mode")).lower()
        if route_mode not in {"pure", "single_query", "multi_query"}:
            if pure_mode:
                route_mode = "pure"
            elif enable_multi_query:
                route_mode = "multi_query"
            else:
                route_mode = "single_query"

        source_priority = plan.get("source_priority")
        if not isinstance(source_priority, list) or not source_priority:
            source_priority = ["vector", "graph", "hierarchical"]

        if not isinstance(retrieval_query_variants, list):
            retrieval_query_variants = state.get("retrieval_query_variants")
        variants_payload = _coerce_variant_payload(retrieval_query_variants, query=query)

        skip_intent_router = _as_bool(
            plan.get("skip_intent_router"),
            default=(pure_mode and bool(disable_intent_router_when_pure)),
        )

        return {
            "route_source": route_source,
            "route_mode": route_mode,
            "query": query,
            "query_source": resolved_query_source,
            "retrieval_query_variants": variants_payload,
            "top_k": top_k,
            "index_scope": index_scope,
            "use_rerank": use_rerank,
            "rerank_threshold": rerank_threshold,
            "pure_mode": pure_mode,
            "enable_multi_query": enable_multi_query,
            "enable_graph_rag": enable_graph_rag,
            "fusion_method": fusion_method,
            "source_priority": source_priority,
            "skip_intent_router": skip_intent_router,
        }

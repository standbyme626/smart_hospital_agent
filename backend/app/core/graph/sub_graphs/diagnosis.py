import asyncio
import hashlib
import time
import uuid
from typing import Literal, Dict, Any, List, Optional, Sequence
from app.core.llm.llm_factory import get_smart_llm
from langchain_core.messages import SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from app.core.config import settings
from app.domain.states.sub_states import DiagnosisState
from app.core.tool_registry import registry
from app.core.prompts.diagnosis import diagnosis_prompt
from app.core.department_normalization import (
    build_department_result,
    extract_department_mentions,
    normalize_department_candidates,
)
from app.rag.graph_rag_service import graph_rag_service
from app.rag.dspy_modules import MedicalConsultant
from app.rag.retrieval_planner import build_retrieval_plan
from app.rag.adapters.query_expander_adapter import QueryExpanderAdapter, extract_variant_texts
from app.rag.adapters.multi_query_retriever_adapter import MultiQueryRetrieverAdapter
from app.rag.adapters.context_window_adapter import ContextWindowAdapter
from app.rag.adapters.json_schema_guardrail import JsonSchemaGuardrail
from app.rag.adapters.retrieval_router_adapter import RetrievalRouterAdapter
import structlog
import dspy

logger = structlog.get_logger(__name__)

# Initialize DSPy module
medical_consultant = MedicalConsultant()

from app.core.services.config_manager import config_manager

# =================================================================
# Node 1: State_Sync
# 职责: 同步 UserProfile（既往史）到诊断上下文，并加载科室配置
# =================================================================
async def state_sync_node(state: DiagnosisState):
    logger.info("Diagnosis Node: State_Sync Start", state_keys=list(state.keys()))
    messages = state.get("messages", [])
    pure_mode = _is_pure_retrieval_mode(state)
    
    # 尝试从消息历史中提取画像和历史（通常由 Ingress 注入到 SystemMessage）
    history_text = "无历史记录"
    profile_text = "未知患者"
    user_profile_obj = state.get("user_profile")
    
    for msg in messages:
        if isinstance(msg, SystemMessage) and "Patient Profile" in msg.content:
            profile_text = msg.content
            break
    
    # 动态加载科室配置
    # 假设 Triage 阶段已经确定了 intent 或 department，如果没有，默认为全科
    department = state.get("department", "general") # 需要确保 Triage 或上游设置了这个字段
    
    # 如果 State 中没有 department，尝试从 user_profile 或上下文推断
    # 这里为了演示，我们假设如果找不到就用 cardiology (Mock) 或者 general
    if department == "general" and not pure_mode:
        # 如果未指定科室，尝试根据 User Profile 或消息内容动态判定
        try:
            logger.info("department_classification_start", profile_snippet=profile_text[:50])
            llm = get_smart_llm(temperature=0.0)
            
            # Simple classification prompt
            prompt = f"""Based on the patient profile and messages, classify the most likely medical department.
            
Profile: {profile_text}
History: {history_text}

Available Departments: Cardiology, Respiratory, Gastroenterology, Neurology, Orthopedics, Dermatology, Pediatrics, General.

Output ONLY the department name in English. If uncertain, output 'General'."""

            ai_msg = await asyncio.wait_for(llm.ainvoke(prompt), timeout=_triage_tool_timeout_s())
            inferred_dept = ai_msg.content.strip().replace(".", "")
            
            # Map back to standard keys if needed (simple normalization)
            if inferred_dept.lower() in ["cardiology", "respiratory", "gastroenterology", "neurology", "orthopedics", "dermatology", "pediatrics"]:
                department = inferred_dept.lower()
                logger.info("department_inferred", department=department)
            else:
                department = "general"
                
        except asyncio.TimeoutError:
            logger.warning("department_inference_timeout", timeout_s=_triage_tool_timeout_s())
            department = "general"
        except Exception as e:
            logger.warning("department_inference_failed", error=str(e))
            department = "general" # Fallback 
    elif pure_mode:
        logger.info("diagnosis_state_sync_pure_mode_skip_department_llm")
        
    dept_config = config_manager.get_config(department)
    system_prompt = config_manager.get_system_prompt(department)
    
    logger.info("diagnosis_config_loaded", department=department, has_config=bool(dept_config))

    # Keep structured user_profile object unchanged to avoid breaking other nodes.
    # Put textual profile into a separate transient field for diagnosis reasoning.
    return {
        "profile_text": profile_text,
        "department": department,
        "system_prompt": system_prompt, # 注入到 State 中供后续节点使用
        "rag_pure_mode": pure_mode,
    }


def _extract_retrieval_query(state: DiagnosisState) -> tuple[str, str]:
    msgs = state.get("messages", [])

    # [Fix] 增强消息提取逻辑，支持 Object, Dict, Tuple, List
    for msg in reversed(msgs):
        msg_type = None
        msg_content = None

        if hasattr(msg, "type") and hasattr(msg, "content"):
            msg_type = msg.type
            msg_content = msg.content
        elif isinstance(msg, dict):
            msg_type = msg.get("type") or msg.get("role")
            msg_content = msg.get("content")
        elif isinstance(msg, (tuple, list)) and len(msg) >= 2:
            msg_type = msg[0]
            msg_content = msg[1]

        if msg_type == "user":
            msg_type = "human"
        if msg_type == "assistant":
            msg_type = "ai"

        if msg_type in ["user", "human"] and msg_content:
            return str(msg_content), "messages"

    # 多级兜底，防止 query 丢失导致 RAG 退化
    retrieval_query = state.get("retrieval_query")
    if isinstance(retrieval_query, str) and retrieval_query.strip():
        return retrieval_query.strip(), "retrieval_query"

    event = state.get("event", {})
    if isinstance(event, dict):
        raw_input = event.get("raw_input")
        if isinstance(raw_input, str) and raw_input.strip():
            return raw_input.strip(), "event.raw_input"

    current_turn_input = state.get("current_turn_input")
    if isinstance(current_turn_input, str) and current_turn_input.strip():
        return current_turn_input.strip(), "current_turn_input"

    user_input = state.get("user_input")
    if isinstance(user_input, str) and user_input.strip():
        return user_input.strip(), "user_input"

    symptoms = state.get("symptoms")
    if isinstance(symptoms, str) and symptoms.strip():
        return symptoms.strip(), "symptoms"

    return "", "none"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    if parsed < 0:
        return 0.0
    if parsed > 1:
        if parsed <= 100:
            return round(parsed / 100.0, 4)
        return 1.0
    return round(parsed, 4)


def _triage_tool_timeout_s() -> float:
    raw = getattr(settings, "TRIAGE_TOOL_TIMEOUT_SECONDS", 2.8)
    try:
        timeout_s = float(raw)
    except Exception:
        timeout_s = 2.8
    return min(max(timeout_s, 1.0), 6.0)


def _triage_fast_conf_threshold() -> float:
    raw = getattr(settings, "TRIAGE_FAST_CONFIDENCE_THRESHOLD", 0.62)
    return min(max(_safe_float(raw, default=0.62), 0.0), 1.0)


def _is_pure_retrieval_mode(state: Dict[str, Any] | None = None) -> bool:
    if isinstance(state, dict):
        state_flag = state.get("rag_pure_mode")
        if isinstance(state_flag, bool):
            return state_flag
    return bool(getattr(settings, "RAG_PURE_RETRIEVAL_MODE", False))


def _default_topk_source_ratio() -> Dict[str, Any]:
    return {
        "original": 1.0,
        "expanded": 0.0,
        "original_count": 0,
        "expanded_count": 0,
    }


def _normalize_fusion_method(raw: Any, default: str = "weighted_rrf") -> str:
    candidate = str(raw or default).strip().lower()
    if candidate not in {"weighted_rrf", "rrf", "concat_merge"}:
        return default
    return candidate


def _default_source_priority() -> List[str]:
    return ["vector", "graph", "hierarchical"]


def _diagnosis_graph_version() -> str:
    return str(getattr(settings, "DIAGNOSIS_GRAPH_VERSION", getattr(settings, "VERSION", "v1")) or "v1")


def _diagnosis_data_contract_version() -> str:
    return str(getattr(settings, "DIAGNOSIS_DATA_CONTRACT_VERSION", "v1") or "v1")


def _is_debug_snapshot_enabled(state: Dict[str, Any] | None = None) -> bool:
    if isinstance(state, dict):
        state_flag = state.get("enable_debug_snapshot")
        if isinstance(state_flag, bool):
            return state_flag
    return bool(getattr(settings, "ENABLE_DEBUG_SNAPSHOT", False))


def _parse_debug_include_nodes(raw: Any) -> List[str]:
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item or "").strip()]
        return list(dict.fromkeys(items))
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
        items = [item for item in parts if item]
        return list(dict.fromkeys(items))
    return []


def _resolve_debug_include_nodes(state: Dict[str, Any] | None = None) -> List[str]:
    if isinstance(state, dict):
        state_nodes = _parse_debug_include_nodes(state.get("debug_include_nodes"))
        if state_nodes:
            return state_nodes
    return _parse_debug_include_nodes(getattr(settings, "DEBUG_INCLUDE_NODES", ""))


def _resolve_request_id(state: Dict[str, Any] | None = None) -> str:
    if isinstance(state, dict):
        candidate = state.get("request_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        event = state.get("event")
        if isinstance(event, dict):
            payload = event.get("payload")
            if isinstance(payload, dict):
                payload_id = payload.get("request_id")
                if isinstance(payload_id, str) and payload_id.strip():
                    return payload_id.strip()
            event_id = event.get("request_id")
            if isinstance(event_id, str) and event_id.strip():
                return event_id.strip()
        session_id = state.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            return f"{session_id.strip()}-{uuid.uuid4().hex[:8]}"
    return f"req-{uuid.uuid4().hex}"


def _hash_value(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _query_ref(value: Any) -> Dict[str, Any]:
    text = str(value or "").strip()
    return {"hash": _hash_value(text), "len": len(text)}


def _safe_doc_refs(docs: Any, *, max_items: int = 8) -> List[Dict[str, Any]]:
    if not isinstance(docs, list):
        return []

    refs: List[Dict[str, Any]] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("doc_id") or item.get("source_id") or "unknown").strip()
        chunk_id = str(item.get("chunk_id") or item.get("id") or doc_id).strip()
        content_hash = str(item.get("hash") or "").strip()
        if not content_hash:
            content = str(item.get("content") or item.get("text") or "").strip()
            content_hash = _hash_value(content) if content else ""

        ref: Dict[str, Any] = {
            "doc_id": doc_id or "unknown",
            "chunk_id": chunk_id or doc_id or "unknown",
            "hash": content_hash,
        }
        source_type = str(item.get("source_type") or item.get("source") or "").strip()
        if source_type:
            ref["source_type"] = source_type
        score = item.get("score")
        if isinstance(score, (int, float)):
            ref["score"] = round(float(score), 6)

        refs.append(ref)
        if len(refs) >= max_items:
            break
    return refs


def _safe_tool_trace(trace: Any) -> Dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    safe = {
        "tool": str(trace.get("tool") or ""),
        "status": str(trace.get("status") or ""),
    }
    timeout_s = trace.get("timeout_s")
    if isinstance(timeout_s, (int, float)):
        safe["timeout_s"] = float(timeout_s)
    error = str(trace.get("error") or "").strip()
    if error:
        safe["error_hash"] = _hash_value(error)
    reason = str(trace.get("reason") or "").strip()
    if reason:
        safe["reason"] = reason
    return safe


def _build_route_snapshot(route_plan: Dict[str, Any]) -> Dict[str, Any]:
    query_variants = route_plan.get("retrieval_query_variants")
    if not isinstance(query_variants, list):
        query_variants = []

    return {
        "route_source": route_plan.get("route_source"),
        "route_mode": route_plan.get("route_mode"),
        "query": _query_ref(route_plan.get("query")),
        "query_source": route_plan.get("query_source"),
        "query_variants": [_query_ref(item.get("text")) for item in query_variants if isinstance(item, dict)],
        "top_k": route_plan.get("top_k"),
        "index_scope": route_plan.get("index_scope"),
        "use_rerank": route_plan.get("use_rerank"),
        "rerank_threshold": route_plan.get("rerank_threshold"),
        "pure_mode": route_plan.get("pure_mode"),
        "enable_multi_query": route_plan.get("enable_multi_query"),
        "fusion_method": route_plan.get("fusion_method"),
        "source_priority": route_plan.get("source_priority"),
        "skip_intent_router": route_plan.get("skip_intent_router"),
    }


def _build_debug_snapshot(
    *,
    state: DiagnosisState,
    node_name: str,
    node_version: str,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not _is_debug_snapshot_enabled(state if isinstance(state, dict) else None):
        return None

    include_nodes = _resolve_debug_include_nodes(state if isinstance(state, dict) else None)
    if not include_nodes:
        return None
    include_nodes_set = {item.lower() for item in include_nodes}
    if node_name.lower() not in include_nodes_set:
        return None

    return {
        "request_id": _resolve_request_id(state if isinstance(state, dict) else None),
        "graph_version": _diagnosis_graph_version(),
        "node_version": str(node_version or "v1"),
        "data_contract_version": _diagnosis_data_contract_version(),
        "schema_version": str(getattr(settings, "DIAGNOSIS_SCHEMA_VERSION", "v1") or "v1"),
        "node": node_name,
        "captured_at": round(time.time(), 6),
        "payload": payload,
    }


def _merge_debug_snapshot(
    *,
    state: DiagnosisState,
    snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}

    existing = state.get("debug_snapshots")
    merged = dict(existing) if isinstance(existing, dict) else {}
    node_name = str(snapshot.get("node") or "unknown")
    merged[node_name] = snapshot
    return {"debug_snapshots": merged}


def _resolve_route_plan(
    *,
    state: DiagnosisState,
    fallback_query: str,
    query_source: str,
) -> Dict[str, Any]:
    configured_fusion_method = _normalize_fusion_method(getattr(settings, "MULTI_QUERY_FUSION_METHOD", "weighted_rrf"))
    default_enable_multi_query = bool(getattr(settings, "ENABLE_MULTI_QUERY", False))
    default_pure_mode = _is_pure_retrieval_mode(state)
    router_adapter = RetrievalRouterAdapter(enabled=bool(getattr(settings, "ENABLE_RETRIEVAL_ROUTER_ADAPTER", False)))
    return router_adapter.resolve(
        state=state if isinstance(state, dict) else {},
        fallback_query=fallback_query,
        query_source=query_source,
        default_top_k=3,
        default_index_scope="paragraph",
        default_fusion_method=configured_fusion_method,
        default_enable_multi_query=default_enable_multi_query,
        default_pure_mode=default_pure_mode,
        disable_intent_router_when_pure=bool(getattr(settings, "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", True)),
    )


def _context_ordering_strategy() -> str:
    raw = str(getattr(settings, "CONTEXT_ORDERING_STRATEGY", "score_desc") or "score_desc").strip().lower()
    if raw not in {"score_desc", "lost_in_middle_mitigate"}:
        return "score_desc"
    return raw


def _context_adapter(*, stage_b_enabled: bool) -> ContextWindowAdapter:
    return ContextWindowAdapter(
        window_size=max(0, int(getattr(settings, "CONTEXT_WINDOW_SIZE", 1))),
        max_evidence=max(1, int(getattr(settings, "CONTEXT_MAX_EVIDENCE", 6))),
        max_per_source=max(1, int(getattr(settings, "CONTEXT_DIVERSITY_MAX_PER_SOURCE", 2))),
        ordering_strategy=_context_ordering_strategy(),
        enable_window=stage_b_enabled,
        enable_merge=stage_b_enabled and bool(getattr(settings, "CONTEXT_AUTOMERGE_ENABLED", True)),
        enable_diversity=stage_b_enabled and bool(getattr(settings, "CONTEXT_DIVERSITY_FILTER_ENABLED", True)),
        max_context_chars=max(400, int(getattr(settings, "CONTEXT_MAX_CHARS", 3200))),
    )


def _coerce_context_docs(
    *,
    vector_docs_override: Optional[Sequence[Dict[str, Any]]],
    fallback_context: str,
) -> List[Dict[str, Any]]:
    if isinstance(vector_docs_override, list) and vector_docs_override:
        return [dict(item) for item in vector_docs_override if isinstance(item, dict)]
    fallback = str(fallback_context or "").strip()
    if not fallback:
        return []
    return [
        {
            "id": "fallback_context",
            "chunk_id": "fallback_context",
            "source_id": "fallback",
            "score": 0.0,
            "content": fallback,
            "source": "fallback",
            "source_type": "fallback",
        }
    ]


def _citations_from_context_pack(state: DiagnosisState, *, max_items: int = 5) -> List[Dict[str, str]]:
    context_pack = state.get("context_pack") if isinstance(state.get("context_pack"), dict) else {}
    evidence = context_pack.get("evidence") if isinstance(context_pack, dict) else None
    if not isinstance(evidence, list):
        return []

    citations: List[Dict[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        doc_id = str(item.get("doc_id") or item.get("source_id") or item.get("chunk_id") or "unknown").strip()
        chunk_id = str(item.get("chunk_id") or item.get("doc_id") or doc_id).strip()
        span = content[:120]
        citations.append({"doc_id": doc_id, "chunk_id": chunk_id, "span": span})
        if len(citations) >= max_items:
            break
    return citations


def _build_diagnosis_output(
    *,
    department_top1: str,
    department_top3: List[str],
    confidence: float,
    reasoning: str,
    citations: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "diagnosis_schema_version": str(getattr(settings, "DIAGNOSIS_SCHEMA_VERSION", "v1") or "v1"),
        "department_top1": str(department_top1 or "Unknown"),
        "department_top3": [str(item) for item in department_top3 if str(item or "").strip()],
        "confidence": _safe_float(confidence, default=0.0),
        "reasoning": str(reasoning or ""),
        "citations": citations,
    }


async def _attach_guarded_diagnosis_output(
    *,
    state: DiagnosisState,
    output: Dict[str, Any],
    diagnosis_output: Dict[str, Any],
) -> Dict[str, Any]:
    guardrail = JsonSchemaGuardrail(
        schema_version=str(getattr(settings, "DIAGNOSIS_SCHEMA_VERSION", "v1") or "v1"),
        enabled=bool(getattr(settings, "ENABLE_JSON_SCHEMA_GUARDRAIL", False)),
    )
    guardrail_result = await guardrail.validate_and_repair(diagnosis_output)
    merged = dict(output)
    merged.update(
        {
            "diagnosis_output": guardrail_result.get("diagnosis_output"),
            "validated": guardrail_result.get("validated"),
            "validation_error": guardrail_result.get("validation_error"),
            "repair_attempted": guardrail_result.get("repair_attempted"),
        }
    )
    return merged


async def _run_tool_with_timeout(
    *,
    name: str,
    coro,
    timeout_s: float,
) -> tuple[Any, Dict[str, Any]]:
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_s)
        return result, {"tool": name, "status": "ok", "timeout_s": timeout_s}
    except asyncio.TimeoutError:
        logger.warning("diagnosis_tool_timeout", tool=name, timeout_s=timeout_s)
        return None, {"tool": name, "status": "timeout", "timeout_s": timeout_s}
    except Exception as exc:
        logger.warning("diagnosis_tool_failed", tool=name, error=str(exc))
        return None, {"tool": name, "status": "error", "error": str(exc)[:300], "timeout_s": timeout_s}


async def _quick_query_hint_tool(query: str) -> Dict[str, Any]:
    _, canonical = extract_department_mentions(query, top_k=3)
    return {"candidates": canonical, "confidence": 0.65 if canonical else 0.0}


async def _quick_retrieval_hint_tool(query: str, top_k: int) -> Dict[str, Any]:
    candidates: List[str] = []

    # Lightweight extractor from SQL pre-filter (pure local parsing).
    try:
        sql_filter = getattr(graph_rag_service.vector_retriever, "sql_filter", None)
        if sql_filter:
            filters = sql_filter.extract_filters(query)
            dept = filters.get("department") if isinstance(filters, dict) else None
            if isinstance(dept, str) and dept.strip():
                candidates.append(dept.strip())
    except Exception as exc:
        logger.warning("quick_retrieval_hint_sql_filter_failed", error=str(exc))

    # Local alias match from query text.
    _, mention_top = extract_department_mentions(query, top_k=max(3, int(top_k)))
    candidates.extend(mention_top)

    canonical_top3 = normalize_department_candidates(candidates, top_k=3)
    if not canonical_top3:
        return {"candidates": [], "confidence": 0.0}

    conf = 0.78 if len(canonical_top3) >= 2 else 0.70
    return {"candidates": canonical_top3, "confidence": conf}


def _merge_fast_hints(query_hint: Dict[str, Any], retrieval_hint: Dict[str, Any]) -> tuple[List[str], float]:
    query_candidates = list(query_hint.get("candidates") or [])
    retrieval_candidates = list(retrieval_hint.get("candidates") or [])
    merged = normalize_department_candidates(query_candidates + retrieval_candidates, top_k=3)
    if not merged:
        return [], 0.0

    query_conf = _safe_float(query_hint.get("confidence"), default=0.0)
    retrieval_conf = _safe_float(retrieval_hint.get("confidence"), default=0.0)
    confidence = max(query_conf, retrieval_conf)

    if query_candidates and retrieval_candidates:
        if query_candidates[0] == retrieval_candidates[0]:
            confidence = min(0.95, confidence + 0.08)
        else:
            confidence = max(0.0, confidence - 0.05)

    return merged, round(confidence, 4)


async def quick_triage_node(state: DiagnosisState):
    query, source = _extract_retrieval_query(state)
    if not query:
        logger.warning("quick_triage_no_query")
        return {"triage_fast_result": None, "triage_fast_ready": False}

    route_plan = _resolve_route_plan(
        state=state,
        fallback_query=query,
        query_source=source,
    )
    resolved_query = str(route_plan.get("query") or query).strip()
    top_k = int(route_plan.get("top_k") or 3)
    timeout_s = _triage_tool_timeout_s()
    threshold = _triage_fast_conf_threshold()

    (query_hint, trace_query), (retrieval_hint, trace_retrieval) = await asyncio.gather(
        _run_tool_with_timeout(
            name="quick_query_hint",
            coro=_quick_query_hint_tool(resolved_query),
            timeout_s=timeout_s,
        ),
        _run_tool_with_timeout(
            name="quick_retrieval_hint",
            coro=_quick_retrieval_hint_tool(resolved_query, int(top_k)),
            timeout_s=timeout_s,
        ),
    )

    merged_top3, confidence = _merge_fast_hints(query_hint or {}, retrieval_hint or {})
    fast_result = build_department_result(top3=merged_top3, confidence=confidence, source="triage_fast_path")
    fast_ready = bool(
        getattr(settings, "TRIAGE_FAST_PATH_ENABLED", True)
        and fast_result
        and _safe_float(fast_result.get("confidence"), default=0.0) >= threshold
    )

    logger.info(
        "quick_triage_result",
        source=source,
        route_source=route_plan.get("route_source"),
        top3=(fast_result or {}).get("department_top3", []),
        confidence=(fast_result or {}).get("confidence", 0.0),
        fast_ready=fast_ready,
        threshold=threshold,
    )

    payload: Dict[str, Any] = {
        "triage_fast_result": fast_result,
        "triage_fast_ready": fast_ready,
        "triage_fast_tool_trace": [trace_query, trace_retrieval],
    }
    if fast_ready and isinstance(fast_result, dict):
        payload.update(
            {
                "department_top1": fast_result.get("department_top1"),
                "department_top3": fast_result.get("department_top3"),
                "recommended_department": fast_result.get("department_top1"),
                "confidence": fast_result.get("confidence"),
            }
        )
    debug_snapshot = _build_debug_snapshot(
        state=state,
        node_name="Quick_Triage",
        node_version="v2",
        payload={
            "route_plan": _build_route_snapshot(route_plan),
            "fast_ready": fast_ready,
            "threshold": threshold,
            "top_k": top_k,
            "triage_fast_result": {
                "department_top1": (fast_result or {}).get("department_top1"),
                "department_top3": (fast_result or {}).get("department_top3"),
                "confidence": (fast_result or {}).get("confidence"),
            },
            "tool_trace": [_safe_tool_trace(trace_query), _safe_tool_trace(trace_retrieval)],
        },
    )
    payload.update(_merge_debug_snapshot(state=state, snapshot=debug_snapshot))
    return payload


async def quick_triage_router(state: DiagnosisState) -> Literal["fast_exit", "deep_diagnosis"]:
    if not getattr(settings, "TRIAGE_FAST_PATH_ENABLED", True):
        return "deep_diagnosis"
    return "fast_exit" if bool(state.get("triage_fast_ready")) else "deep_diagnosis"


# =================================================================
# Node 2: Query_Rewrite
# 职责: Ingress -> Retriever 之间的检索规划（规则优先，模型兜底）
# =================================================================
async def query_rewrite_node(state: DiagnosisState):
    query, source = _extract_retrieval_query(state)
    if not query:
        logger.warning("query_rewrite_no_query_found", available_keys=list(state.keys()))
        return {}

    expander = QueryExpanderAdapter(
        max_variants=max(1, int(getattr(settings, "QUERY_EXPANSION_MAX_VARIANTS", 4))),
        max_query_len_per_variant=max(1, int(getattr(settings, "QUERY_EXPANSION_MAX_QUERY_LEN_PER_VARIANT", 120))),
        rewrite_type_budget=getattr(settings, "QUERY_EXPANSION_REWRITE_TYPE_BUDGET", None),
    )
    default_fusion_method = _normalize_fusion_method(getattr(settings, "MULTI_QUERY_FUSION_METHOD", "weighted_rrf"))
    default_enable_multi_query = bool(getattr(settings, "ENABLE_MULTI_QUERY", False))

    if _is_pure_retrieval_mode(state):
        top_k = state.get("retrieval_top_k", 3) or 3
        raw_top_k_override = state.get("retrieval_top_k_override")
        if isinstance(raw_top_k_override, (int, float)):
            top_k = max(1, min(10, int(raw_top_k_override)))

        raw_use_rerank = state.get("retrieval_use_rerank")
        use_rerank = raw_use_rerank if isinstance(raw_use_rerank, bool) else None

        raw_rerank_threshold = state.get("retrieval_rerank_threshold")
        rerank_threshold = None
        if isinstance(raw_rerank_threshold, (int, float)):
            rerank_threshold = max(0.0, min(1.0, float(raw_rerank_threshold)))

        logger.info(
            "query_rewrite_bypassed_pure_mode",
            source=source,
            query=query[:120],
            top_k=top_k,
            use_rerank=use_rerank,
            rerank_threshold=rerank_threshold,
        )
        expanded_variants = expander.expand(
            query=query,
            planned_variants=[query],
            enable_query_expansion=False,
            original_only=True,
        )
        variant_texts = extract_variant_texts(expanded_variants, original_query=query)
        output = {
            "retrieval_query": variant_texts[0] if variant_texts else query,
            "retrieval_query_variants": expanded_variants,
            "retrieval_top_k": top_k,
            "retrieval_plan": {
                "original_query": query,
                "primary_query": variant_texts[0] if variant_texts else query,
                "query_variants": variant_texts,
                "top_k": top_k,
                "complexity": "pure",
                "rewrite_source": "pure_bypass",
                "index_scope": "paragraph",
                "route_mode": "pure",
                "pure_mode": True,
                "enable_multi_query": default_enable_multi_query,
                "enable_graph_rag": True,
                "source_priority": _default_source_priority(),
                "fusion_method": default_fusion_method,
                "skip_intent_router": bool(getattr(settings, "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", True)),
                "use_rerank": use_rerank,
                "rerank_threshold": rerank_threshold,
                "router_adapter_version": "v1",
            },
            "retrieval_index_scope": "paragraph",
            "retrieval_use_rerank": use_rerank,
            "retrieval_rerank_threshold": rerank_threshold,
            "variant_hits_map": {},
            "topk_source_ratio": _default_topk_source_ratio(),
            "fusion_method": default_fusion_method,
            "rag_pure_mode": True,
        }
        debug_snapshot = _build_debug_snapshot(
            state=state,
            node_name="Query_Rewrite",
            node_version="v2",
            payload={
                "query_source": source,
                "query": _query_ref(query),
                "retrieval_plan": {
                    "top_k": top_k,
                    "index_scope": "paragraph",
                    "route_mode": "pure",
                    "pure_mode": True,
                    "enable_multi_query": default_enable_multi_query,
                    "fusion_method": default_fusion_method,
                    "skip_intent_router": bool(getattr(settings, "RAG_DISABLE_INTENT_ROUTER_WHEN_PURE", True)),
                    "use_rerank": use_rerank,
                    "rerank_threshold": rerank_threshold,
                    "source_priority": _default_source_priority(),
                    "query_variants": [_query_ref(v) for v in variant_texts],
                },
            },
        )
        output.update(_merge_debug_snapshot(state=state, snapshot=debug_snapshot))
        return output

    intent = str(state.get("intent", "") or "")
    plan = await build_retrieval_plan(query=query, intent=intent)
    top_k = plan.top_k
    raw_top_k_override = state.get("retrieval_top_k_override")
    if isinstance(raw_top_k_override, (int, float)):
        top_k = max(1, min(10, int(raw_top_k_override)))

    raw_use_rerank = state.get("retrieval_use_rerank")
    use_rerank = raw_use_rerank if isinstance(raw_use_rerank, bool) else None

    raw_rerank_threshold = state.get("retrieval_rerank_threshold")
    rerank_threshold = None
    if isinstance(raw_rerank_threshold, (int, float)):
        rerank_threshold = max(0.0, min(1.0, float(raw_rerank_threshold)))

    logger.info(
        "query_rewrite_plan",
        source=source,
        primary_query=plan.primary_query[:120],
        top_k=top_k,
        variants=len(plan.query_variants),
        complexity=plan.complexity,
        rewrite_source=plan.rewrite_source,
        use_rerank=use_rerank,
        rerank_threshold=rerank_threshold,
    )

    enable_query_expansion = bool(getattr(settings, "ENABLE_QUERY_EXPANSION", False))
    expanded_variants = expander.expand(
        query=plan.primary_query,
        planned_variants=plan.query_variants,
        enable_query_expansion=enable_query_expansion,
        original_only=not enable_query_expansion,
    )
    variant_texts = extract_variant_texts(expanded_variants, original_query=plan.primary_query)
    retrieval_plan = plan.to_state_dict()
    retrieval_plan["query_variants"] = variant_texts
    retrieval_plan["query_expansion_enabled"] = enable_query_expansion
    retrieval_plan["max_variants"] = expander.max_variants
    retrieval_plan["max_query_len_per_variant"] = expander.max_query_len_per_variant
    retrieval_plan["rewrite_type_budget"] = expander.rewrite_type_budget
    retrieval_plan["route_mode"] = "multi_query" if default_enable_multi_query else "single_query"
    retrieval_plan["pure_mode"] = False
    retrieval_plan["enable_multi_query"] = default_enable_multi_query
    retrieval_plan["enable_graph_rag"] = True
    retrieval_plan["source_priority"] = _default_source_priority()
    retrieval_plan["fusion_method"] = default_fusion_method
    retrieval_plan["skip_intent_router"] = False
    retrieval_plan["use_rerank"] = use_rerank
    retrieval_plan["rerank_threshold"] = rerank_threshold
    retrieval_plan["router_adapter_version"] = "v1"
    output = {
        "retrieval_query": variant_texts[0] if variant_texts else plan.primary_query,
        "retrieval_query_variants": expanded_variants,
        "retrieval_top_k": top_k,
        "retrieval_plan": retrieval_plan,
        "retrieval_index_scope": plan.index_scope,
        "retrieval_use_rerank": use_rerank,
        "retrieval_rerank_threshold": rerank_threshold,
        "variant_hits_map": {},
        "topk_source_ratio": _default_topk_source_ratio(),
        "fusion_method": default_fusion_method,
    }

    debug_snapshot = _build_debug_snapshot(
        state=state,
        node_name="Query_Rewrite",
        node_version="v2",
        payload={
            "query_source": source,
            "query": _query_ref(query),
            "retrieval_plan": {
                "top_k": retrieval_plan.get("top_k"),
                "index_scope": retrieval_plan.get("index_scope"),
                "route_mode": retrieval_plan.get("route_mode"),
                "pure_mode": retrieval_plan.get("pure_mode"),
                "enable_multi_query": retrieval_plan.get("enable_multi_query"),
                "fusion_method": retrieval_plan.get("fusion_method"),
                "skip_intent_router": retrieval_plan.get("skip_intent_router"),
                "use_rerank": retrieval_plan.get("use_rerank"),
                "rerank_threshold": retrieval_plan.get("rerank_threshold"),
                "source_priority": retrieval_plan.get("source_priority"),
                "query_variants": [_query_ref(v) for v in retrieval_plan.get("query_variants", [])],
            },
        },
    )
    output.update(_merge_debug_snapshot(state=state, snapshot=debug_snapshot))
    return output

# =================================================================
# Node 3: Hybrid_Retriever
# 职责: 启动 Neo4j (GraphRAG) 和 Milvus (VectorRAG) 进行知识检索
# =================================================================
async def hybrid_retriever_node(state: DiagnosisState):
    logger.info("Diagnosis Node: Hybrid_Retriever Start")
    planned_query = state.get("retrieval_query")
    if isinstance(planned_query, str) and planned_query.strip():
        last_user_msg = planned_query.strip()
        query_source = "retrieval_query"
    else:
        last_user_msg, query_source = _extract_retrieval_query(state)
    route_plan = _resolve_route_plan(
        state=state,
        fallback_query=last_user_msg,
        query_source=query_source,
    )

    configured_fusion_method = _normalize_fusion_method(getattr(settings, "MULTI_QUERY_FUSION_METHOD", "weighted_rrf"))
    pure_mode = bool(route_plan.get("pure_mode"))
    last_user_msg = str(route_plan.get("query") or "").strip()
    query_source = str(route_plan.get("query_source") or query_source or "none")
    query_variants = route_plan.get("retrieval_query_variants", [])
    variant_texts = extract_variant_texts(query_variants if isinstance(query_variants, list) else None, original_query=last_user_msg)
    top_k = int(route_plan.get("top_k") or 3)
    index_scope = str(route_plan.get("index_scope") or "paragraph")
    use_rerank = route_plan.get("use_rerank")
    rerank_threshold = route_plan.get("rerank_threshold")
    enable_multi_query = bool(route_plan.get("enable_multi_query"))
    route_mode = str(route_plan.get("route_mode") or ("pure" if pure_mode else "single_query"))
    source_priority = route_plan.get("source_priority") if isinstance(route_plan.get("source_priority"), list) else _default_source_priority()
    skip_intent_router = bool(route_plan.get("skip_intent_router"))
    selected_fusion_method = _normalize_fusion_method(
        route_plan.get("fusion_method"),
        default=configured_fusion_method,
    )
    multi_query_adapter = MultiQueryRetrieverAdapter(
        retriever=graph_rag_service.vector_retriever,
        fusion_method=selected_fusion_method,
        rrf_k=max(1, int(getattr(settings, "MULTI_QUERY_RRF_K", 60))),
    )

    logger.info(
        "hybrid_retriever_query",
        query=last_user_msg,
        source=query_source or "none",
        top_k=top_k,
        index_scope=index_scope,
        variant_count=len(variant_texts),
        use_rerank=use_rerank,
        rerank_threshold=rerank_threshold,
        enable_multi_query=enable_multi_query,
        fusion_method=selected_fusion_method,
        pure_mode=pure_mode,
        route_source=route_plan.get("route_source"),
        route_mode=route_mode,
        source_priority=source_priority,
    )

    if not last_user_msg:
        logger.warning("hybrid_retriever_no_query_found", available_keys=list(state.keys()))
        return {}

    # 简单实体提取 (Placeholder for NER)
    entities = [last_user_msg]

    timeout_s = _triage_tool_timeout_s()
    variant_hits_map: Dict[str, List[Dict[str, Any]]] = {}
    topk_source_ratio = _default_topk_source_ratio()
    fusion_method = selected_fusion_method
    vector_docs_override: Optional[List[Dict[str, Any]]] = None
    stage_b_context_enabled = bool(getattr(settings, "ENABLE_CONTEXT_WINDOW_AUTOMERGE", False))
    context_adapter = _context_adapter(stage_b_enabled=stage_b_context_enabled)
    multi_trace: Dict[str, Any] = {
        "tool": "multi_query_retriever",
        "status": "skipped",
        "reason": "feature_flag_disabled",
    }

    if enable_multi_query and variant_texts:
        multi_result, multi_trace = await _run_tool_with_timeout(
            name="multi_query_retriever",
            coro=multi_query_adapter.retrieve(
                query=last_user_msg,
                retrieval_query_variants=query_variants if isinstance(query_variants, list) else None,
                top_k=max(1, int(top_k)),
                enable_multi_query=enable_multi_query,
                original_only=not enable_multi_query,
                use_rerank=use_rerank if isinstance(use_rerank, bool) else None,
                rerank_threshold=float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
                pure_mode=pure_mode,
                fusion_method=selected_fusion_method,
            ),
            timeout_s=timeout_s,
        )
        if isinstance(multi_result, dict):
            if isinstance(multi_result.get("fused_docs"), list):
                vector_docs_override = multi_result.get("fused_docs")
            if isinstance(multi_result.get("variant_hits_map"), dict):
                variant_hits_map = multi_result.get("variant_hits_map")
            if isinstance(multi_result.get("topk_source_ratio"), dict):
                topk_source_ratio = multi_result.get("topk_source_ratio")
            fusion_method = str(multi_result.get("fusion_method") or fusion_method)
        if not variant_hits_map:
            variant_hits_map = {text: [] for text in variant_texts}
    elif variant_texts:
        variant_hits_map = {variant_texts[0]: []}
        fusion_method = "original_only"

    query_variants_for_graph = None
    if enable_multi_query and not vector_docs_override:
        query_variants_for_graph = variant_texts

    result, trace = await _run_tool_with_timeout(
        name="hybrid_retriever",
        coro=graph_rag_service.search(
            query=last_user_msg,
            extracted_entities=entities,
            top_k=max(1, int(top_k)),
            query_variants=query_variants_for_graph,
            vector_docs_override=vector_docs_override if isinstance(vector_docs_override, list) else None,
            index_scope=index_scope,
            use_rerank=use_rerank if isinstance(use_rerank, bool) else None,
            rerank_threshold=float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
            skip_intent_router=skip_intent_router,
            pure_mode=pure_mode,
        ),
        timeout_s=timeout_s,
    )
    if isinstance(result, str) and result.strip():
        context = result
        logger.info("hybrid_retriever_success", context_len=len(context))
    else:
        fast_hint = state.get("triage_fast_result") if isinstance(state.get("triage_fast_result"), dict) else {}
        fast_top3 = fast_hint.get("department_top3") if isinstance(fast_hint.get("department_top3"), list) else []
        fast_text = f"Fast triage candidates: {', '.join(fast_top3)}." if fast_top3 else "Fast triage unavailable."
        context = f"{fast_text}\nKnowledge retrieval timeout/degraded."
        logger.warning("hybrid_retriever_degraded", trace=trace)
    
    context_docs = _coerce_context_docs(
        vector_docs_override=vector_docs_override,
        fallback_context=context,
    )
    context_pack = context_adapter.build_context_pack(
        docs=context_docs,
        fusion_method=fusion_method,
        fallback_context=context,
        ordering_strategy=_context_ordering_strategy(),
    )
    context_for_prompt = (
        context_adapter.render_context(context_pack=context_pack, fallback_context=context)
        if stage_b_context_enabled
        else context
    )

    resolved_retrieval_plan = dict(state.get("retrieval_plan") if isinstance(state.get("retrieval_plan"), dict) else {})
    resolved_retrieval_plan.update(
        {
            "primary_query": last_user_msg,
            "query_variants": variant_texts,
            "top_k": max(1, int(top_k)),
            "index_scope": index_scope,
            "fusion_method": fusion_method,
            "pure_mode": pure_mode,
            "enable_multi_query": enable_multi_query,
            "enable_graph_rag": bool(route_plan.get("enable_graph_rag", True)),
            "source_priority": source_priority,
            "skip_intent_router": skip_intent_router,
            "use_rerank": use_rerank if isinstance(use_rerank, bool) else None,
            "rerank_threshold": float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
            "route_mode": route_mode,
            "route_source": route_plan.get("route_source"),
            "router_adapter_version": "v1",
        }
    )

    # 将检索结果作为 SystemMessage 注入上下文，供 DSPy 使用
    # 使用特定前缀以便 DSPy 节点识别
    output = {
        "messages": [SystemMessage(content=f"Medical Context:\n{context_for_prompt}")],
        "retrieval_query": last_user_msg,
        "retrieval_query_variants": query_variants if isinstance(query_variants, list) else [],
        "retrieval_top_k": max(1, int(top_k)),
        "retrieval_plan": resolved_retrieval_plan,
        "retrieval_index_scope": index_scope,
        "retrieval_use_rerank": use_rerank if isinstance(use_rerank, bool) else None,
        "retrieval_rerank_threshold": float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
        "variant_hits_map": variant_hits_map,
        "topk_source_ratio": topk_source_ratio,
        "fusion_method": fusion_method,
        "context_pack": context_pack,
        "triage_tool_trace": [multi_trace, trace],
        "rag_pure_mode": pure_mode,
    }
    if pure_mode:
        fast_result = state.get("triage_fast_result") if isinstance(state.get("triage_fast_result"), dict) else {}
        candidates: List[str] = []
        if isinstance(fast_result.get("department_top3"), list):
            candidates.extend([str(x) for x in fast_result.get("department_top3", []) if str(x or "").strip()])
        _, mention_top = extract_department_mentions(f"{last_user_msg}\n{context_for_prompt}", top_k=3)
        candidates.extend(mention_top)
        dept_result = build_department_result(top3=candidates or ["全科"], source="diagnosis_pure_rag")
        if dept_result:
            output.update(
                {
                    "pure_retrieval_result": dept_result,
                    "recommended_department": dept_result.get("department_top1"),
                    "department_top1": dept_result.get("department_top1"),
                    "department_top3": dept_result.get("department_top3"),
                    "confidence": dept_result.get("confidence"),
                }
            )
            output["last_tool_result"] = {
                "diagnosis": dept_result.get("department_top1"),
                "confidence": dept_result.get("confidence"),
                "reasoning": "Pure RAG mode: skipped DSPy reasoning, used retrieval-only department extraction.",
                "follow_ups": [],
                "department_top3": dept_result.get("department_top3"),
            }

    variant_hits_summary: List[Dict[str, Any]] = []
    for variant, docs in (variant_hits_map or {}).items():
        if not isinstance(docs, list):
            continue
        variant_hits_summary.append(
            {
                "query": _query_ref(variant),
                "hit_count": len(docs),
                "doc_refs": _safe_doc_refs(docs, max_items=2),
            }
        )
    evidence_refs = _safe_doc_refs(context_pack.get("evidence") if isinstance(context_pack, dict) else [], max_items=8)

    debug_snapshot = _build_debug_snapshot(
        state=state,
        node_name="Hybrid_Retriever",
        node_version="v2",
        payload={
            "route_plan": _build_route_snapshot(route_plan),
            "variant_count": len(variant_texts),
            "variant_hits_map": variant_hits_summary,
            "topk_source_ratio": topk_source_ratio,
            "fusion_method": fusion_method,
            "pure_mode": pure_mode,
            "context_pack": {
                "ordering": context_pack.get("ordering") if isinstance(context_pack, dict) else None,
                "truncation": context_pack.get("truncation") if isinstance(context_pack, dict) else None,
                "evidence_refs": evidence_refs,
            },
            "tool_trace": [_safe_tool_trace(multi_trace), _safe_tool_trace(trace)],
        },
    )
    output.update(_merge_debug_snapshot(state=state, snapshot=debug_snapshot))
    return output

# =================================================================
# Node 4: DSPy_Reasoner
# 职责: 使用 DSPy 结合知识和症状进行结构化推理
# =================================================================
async def dspy_reasoner_node(state: DiagnosisState):
    logger.info("Diagnosis Node: DSPy_Reasoner Start")
    current_loop = state.get("loop_count", 0) + 1
    
    # 1. 准备输入
    # 从 State 中获取 Profile (由 State_Sync 同步)
    profile_text = state.get("profile_text", "")
    if not profile_text:
        user_profile = state.get("user_profile")
        if isinstance(user_profile, str):
            profile_text = user_profile
        elif isinstance(user_profile, dict):
            profile_text = str(user_profile)
        elif user_profile is not None:
            profile_text = str(user_profile)
        else:
            profile_text = "未知患者"
    
    # 从 Messages 中获取 Symptoms 和 Context
    current_symptoms = ""
    conversation_history = ""
    retrieved_knowledge = "未检索到具体知识"
    
    msgs = state.get("messages", [])
    if msgs:
        # Build Conversation History
        history_lines = []
        for msg in msgs:
            role = "unknown"
            if hasattr(msg, 'type'):
                if msg.type in ["user", "human"]: role = "Patient"
                elif msg.type in ["ai", "assistant"]: role = "Doctor"
            
            if role in ["Patient", "Doctor"] and hasattr(msg, 'content'):
                history_lines.append(f"{role}: {msg.content}")
        
        conversation_history = "\n".join(history_lines)

        # Symptoms
        for msg in reversed(msgs):
            if hasattr(msg, 'type') and msg.type in ["user", "human"]:
                current_symptoms = msg.content
                break
        # Context (Look for SystemMessage from Retriever)
        for msg in reversed(msgs):
            if isinstance(msg, SystemMessage) and msg.content and msg.content.startswith("Medical Context"):
                retrieved_knowledge = msg.content
                break

    # 2. 配置 DSPy (Safety Check)
    if not dspy.settings.lm:
         lm = dspy.LM('openai/' + settings.OPENAI_MODEL_SMART, api_key=settings.OPENAI_API_KEY, api_base=settings.OPENAI_API_BASE)
         dspy.configure(lm=lm)

    # 3. 执行推理
    try:
        # [Config] 如果存在动态加载的 System Prompt，应该在这里影响 DSPy 的行为
        # 目前 DSPy 的 Signature 是静态定义的。
        # 我们可以通过 context 参数传递额外指令，或者在 medical_consultant 内部处理。
        # 这里我们将 system_prompt 作为 context 的一部分前置。
        
        system_prompt = state.get("system_prompt", "")
        if system_prompt:
             retrieved_knowledge = f"【系统指令】\n{system_prompt}\n\n【检索知识】\n{retrieved_knowledge}"

        prediction = await asyncio.wait_for(
            asyncio.to_thread(
                medical_consultant,
                patient_profile=profile_text,
                medical_history=profile_text, # 暂时复用 Profile
                conversation_history=conversation_history or "无历史记录",
                current_symptoms=current_symptoms or "无",
                retrieved_knowledge=retrieved_knowledge,
            ),
            timeout=_triage_tool_timeout_s(),
        )
        
        # 4. 保存原始结果到 State
        # 我们需要将 DSPy 的结果传递给 Evaluator
        # 由于 State 类型限制，我们可以将其序列化存入 last_tool_result 或者构造一个临时的 AIMessage
        # 这里我们选择构造一个特殊的 AIMessage，包含 reasoning 元数据
        
        reasoning = prediction.reasoning
        diagnosis_list = prediction.suggested_diagnosis
        confidence = float(prediction.confidence_score) if prediction.confidence_score else 0.0
        follow_ups = prediction.follow_up_questions
        
        diag_str = ", ".join(diagnosis_list) if isinstance(diagnosis_list, list) else str(diagnosis_list)
        
        # 将结果暂存，不在 User 可见的消息流中显示（Evaluator 决定是否显示）
        # 但 LangGraph 的 messages 是追加的。
        # 我们这里返回一个 dict 更新 state 的临时字段，或者使用 last_tool_result 作为一个通用的 payload 容器
        
        result_payload = {
            "diagnosis": diag_str,
            "confidence": confidence,
            "reasoning": reasoning,
            "follow_ups": follow_ups
        }
        
        logger.info("DSPy Reasoning Result", **result_payload)
        
        return {
            "loop_count": current_loop,
            "last_tool_result": result_payload # HACK: Reuse this field to pass data to next node
        }

    except asyncio.TimeoutError:
        fast_result = state.get("triage_fast_result") if isinstance(state.get("triage_fast_result"), dict) else {}
        top3 = fast_result.get("department_top3") if isinstance(fast_result.get("department_top3"), list) else []
        top1 = fast_result.get("department_top1") if isinstance(fast_result.get("department_top1"), str) else "Unknown"
        conf = _safe_float(fast_result.get("confidence"), default=0.55)
        logger.warning("dspy_reasoner_timeout_fallback", timeout_s=_triage_tool_timeout_s(), fast_top3=top3)
        return {
            "loop_count": current_loop,
            "last_tool_result": {
                "diagnosis": top1,
                "confidence": conf,
                "reasoning": f"DSPy超时，使用快速分诊结果兜底（timeout={_triage_tool_timeout_s():.1f}s）",
                "follow_ups": [],
                "department_top3": top3,
            },
        }

    except Exception as e:
        logger.error("DSPy Execution Failed", error=str(e))
        return {"loop_count": current_loop, "last_tool_result": {"error": str(e), "confidence": 0.0}}

def _decision_confidence_threshold() -> float:
    raw = getattr(settings, "DIAGNOSIS_DECISION_CONFIDENCE_THRESHOLD", 0.8)
    return min(max(_safe_float(raw, default=0.8), 0.0), 1.0)


def _decision_grounded_min_evidence() -> int:
    raw = getattr(settings, "DIAGNOSIS_DECISION_MIN_EVIDENCE", 1)
    try:
        return max(1, int(raw))
    except Exception:
        return 1


def _decision_high_risk_keywords() -> List[str]:
    raw = str(
        getattr(
            settings,
            "DIAGNOSIS_DECISION_HIGH_RISK_KEYWORDS",
            "紧急,急性,胸痛,呼吸困难,抽搐,昏迷,休克,high risk,emergency",
        )
        or ""
    )
    keywords = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return keywords or ["紧急", "high risk", "emergency"]


def _decision_has_conflict(payload: Dict[str, Any]) -> bool:
    reasoning = str(payload.get("reasoning") or "").lower()
    diagnosis = str(payload.get("diagnosis") or "").lower()
    if not reasoning and not diagnosis:
        return False
    conflict_markers = ["矛盾", "冲突", "conflict", "inconsistent", "不一致"]
    text = f"{diagnosis}\n{reasoning}"
    return any(marker in text for marker in conflict_markers)


def _decision_is_high_risk(payload: Dict[str, Any]) -> bool:
    text = f"{payload.get('diagnosis', '')}\n{payload.get('reasoning', '')}".lower()
    return any(keyword in text for keyword in _decision_high_risk_keywords())


def _build_decision_contract(*, state: DiagnosisState, payload: Dict[str, Any]) -> Dict[str, Any]:
    confidence_score = _safe_float(payload.get("confidence"), default=0.0)
    evidence = []
    context_pack = state.get("context_pack")
    if isinstance(context_pack, dict):
        evidence = context_pack.get("evidence") if isinstance(context_pack.get("evidence"), list) else []

    evidence_refs = _safe_doc_refs(evidence, max_items=4)
    grounded_flag = len(evidence_refs) >= _decision_grounded_min_evidence() and confidence_score >= 0.4

    if _decision_is_high_risk(payload):
        decision_action = "human_review"
        decision_reason = "high_risk"
    elif len(evidence_refs) < _decision_grounded_min_evidence():
        decision_action = "retrieve_more"
        decision_reason = "insufficient_evidence"
    elif _decision_has_conflict(payload):
        decision_action = "clarify"
        decision_reason = "conflicting_evidence"
    elif confidence_score < _decision_confidence_threshold():
        decision_action = "clarify"
        decision_reason = "low_confidence"
    else:
        decision_action = "end_diagnosis"
        decision_reason = "sufficient_evidence"

    return {
        "decision_action": decision_action,
        "decision_reason": decision_reason,
        "confidence_score": confidence_score,
        "grounded_flag": bool(grounded_flag),
        "evidence_refs": evidence_refs,
    }


# =================================================================
# Node 4: Decision_Judge
# 职责: 统一置信裁决治理（Evidence/Risk/Action）
# =================================================================
async def decision_judge_node(state: DiagnosisState):
    logger.info("Diagnosis Node: Decision_Judge Start")
    payload = state.get("last_tool_result")
    payload = dict(payload) if isinstance(payload, dict) else {}
    decision = _build_decision_contract(state=state, payload=payload)
    payload.update({k: decision[k] for k in ("decision_action", "decision_reason", "confidence_score", "grounded_flag")})

    logger.info(
        "diagnosis_decision_judged",
        action=decision.get("decision_action"),
        reason=decision.get("decision_reason"),
        confidence_score=decision.get("confidence_score"),
        grounded_flag=decision.get("grounded_flag"),
    )

    output = {
        "last_tool_result": payload,
        "decision_action": decision.get("decision_action"),
        "decision_reason": decision.get("decision_reason"),
        "confidence_score": decision.get("confidence_score"),
        "grounded_flag": decision.get("grounded_flag"),
    }
    debug_snapshot = _build_debug_snapshot(
        state=state,
        node_name="Decision_Judge",
        node_version="v1",
        payload={
            "decision_action": decision.get("decision_action"),
            "decision_reason": decision.get("decision_reason"),
            "confidence_score": decision.get("confidence_score"),
            "grounded_flag": decision.get("grounded_flag"),
            "evidence_refs": decision.get("evidence_refs"),
            "diagnosis": _query_ref(payload.get("diagnosis")),
            "reasoning": _query_ref(payload.get("reasoning")),
        },
    )
    output.update(_merge_debug_snapshot(state=state, snapshot=debug_snapshot))
    return output


# =================================================================
# Node 5: Confidence_Evaluator
# 职责: 将裁决动作映射到子图分支
# =================================================================
async def confidence_evaluator_node(state: DiagnosisState) -> Literal["end_diagnosis", "clarify_question"]:
    logger.info("Diagnosis Node: Confidence_Evaluator Start")

    payload = state.get("last_tool_result", {})
    action = state.get("decision_action")
    if not isinstance(action, str) or not action.strip():
        if isinstance(payload, dict):
            action = str(payload.get("decision_action") or "").strip()
        else:
            action = ""

    if action in {"end_diagnosis", "human_review", "reject"}:
        return "end_diagnosis"
    if action in {"retrieve_more", "clarify"}:
        return "clarify_question"

    confidence = _safe_float(payload.get("confidence"), default=0.0) if isinstance(payload, dict) else 0.0
    diagnosis_str = str(payload.get("diagnosis") or "") if isinstance(payload, dict) else ""
    if confidence > _decision_confidence_threshold():
        logger.info("diagnosis_confirmed", confidence=confidence, diagnosis=diagnosis_str)
        return "end_diagnosis"
    if "紧急" in diagnosis_str or "Emergency" in diagnosis_str:
        logger.warning("emergency_detected_in_diagnosis", diagnosis=diagnosis_str)
        return "end_diagnosis"
    logger.info("diagnosis_uncertain_clarify", confidence=confidence)
    return "clarify_question"

# =================================================================
# Helper Nodes for Outputs
# =================================================================
def _decision_fields_from_state(state: DiagnosisState, *, fallback_confidence: float = 0.0) -> Dict[str, Any]:
    payload = state.get("last_tool_result") if isinstance(state.get("last_tool_result"), dict) else {}
    action = str(state.get("decision_action") or payload.get("decision_action") or "").strip()
    reason = str(state.get("decision_reason") or payload.get("decision_reason") or "").strip()
    confidence_score = _safe_float(
        state.get("confidence_score") if state.get("confidence_score") is not None else payload.get("confidence_score"),
        default=fallback_confidence,
    )
    grounded_flag_raw = state.get("grounded_flag")
    if grounded_flag_raw is None:
        grounded_flag_raw = payload.get("grounded_flag")
    grounded_flag = bool(grounded_flag_raw) if isinstance(grounded_flag_raw, bool) else None
    return {
        "decision_action": action or None,
        "decision_reason": reason or None,
        "confidence_score": confidence_score,
        "grounded_flag": grounded_flag,
    }


async def generate_report_node(state: DiagnosisState):
    citations = _citations_from_context_pack(state, max_items=5)

    if _is_pure_retrieval_mode(state):
        dept_result = state.get("pure_retrieval_result") if isinstance(state.get("pure_retrieval_result"), dict) else None
        if not dept_result:
            raw_candidates = state.get("department_top3") if isinstance(state.get("department_top3"), list) else []
            dept_result = build_department_result(top3=raw_candidates or ["全科"], source="diagnosis_pure_rag")
        if dept_result:
            top1 = str(dept_result.get("department_top1") or "")
            top3 = [str(x) for x in dept_result.get("department_top3") or []]
            conf = _safe_float(dept_result.get("confidence"), default=0.7)
            report = (
                "【纯检索分诊建议】\n"
                f"推荐科室: {top1}\n"
                f"候选科室Top3: {', '.join(top3)}\n"
                "说明: 已启用 pure RAG，关闭 Query Rewrite / DSPy 推理，仅基于检索结果。"
            )
            output = {
                "confirmed_diagnosis": top1,
                "clinical_report": report,
                "is_diagnosis_confirmed": True,
                "recommended_department": top1,
                "department_top1": top1,
                "department_top3": top3,
                "confidence": conf,
                "messages": [AIMessage(content=report)],
            }
            output.update(_decision_fields_from_state(state, fallback_confidence=conf))
            diagnosis_output = _build_diagnosis_output(
                department_top1=top1,
                department_top3=top3,
                confidence=conf,
                reasoning="Pure RAG mode: retrieval-only department extraction.",
                citations=citations,
            )
            return await _attach_guarded_diagnosis_output(
                state=state,
                output=output,
                diagnosis_output=diagnosis_output,
            )

    if bool(state.get("triage_fast_ready")):
        fast_result = state.get("triage_fast_result") if isinstance(state.get("triage_fast_result"), dict) else {}
        top3 = fast_result.get("department_top3") if isinstance(fast_result.get("department_top3"), list) else []
        top1 = fast_result.get("department_top1") if isinstance(fast_result.get("department_top1"), str) else ""
        conf = _safe_float(fast_result.get("confidence"), default=0.0)
        if top1 and top3:
            report = (
                f"【快速分诊建议】\n"
                f"推荐科室: {top1}\n"
                f"候选科室Top3: {', '.join(top3)}\n"
                f"置信度: {conf:.2f}"
            )
            output = {
                "confirmed_diagnosis": top1,
                "clinical_report": report,
                "is_diagnosis_confirmed": True,
                "recommended_department": top1,
                "department_top1": top1,
                "department_top3": top3,
                "confidence": conf,
                "messages": [AIMessage(content=report)],
            }
            output.update(_decision_fields_from_state(state, fallback_confidence=conf))
            diagnosis_output = _build_diagnosis_output(
                department_top1=top1,
                department_top3=[str(item) for item in top3],
                confidence=conf,
                reasoning="Fast triage path result.",
                citations=citations,
            )
            return await _attach_guarded_diagnosis_output(
                state=state,
                output=output,
                diagnosis_output=diagnosis_output,
            )

    payload = state.get("last_tool_result", {})
    diag_str = payload.get("diagnosis", "Unknown")
    reasoning = payload.get("reasoning", "")
    confidence = _safe_float(payload.get("confidence"), default=0.0)
    
    report = f"【DSPy 诊断报告】\n诊断: {diag_str}\n置信度: {confidence}\n依据: {reasoning}"
    
    # [Smoke Test Requirement] Auto-switch to REGISTRATION if diagnosis is confirmed
    # This allows the flow to proceed to Service/Booking
    intent_update = "REGISTRATION" if diag_str and diag_str != "Unknown" else None

    dept_candidates = payload.get("department_top3") if isinstance(payload.get("department_top3"), list) else []
    if not dept_candidates:
        _, dept_candidates = extract_department_mentions(f"{diag_str}\n{reasoning}", top_k=3)
    dept_result = build_department_result(top3=dept_candidates, top1=str(diag_str), confidence=confidence, source="diagnosis_report")

    output = {
        "confirmed_diagnosis": diag_str,
        "clinical_report": report,
        "is_diagnosis_confirmed": True,
        "intent": intent_update, # Update intent to trigger router
        "messages": [AIMessage(content=report)]
    }
    output.update(_decision_fields_from_state(state, fallback_confidence=confidence))
    if dept_result:
        output.update(
            {
                "recommended_department": dept_result.get("department_top1"),
                "department_top1": dept_result.get("department_top1"),
                "department_top3": dept_result.get("department_top3"),
                "confidence": dept_result.get("confidence"),
            }
        )
    diagnosis_top1 = str(output.get("department_top1") or diag_str or "Unknown")
    diagnosis_top3 = [str(item) for item in output.get("department_top3", [])] if isinstance(output.get("department_top3"), list) else []
    diagnosis_conf = _safe_float(output.get("confidence"), default=confidence)
    diagnosis_output = _build_diagnosis_output(
        department_top1=diagnosis_top1,
        department_top3=diagnosis_top3,
        confidence=diagnosis_conf,
        reasoning=str(reasoning or ""),
        citations=citations,
    )
    return await _attach_guarded_diagnosis_output(
        state=state,
        output=output,
        diagnosis_output=diagnosis_output,
    )

async def generate_question_node(state: DiagnosisState):
    payload = state.get("last_tool_result", {})
    follow_ups = payload.get("follow_ups", [])
    confidence = payload.get("confidence", 0.0)
    decision_reason = str(state.get("decision_reason") or payload.get("decision_reason") or "").strip()
    
    question = ""
    if follow_ups:
        if isinstance(follow_ups, list) and len(follow_ups) > 0:
            question = follow_ups[0]
        elif isinstance(follow_ups, str) and len(follow_ups.strip()) > 0:
            question = follow_ups
            
    if not question:
         question = "请您详细描述一下您的主要症状，包括发病时间、持续时间以及是否有其他伴随不适？"
         
    reason_text = f"（原因：{decision_reason}）" if decision_reason else ""
    msg = f"基于目前信息，我尚不能完全确定诊断（置信度 {confidence:.2f}）{reason_text}。\n建议补充：{question}"
    
    return {
        "messages": [AIMessage(content=msg)]
        # 不设置 is_diagnosis_confirmed，也不增加 loop_count (reasoner 已增加)
    }


async def post_retrieval_router(state: DiagnosisState) -> Literal["pure_report", "dspy_reasoner"]:
    if _is_pure_retrieval_mode(state):
        return "pure_report"
    return "dspy_reasoner"


def build_diagnosis_graph():
    """
    构建诊断子图 (Diagnosis Subgraph) - [Phase 3 Refactor]
    Nodes: State_Sync -> Query_Rewrite -> Hybrid_Retriever -> DSPy_Reasoner -> Confidence_Evaluator
    """
    workflow = StateGraph(DiagnosisState)
    
    # 1. 初始化模型 (Legacy fallback)
    llm = get_smart_llm(temperature=0.0)

    # =================================================================
    # Graph Construction
    # =================================================================
    workflow.add_node("State_Sync", state_sync_node)
    workflow.add_node("Query_Rewrite", query_rewrite_node)
    workflow.add_node("Quick_Triage", quick_triage_node)
    workflow.add_node("Hybrid_Retriever", hybrid_retriever_node)
    workflow.add_node("DSPy_Reasoner", dspy_reasoner_node)
    decision_governance_enabled = bool(getattr(settings, "ENABLE_DECISION_GOVERNANCE", False))
    if decision_governance_enabled:
        workflow.add_node("Decision_Judge", decision_judge_node)
    
    # 动作节点
    workflow.add_node("Diagnosis_Report", generate_report_node)
    workflow.add_node("Clarify_Question", generate_question_node)

    # 边连接
    workflow.set_entry_point("State_Sync")
    workflow.add_edge("State_Sync", "Query_Rewrite")
    workflow.add_edge("Query_Rewrite", "Quick_Triage")
    workflow.add_conditional_edges(
        "Quick_Triage",
        quick_triage_router,
        {
            "fast_exit": "Diagnosis_Report",
            "deep_diagnosis": "Hybrid_Retriever",
        },
    )
    workflow.add_conditional_edges(
        "Hybrid_Retriever",
        post_retrieval_router,
        {
            "pure_report": "Diagnosis_Report",
            "dspy_reasoner": "DSPy_Reasoner",
        },
    )
    
    # 条件边
    if decision_governance_enabled:
        workflow.add_edge("DSPy_Reasoner", "Decision_Judge")
        workflow.add_conditional_edges(
            "Decision_Judge",
            confidence_evaluator_node,
            {
                "end_diagnosis": "Diagnosis_Report",
                "clarify_question": "Clarify_Question"
            }
        )
    else:
        workflow.add_conditional_edges(
            "DSPy_Reasoner",
            confidence_evaluator_node,
            {
                "end_diagnosis": "Diagnosis_Report",
                "clarify_question": "Clarify_Question"
            }
        )
    
    workflow.add_edge("Diagnosis_Report", END)
    workflow.add_edge("Clarify_Question", END) # 返回给用户等待输入

    return workflow.compile()

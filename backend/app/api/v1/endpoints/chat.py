import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.graph.workflow import app as graph_app
from app.core.monitoring.langfuse_bridge import langfuse_bridge
from app.core.department_normalization import (
    build_department_result,
    extract_department_mentions,
)
from app.core.stream_compat import extract_doctor_slots, extract_ui_payment
from app.core.stream_schema import build_stream_payload
from app.services.mcp.his_server import HISService

router = APIRouter()
logger = logging.getLogger(__name__)

class RagTuningRequest(BaseModel):
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    use_rerank: Optional[bool] = None
    rerank_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    rag: Optional[RagTuningRequest] = None
    request_id: Optional[str] = None
    debug_include_nodes: Optional[list[str]] = None
    rewrite_timeout: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    crisis_fastlane: Optional[bool] = None


def _dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    return model.dict(exclude_none=True)


def _parse_debug_include_nodes(raw: Any) -> List[str]:
    if isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item or "").strip()]
        return list(dict.fromkeys(items))
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return list(dict.fromkeys(items))
    return []


def _default_debug_include_nodes() -> List[str]:
    return _parse_debug_include_nodes(getattr(settings, "DEBUG_INCLUDE_NODES", ""))


def _resolve_rewrite_timeout_s(override: Any = None) -> float:
    raw = override if isinstance(override, (int, float)) else getattr(settings, "QUERY_REWRITE_TIMEOUT_SECONDS", 4.0)
    try:
        timeout_s = float(raw)
    except Exception:
        timeout_s = 4.0
    return min(max(timeout_s, 1.0), 10.0)


def _resolve_runtime_config(
    *,
    rag_options: Dict[str, Any],
    debug_include_nodes: Any,
    rewrite_timeout: Any,
    crisis_fastlane: Any,
) -> Dict[str, Dict[str, Any]]:
    requested: Dict[str, Any] = {}
    effective: Dict[str, Any] = {}

    requested_nodes = _parse_debug_include_nodes(debug_include_nodes)
    effective_nodes = requested_nodes or _default_debug_include_nodes()
    requested["debug_include_nodes"] = requested_nodes
    effective["debug_include_nodes"] = effective_nodes

    requested_top_k = rag_options.get("top_k")
    if isinstance(requested_top_k, (int, float)):
        requested["top_k"] = int(requested_top_k)
        effective["top_k"] = max(1, min(10, int(requested_top_k)))
    else:
        requested["top_k"] = None
        effective["top_k"] = 3

    requested_rerank_threshold = rag_options.get("rerank_threshold")
    if isinstance(requested_rerank_threshold, (int, float)):
        requested["rerank_threshold"] = float(requested_rerank_threshold)
        effective["rerank_threshold"] = max(0.0, min(1.0, float(requested_rerank_threshold)))
    else:
        requested["rerank_threshold"] = None
        effective["rerank_threshold"] = None

    requested["rewrite_timeout_s"] = float(rewrite_timeout) if isinstance(rewrite_timeout, (int, float)) else None
    effective["rewrite_timeout_s"] = _resolve_rewrite_timeout_s(rewrite_timeout)

    if isinstance(crisis_fastlane, bool):
        requested_fastlane = crisis_fastlane
    else:
        requested_fastlane = None
    requested["crisis_fastlane"] = requested_fastlane
    effective["crisis_fastlane"] = (
        requested_fastlane
        if isinstance(requested_fastlane, bool)
        else bool(getattr(settings, "CRISIS_FASTLANE_ENABLED", True))
    )

    return {"requested": requested, "effective": effective}

# Nodes that should never stream model tokens to user.
EXCLUDED_STREAM_NODES = {
    "cache_lookup",
    "persistence",
    "pii_filter",
    "multimodal_processor",
    "history_injector",
    "guard",
    "intent_classifier",
    "hybrid_retriever",
    "State_Sync",
    "DSPy_Reasoner",
    "quality_gate",
}

STATUS_MAP = {
    "unified_preprocessor": "正在分析病情...",
    "fast_track": "正在生成回答...",
    "fast_reply": "正在生成回答...",
    "standard_consultant": "全科医生正在思考...",
    "expert_crew": "专家组正在会诊...",
    "expert_aggregation": "正在整合诊断结果...",
    "service": "正在处理挂号服务...",
    "service_agent": "正在处理挂号服务...",
    "tools": "正在处理挂号工具调用...",
    "intent_classifier": "正在识别意图...",
    "diagnosis": "正在进行诊断分析...",
    "Hybrid_Retriever": "正在检索医学知识...",
    "State_Sync": "正在同步患者上下文...",
    "triage_node": "正在分诊...",
    "DSPy_Reasoner": "专家系统正在推理...",
    "Diagnosis_Report": "正在生成诊断报告...",
    "Clarify_Question": "正在生成追问..."
}

STAGE_NODE_MAP = {
    "Query_Rewrite": "rewrite",
    "Quick_Triage": "rewrite",
    "Hybrid_Retriever": "retrieve",
    "retriever": "retrieve",
    "DSPy_Reasoner": "judge",
    "Decision_Judge": "judge",
    "Diagnosis_Report": "respond",
    "Clarify_Question": "respond",
    "service": "respond",
    "fast_reply": "respond",
    "booking_shortcut": "respond",
}


def _resolve_stage(node: str, event_type: str) -> str:
    node_name = str(node or "").strip()
    if node_name in STAGE_NODE_MAP:
        return STAGE_NODE_MAP[node_name]
    if event_type in {"token", "final"}:
        return "respond"
    if event_type in {"thought", "status", "ping"}:
        return "route"
    return ""


def _resolve_node_name(event: dict) -> str:
    metadata = event.get("metadata", {}) or {}
    return metadata.get("langgraph_node") or event.get("name", "") or ""


def _extract_message_text(msg) -> str:
    if msg is None:
        return ""
    if hasattr(msg, "content"):
        return str(getattr(msg, "content") or "")
    if isinstance(msg, dict):
        return str(msg.get("content") or "")
    return ""


def _message_is_ai(msg) -> bool:
    if msg is None:
        return False
    msg_type = ""
    role = ""
    if hasattr(msg, "type"):
        msg_type = str(getattr(msg, "type") or "").lower()
    if hasattr(msg, "role"):
        role = str(getattr(msg, "role") or "").lower()
    if isinstance(msg, dict):
        msg_type = str(msg.get("type") or "").lower()
        role = str(msg.get("role") or "").lower()
    return msg_type in {"ai", "assistant"} or role == "assistant"


def _extract_texts_from_output(output) -> list[str]:
    texts: list[str] = []
    if output is None:
        return texts

    if hasattr(output, "content"):
        content = _extract_message_text(output).strip()
        if content:
            texts.append(content)
        return texts

    if isinstance(output, dict):
        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if _message_is_ai(msg):
                    content = _extract_message_text(msg).strip()
                    if content:
                        texts.append(content)
                        break

        for key in ("final_output", "clinical_report", "diagnosis_report", "final_response", "response", "content"):
            value = output.get(key)
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())

    if isinstance(output, list):
        for item in output:
            content = _extract_message_text(item).strip()
            if content:
                texts.append(content)

    deduped = []
    seen = set()
    for t in texts:
        norm = " ".join(t.split())
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(t)
    return deduped


def _extract_department_result_from_output(output: Any, text: str = "") -> dict | None:
    candidates: list[str] = []
    confidence: Any = None

    def _add_candidate(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())

    def _add_candidates(values: Any) -> None:
        if isinstance(values, list):
            for item in values:
                _add_candidate(item)

    if isinstance(output, dict):
        triage_result = output.get("triage_result")
        if isinstance(triage_result, dict):
            _add_candidate(triage_result.get("department_top1"))
            _add_candidate(triage_result.get("recommended_department"))
            _add_candidates(triage_result.get("department_top3"))
            if confidence is None:
                confidence = triage_result.get("confidence")

        triage_fast_result = output.get("triage_fast_result")
        if isinstance(triage_fast_result, dict):
            _add_candidate(triage_fast_result.get("department_top1"))
            _add_candidate(triage_fast_result.get("recommended_department"))
            _add_candidates(triage_fast_result.get("department_top3"))
            if confidence is None:
                confidence = triage_fast_result.get("confidence")

        _add_candidate(output.get("department"))
        _add_candidate(output.get("recommended_department"))
        _add_candidate(output.get("department_top1"))
        _add_candidates(output.get("departments"))
        _add_candidates(output.get("department_top3"))
        if confidence is None:
            confidence = output.get("confidence")

    if text:
        _, canonical_top = extract_department_mentions(text, top_k=3)
        candidates.extend(canonical_top)

    return build_department_result(
        top3=candidates,
        confidence=confidence,
        source="chat_stream",
    )


def _sse_payload(
    *,
    event_type: str,
    content: str,
    session_id: str,
    request_id: str,
    seq: int,
    node: str,
    stage: str = "",
    meta: dict | None = None,
) -> str:
    payload = json.dumps(
        build_stream_payload(
            event_type=event_type,  # type: ignore[arg-type]
            content=content,
            session_id=session_id,
            request_id=request_id,
            seq=seq,
            stage=stage or _resolve_stage(node, event_type),
            node=node,
            meta=meta or {},
        ),
        ensure_ascii=False,
    )
    return f"data: {payload}\n\n"


def _extract_command_payload(message: str, prefix: str) -> str:
    if not message:
        return ""
    up = message.upper()
    if not up.startswith(prefix):
        return ""
    return message.split(":", 1)[1].strip() if ":" in message else ""


async def _emit_booking_shortcut(message: str, session_id: str, request_id: str) -> AsyncGenerator[str, None]:
    """
    Deterministic booking path for BOOK:/PAY: commands.
    Avoids routing back to diagnosis chain.
    """
    seq = 0

    def emit(event_type: str, content: str, *, node: str, meta: dict | None = None) -> str:
        nonlocal seq
        seq += 1
        return _sse_payload(
            event_type=event_type,
            content=content,
            session_id=session_id,
            request_id=request_id,
            seq=seq,
            node=node,
            meta=meta,
        )

    def done_payload() -> str:
        return "data: [DONE]\n\n"

    slot_id = _extract_command_payload(message, "BOOK:")
    if slot_id:
        yield emit("thought", "正在锁定号源...", node="booking_shortcut")
        try:
            lock_res = await HISService.lock_slot.ainvoke({"slot_id": slot_id, "patient_id": "guest"})
        except Exception as exc:
            yield emit(
                "booking_error",
                "锁号失败",
                node="booking_shortcut",
                meta={"data": {"message": str(exc), "slot_id": slot_id}},
            )
            yield emit("final", "锁号失败，请稍后重试。", node="booking_shortcut")
            yield done_payload()
            return

        if lock_res.get("status") != "success":
            fail_text = f"号源锁定失败：{lock_res.get('message', 'unknown_error')}。请重试。"
            yield emit("booking_error", "锁号失败", node="booking_shortcut", meta={"data": lock_res})
            yield emit("token", fail_text, node="booking_shortcut")
            yield emit("final", fail_text, node="booking_shortcut")
            yield done_payload()
            return

        preview = {
            "slot_id": slot_id,
            "order_id": lock_res.get("order_id"),
            "slot_info": lock_res.get("slot_info", {}),
            "payment_required": lock_res.get("payment_required"),
        }
        yield emit("booking_preview", "booking_preview", node="booking_shortcut", meta={"data": preview})
        yield emit("payment_required", "payment_required", node="booking_shortcut", meta={"data": preview})
        amount = preview.get("payment_required")
        order_id = preview.get("order_id")
        done_text = f"已锁定号源，订单号 {order_id}，待支付金额 ¥{amount}。请确认支付。"
        yield emit("token", done_text, node="booking_shortcut")
        yield emit("final", done_text, node="booking_shortcut")
        yield done_payload()
        return

    order_id = _extract_command_payload(message, "PAY:")
    if order_id:
        yield emit("thought", "正在确认支付...", node="booking_shortcut")
        try:
            confirm_res = await HISService.confirm_appointment.ainvoke({"order_id": order_id})
        except Exception as exc:
            yield emit(
                "booking_error",
                "支付确认失败",
                node="booking_shortcut",
                meta={"data": {"message": str(exc), "order_id": order_id}},
            )
            yield emit("final", "支付确认失败，请稍后重试。", node="booking_shortcut")
            yield done_payload()
            return

        if confirm_res.get("status") != "success":
            fail_text = f"支付确认失败：{confirm_res.get('message', 'unknown_error')}。请重试。"
            yield emit("booking_error", "支付确认失败", node="booking_shortcut", meta={"data": confirm_res})
            yield emit("token", fail_text, node="booking_shortcut")
            yield emit("final", fail_text, node="booking_shortcut")
            yield done_payload()
            return

        confirmed = {"order_id": order_id, "details": confirm_res.get("details", {})}
        done_text = f"支付完成，预约成功。订单号 {order_id}。"
        yield emit("booking_confirmed", "booking_confirmed", node="booking_shortcut", meta={"data": confirmed})
        yield emit("token", done_text, node="booking_shortcut")
        yield emit("final", done_text, node="booking_shortcut")
        yield done_payload()
        return

    yield emit(
        "booking_error",
        "命令格式错误",
        node="booking_shortcut",
        meta={"data": {"message": "expected BOOK:slot_id or PAY:order_id"}},
    )
    yield emit("final", "命令格式错误", node="booking_shortcut")
    yield done_payload()


def _is_booking_shortcut(message: str) -> bool:
    text = (message or "").strip().upper()
    return text.startswith("BOOK:") or text.startswith("PAY:")


async def event_generator(
    message: str,
    session_id: str,
    rag_options: Optional[dict] = None,
    request_id: Optional[str] = None,
    debug_include_nodes: Optional[list[str]] = None,
    rewrite_timeout: Optional[float] = None,
    crisis_fastlane: Optional[bool] = None,
) -> AsyncGenerator[str, None]:
    """
    SSE Generator using astream_events (v2) for real-time token streaming.
    [V6.7] 过滤 <think> 推理块，仅输出最终回答
    """
    resolved_request_id = str(request_id or "").strip() or f"req-{uuid.uuid4().hex}"
    if _is_booking_shortcut(message):
        async for item in _emit_booking_shortcut(message, session_id, resolved_request_id):
            yield item
        return

    rag_options = rag_options or {}
    top_k_override = rag_options.get("top_k")
    use_rerank = rag_options.get("use_rerank")
    rerank_threshold = rag_options.get("rerank_threshold")
    runtime_config = _resolve_runtime_config(
        rag_options=rag_options,
        debug_include_nodes=debug_include_nodes,
        rewrite_timeout=rewrite_timeout,
        crisis_fastlane=crisis_fastlane,
    )
    runtime_requested = runtime_config["requested"]
    runtime_effective = runtime_config["effective"]
    resolved_debug_nodes = runtime_effective.get("debug_include_nodes")
    if not isinstance(resolved_debug_nodes, list):
        resolved_debug_nodes = []

    inputs = {
        "symptoms": message,
        "user_input": message,            # 兼容 unified_preprocessor
        "current_turn_input": message,    # 本轮输入（防止读到上一轮 AI）
        "retrieval_query": message,       # 统一检索入口字段
        "request_id": resolved_request_id,
        "debug_include_nodes": resolved_debug_nodes,
        "retrieval_top_k_override": int(top_k_override) if isinstance(top_k_override, (int, float)) else None,
        "retrieval_use_rerank": use_rerank if isinstance(use_rerank, bool) else None,
        "retrieval_rerank_threshold": float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
        "query_rewrite_timeout_override_s": runtime_effective.get("rewrite_timeout_s"),
        "crisis_fastlane_override": runtime_effective.get("crisis_fastlane"),
        "runtime_config_requested": runtime_requested,
        "runtime_config_effective": runtime_effective,
        "event": {
            "event_type": "SYMPTOM_DESCRIPTION",
            "payload": {"session_id": session_id, "request_id": resolved_request_id},
            "raw_input": message,
            "timestamp": time.time(),
        },
        "messages": [HumanMessage(content=message)],
        "patient_id": "guest",            # 默认游客
        "session_id": session_id,
        "medical_history": "",            # 初始化空字符串
        "intent": "",
        "status": "",
        "final_output": "",
        "clinical_report": "",
        "diagnostician_output": "",
        "pharmacist_output": "",
        "auditor_output": "",
        "evidence_chain": [],
        "persona_update_proposals": [],
        "usage_statistics": [],
        "images": [],
        "audit_retry_count": 0,
        "departments": [],
        "persona": {},
    }
    config = {"configurable": {"thread_id": session_id}}

    in_think_block = False
    emitted_nodes = set()
    emitted_text_keys = set()
    emitted_slots_keys = set()
    emitted_department_keys = set()
    final_parts: list[str] = []
    emitted_token_count = 0
    emitted_status_count = 0
    seq = 0
    stream_started = time.perf_counter()
    first_event_ms: int | None = None
    first_token_ms: int | None = None
    runtime_config_final = dict(runtime_effective)
    cancelled = False

    def emit(
        event_type: str,
        content: str,
        *,
        node: str,
        meta: dict | None = None,
        stage: str = "",
    ) -> str:
        nonlocal seq, first_event_ms, first_token_ms, emitted_status_count
        if first_event_ms is None:
            first_event_ms = int((time.perf_counter() - stream_started) * 1000)
        if event_type == "token" and first_token_ms is None:
            first_token_ms = int((time.perf_counter() - stream_started) * 1000)
        if event_type == "status":
            emitted_status_count += 1
        seq += 1
        return _sse_payload(
            event_type=event_type,
            content=content,
            session_id=session_id,
            request_id=resolved_request_id,
            seq=seq,
            stage=stage,
            node=node,
            meta=meta,
        )

    ping_interval = float(getattr(settings, "SSE_PING_INTERVAL_SECONDS", 8.0) or 8.0)
    ping_interval = min(max(ping_interval, 2.0), 30.0)

    langfuse_bridge.ensure_trace(
        request_id=resolved_request_id,
        session_id=session_id,
        user_intent="",
        metadata={
            "entry": "chat_stream",
            "request_id": resolved_request_id,
            "message_len": len(str(message or "")),
            "runtime_config_requested": runtime_requested,
            "runtime_config_effective": runtime_effective,
        },
    )

    yield emit(
        "status",
        "stream_opened",
        node="chat_stream",
        meta={
            "phase": "opened",
            "metrics": {"t_connect_ms": 0},
            "runtime_config_requested": runtime_requested,
            "runtime_config_effective": runtime_effective,
        },
        stage="route",
    )

    try:
        events_iter = graph_app.astream_events(inputs, config=config, version="v2").__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(events_iter.__anext__(), timeout=ping_interval)
            except asyncio.TimeoutError:
                yield emit(
                    "ping",
                    "keep_alive",
                    node="chat_stream",
                    meta={"elapsed_ms": int((time.perf_counter() - stream_started) * 1000)},
                    stage="route",
                )
                continue
            except StopAsyncIteration:
                break

            kind = event.get("event")
            if kind == "on_chat_model_stream":
                node_name = _resolve_node_name(event)
                if node_name in EXCLUDED_STREAM_NODES:
                    continue

                chunk = event.get("data", {}).get("chunk")
                if hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    
                    # [V6.7] 过滤 <think> 块
                    if "<think>" in content:
                        in_think_block = True
                        content = content.split("<think>")[0]
                    if "</think>" in content:
                        in_think_block = False
                        content = content.split("</think>")[-1]
                        
                    # 在 think 块内，不输出
                    if in_think_block:
                        continue
                        
                    # 过滤空内容和单独的换行
                    if content and content.strip():
                        yield emit("token", content, node=node_name)
                        emitted_token_count += 1
                        final_parts.append(content)

            elif kind == "on_chain_start":
                node_name = _resolve_node_name(event)
                if not node_name:
                    continue
                if node_name in STATUS_MAP and node_name not in emitted_nodes:
                    emitted_nodes.add(node_name)
                    status_text = STATUS_MAP[node_name]
                    yield emit("thought", status_text, node=node_name)
                elif node_name not in emitted_nodes and node_name not in {"LangGraph"}:
                    emitted_nodes.add(node_name)
                    yield emit("thought", f"正在处理 {node_name}...", node=node_name)

            elif kind == "on_chain_end":
                node_name = _resolve_node_name(event)
                if node_name in {"cache_lookup", "persistence"}:
                    continue

                output = event.get("data", {}).get("output", {})
                if node_name == "Query_Rewrite" and isinstance(output, dict):
                    retrieval_plan = output.get("retrieval_plan")
                    if isinstance(retrieval_plan, dict):
                        rewrite_fallback = bool(retrieval_plan.get("rewrite_fallback"))
                        fallback_reason = str(retrieval_plan.get("rewrite_fallback_reason") or "").strip()
                        crisis_fastlane = bool(retrieval_plan.get("crisis_fastlane"))
                        effective_runtime_config = retrieval_plan.get("effective_runtime_config")
                        if isinstance(effective_runtime_config, dict):
                            runtime_config_final = dict(effective_runtime_config)
                            yield emit(
                                "status",
                                "runtime_config_applied",
                                node=node_name,
                                meta={"runtime_config_effective": runtime_config_final},
                                stage="rewrite",
                            )
                            langfuse_bridge.annotate_trace(
                                resolved_request_id,
                                metadata={"runtime_config_effective": runtime_config_final},
                            )
                        if rewrite_fallback or crisis_fastlane:
                            rewrite_meta = {
                                "rewrite_fallback": rewrite_fallback,
                                "fallback_reason": fallback_reason or "",
                                "crisis_fastlane": crisis_fastlane,
                                "runtime_config_effective": runtime_config_final,
                            }
                            yield emit(
                                "status",
                                "rewrite_path",
                                node=node_name,
                                meta=rewrite_meta,
                                stage="rewrite",
                            )
                            langfuse_bridge.annotate_trace(
                                resolved_request_id,
                                metadata={"query_rewrite": rewrite_meta},
                            )

                dept_result = _extract_department_result_from_output(output=output)
                if dept_result:
                    dept_key = json.dumps(dept_result, ensure_ascii=False, sort_keys=True)
                    if dept_key not in emitted_department_keys:
                        emitted_department_keys.add(dept_key)
                        yield emit(
                            "department_result",
                            "department_result",
                            node=node_name,
                            meta={"department_result": dept_result, "data": dept_result},
                        )

                texts = _extract_texts_from_output(output)
                for text in texts:
                    key = " ".join(text.split())
                    if not key or key in emitted_text_keys:
                        continue
                    emitted_text_keys.add(key)
                    yield emit("token", text, node=node_name)
                    emitted_token_count += 1
                    final_parts.append(text)

                    dept_result = _extract_department_result_from_output(output=output, text=text)
                    if dept_result:
                        dept_key = json.dumps(dept_result, ensure_ascii=False, sort_keys=True)
                        if dept_key not in emitted_department_keys:
                            emitted_department_keys.add(dept_key)
                            yield emit(
                                "department_result",
                                "department_result",
                                node=node_name,
                                meta={"department_result": dept_result, "data": dept_result},
                            )

                    for slots_payload in extract_doctor_slots(text):
                        slots_key = json.dumps(slots_payload, ensure_ascii=False, sort_keys=True, default=str)
                        if slots_key in emitted_slots_keys:
                            continue
                        emitted_slots_keys.add(slots_key)
                        yield emit(
                            "doctor_slots",
                            "doctor_slots",
                            node=node_name,
                            meta={"data": slots_payload, "slots": slots_payload},
                        )

                    for payment_payload in extract_ui_payment(text):
                        yield emit(
                            "payment_required",
                            "payment_required",
                            node=node_name,
                            meta={"data": payment_payload},
                        )
    except asyncio.CancelledError:
        cancelled = True
        logger.info("chat_stream_cancelled", request_id=resolved_request_id, session_id=session_id)
    except Exception as e:
        logger.error("chat_stream_error", request_id=resolved_request_id, error=str(e))
        yield emit(
            "error",
            str(e),
            node="chat_stream",
            meta={"request_id": resolved_request_id},
            stage="route",
        )
    finally:
        total_ms = int((time.perf_counter() - stream_started) * 1000)
        metrics = {
            "t_connect_ms": 0,
            "t_first_event_ms": first_event_ms if first_event_ms is not None else -1,
            "t_first_token_ms": first_token_ms if first_token_ms is not None else -1,
            "t_final_ms": total_ms,
            "emitted_token_count": emitted_token_count,
            "emitted_status_count": emitted_status_count,
            "cancelled": cancelled,
        }
        langfuse_bridge.annotate_trace(
            resolved_request_id,
            metadata={
                "sse_metrics": metrics,
                "runtime_config_effective": runtime_config_final,
            },
        )
        langfuse_bridge.finish_trace(
            request_id=resolved_request_id,
            output={
                "token_count": emitted_token_count,
                "cancelled": cancelled,
            },
            metadata={
                "sse_metrics": metrics,
                "runtime_config_effective": runtime_config_final,
            },
        )

        if not cancelled:
            final_text = "".join(final_parts).strip()
            yield emit(
                "final",
                final_text or "stream_completed",
                node="chat_stream",
                meta={"metrics": metrics},
                stage="respond",
            )
            yield emit(
                "status",
                "stream_closed",
                node="chat_stream",
                meta={
                    "phase": "closed",
                    "metrics": metrics,
                    "runtime_config_effective": runtime_config_final,
                },
                stage="route",
            )
            yield "data: [DONE]\n\n"

@router.post("/stream")
async def stream_chat(payload: ChatRequest, request: Request):
    """
    SSE Endpoint for Real-time Chat
    """
    resolved_request_id = (
        str(
            payload.request_id
            or request.headers.get("X-Request-ID")
            or getattr(getattr(request, "state", object()), "request_id", "")
        ).strip()
        or f"req-{uuid.uuid4().hex}"
    )
    return StreamingResponse(
        event_generator(
            payload.message,
            payload.session_id,
            _dump_model(payload.rag) if payload.rag else None,
            resolved_request_id,
            payload.debug_include_nodes,
            payload.rewrite_timeout,
            payload.crisis_fastlane,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": resolved_request_id,
        },
    )

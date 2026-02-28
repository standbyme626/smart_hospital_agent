import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.graph.workflow import app as graph_app
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


def _dump_model(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)  # type: ignore[attr-defined]
    return model.dict(exclude_none=True)

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
    node: str,
    meta: dict | None = None,
) -> str:
    payload = json.dumps(
        build_stream_payload(
            event_type=event_type,  # type: ignore[arg-type]
            content=content,
            session_id=session_id,
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


async def _emit_booking_shortcut(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    Deterministic booking path for BOOK:/PAY: commands.
    Avoids routing back to diagnosis chain.
    """
    slot_id = _extract_command_payload(message, "BOOK:")
    if slot_id:
        yield _sse_payload(
            event_type="thought",
            content="正在锁定号源...",
            session_id=session_id,
            node="booking_shortcut",
        )
        try:
            lock_res = await HISService.lock_slot.ainvoke({"slot_id": slot_id, "patient_id": "guest"})
        except Exception as exc:
            yield _sse_payload(
                event_type="booking_error",
                content="锁号失败",
                session_id=session_id,
                node="booking_shortcut",
                meta={"data": {"message": str(exc), "slot_id": slot_id}},
            )
            yield "data: [DONE]\n\n"
            return

        if lock_res.get("status") != "success":
            yield _sse_payload(
                event_type="booking_error",
                content="锁号失败",
                session_id=session_id,
                node="booking_shortcut",
                meta={"data": lock_res},
            )
            yield _sse_payload(
                event_type="token",
                content=f"号源锁定失败：{lock_res.get('message', 'unknown_error')}。请重试。",
                session_id=session_id,
                node="booking_shortcut",
            )
            yield "data: [DONE]\n\n"
            return

        preview = {
            "slot_id": slot_id,
            "order_id": lock_res.get("order_id"),
            "slot_info": lock_res.get("slot_info", {}),
            "payment_required": lock_res.get("payment_required"),
        }
        yield _sse_payload(
            event_type="booking_preview",
            content="booking_preview",
            session_id=session_id,
            node="booking_shortcut",
            meta={"data": preview},
        )
        yield _sse_payload(
            event_type="payment_required",
            content="payment_required",
            session_id=session_id,
            node="booking_shortcut",
            meta={"data": preview},
        )
        amount = preview.get("payment_required")
        order_id = preview.get("order_id")
        yield _sse_payload(
            event_type="token",
            content=f"已锁定号源，订单号 {order_id}，待支付金额 ¥{amount}。请确认支付。",
            session_id=session_id,
            node="booking_shortcut",
        )
        yield "data: [DONE]\n\n"
        return

    order_id = _extract_command_payload(message, "PAY:")
    if order_id:
        yield _sse_payload(
            event_type="thought",
            content="正在确认支付...",
            session_id=session_id,
            node="booking_shortcut",
        )
        try:
            confirm_res = await HISService.confirm_appointment.ainvoke({"order_id": order_id})
        except Exception as exc:
            yield _sse_payload(
                event_type="booking_error",
                content="支付确认失败",
                session_id=session_id,
                node="booking_shortcut",
                meta={"data": {"message": str(exc), "order_id": order_id}},
            )
            yield "data: [DONE]\n\n"
            return

        if confirm_res.get("status") != "success":
            yield _sse_payload(
                event_type="booking_error",
                content="支付确认失败",
                session_id=session_id,
                node="booking_shortcut",
                meta={"data": confirm_res},
            )
            yield _sse_payload(
                event_type="token",
                content=f"支付确认失败：{confirm_res.get('message', 'unknown_error')}。请重试。",
                session_id=session_id,
                node="booking_shortcut",
            )
            yield "data: [DONE]\n\n"
            return

        confirmed = {"order_id": order_id, "details": confirm_res.get("details", {})}
        yield _sse_payload(
            event_type="booking_confirmed",
            content="booking_confirmed",
            session_id=session_id,
            node="booking_shortcut",
            meta={"data": confirmed},
        )
        yield _sse_payload(
            event_type="token",
            content=f"支付完成，预约成功。订单号 {order_id}。",
            session_id=session_id,
            node="booking_shortcut",
        )
        yield "data: [DONE]\n\n"
        return

    yield _sse_payload(
        event_type="booking_error",
        content="命令格式错误",
        session_id=session_id,
        node="booking_shortcut",
        meta={"data": {"message": "expected BOOK:slot_id or PAY:order_id"}},
    )
    yield "data: [DONE]\n\n"


def _is_booking_shortcut(message: str) -> bool:
    text = (message or "").strip().upper()
    return text.startswith("BOOK:") or text.startswith("PAY:")

async def event_generator(
    message: str,
    session_id: str,
    rag_options: Optional[dict] = None,
    request_id: Optional[str] = None,
    debug_include_nodes: Optional[list[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    SSE Generator using astream_events (v2) for real-time token streaming.
    [V6.7] 过滤 <think> 推理块，仅输出最终回答
    """
    if _is_booking_shortcut(message):
        async for item in _emit_booking_shortcut(message, session_id):
            yield item
        return

    # [Fix] 完善入口状态初始化，确保所有必需字段存在
    rag_options = rag_options or {}
    top_k_override = rag_options.get("top_k")
    use_rerank = rag_options.get("use_rerank")
    rerank_threshold = rag_options.get("rerank_threshold")

    resolved_request_id = str(request_id or "").strip() or f"req-{uuid.uuid4().hex}"
    inputs = {
        "symptoms": message,
        "user_input": message,            # 兼容 unified_preprocessor
        "current_turn_input": message,    # 本轮输入（防止读到上一轮 AI）
        "retrieval_query": message,       # 统一检索入口字段
        "request_id": resolved_request_id,
        "debug_include_nodes": debug_include_nodes if isinstance(debug_include_nodes, list) else None,
        "retrieval_top_k_override": top_k_override if isinstance(top_k_override, int) else None,
        "retrieval_use_rerank": use_rerank if isinstance(use_rerank, bool) else None,
        "retrieval_rerank_threshold": float(rerank_threshold) if isinstance(rerank_threshold, (int, float)) else None,
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
    
    # [V6.7] 状态追踪：过滤 <think> 块
    in_think_block = False
    emitted_nodes = set()  # 避免重复发送 thought
    emitted_text_keys = set()
    emitted_slots_keys = set()
    emitted_department_keys = set()
    emitted_token_count = 0
    
    try:
        # Use astream_events to capture internal events
        async for event in graph_app.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]
            
            # 1. Token Streaming (Real-time text)
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
                        payload = json.dumps(
                            build_stream_payload(
                                event_type="token",
                                content=content,
                                session_id=session_id,
                                node=node_name,
                            ),
                            ensure_ascii=False,
                        )
                        yield f"data: {payload}\n\n"
                        emitted_token_count += 1

            # 2. Node Status Updates (Thought chain)
            elif kind == "on_chain_start":
                node_name = _resolve_node_name(event)
                if not node_name:
                    continue
                if node_name in STATUS_MAP and node_name not in emitted_nodes:
                    emitted_nodes.add(node_name)
                    status_text = STATUS_MAP[node_name]
                    payload = json.dumps(
                        build_stream_payload(
                            event_type="thought",
                            content=status_text,
                            session_id=session_id,
                            node=node_name,
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
                elif node_name not in emitted_nodes and node_name not in {"LangGraph"}:
                    emitted_nodes.add(node_name)
                    payload = json.dumps(
                        build_stream_payload(
                            event_type="thought",
                            content=f"正在处理 {node_name}...",
                            session_id=session_id,
                            node=node_name,
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"

            # 3. Handle non-streaming nodes that produce final output
            elif kind == "on_chain_end":
                node_name = _resolve_node_name(event)
                if node_name in {"cache_lookup", "persistence"}:
                    continue

                output = event.get("data", {}).get("output", {})
                dept_result = _extract_department_result_from_output(output=output)
                if dept_result:
                    dept_key = json.dumps(dept_result, ensure_ascii=False, sort_keys=True)
                    if dept_key not in emitted_department_keys:
                        emitted_department_keys.add(dept_key)
                        dept_event = json.dumps(
                            build_stream_payload(
                                event_type="department_result",
                                content="department_result",
                                session_id=session_id,
                                node=node_name,
                                meta={"department_result": dept_result, "data": dept_result},
                            ),
                            ensure_ascii=False,
                        )
                        yield f"data: {dept_event}\n\n"

                texts = _extract_texts_from_output(output)
                for text in texts:
                    key = " ".join(text.split())
                    if not key or key in emitted_text_keys:
                        continue
                    emitted_text_keys.add(key)
                    payload = json.dumps(
                        build_stream_payload(
                            event_type="token",
                            content=text,
                            session_id=session_id,
                            node=node_name,
                        ),
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
                    emitted_token_count += 1

                    dept_result = _extract_department_result_from_output(output=output, text=text)
                    if dept_result:
                        dept_key = json.dumps(dept_result, ensure_ascii=False, sort_keys=True)
                        if dept_key not in emitted_department_keys:
                            emitted_department_keys.add(dept_key)
                            dept_event = json.dumps(
                                build_stream_payload(
                                    event_type="department_result",
                                    content="department_result",
                                    session_id=session_id,
                                    node=node_name,
                                    meta={"department_result": dept_result, "data": dept_result},
                                ),
                                ensure_ascii=False,
                            )
                            yield f"data: {dept_event}\n\n"

                    # Compatibility layer: emit structured slots event when model returns <ui_slots>.
                    for slots_payload in extract_doctor_slots(text):
                        slots_key = json.dumps(slots_payload, ensure_ascii=False, sort_keys=True, default=str)
                        if slots_key in emitted_slots_keys:
                            continue
                        emitted_slots_keys.add(slots_key)
                        slots_event = json.dumps(
                            build_stream_payload(
                                event_type="doctor_slots",
                                content="doctor_slots",
                                session_id=session_id,
                                node=node_name,
                                meta={"data": slots_payload, "slots": slots_payload},
                            ),
                            ensure_ascii=False,
                        )
                        yield f"data: {slots_event}\n\n"

                    for payment_payload in extract_ui_payment(text):
                        payment_event = json.dumps(
                            build_stream_payload(
                                event_type="payment_required",
                                content="payment_required",
                                session_id=session_id,
                                node=node_name,
                                meta={"data": payment_payload},
                            ),
                            ensure_ascii=False,
                        )
                        yield f"data: {payment_event}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_payload = json.dumps(
            build_stream_payload(
                event_type="error",
                content=str(e),
                session_id=session_id,
                node="chat_stream",
            ),
            ensure_ascii=False,
        )
        yield f"data: {error_payload}\n\n"
    
    # End of stream
    yield "data: [DONE]\n\n"

@router.post("/stream")
async def stream_chat(request: ChatRequest):
    """
    SSE Endpoint for Real-time Chat
    """
    return StreamingResponse(
        event_generator(
            request.message,
            request.session_id,
            _dump_model(request.rag) if request.rag else None,
            request.request_id,
            request.debug_include_nodes,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

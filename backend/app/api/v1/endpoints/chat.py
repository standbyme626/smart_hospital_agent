import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.core.graph.workflow import app as graph_app
from app.core.graph.state import AgentState
from app.core.stream_schema import build_stream_payload

router = APIRouter()
logger = logging.getLogger(__name__)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

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

async def event_generator(message: str, session_id: str) -> AsyncGenerator[str, None]:
    """
    SSE Generator using astream_events (v2) for real-time token streaming.
    [V6.7] 过滤 <think> 推理块，仅输出最终回答
    """
    # [Fix] 完善入口状态初始化，确保所有必需字段存在
    inputs = {
        "symptoms": message,
        "user_input": message,            # 兼容 unified_preprocessor
        "current_turn_input": message,    # 本轮输入（防止读到上一轮 AI）
        "retrieval_query": message,       # 统一检索入口字段
        "event": {
            "event_type": "SYMPTOM_DESCRIPTION",
            "payload": {"session_id": session_id},
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
        event_generator(request.message, request.session_id),
        media_type="text/event-stream"
    )

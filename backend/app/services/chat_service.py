import json
import time
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
import structlog
from langchain_core.messages import HumanMessage

from app.core.graph.workflow import build_medical_graph

logger = structlog.get_logger(__name__)

class ChatService:
    """
    Chat 业务服务层
    负责协调 LangGraph 执行、事件解析、UX 优化逻辑。
    [V2.0 Refactor] 支持双轨并行流式 (Fast Track + Expert Track)
    """
    def __init__(self, graph=None):
        # 支持依赖注入，默认使用全局 Graph
        self.graph = graph if graph else build_medical_graph()

    async def stream_events(self, message: str, session_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行对话流并生成业务事件
        """
        inputs = {
            "symptoms": message,
            "user_input": message,
            "current_turn_input": message,
            "retrieval_query": message,
            "event": {
                "event_type": "SYMPTOM_DESCRIPTION",
                "payload": {"session_id": session_id},
                "raw_input": message,
                "timestamp": time.time(),
            },
            "messages": [HumanMessage(content=message)],
            "patient_id": "p_guest",
            "session_id": session_id or "unknown",
            "medical_history": "" 
        }
        
        # 1. Yield Initial Status
        yield {"type": "status", "content": "系统已接收请求，正在启动双轨分析..."}
        
        node_start_times = {}
        fast_track_done = False
        
        try:
            # 2. Stream LangGraph Events
            config = {"configurable": {"thread_id": session_id}}
            async for event in self.graph.astream_events(inputs, config=config, version="v2"):

                kind = event["event"]
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                run_id = event.get("run_id")
                
                # [Logic] Node Timing Start
                if kind == "on_chain_start" and node_name:
                    node_start_times[run_id] = time.time()
                    if self._is_key_node(node_name):
                        yield {"type": "thought", "content": f"➡️ 进入节点: {node_name}"}

                # [Logic] Node Timing End
                if kind == "on_chain_end" and node_name:
                    if run_id in node_start_times:
                        duration = time.time() - node_start_times[run_id]
                        if self._is_key_node(node_name):
                            yield {"type": "thought", "content": f"✅ 节点完成: {node_name} (耗时: {duration:.4f}s)"}

                # [UX] Status Updates
                if kind == "on_chain_start":
                    status_msg = self._get_status_message(node_name)
                    if status_msg:
                        yield {"type": "status", "content": status_msg}
                
                # [Logic] Fast Track Streaming (Real-time tokens)
                # 监听 fast_track 节点内部的 LLM 流式输出
                if kind == "on_chat_model_stream" and node_name == "fast_track":
                    content = event["data"]["chunk"].content
                    if content:
                        yield {"type": "token", "content": content}
                        fast_track_done = True

                # [Logic] Fast Track Completion
                if kind == "on_chain_end" and node_name == "fast_track":
                    # Fallback: 如果 LLM 不支持流式，则一次性输出结果
                    if not fast_track_done:
                        output = event["data"].get("output")
                        content = ""
                        if output and isinstance(output, dict):
                            content = output.get("fast_response") or output.get("content")
                        
                        if content:
                            yield {"type": "token", "content": content}
                    
                    # [UX] 视觉占位优化：Fast Track 结束后，立即通知前端展示 "专家研判中" 状态
                    # 填补 Fast Track 与 Expert Crew 之间的 10s+ 空白期
                    yield {"type": "status", "content": "expert_calculating"}

                # [Logic] Expert Crew Output (Final Result)
                # 当 expert_crew 完成时，获取其输出并展示
                if kind == "on_chain_end" and node_name == "expert_crew":
                    output = event["data"].get("output")
                    # CrewAI/LangGraph 这里的输出结构通常是 state 的更新
                    # 如果是 expert_crew，它返回的是 {"messages": [AIMessage(...)]}
                    
                    final_msg = ""
                    if output and isinstance(output, dict):
                        if "messages" in output:
                            msgs = output["messages"]
                            if msgs and len(msgs) > 0:
                                final_msg = msgs[-1].content
                        # 兼容直接返回 content 的情况
                        elif "content" in output:
                            final_msg = output["content"]

                    if final_msg:
                        # 在 Fast Response 后追加专家分析
                        separator = "\n\n---\n\n**🏥 三甲专家组会诊报告**:\n\n"
                        yield {"type": "token", "content": separator}
                        
                        # 模拟打字机效果输出专家长文，避免瞬间刷屏
                        chunk_size = 20
                        for i in range(0, len(final_msg), chunk_size):
                            chunk = final_msg[i:i+chunk_size]
                            yield {"type": "token", "content": chunk}
                            await asyncio.sleep(0.01)

                # [Logic] Guardrail Block
                if kind == "on_chain_end" and (node_name == "guard" or node_name == "safety_audit"):
                     output = event["data"].get("output")
                     if isinstance(output, dict) and output.get("status") == "blocked":
                         yield {"type": "error", "content": "请求被医疗安全护栏拦截"}

        except Exception as e:
            logger.error("chat_stream_error", error=str(e))
            yield {"type": "error", "content": f"系统异常: {str(e)}"}
            
        yield {"type": "done", "content": "COMPLETE"}

    def _is_key_node(self, node_name: str) -> bool:
        """Helper: 决定是否显示节点耗时"""
        return node_name in ["guard", "triage_router", "fast_track", "expert_crew", "quality_gate", "persistence", "safety_audit"]

    def _get_status_message(self, node_name: str) -> Optional[str]:
        """Helper: 获取节点状态文案"""
        mapping = {
            "guard": "正在进行安全合规检查...",
            "safety_audit": "正在进行二次医疗与用药安全审查...",
            "triage_router": "正在进行智能分诊...",
            "fast_track": "⚡ 正在调用本地知识库快速响应...",
            "expert_crew": "🏥 专家组正在进行多学科会诊(MDT)...",
            "quality_gate": "正在进行医疗质控...",
            "summarize_history": "正在汇总历史病历..."
        }
        return mapping.get(node_name)

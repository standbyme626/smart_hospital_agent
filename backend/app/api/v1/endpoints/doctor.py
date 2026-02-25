from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.agents.doctor_graph import doctor_graph
from app.models.requests import ChatRequest
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/workflow")
async def doctor_workflow(request: ChatRequest):
    """
    医生工作流接口 (Doctor Workflow Endpoint) - 流式响应
    
    调用 Doctor LangGraph 并流式返回执行事件 (Events):
    - 'message': AI 的回复内容。
    - 'tool_call': 工具调用请求。
    - 'phase': 状态阶段流转 (如 diagnosis -> prescription)。
    """
    
    async def event_generator():
        try:
            # 1. 准备输入状态
            # 目前假设 request.message 是最新的用户输入
            # 在真实应用中，可能需要加载完整的聊天记录或传递 history
            input_message = HumanMessage(content=request.message)
            
            # 配置 Checkpointing (Session ID) 用于持久化记忆
            config = {"configurable": {"thread_id": request.session_id}}
            
            # 2. 调用 Graph 进行流式生成 (Streaming)
            async for event in doctor_graph.astream({"messages": [input_message]}, config=config):
                
                # 处理 'diagnosis_node' 节点输出 (AI 回复)
                if "diagnosis_node" in event:
                    msg = event["diagnosis_node"]["messages"][0]
                    content = msg.content
                    
                    if content:
                         yield {
                            "event": "message", 
                            "data": json.dumps({"content": content})
                        }
                    
                    # 如果有工具调用，通知前端
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield {
                                "event": "tool_call",
                                "data": json.dumps({"name": tc["name"], "args": tc["args"]})
                            }
                            
                # 处理 'tools' 节点输出 (工具执行结果)
                elif "tools" in event:
                    msg = event["tools"]["messages"][0]
                    content = msg.content
                    yield {
                        "event": "tool_output",
                        "data": json.dumps({"content": content[:200] + "..."}) # 截断过长的工具输出
                    }
                    
                # 处理 'state_updater' 节点输出 (阶段流转)
                elif "state_updater" in event:
                    if event["state_updater"] and "phase" in event["state_updater"]:
                        new_phase = event["state_updater"]["phase"]
                        yield {
                            "event": "phase",
                            "data": json.dumps({"phase": new_phase})
                        }
                        
            # 流结束
            yield {"event": "done", "data": "Stream finished"}
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Doctor workflow error: {error_msg}")
            yield {
                "event": "error", 
                "data": json.dumps({"error": error_msg})
            }

    return EventSourceResponse(event_generator())

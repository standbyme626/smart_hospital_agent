import json
import logging

from fastapi import APIRouter, Query
from langchain_core.messages import HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.agents.doctor_graph import doctor_graph
from app.core.config import settings
from app.core.stream_schema import build_stream_payload
from app.models.requests import ChatRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/workflow")
async def doctor_workflow(
    request: ChatRequest,
    schema_mode: str = Query(default="legacy", pattern="^(legacy|unified)$"),
):
    """
    医生工作流接口 (Doctor Workflow Endpoint) - 流式响应

    兼容协议:
    - legacy: 保持旧的 event/data 结构
    - unified: data 使用统一 envelope（与 /chat/stream 对齐）
    """

    force_unified = schema_mode == "unified" or settings.ENABLE_UNIFIED_STREAM_SCHEMA

    def _payload(event_type: str, content: str, node: str = "", meta: dict | None = None) -> str:
        return json.dumps(
            build_stream_payload(
                event_type=event_type,  # type: ignore[arg-type]
                content=content,
                session_id=request.session_id,
                node=node,
                meta=meta or {},
                force_unified=force_unified,
            ),
            ensure_ascii=False,
        )

    async def event_generator():
        try:
            input_message = HumanMessage(content=request.message)
            config = {"configurable": {"thread_id": request.session_id}}

            async for event in doctor_graph.astream({"messages": [input_message]}, config=config):
                if "diagnosis_node" in event:
                    msg = event["diagnosis_node"]["messages"][0]
                    content = str(getattr(msg, "content", "") or "")

                    if content:
                        yield {
                            "event": "message",
                            "data": _payload("token", content, node="diagnosis_node"),
                        }

                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            yield {
                                "event": "tool_call",
                                "data": _payload(
                                    "tool_call",
                                    content=tc.get("name", "tool_call"),
                                    node="diagnosis_node",
                                    meta={"args": tc.get("args", {})},
                                ),
                            }

                elif "tools" in event:
                    msg = event["tools"]["messages"][0]
                    content = str(getattr(msg, "content", "") or "")
                    if len(content) > 200:
                        content = f"{content[:200]}..."
                    yield {
                        "event": "tool_output",
                        "data": _payload("tool_output", content, node="tools"),
                    }

                elif "state_updater" in event:
                    updater = event["state_updater"]
                    if updater and "phase" in updater:
                        new_phase = str(updater["phase"])
                        yield {
                            "event": "phase",
                            "data": _payload(
                                "phase",
                                content=f"phase={new_phase}",
                                node="state_updater",
                                meta={"phase": new_phase},
                            ),
                        }

            yield {
                "event": "done",
                "data": _payload("final", "Stream finished", node="doctor_workflow"),
            }

        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Doctor workflow error: {error_msg}")
            yield {
                "event": "error",
                "data": _payload("error", error_msg, node="doctor_workflow"),
            }

    return EventSourceResponse(event_generator())

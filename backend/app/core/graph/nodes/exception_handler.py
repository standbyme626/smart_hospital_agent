import structlog
import functools
import asyncio
from typing import Callable, Any, Dict
from app.core.graph.state import AgentState

logger = structlog.get_logger(__name__)

class GraphExceptionHandler:
    """
    健壮性与异常处理模块 (Phase 4)
    为 Graph Node 提供统一的 Try-Except 包装与补偿逻辑。
    """

    @staticmethod
    def safe_node(node_name: str, fallback_return: Dict = None):
        """
        Decorator to wrap graph nodes with exception handling.
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(state: AgentState, *args, **kwargs):
                try:
                    return await func(state, *args, **kwargs)
                except Exception as e:
                    logger.error(f"node_execution_failed", node=node_name, error=str(e))
                    
                    # 记录错误到 State
                    error_msg = f"System Error in {node_name}: {str(e)}"
                    
                    # 补偿逻辑 (Compensation Logic)
                    if node_name == "service_agent" and "lock_slot" in str(e):
                        # 号源抢占冲突补偿
                        return {
                            "messages": [{"role": "assistant", "content": "抱歉，该时段刚刚被预约，正在为您刷新其他可用时间..."}],
                            "error": "SlotConflict"
                        }
                    
                    # 默认降级
                    if fallback_return:
                        return fallback_return
                    
                    # 通用错误状态更新
                    return {"error": error_msg}
            return wrapper
        return decorator

    @staticmethod
    async def handle_mcp_error(state: AgentState, error: Exception) -> Dict:
        """
        专门处理 MCP 调用异常
        """
        logger.error("mcp_call_failed", error=str(error))
        return {
            "messages": [{"role": "assistant", "content": "系统连接繁忙，请稍后再试。"}],
            "error": str(error)
        }

exception_handler = GraphExceptionHandler()

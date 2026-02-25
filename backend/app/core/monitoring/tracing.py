import functools
import time
import asyncio
from typing import Any, Callable, Dict, List
from langsmith import traceable
import structlog
from langchain_core.messages import BaseMessage

logger = structlog.get_logger(__name__)

from app.core.safety.pii import PIIService
pii_service = PIIService()
from app.core.utils.tokenizer import global_tokenizer
from app.core.monitoring.cost_tracker import CostTracker
from app.core.config import settings

# [Pain Point #23] Node to Model Mapping for Cost Estimation
# This helps us estimate costs even if the node doesn't return exact token counts
NODE_MODEL_MAPPING = {
    "anamnesis": settings.MODEL_FAST, # Fast LLM
    "expert_consultation": settings.MODEL_SMART, # Usually uses the main model
    "diagnosis_node": settings.MODEL_SMART,
    "medication_node": settings.MODEL_SMART,
    "ingress": settings.LOCAL_SLM_MODEL,
    "medical_core": settings.LOCAL_SLM_MODEL,
    "guardrails": settings.LOCAL_SLM_MODEL,
}

def monitor_node(name: str):
    """
    LangGraph 节点监控装饰器
    
    功能：
    1. 显性追踪：打印 [DEBUG] Node {name} Start
    2. LangSmith 集成：自动记录追踪信息
    3. 内容校验：AI 生成内容 < 20 字符抛出 ValueError
    4. 异常透传：确保错误向上抛出
    5. PII 脱敏：在进入 traceable 前对输入进行脱敏
    6. [Pain Point #23] 资源开销与 Token 成本审计
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            print(f"[DEBUG] Node {name} Start")
            
            # [Task 4] PII Masking before traceable
            scrubbed_args = list(args)
            if len(scrubbed_args) > 0 and isinstance(scrubbed_args[0], dict):
                # LangGraph state is typically the first arg
                state = scrubbed_args[0].copy()
                if "symptoms" in state and isinstance(state["symptoms"], str):
                    state["symptoms"] = pii_service.scrub(state["symptoms"])
                scrubbed_args[0] = state
            
            # Use inner traceable to capture scrubbed inputs
            @traceable(name=name)
            async def _traced_func(*a, **kw):
                start_time = time.time()
                try:
                    # 执行原函数
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*a, **kw)
                    else:
                        result = func(*a, **kw)
                    
                    # 内容校验 (Anti-Lazy Rule)
                    _validate_content(name, result)
                    
                    duration = time.time() - start_time
                    logger.info(f"node_completed", node=name, duration=f"{duration:.2f}s")
                    
                    # 注入耗时数据到结果中，以便 SSE 捕获
                    if isinstance(result, dict):
                        # [V12.0] 记录节点耗时
                        if "node_telemetry" not in result:
                            result["node_telemetry"] = {}
                        result["node_telemetry"][name] = round(duration, 2)

                        # [Pain Point #23] Token Estimation & Cost Audit
                        input_str = str(scrubbed_args[0]) if scrubbed_args else ""
                        output_str = str(result)
                        
                        est_input = global_tokenizer.count_tokens(input_str)
                        est_output = global_tokenizer.count_tokens(output_str)
                        
                        # Determine model
                        model_name = NODE_MODEL_MAPPING.get(name, "local_slm")
                        if model_name == "qwen-plus":
                             # Dynamic fallback to settings if needed
                             model_name = settings.OPENAI_MODEL_NAME or "qwen-plus"
                        
                        usage_record = CostTracker.create_usage_record(
                            node_name=name,
                            model_name=model_name,
                            input_tokens=est_input,
                            output_tokens=est_output,
                            duration_s=duration
                        )
                        
                        # Use the new standard field: usage_statistics
                        if "usage_statistics" not in result:
                            result["usage_statistics"] = []
                        
                        if isinstance(result["usage_statistics"], list):
                            result["usage_statistics"].append(usage_record)
                        else:
                            result["usage_statistics"] = [usage_record]

                    return result
                except Exception as e:
                    logger.error(f"node_failed", node=name, error=str(e))
                    raise e
            
            return await _traced_func(*scrubbed_args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            print(f"[DEBUG] Node {name} Start")
            
            # [Task 4] PII Masking
            scrubbed_args = list(args)
            if len(scrubbed_args) > 0 and isinstance(scrubbed_args[0], dict):
                state = scrubbed_args[0].copy()
                if "symptoms" in state and isinstance(state["symptoms"], str):
                    state["symptoms"] = pii_service.scrub(state["symptoms"])
                scrubbed_args[0] = state

            @traceable(name=name)
            def _traced_func(*a, **kw):
                start_time = time.time()
                try:
                    result = func(*a, **kw)
                    _validate_content(name, result)
                    duration = time.time() - start_time
                    logger.info(f"node_completed", node=name, duration=f"{duration:.2f}s")

                    # 注入耗时数据到结果中，以便 SSE 捕获
                    if isinstance(result, dict):
                        # [V12.0] 记录节点耗时
                        if "node_telemetry" not in result:
                            result["node_telemetry"] = {}
                        result["node_telemetry"][name] = round(duration, 2)

                        # [Pain Point #23] Token Estimation & Cost Audit
                        input_str = str(scrubbed_args[0]) if scrubbed_args else ""
                        output_str = str(result)
                        
                        est_input = global_tokenizer.count_tokens(input_str)
                        est_output = global_tokenizer.count_tokens(output_str)
                        
                        # Determine model
                        model_name = NODE_MODEL_MAPPING.get(name, "local_slm")
                        if model_name == "qwen-plus":
                             # Dynamic fallback to settings if needed
                             model_name = settings.OPENAI_MODEL_NAME or "qwen-plus"
                        
                        usage_record = CostTracker.create_usage_record(
                            node_name=name,
                            model_name=model_name,
                            input_tokens=est_input,
                            output_tokens=est_output,
                            duration_s=duration
                        )
                        
                        # Use the new standard field: usage_statistics
                        if "usage_statistics" not in result:
                            result["usage_statistics"] = []
                        
                        if isinstance(result["usage_statistics"], list):
                            result["usage_statistics"].append(usage_record)
                        else:
                            result["usage_statistics"] = [usage_record]
                    return result
                except Exception as e:
                    logger.error(f"node_failed", node=name, error=str(e))
                    raise e
            
            return _traced_func(*scrubbed_args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def _validate_content(node_name: str, result: Dict[str, Any]):
    """
    校验 AI 生成内容长度
    """
    if not isinstance(result, dict):
        return

    # 检查常见的输出字段
    check_fields = ["messages", "diagnosis_report", "final_response", "clinical_report", "final_output"]
    
    # [V6.2 Fix] 增加豁免名单，针对缓存命中、分类等短回复节点放宽限制
    exempt_nodes = [
        "cache_lookup",
        "classifier",
        "guard",
        "triage_router",
        "quality_gate",
        "parallel_processing",
        "memory_management",
        "summarize_history",
        "expert_aggregation",
    ]
    if node_name in exempt_nodes:
        min_length = 2 # 极短回复（如“你好”）也允许
    else:
        min_length = 20

    for field in check_fields:
        content = result.get(field)
        if content:
            text = ""
            if isinstance(content, list):
                # 提取最后一条消息的内容
                if content and hasattr(content[-1], "content"):
                    text = content[-1].content
                elif content and isinstance(content[-1], dict) and "content" in content[-1]:
                    text = content[-1]["content"]
            elif isinstance(content, str):
                text = content
            
            if text and len(text.strip()) < min_length:
                # 只有当确实有内容但太短时才报错，排除空返回（有些节点可能不产出这些字段）
                raise ValueError(f"AI content in node '{node_name}' field '{field}' is too short (< {min_length} chars): '{text}'")

def get_langsmith_callbacks():
    """
    获取 LangSmith 回调，用于传递给 CrewAI 或其他非 LangChain 组件
    """
    try:
        from langchain_core.tracers.context import get_tracing_integration_callback
        return [get_tracing_integration_callback()]
    except ImportError:
        return []

def setup_langsmith():
    """
    从 settings 中读取并设置 LangSmith 环境变量
    """
    from app.core.config import settings
    import os
    
    # 确保环境变量被正确设置（即使 settings 初始化时已经设置过，这里作为显式确认）
    if settings.LANGCHAIN_TRACING_V2.lower() == "true":
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
        logger.info("langsmith_setup", status="enabled", project=settings.LANGCHAIN_PROJECT)
    else:
        logger.info("langsmith_setup", status="disabled")

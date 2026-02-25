from typing import List, Dict
import random
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import structlog

logger = structlog.get_logger()

class DrugServiceError(Exception):
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(DrugServiceError),
    reraise=False # 降级处理，不抛出异常
)
def check_drug_interaction(drugs: List[str]) -> str:
    """
    检查药物相互作用 (Mock API)。
    模拟20%的调用失败率，演示重试机制。
    
    Args:
        drugs: 药物名称列表，如 ["阿司匹林", "头孢"]
        
    Returns:
        str: 相互作用警告信息
    """
    try:
        logger.info("tool_call.check_drug_interaction", drugs=drugs)
        
        # 模拟外部 API 调用延迟
        time.sleep(0.1)
        
        # 模拟随机故障
        if random.random() < 0.2:
            logger.warning("mock_api_failure", service="drug_interaction")
            raise DrugServiceError("Drug interaction service temporarily unavailable")

        # Mock 业务逻辑
        interactions = []
        if "阿司匹林" in drugs and "华法林" in drugs:
            interactions.append("⚠️ 【高风险】阿司匹林与华法林联用可能增加出血风险！")
        if "头孢" in drugs and "酒精" in drugs:
            interactions.append("⚠️ 【严重】双硫仑样反应风险！严禁饮酒！")
            
        if not interactions:
            return "✅ 未发现已知相互作用。"
            
        return "\n".join(interactions)
        
    except DrugServiceError as e:
        # Tenacity 会捕获此异常并重试
        # 如果重试耗尽（reraise=False），代码会继续向下执行（如果不抛出？）
        # 注意: tenacity 默认 decorator 行为：如果 reraise=True，则抛出 RetryError
        # 如果要实现 Fallback，通常需要 retry_error_callback
        raise e
    except Exception as e:
        # 其他异常直接抛出
        raise e

# 真正的 Fallback 包装函数（因为 decorator 比较难做复杂的 fallback）
def check_drug_interaction_with_fallback(drugs: List[str]) -> str:
    try:
        return check_drug_interaction(drugs)
    except Exception:
        logger.error("tool_fallback.check_drug_interaction", drugs=drugs)
        return "⚠️ 药品相互作用检查服务当前不可用，请人工核查说明书。"

import os
from typing import Dict, List, Optional
from app.core.config import settings

class ModelManager:
    """
    统一模型管理器 (Unified Model Manager)
    [V6.4.2] 支持从本地环境变量随时更换模型，并提供默认映射。
    """
    
    # 模型分类映射 (Model Classifications)
    # [Refactor] 移除硬编码的映射表，完全由 config.py 和环境变量驱动
    # _MODEL_MAP 已被移除

    @classmethod
    def get_model(cls, model_type: str = "smart") -> str:
        """
        根据类型获取模型名称
        
        Args:
            model_type: "smart", "fast", "coder", "long", "local"
        Returns:
            str: 实际调用的模型字符串
        """
        # 1. 优先从环境变量获取 (MODEL_SMART, MODEL_FAST 等)
        # 实际上 settings 已经帮我们做了这件事，这里只是做一个路由
        env_key = f"MODEL_{model_type.upper()}"
        env_val = os.getenv(env_key)
        if env_val:
            return env_val

        # 2. 从 settings 获取配置值
        if model_type == "smart":
            return settings.MODEL_SMART
        elif model_type == "fast":
            return settings.MODEL_FAST
        elif model_type == "coder":
            return settings.MODEL_CODER
        elif model_type == "long":
            return settings.MODEL_LONG
        elif model_type == "local":
            return settings.LOCAL_SLM_MODEL
            
        # 3. 兜底
        return settings.MODEL_SMART

    @classmethod
    def get_all_candidates(cls) -> List[str]:
        """获取所有候选模型列表 (用于轮换策略)"""
        return settings.MODEL_CANDIDATES_LIST

    @classmethod
    def get_local_model(cls) -> str:
        """获取本地模型名称"""
        return cls.get_model("local")

# 导出单例或静态访问
model_manager = ModelManager

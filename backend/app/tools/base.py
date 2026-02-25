"""
工具基类 (Base Tool Class)

定义所有工具的通用接口和行为
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog

logger = structlog.get_logger()


class BaseTool(ABC):
    """
    工具基类
    
    所有工具必须实现:
    1. name: 工具名称
    2. description: 工具描述
    3. _execute: 核心逻辑
    4. _fallback: 降级方案
    """
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.call_count = 0
        self.error_count = 0
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述(供LLM理解)"""
        pass
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        核心执行逻辑
        
        返回:
            统一格式: {
                'success': bool,
                'data': Any,
                'source': str,
                'confidence': float
            }
        """
        pass
    
    @abstractmethod
    def _fallback(self, error: Exception, **kwargs) -> Dict[str, Any]:
        """
        降级方案
        
        当_execute失败时调用
        """
        pass
    
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        统一入口
        
        包含:
        - 重试逻辑
        - 降级处理
        - 日志记录
        - 指标埋点
        """
        self.call_count += 1
        
        try:
            logger.info(
                f"tool.execute",
                tool=self.name,
                params=kwargs
            )
            
            result = await self._execute(**kwargs)
            
            logger.info(
                f"tool.success",
                tool=self.name,
                source=result.get('source')
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            
            logger.error(
                f"tool.error",
                tool=self.name,
                error=str(e),
                fallback=True
            )
            
            # 调用降级方案
            return self._fallback(error=e, **kwargs)

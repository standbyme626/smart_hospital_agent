from typing import Dict, Any, Callable, List, Optional
from langchain_core.tools import BaseTool, tool
import structlog

logger = structlog.get_logger(__name__)

class ToolRegistry:
    """
    工具注册表 (Tool Registry)
    
    用于管理 Agent 使用的所有工具，支持依赖注入和动态加载。
    避免在 Graph 定义中硬编码工具列表。
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        
    def register(self, name: str, tool_instance: BaseTool):
        """注册一个工具实例"""
        if name in self._tools:
            logger.warning("tool_registry.overwrite", name=name)
        self._tools[name] = tool_instance
        logger.info("tool_registry.registered", name=name)
        
    def get(self, name: str) -> BaseTool:
        """获取指定名称的工具"""
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry.")
        return self._tools[name]
        
    def get_all(self) -> List[BaseTool]:
        """获取所有注册的工具"""
        return list(self._tools.values())
    
    def get_subset(self, names: List[str]) -> List[BaseTool]:
        """获取指定名称列表的工具子集"""
        return [self.get(name) for name in names]

# Global instance for easy access (can be overridden in tests)
registry = ToolRegistry()

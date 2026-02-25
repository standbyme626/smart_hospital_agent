import yaml
import os
from typing import Dict, Any, Optional
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)

class ConfigManager:
    """
    科室配置管理器 (Department Configuration Manager)
    负责加载 /config/departments/ 下的 YAML 配置文件，
    实现“一套代码，多科室复用”的架构目标。
    """
    
    _configs: Dict[str, Any] = {}
    _config_path: Path = Path(__file__).parent.parent.parent.parent / "config" / "departments"

    @classmethod
    def load_all(cls):
        """加载所有科室配置"""
        if not cls._config_path.exists():
            logger.error("config_path_not_found", path=str(cls._config_path))
            return

        for file_path in cls._config_path.glob("*.yaml"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    dept_name = config.get("name")
                    if dept_name:
                        cls._configs[dept_name] = config
                        logger.info("department_config_loaded", department=dept_name)
            except Exception as e:
                logger.error("config_load_failed", file=file_path.name, error=str(e))

    @classmethod
    def get_config(cls, department_name: str) -> Optional[Dict[str, Any]]:
        """获取指定科室的配置"""
        if not cls._configs:
            cls.load_all()
        
        # 支持别名或模糊匹配逻辑（可选）
        return cls._configs.get(department_name)

    @classmethod
    def get_system_prompt(cls, department_name: str) -> str:
        """获取科室特定的 System Prompt"""
        config = cls.get_config(department_name)
        if not config:
            return "你是一名全科医生，负责回答患者的医疗咨询。"
        
        prompts = config.get("prompts", {})
        role = prompts.get("role_definition", "")
        guidelines = prompts.get("diagnostic_guidelines", "")
        
        return f"{role}\n\n诊断指南:\n{guidelines}"

# Global Instance
config_manager = ConfigManager()

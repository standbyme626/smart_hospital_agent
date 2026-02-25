import re
import logging
from typing import Dict, Tuple, List

class SensitiveFormatter(logging.Formatter):
    """
    日志脱敏格式化器
    自动过滤日志中的敏感信息（API Key、密码等）
    """
    
    # 敏感信息正则模式
    PATTERNS: List[Tuple[str, str]] = [
        # API Keys
        (r'sk-[a-zA-Z0-9]{32,}', 'sk-***'),
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'api_key=***'),
        
        # Passwords
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'password=***'),
        (r'passwd["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'passwd=***'),
        
        # Tokens
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\s]{20,})', 'token=***'),
        (r'bearer\s+([a-zA-Z0-9_\-\.]+)', 'bearer ***'),
        
        # Database URLs
        (r'postgresql://([^:]+):([^@]+)@', 'postgresql://***:***@'),
        (r'mysql://([^:]+):([^@]+)@', 'mysql://***:***@'),
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录并脱敏"""
        msg = super().format(record)
        
        # 应用所有脱敏规则
        for pattern, replacement in self.PATTERNS:
            msg = re.sub(pattern, replacement, msg, flags=re.IGNORECASE)
        
        return msg


def configure_sensitive_logging():
    """
    配置全局日志脱敏
    
    Usage:
        from app.core.middleware.logging import configure_sensitive_logging
        configure_sensitive_logging()
    """
    # 获取 root logger
    root_logger = logging.getLogger()
    
    # 应用脱敏格式化器到所有 handler
    for handler in root_logger.handlers:
        formatter = SensitiveFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

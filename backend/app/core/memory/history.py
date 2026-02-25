from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.core.config import settings
import structlog

logger = structlog.get_logger()

def get_message_history(session_id: str) -> RedisChatMessageHistory:
    """
    获取 Redis 聊天记录对象。
    
    Args:
        session_id: 会话唯一标识
        
    Returns:
        RedisChatMessageHistory: LangChain 历史记录对象
    """
    try:
        # 确保 Redis URL 包含数据库索引（如果有），通常 settings.REDIS_URL 格式如 redis://localhost:6379/0
        history = RedisChatMessageHistory(
            url=settings.REDIS_URL,
            session_id=f"chat_history:{session_id}",
            ttl=3600 * 24  # 聊天记录保留 24 小时
        )
        return history
    except Exception as e:
        logger.error("memory.init_failed", error=str(e))
        raise e

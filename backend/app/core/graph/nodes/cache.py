import time
import hashlib
import json
import re
import asyncio
import structlog
from typing import Dict, Any
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.core.cache.redis_pool import get_redis_client
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig

logger = structlog.get_logger(__name__)

@monitor_node("cache_lookup")
async def cache_lookup_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：语义缓存 (Semantic Cache)
    使用 Redis 对用户问题进行 Hash 查找。
    """
    print(f"[DEBUG] Node CacheLookup Start")
    logger.info("node_start", node="cache_lookup")
    start = time.time()
    
    user_input = state.get("symptoms", "")
    if not user_input:
        return {}
            
    # Key Normalization (Strict)
    clean_query = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", user_input)).lower().strip()
    
    # MD5 Hash
    query_hash = hashlib.md5(clean_query.encode("utf-8")).hexdigest()
    cache_key = f"cache:{query_hash}"
    
    redis_client = get_redis_client()
    
    try:
        # Redis Fail-Fast (Timeout 0.5s)
        cached_data = await asyncio.wait_for(redis_client.get(cache_key), timeout=0.5)
    except (asyncio.TimeoutError, Exception) as e:
        logger.warning("cache_lookup_unavailable", error=str(e))
        cached_data = None
    
    if cached_data:
        data = json.loads(cached_data)
        logger.info("cache_hit", session_id=state.get("session_id"), query_hash=query_hash, duration=f"{time.time() - start:.2f}s")
        return {
            "final_output": data.get("response", ""),
            "messages": [AIMessage(content=data.get("response", ""))],
            "status": "cached", 
            "cache_hit": True
        }
    
    return {"cache_hit": False}

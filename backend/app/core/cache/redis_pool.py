import redis.asyncio as redis
from app.core.config import settings

# Global Connection Pools
_pool = None
_binary_pool = None

def get_redis_client():
    """获取文本 Redis 客户端 (decode_responses=True)"""
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=True,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
    return redis.Redis(connection_pool=_pool)

def get_redis_binary_client():
    """获取二进制 Redis 客户端 (decode_responses=False)"""
    global _binary_pool
    if _binary_pool is None:
        _binary_pool = redis.ConnectionPool(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=0,
            decode_responses=False,
            max_connections=settings.REDIS_MAX_CONNECTIONS
        )
    return redis.Redis(connection_pool=_binary_pool)

async def close_redis_pool():
    """关闭所有连接池"""
    global _pool, _binary_pool
    if _pool:
        await _pool.disconnect()
        _pool = None
    if _binary_pool:
        await _binary_pool.disconnect()
        _binary_pool = None
    print("[Redis] All Connection Pools Closed.")

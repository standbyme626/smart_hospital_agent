
import asyncio
import structlog
import time
from enum import IntEnum
from typing import Dict, Any, Optional

logger = structlog.get_logger(__name__)

class Priority(IntEnum):
    EMERGENCY = 0  # 核心急诊/生命体征 (最高优先级)
    URGENT = 1     # 挂号/就医咨询
    NORMAL = 2     # 一般健康咨询/百科
    LOW = 3        # 闲聊/非医学

class PriorityGuard:
    """
    [Phase 6.4] 优先级调度器 (Priority Scheduler).
    控制并发执行槽位，确保高优先级请求优先获得算力资源。
    """
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.current_running = 0
        self._lock = asyncio.Lock()
        self._queue = asyncio.PriorityQueue()
        self._waiting_count = 0

    async def acquire(self, priority: Priority, request_id: str):
        """申请执行槽位"""
        start_wait = time.time()
        
        async with self._lock:
            if self.current_running < self.max_concurrent:
                self.current_running += 1
                logger.debug("priority_guard_acquire_immediate", 
                           request_id=request_id, 
                           priority=priority.name,
                           running=self.current_running)
                return True
            
            # 进入等待队列
            self._waiting_count += 1
            waiter = asyncio.Event()
            # PriorityQueue 存储 (priority_value, timestamp, waiter_event)
            # 使用 timestamp 确保同优先级 FIFO
            await self._queue.put((int(priority), time.time(), waiter))
            
        logger.info("priority_guard_queued", 
                  request_id=request_id, 
                  priority=priority.name, 
                  waiting=self._waiting_count)
        
        # 等待唤醒
        await waiter.wait()
        
        wait_duration = time.time() - start_wait
        logger.info("priority_guard_acquired_after_wait", 
                  request_id=request_id, 
                  wait_sec=f"{wait_duration:.2f}s")
        return True

    async def release(self):
        """释放执行槽位"""
        async with self._lock:
            if not self._queue.empty():
                # 唤醒最高优先级的等待者
                _, _, waiter = await self._queue.get()
                self._waiting_count -= 1
                waiter.set()
            else:
                self.current_running = max(0, self.current_running - 1)
            
            logger.debug("priority_guard_released", 
                       running=self.current_running, 
                       waiting=self._waiting_count)

    from contextlib import asynccontextmanager
    @asynccontextmanager
    async def scope(self, priority: Priority, request_id: str):
        """上下文管理器封装"""
        await self.acquire(priority, request_id)
        try:
            yield
        finally:
            await self.release()

# 全局单例
_global_guard = PriorityGuard(max_concurrent=3) # RTX 4060 建议并发 2-4

def get_priority_guard() -> PriorityGuard:
    return _global_guard

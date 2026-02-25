import structlog
import asyncio
import logging
import time
import torch
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from fastapi import FastAPI
from pymilvus import connections
from app.core.config import settings
from app.core.cache.redis_pool import get_redis_client, close_redis_pool

from app.core.monitoring.metrics import GPU_MEMORY_USAGE, MODEL_POOL_ACTIVE, MODEL_OFFLOAD_EVENTS

logger = structlog.get_logger(__name__)

class ModelPoolManager:
    """
    [Phase 6.2] ONNX 动态模型卸载管理器 (Memory Offloading Strategy).
    管理 Embedding, Reranker 等非 vLLM 模型，在闲置时自动从 GPU 卸载以腾出显存。
    """
    _models: Dict[str, Any] = {}
    _last_used: Dict[str, float] = {}
    _lock = asyncio.Lock()
    _monitor_task: Optional[asyncio.Task] = None
    
    IDLE_TIMEOUT = 300  # 闲置 5 分钟自动卸载
    VRAM_CRITICAL_THRESHOLD = 0.90 # 显存占用 90% 时强制卸载闲置模型

    @classmethod
    def register(cls, name: str, model_instance: Any):
        """注册模型实例"""
        cls._models[name] = model_instance
        cls._last_used[name] = time.time()
        # 初始化 Metrics
        MODEL_POOL_ACTIVE.labels(model_type=name).set(1)
        logger.info("model_pool_registered", name=name)

    @classmethod
    def mark_used(cls, name: str):
        """标记模型正在被使用"""
        cls._last_used[name] = time.time()
        MODEL_POOL_ACTIVE.labels(model_type=name).set(1)

    @classmethod
    async def monitor_loop(cls):
        """后台监控循环：检查模型闲置状态"""
        logger.info("model_pool_monitor_started")
        while True:
            try:
                await asyncio.sleep(30) # 缩短检查间隔至 30 秒，更灵敏
                now = time.time()
                
                # 检查 GPU 显存压力
                vram_usage = 0.0
                if torch.cuda.is_available():
                    vram_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    GPU_MEMORY_USAGE.set(vram_usage) # 更新 Metrics
                
                async with cls._lock:
                    for name, model_instance in cls._models.items():
                        idle_time = now - cls._last_used.get(name, 0)
                        
                        # 卸载逻辑：超过闲置时间 OR 显存压力过大
                        if hasattr(model_instance, 'is_on_gpu') and model_instance.is_on_gpu:
                            if idle_time > cls.IDLE_TIMEOUT or (vram_usage > cls.VRAM_CRITICAL_THRESHOLD and idle_time > 60):
                                logger.info("model_pool_offloading_idle_model", 
                                          name=name, 
                                          idle_sec=f"{idle_time:.1f}",
                                          vram_usage=f"{vram_usage:.2%}")
                                if hasattr(model_instance, 'offload'):
                                    await asyncio.to_thread(model_instance.offload)
                                    # 更新 Metrics
                                    MODEL_POOL_ACTIVE.labels(model_type=name).set(0)
                                    MODEL_OFFLOAD_EVENTS.labels(model_name=name).inc()
                                    # 强制清理显存
                                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error("model_pool_monitor_error", error=str(e))

    @classmethod
    async def start_monitor(cls):
        """启动监控任务"""
        if cls._monitor_task is None:
            cls._monitor_task = asyncio.create_task(cls.monitor_loop())

    @classmethod
    async def stop_monitor(cls):
        """停止监控任务"""
        if cls._monitor_task:
            cls._monitor_task.cancel()
            cls._monitor_task = None

    @classmethod
    async def clear_all(cls):
        """强制卸载所有模型并清理 (VRAM 深度释放)"""
        async with cls._lock:
            logger.info("model_pool_clearing_all", count=len(cls._models))
            for name, model_instance in list(cls._models.items()):
                try:
                    if hasattr(model_instance, 'offload'):
                        await asyncio.to_thread(model_instance.offload)
                    MODEL_POOL_ACTIVE.labels(model_type=name).set(0)
                except Exception as e:
                    logger.error("model_pool_offload_failed", name=name, error=str(e))
            
            cls._models.clear()
            cls._last_used.clear()
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info("model_pool_cleared")

class CloudHealthCheck:
    """
    [Phase 6.5] Cloud LLM Connectivity Check
    """
    STATUS = "PENDING"
    MESSAGE = "Initializing..."
    
    @classmethod
    async def run_check(cls):
        from app.core.llm.llm_factory import get_fast_llm
        from langchain_core.messages import HumanMessage
        logger.info("cloud_health_check_starting")
        try:
            # Cloud-path check (endpoint can be OpenAI-compatible local service like Ollama).
            llm = get_fast_llm(max_tokens=5, allow_local=False)
            await llm.ainvoke([HumanMessage(content="Ping")])
            cls.STATUS = "ONLINE"
            cls.MESSAGE = f"Model: {settings.OPENAI_MODEL_NAME_DYN}"
            logger.info("cloud_health_check_success")
        except Exception as e:
            cls.STATUS = "FAILED"
            cls.MESSAGE = str(e)[:30] + "..."
            logger.error("cloud_health_check_failed", error=str(e))

class InfrastructureManager:
    """
    全局基础设施管理器 (Infrastructure Manager).
    统一管理 Redis, Milvus, 数据库等连接池的生命周期。
    """
    
    @staticmethod
    async def init_milvus():
        """初始化 Milvus 连接池"""
        try:
            if not connections.has_connection("default"):
                logger.info("infra_milvus_connecting", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT
                )
            logger.info("infra_milvus_connected")
        except Exception as e:
            logger.error("infra_milvus_connection_failed", error=str(e))

    @staticmethod
    async def close_milvus():
        """关闭 Milvus 连接"""
        try:
            connections.disconnect("default")
            logger.info("infra_milvus_disconnected")
        except Exception as e:
            logger.error("infra_milvus_disconnect_failed", error=str(e))

    @staticmethod
    async def init_redis():
        """初始化 Redis 连接池"""
        try:
            from app.core.cache.redis_pool import get_redis_client, get_redis_binary_client
            client = get_redis_client()
            binary_client = get_redis_binary_client()
            await asyncio.gather(client.ping(), binary_client.ping())
            logger.info("infra_redis_connected", pools=["text", "binary"])
        except Exception as e:
            logger.error("infra_redis_connection_failed", error=str(e))

    @staticmethod
    async def close_redis():
        """关闭 Redis 连接池"""
        await close_redis_pool()

    @staticmethod
    async def close_resources():
        """关闭所有资源连接池"""
        from app.db.session import close_db
        await asyncio.gather(
            close_redis_pool(),
            close_db(),
            InfrastructureManager.close_milvus()
        )
        await ModelPoolManager.stop_monitor()
        logger.info("infrastructure_shutdown_complete")

    @staticmethod
    async def init_resources():
        """初始化核心业务资源 (Embedding, Retriever, Jieba)"""
        import jieba
        from app.core.tools.medical_tools import get_retriever
        from app.services.embedding import EmbeddingService
        from app.core.monitoring.tracing import setup_langsmith

        # 1. Jieba Warmup
        jieba.setLogLevel(logging.INFO)
        await asyncio.to_thread(jieba.initialize)
        logger.info("infra_jieba_initialized")

        # 2. LangSmith Tracing
        setup_langsmith()

        # 3. Embedding & Retriever
        # [Fix] 移除 asyncio.to_thread，确保在当前 Event Loop 中初始化
        from app.services.embedding import EmbeddingService
        from app.core.tools.medical_tools import get_retriever
        
        # 预热模型与检索器
        EmbeddingService() 
        retriever = get_retriever()
        await retriever.initialize() # [New] Async Initialization for BM25
        
        logger.info("infra_resources_initialized")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan 管理器。
    """
    logger.info("app_lifespan_starting")
    
    # 并行初始化基础设施
    logger.info("infra_init_starting")
    try:
        await asyncio.wait_for(asyncio.gather(
            InfrastructureManager.init_redis(),
            InfrastructureManager.init_milvus()
        ), timeout=10.0) # Add timeout
    except asyncio.TimeoutError:
        logger.error("infra_init_timeout_skipping")
    except Exception as e:
        logger.error("infra_init_failed", error=str(e))
    logger.info("infra_init_completed")
    
    # 初始化核心业务资源
    logger.info("resources_init_starting")
    try:
        await asyncio.wait_for(InfrastructureManager.init_resources(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.error("resources_init_timeout_skipping")
    except Exception as e:
        logger.error("resources_init_failed", error=str(e))
    logger.info("resources_init_completed")
    
    # [Phase 6.2] 启动模型卸载监控
    await ModelPoolManager.start_monitor()
    
    # [Phase 6.5] Cloud Health Check
    # 启动时不阻塞主流程，避免网络不可达导致启动超时
    asyncio.create_task(CloudHealthCheck.run_check())
    
    # 打印启动看板
    try:
        from app.core.infra_dashboard import print_startup_dashboard
        print_startup_dashboard()
    except Exception as e:
        logger.warning("dashboard_print_failed", error=str(e))
    
    yield
    
    # [Phase 6.2] 停止模型卸载监控
    await ModelPoolManager.stop_monitor()
    
    # 关闭基础设施
    logger.info("app_lifespan_stopping")
    await asyncio.gather(
        InfrastructureManager.close_redis(),
        InfrastructureManager.close_milvus()
    )
    logger.info("app_lifespan_stopped")

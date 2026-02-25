import time
import structlog
import asyncio
from typing import Dict, Any
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node

logger = structlog.get_logger(__name__)

async def _async_persistence_task(state: AgentState):
    """
    异步执行持久化任务 (Fire-and-Forget)
    """
    try:
        start_time = time.time()
        # 延迟导入以避免测试环境下的 DB 连接问题
        from app.db.session import AsyncSessionLocal
        from app.services.medical_record import MedicalRecordService
        from app.core.memory.patient_memory import PatientMemoryRetriever
        
        status = state.get("status")
        patient_id = state.get("patient_id", "guest")
        session_id = state.get("session_id", "unknown")
        # [Pain Point #3] 优先使用归一化的 clinical_report 字段
        diagnosis_report = state.get("clinical_report") or state.get("diagnosis_report", "")
        
        # 序列化消息
        serialized_history = []
        for msg in state.get("messages", []):
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            serialized_history.append({"role": role, "content": content})
        
        # 1. PostgreSQL 持久化
        async with AsyncSessionLocal() as session:
            try:
                service = MedicalRecordService(session)
                try:
                    await service.create_consultation(
                        patient_id=patient_id,
                        session_id=session_id,
                        symptoms=state.get("symptoms", ""),
                        medical_history=state.get("medical_history", "")
                    )
                except Exception:
                    await session.rollback()
                    pass # 忽略重复创建错误
                
                await service.update_diagnosis(
                    session_id=session_id,
                    diagnosis_result=diagnosis_report,
                    dialogue_history=serialized_history
                )
                logger.info("persistence_success", target="postgres", session_id=session_id)
            except Exception as db_err:
                logger.error("persistence_error", target="postgres", error=str(db_err))
                await session.rollback()

        # 2. Milvus 记忆持久化 (仅当 approved 或 fast_track)
        if status in ["approved", "fast_track"] and diagnosis_report:
            try:
                memory_retriever = PatientMemoryRetriever()
                memory_content = f"Symptoms: {state.get('symptoms')}\nDiagnosis: {diagnosis_report}"
                if len(memory_content) > 1000:
                    memory_content = memory_content[:1000] + "...(truncated)"
                
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, 
                    lambda: memory_retriever.add_memory(patient_id, session_id, memory_content)
                )
                logger.info("persistence_success", target="milvus", session_id=session_id)
                
            except Exception as e:
                logger.error("milvus_persistence_failed", error=str(e))
                
        # 3. 语义缓存持久化 (Redis)
        if status in ["approved", "fast_track"] and diagnosis_report:
            try:
                from app.core.cache.redis_pool import get_redis_client
                import hashlib
                import re
                import json
                
                user_input = state.get("symptoms", "")
                if user_input:
                    clean_query = "".join(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]", user_input)).lower()
                    query_hash = hashlib.md5(clean_query.encode("utf-8")).hexdigest()
                    cache_key = f"cache:{query_hash}"
                    
                    redis_client = get_redis_client()
                    cache_data = {
                        "response": diagnosis_report,
                        "timestamp": time.time(),
                        "session_id": session_id
                    }
                    
                    await redis_client.setex(
                        cache_key,
                        3600 * 24,
                        json.dumps(cache_data)
                    )
                    logger.info("persistence_success", target="redis", key=cache_key)
            except Exception as cache_err:
                logger.error("redis_persistence_failed", error=str(cache_err))

        duration = time.time() - start_time
        logger.info("async_persistence_completed", duration=f"{duration:.2f}s")
        
    except Exception as e:
        logger.error("async_persistence_critical_error", error=str(e))

@monitor_node("persistence")
async def persistence_node(state: AgentState):
    """
    节点：持久化 (Persistence)
    启动异步任务保存数据，不阻塞回复。
    """
    print(f"[DEBUG] Node persistence Start")
    logger.info("node_start", node="persistence")
    
    # 启动 Fire-and-Forget 异步任务
    asyncio.create_task(_async_persistence_task(state.copy()))
    
    return {"status": "persisted"}

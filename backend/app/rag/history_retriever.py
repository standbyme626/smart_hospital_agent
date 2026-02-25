
import asyncio
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.medical_record import MedicalRecordService
from app.core.memory.patient_memory import PatientMemoryRetriever

from langsmith import traceable

class HistoryRetriever:
    """
    历史病历检索器 (History Retriever)
    聚合 PostgreSQL (结构化近期记录) 和 Milvus (语义化长期记忆)
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db_service = MedicalRecordService(db_session)
        # PatientMemoryRetriever 是单例或轻量级，可以在这里初始化
        # [Resilience Fix] Wrap in try/except to prevent Milvus failure from blocking PG retrieval
        try:
            self.memory_retriever = PatientMemoryRetriever()
        except Exception as e:
            print(f"[HistoryRetriever] Warning: Failed to initialize PatientMemoryRetriever (Milvus might be down). Error: {e}")
            self.memory_retriever = None

    # @traceable(run_type="retriever", name="History_Context_Retrieval")
    async def get_patient_context(self, patient_id: str, query: str = "", config: Any = None) -> str:
        """
        获取患者综合上下文
        """
        context_parts = []
        
        # 1. 获取近期诊疗记录 (PostgreSQL)
        try:
            # 获取最近 5 次
            recent_consultations = await self.db_service.get_consultation_history(patient_id)
            
            # 按时间倒序
            recent_consultations.sort(key=lambda x: x.created_at, reverse=True)
            recent_5 = recent_consultations[:5]
            
            if recent_5:
                context_parts.append("=== Recent Medical History (Last 5 Visits) ===")
                for cons in recent_5:
                    date_str = cons.created_at.strftime("%Y-%m-%d")
                    context_parts.append(f"Date: {date_str} | Session ID: {cons.session_id} | Diagnosis: {cons.diagnosis_result or 'Pending'}")
                    if cons.prescriptions:
                        meds = ", ".join([p.medication_name for p in cons.prescriptions])
                        context_parts.append(f"Prescriptions: {meds}")
                    context_parts.append("---")
        except Exception as e:
            import logging
            logging.getLogger("app.rag.history_retriever").error(f"Error fetching PG history: {e}")
        
        # 2. 获取语义相关记忆 (Milvus)
        # 如果有具体查询症状，尝试检索相关过往 (e.g. "Previous headaches")
        if query and self.memory_retriever:
            try:
                # [V6.2 Fix] Use to_thread with timeout to prevent blocking the event loop on Milvus calls
                semantic_memories = await asyncio.wait_for(
                    asyncio.to_thread(self.memory_retriever.search_memory, patient_id, query, top_k=3),
                    timeout=3.0
                )
            except Exception as e:
                import logging
                logging.getLogger("uvicorn.error").warning(f"Semantic memory retrieval failed: {e}")
                semantic_memories = []

            if semantic_memories:
                context_parts.append("\n=== Relevant Past History (Semantic Search) ===")
                for mem in semantic_memories:
                    # 过滤掉已经是"最近记录"的重复项 (简单通过 session_id 判断)
                    recent_session_ids = [c.session_id for c in recent_5]
                    if mem['session_id'] not in recent_session_ids:
                        context_parts.append(f"Note: {mem['content']} (Score: {mem['score']:.2f})")

        if not context_parts:
            return "No medical history found."
            
        return "\n".join(context_parts)

    async def save_session_summary(self, patient_id: str, session_id: str, summary: str):
        """
        保存会话摘要到长期记忆
        """
        # [V6.2 Fix] Use to_thread for synchronous Milvus operation
        try:
            await asyncio.to_thread(self.memory_retriever.add_memory, patient_id, session_id, summary)
        except Exception as e:
            import logging
            logging.getLogger("uvicorn.error").error(f"Failed to save session summary: {e}")

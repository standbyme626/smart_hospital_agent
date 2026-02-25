import json
import time
import structlog
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain_core.messages import AIMessage
from app.core.llm.llm_factory import SmartRotatingLLM
from app.core.config import settings

logger = structlog.get_logger(__name__)

class AuditSystem:
    """
    审计代理 (The Referee) - V2.0 异步/非侵入式架构
    - 评估医学准确性 (Accuracy)
    - 监控性能 (Latency, Tokens)
    - 计算全局评分 (Global Score)
    - 支撑 RAG 驱动的执业标准校验
    """
    
    def __init__(self):
        self.llm = SmartRotatingLLM(
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=0.0 # Strict mode
        )
        # 加载标准映射表
        self.standards_path = Path(settings.PROJECT_ROOT) / "backend/data/knowledge_base/EXAM_STANDARDS_MAPPING.json"
        self.standards = self._load_standards()
        
    def _load_standards(self) -> Dict[str, Any]:
        if self.standards_path.exists():
            try:
                with open(self.standards_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("exam_standards", {}).get("dimensions", {})
            except Exception as e:
                logger.error("audit.standards_load_failed", error=str(e))
        return {}

    async def run_async_audit(self, case_data: Dict, chat_history: List[Dict], metadata: Dict[str, Any] = None):
        """
        [Async Entry Point] 异步审计入口
        通常由 BackgroundTasks 或 CallbackHandler 调用，不阻塞主线程。
        """
        try:
            # 仅在 Evolution Mode 或明确要求时运行完整审计
            if not settings.EVOLUTION_MODE:
                # 生产模式仅记录基础日志，不做耗时 LLM 评分
                logger.info("audit.production_log", case_id=case_data.get("id"), metadata=metadata)
                return
                
            logger.info("audit.async_start", case_id=case_data.get("id"))
            
            # 模拟获取 Patient Score (在真实场景中可能来自 metadata 或另外的反馈渠道)
            patient_score = metadata.get("ux_score", 75) 
            
            result = await self.audit_session(case_data, chat_history, patient_score, metadata)
            
            # 如果是低分案例，写入优化日志
            if result['global_score'] < 70:
                self._log_for_optimization(result, chat_history)
                
        except Exception as e:
            logger.error("audit.async_failed", error=str(e))

    async def audit_session(self, case_data: Dict, chat_history: List[Dict], patient_score: int, metadata: Dict = None) -> Dict[str, Any]:
        """
        对一次完整的会话进行深度审计
        """
        metadata = metadata or {}
        
        # 1. 提取上下文
        final_diagnosis = "未给出明确结论"
        doctor_replies = []
        # [NEW] Check for internal_monologue/thought to verify Expert Group involvement
        has_thought_process = False
        
        for msg in chat_history:
            if msg["role"] == "doctor":
                final_diagnosis = msg["content"]
                doctor_replies.append(msg["content"])
            # Check if any message or metadata indicates thought process (from DSPy/Expert)
            if msg.get("type") == "thought" or "reasoning" in msg:
                has_thought_process = True
        
        # [NEW] Repetition Check
        is_repetitive = False
        if len(doctor_replies) >= 2:
             from difflib import SequenceMatcher
             last_reply = doctor_replies[-1]
             prev_reply = doctor_replies[-2]
             similarity = SequenceMatcher(None, last_reply, prev_reply).ratio()
             if similarity > 0.9:
                 is_repetitive = True
                 logger.warning("audit.repetition_detected", similarity=similarity)
        
        full_dialogue = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
        
        # 2. 多维度评分 (Ethics, Anamnesis, Diagnostic Logic, Safety)
        # 使用 RAG/Standards 注入 Prompt
        scores, reasons = await self._evaluate_multidimensional(case_data, full_dialogue, final_diagnosis)
        
        # [NEW] Penalize Repetition
        if is_repetitive:
            scores["diagnostic_logic"] = 0
            reasons["diagnostic_logic"] = reasons.get("diagnostic_logic", "") + " [FATAL] Logic Loop Detected (Repetitive Responses)."
            
        # [NEW] Penalize Skipping Expert Group
        # 如果没有思维链，且不是简单的寒暄，说明跳过了专家组
        if not has_thought_process and len(chat_history) > 2:
             scores["diagnostic_logic"] = max(0, scores.get("diagnostic_logic", 0) - 50)
             reasons["diagnostic_logic"] = reasons.get("diagnostic_logic", "") + " [WARNING] EXPERT_GROUP_SKIPPED: No reasoning trace found."
             logger.warning("audit.expert_group_skipped", case_id=case_data.get("id"))
        
        # 3. 性能指标审计
        latency = metadata.get("latency_ms", 0)
        tokens = metadata.get("total_tokens", 0)
        efficiency_score = self._calculate_efficiency_score(latency, tokens)
        
        # 4. 计算全局分
        # Global = (Ethics*0.15 + Anamnesis*0.3 + Logic*0.35 + Safety*0.2) * 0.8 + UX*0.2
        # (这里简化权重公式以适应旧接口，或者使用 standards 中的权重)
        
        medical_score = (
            scores.get("ethics", 0) * 0.15 +
            scores.get("anamnesis", 0) * 0.30 +
            scores.get("diagnostic_logic", 0) * 0.35 +
            scores.get("safety", 0) * 0.20
        )
        
        global_score = (medical_score * 0.7) + (patient_score * 0.2) + (efficiency_score * 0.1)
        
        result = {
            "case_id": case_data.get("id"),
            "global_score": round(global_score, 2),
            "breakdown": {
                "medical_score": round(medical_score, 2),
                "dimensions": scores,
                "ux_score": patient_score,
                "efficiency": efficiency_score
            },
            "audit_reasons": reasons,
            "timestamp": time.time()
        }
        
        logger.info("audit_complete", result=result)
        return result

    async def _evaluate_multidimensional(self, case_data: Dict, dialogue: str, diagnosis: str) -> (Dict[str, float], Dict[str, str]):
        """
        基于 EXAM_STANDARDS_MAPPING 执行多维度评分
        """
        # 构建 Prompt，注入标准
        standards_desc = json.dumps(self.standards, ensure_ascii=False, indent=2)
        
        prompt = f"""
你是一名国家医师资格考试的考官。请根据《医师资格考试暂行办法》评分标准，对考生的临床表现进行打分。

【评分标准 (JSON)】:
{standards_desc}

【标准病例】:
- 预期诊断: {case_data.get('diagnosis_standard')}
- 隐藏信息/关键点: {case_data.get('hidden_info')}

【考生(医生)与患者对话记录】:
{dialogue}

请对以下四个维度进行评分 (0-100)：
1. 职业素质 (ethics)
2. 病史采集 (anamnesis)
3. 临床思辨 (diagnostic_logic)
4. 医疗安全 (safety)

输出必须为严格的 JSON 格式：
{{
  "scores": {{
    "ethics": <int>,
    "anamnesis": <int>,
    "diagnostic_logic": <int>,
    "safety": <int>
  }},
  "reasons": {{
    "ethics": "<扣分理由>",
    "anamnesis": "<扣分理由>",
    "diagnostic_logic": "<扣分理由>",
    "safety": "<扣分理由>"
  }}
}}
"""
        try:
            response = await self.llm.ainvoke(prompt)
            # 解析 JSON
            import re
            content = response.content.strip()
            # 尝试提取 JSON 块
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("scores", {}), data.get("reasons", {})
            return {}, {}
            
        except Exception as e:
            logger.error("audit.llm_eval_failed", error=str(e))
            # [Fix] 异常透传：严禁空 try-except。错误必须记录并向上抛出。
            # 如果是严重的鉴权或模型错误，不应掩盖，否则导致死循环重试
            error_str = str(e).lower()
            if "401" in error_str or "403" in error_str or "insufficient_quota" in error_str:
                raise e
                
            # Fallback - Ensure non-zero scores to prevent loop death
            return {
                "ethics": 50, 
                "anamnesis": 50, 
                "diagnostic_logic": 50, 
                "safety": 50
            }, {"error": str(e), "note": "Audit LLM Failed - Using Fallback Scores"}

    def _calculate_efficiency_score(self, latency_ms: float, tokens: int) -> int:
        """
        计算效率分。
        Latency > 15000ms (15s) 强制扣分。
        """
        score = 100
        if latency_ms > 15000:
            score -= 20
        elif latency_ms > 8000:
            score -= 10
            
        if tokens > 4000: # 过度啰嗦
            score -= 10
            
        return max(0, score)

    def _log_for_optimization(self, result: Dict, chat_history: List):
        """
        记录低分案例到 candidates/ 目录供进化引擎使用
        """
        try:
            log_dir = Path(settings.PROJECT_ROOT) / "backend/data/audit_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            filename = f"failed_case_{int(time.time())}_{result['case_id']}.json"
            data = {
                "audit_result": result,
                "chat_history": chat_history
            }
            with open(log_dir / filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("audit.log_failed", error=str(e))

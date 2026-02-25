import json
import random
import asyncio
import os
from typing import Dict, Any, List, Optional
import structlog
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.llm.llm_factory import SmartRotatingLLM
from app.core.config import settings

logger = structlog.get_logger(__name__)

class PatientAdversary:
    """
    模拟患者代理 (Red Team)
    - 负责加载真实/合成病例
    - 模拟患者对话行为（隐瞒信息、情绪化）
    - 给出满意度评分
    """
    
    def __init__(self, case_id: str = None, department: str = None):
        self.llm = SmartRotatingLLM(
            model_name=settings.OPENAI_MODEL_NAME, 
            temperature=0.7 # High temperature for diverse persona
        )
        self.case_data = self._load_case(case_id, department)
        self.persona = self._generate_persona()
        self.chat_history = []
        self.step_count = 0
        
        logger.info("patient_agent_initialized", 
                    case_id=self.case_data.get("id"), 
                    persona=self.persona['type'])

    def _load_case(self, case_id: str, department: str) -> Dict[str, Any]:
        """
        从 backend/data/test_cases/standard_cases.json 加载病例
        """
        cases = []
        file_path = os.path.join(settings.PROJECT_ROOT, "backend/data/test_cases/standard_cases.json")
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    cases = json.load(f)
                logger.info("patient_cases_loaded", count=len(cases), source=file_path)
            else:
                logger.warning("patient_cases_file_missing", path=file_path)
        except Exception as e:
            logger.error("patient_cases_load_failed", error=str(e))

        # Fallback if file load fails or empty
        if not cases:
            logger.warning("using_fallback_hardcoded_cases")
            cases = [
                {
                    "id": "CASE_CARDIO_001",
                    "department": "Cardiology",
                    "chief_complaint": "胸口疼",
                    "hidden_info": "疼痛向左肩放射，伴有出汗。既往有高血压史，未规律服药。",
                    "diagnosis_standard": "不稳定型心绞痛或急性心梗",
                    "keywords": ["心梗", "急诊", "心电图"]
                }
            ]
        
        if case_id:
            for c in cases:
                if c["id"] == case_id:
                    return c
        
        # Random pick by department if specified
        if department:
            filtered = [c for c in cases if c["department"].lower() == department.lower()]
            if filtered:
                return random.choice(filtered)
            else:
                logger.warning("no_case_found_for_department", department=department)
                
        return random.choice(cases)

    def _generate_persona(self) -> Dict[str, str]:
        """
        随机生成性格特征
        """
        personas = [
            {"type": "Anxious", "desc": "非常焦虑，担心是大病，语速快，不断追问。"},
            {"type": "Skeptical", "desc": "怀疑医生的水平，语气冷淡，只回答必要问题。"},
            {"type": "Confused", "desc": "表达不清，逻辑混乱，需要医生引导。"},
            {"type": "Cooperative", "desc": "配合度高，如实回答。"}
        ]
        return random.choice(personas)

    async def speak(self, doctor_message: str = "") -> str:
        """
        生成患者回复
        """
        self.step_count += 1
        
        # 第一轮只说主诉
        if self.step_count == 1 and not doctor_message:
            return f"医生，我{self.case_data['chief_complaint']}。"
            
        # 后续轮次：根据医生提问释放信息
        system_prompt = f"""
你正在扮演一名患者。
【你的基本情况】：
- 主诉：{self.case_data['chief_complaint']}
- 隐藏信息（只有医生问到相关问题时才说）：{self.case_data['hidden_info']}

【你的性格】：{self.persona['type']} - {self.persona['desc']}

【当前对话】：
医生问："{doctor_message}"

请根据你的性格生成回复。
规则：
1. 如果医生问到了隐藏信息相关的内容，请释放一部分信息。
2. 如果医生没问到，不要主动说隐藏信息。
3. 保持口语化，不要太专业。
4. 回复简短有力，不要超过 50 字。
"""
        response = await self.llm.ainvoke(system_prompt)
        content = response.content
        self.chat_history.append({"role": "doctor", "content": doctor_message})
        self.chat_history.append({"role": "patient", "content": content})
        
        # [Fix] Dead Loop Circuit Breaker
        # 如果收到重复的医生回复，强制终止对话并标记失败
        if self.step_count > 10:  # Increased from 5 to 10 to allow longer conversations
             doctor_msgs = [m["content"] for m in self.chat_history if m["role"] == "doctor"]
             if len(doctor_msgs) >= 3 and doctor_msgs[-1] == doctor_msgs[-2] == doctor_msgs[-3]: # Strict check: 3 repeats
                 logger.error("patient_agent_dead_loop_detected", action="terminate")
                 # Return a specific message to break the loop instead of raising error
                 return "医生，您好像在重复同样的话。请问还有什么建议吗？"
                 
        return content

    async def evaluate_satisfaction(self) -> int:
        """
        对话结束后，自我评价满意度 (0-100)
        """
        system_prompt = f"""
你是一名患者，刚刚结束了与医生的对话。
请根据以下对话历史，给医生的服务打分（0-100）。

【评分标准】：
- 80-100：医生专业，态度好，解决了我的疑问。
- 60-79：医生一般，回答中规中矩。
- 0-59：医生没听懂我的话，或者态度恶劣，或者没解决问题。

【对话历史】：
{json.dumps(self.chat_history, ensure_ascii=False)}

只输出一个数字，不要其他内容。
"""
        try:
            response = await self.llm.ainvoke(system_prompt)
            score = int(''.join(filter(str.isdigit, response.content)))
            score = max(0, min(100, score)) # Clamp
            return score
        except Exception as e:
            logger.error("patient_eval_failed", error=str(e))
            return 50 # Default

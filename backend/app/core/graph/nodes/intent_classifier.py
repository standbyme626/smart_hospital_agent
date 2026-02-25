from typing import Dict, Any
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import structlog
from app.core.graph.state import AgentState
from app.core.llm.llm_factory import get_fast_llm
from app.core.models.local_slm import LocalSLMService
from app.core.monitoring.quality_collector import QualityCollector
from app.core.prompts.triage import get_intent_classification_prompt

logger = structlog.get_logger(__name__)

class IntentClassifier:
    """
    意图识别节点 (Intent Classifier)
    
    [V11.29 优化]
    优先使用本地 0.6B DPO 模型进行医疗意图识别，
    利用其“医生本能” (CoT) 进行科室建议。
    """
    
    def __init__(self):
        self.llm = get_fast_llm(temperature=0.0) # 备用快速模型
        self.slm = LocalSLMService() # 核心本地模型
        self.quality_collector = QualityCollector() # [Plan 5] MLOps 采集器
        
        # 定义 Prompt (仅作为备用)
        self.prompt = ChatPromptTemplate.from_template(
            get_intent_classification_prompt(user_input="{input}")
        )
        self.chain = self.prompt | self.llm | StrOutputParser()
        
    def _extract_user_input(self, state: AgentState) -> str:
        """只抽取本轮用户输入，避免误读到上一轮 AI 回复。"""
        for key in ("current_turn_input", "retrieval_query", "user_input", "symptoms"):
            value = state.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        event = state.get("event", {})
        if isinstance(event, dict):
            raw_input = event.get("raw_input")
            if isinstance(raw_input, str) and raw_input.strip():
                return raw_input.strip()

        messages = state.get("messages", [])
        for msg in reversed(messages):
            msg_type = None
            msg_content = None
            if hasattr(msg, "type") and hasattr(msg, "content"):
                msg_type = getattr(msg, "type", None)
                msg_content = getattr(msg, "content", None)
            elif isinstance(msg, dict):
                msg_type = msg.get("type") or msg.get("role")
                msg_content = msg.get("content")

            if msg_type == "user":
                msg_type = "human"
            if msg_type == "human" and isinstance(msg_content, str) and msg_content.strip():
                return msg_content.strip()
        return ""

    def _check_fast_track(self, text: str) -> str | None:
        """
        [Layer 1] Fast Track Analysis
        Regex/Keyword based detection for immediate routing.
        """
        text = text.lower()
        
        # 1. CRISIS (Highest Priority)
        crisis_keywords = [
            "救命", "胸痛", "昏迷", "呼吸困难", "心脏病", "中风", "大出血", "120", "急救",
            "dying", "stroke", "想死", "自杀", "suicide", "不想活", "轻生"
        ]
        if any(kw in text for kw in crisis_keywords):
            return "CRISIS"
            
        # 2. Explicit Medical Services
        service_keywords = ["挂号", "预约", "看病", "医生", "门诊", "急诊", "科室", "register", "appointment", "booking"]
        service_patterns = [
            r"挂.{0,8}号",      # 挂号 / 挂明天下午心内科号
            r"约.{0,8}号",      # 约号
            r"预约.{0,12}(门诊|医生|科室|号)",
        ]
        if any(kw in text for kw in service_keywords) or any(re.search(p, text) for p in service_patterns):
            return "REGISTRATION"
            
        # 3. Explicit Info/Location
        info_keywords = ["哪里", "地址", "位置", "几点", "时间", "电话", "contact", "address", "where"]
        if any(kw in text for kw in info_keywords):
            return "INFO"

        # 4. Explicit Quit/End
        if text in ["退出", "结束", "再见", "bye", "exit"]:
            return "GREETING" # Will be handled by fast_reply to say goodbye
            
        return None

    def _rule_based_fallback(self, text: str) -> str:
        """基于规则的兜底分类，用于 LLM 不可用或微调时"""
        text = text.lower()
        
        # CRISIS 关键词
        crisis_keywords = [
            "救命", "胸痛", "昏迷", "呼吸困难", "心脏病发作", "中风", "大出血", "120", "急救",
            "dying", "heart attack", "stroke", "想死", "自杀", "suicide", "不想活", "轻生"
        ]
        for kw in crisis_keywords:
            if kw in text:
                return "CRISIS"

        # 挂号服务关键词
        service_keywords = ["挂号", "预约", "门诊", "找医生", "看医生", "科室", "register", "appointment", "booking"]
        service_patterns = [
            r"挂.{0,8}号",
            r"约.{0,8}号",
            r"预约.{0,12}(门诊|医生|科室|号)",
        ]
        if any(kw in text for kw in service_keywords) or any(re.search(p, text) for p in service_patterns):
            return "REGISTRATION"

        # 信息查询关键词
        info_keywords = ["哪里", "地址", "位置", "几点", "时间", "电话", "contact", "address", "where"]
        if any(kw in text for kw in info_keywords):
            return "INFO"
                
        # GREETING 关键词
        greeting_keywords = ["你好", "hello", "hi", "早安", "在吗", "who are you"]
        if len(text) < 10 and any(kw in text for kw in greeting_keywords):
            return "GREETING"
            
        # VAGUE 关键词
        if len(text) < 5:
            return "VAGUE_SYMPTOM"
            
        return "MEDICAL_CONSULT"

    def _normalize_intent(self, raw_intent: str, user_input: str, raw_output: str = "") -> str:
        intent = (raw_intent or "").strip().upper()

        alias_map = {
            "SERVICE_BOOKING": "REGISTRATION",
            "APPOINTMENT": "REGISTRATION",
            "BOOKING": "REGISTRATION",
            "SERVICE": "REGISTRATION",
            "SYMPTOM": "MEDICAL_CONSULT",
            "MEDICAL": "MEDICAL_CONSULT",
            "COMPLEX_SYMPTOM": "MEDICAL_CONSULT",
            "STANDARD": "MEDICAL_CONSULT",
            "VAGUE": "VAGUE_SYMPTOM",
        }

        if intent in alias_map:
            intent = alias_map[intent]

        valid = {"CRISIS", "REGISTRATION", "MEDICAL_CONSULT", "GREETING", "INFO", "VAGUE_SYMPTOM"}
        if intent in valid:
            return intent

        # 兜底：优先基于用户原句，避免模型输出格式漂移导致空分类
        fallback_intent = self._rule_based_fallback(user_input or raw_output)
        return fallback_intent

    async def run(self, state: AgentState) -> Dict[str, Any]:
        """节点执行入口"""
        print(f"[DEBUG] Node IntentClassifier Start")
        logger.info("node_start", node="intent_classifier")
        
        # [Fix] 优先读取本轮输入，避免上下文串轮
        user_input = self._extract_user_input(state)
        
        # Ensure user_input is a string
        if not isinstance(user_input, str):
            user_input = str(user_input)
        user_input = user_input.strip()

        intent = "MEDICAL_CONSULT"
        raw_output = ""
        source = "local_slm"
        
        try:
            if not self.slm:
                raise RuntimeError("Local SLM disabled")

            # [Layer 1] Fast Track Check (Regex/Keyword)
            fast_intent = self._check_fast_track(user_input)
            if fast_intent:
                logger.info("fast_track_hit", intent=fast_intent)
                fast_status = "crisis" if fast_intent == "CRISIS" else "classified"
                return {
                    "intent": fast_intent,
                    "raw_triage_output": f"[FastTrack] Matched keyword in: {user_input}",
                    "current_turn_input": user_input,
                    "retrieval_query": user_input,
                    "status": fast_status
                }

            # [Layer 2] Deep Analysis with Local SLM (Int8 Quantized)
            # 构造能够打破闲聊僵局的 System Prompt
            system_prompt = (
                "你是一个专业的医疗分诊助手。你的任务是分析用户的输入并分类。\n"
                "类别说明：\n"
                "1. CRISIS: 危及生命的紧急情况（如昏迷、心脏病、大出血）。\n"
                "2. MEDICAL_CONSULT: 包含任何医疗症状（痛、痒、晕等）或医疗服务请求（挂号、开药、看医生）。\n"
                "3. GREETING: 纯粹的打招呼或闲聊，不包含任何医疗内容。\n"
                "4. INFO: 询问医院信息（地址、时间、医生介绍）。\n"
                "5. VAGUE_SYMPTOM: 描述模糊，需要追问。\n"
                "\n"
                "重要规则：\n"
                "- 即使上下文是闲聊，只要用户提到任何身体不适或医疗需求，必须立即归类为 MEDICAL_CONSULT。\n"
                "- 宁可误判为医疗咨询，也不要漏掉病人的求助。\n"
                "- 如果用户描述了具体症状（如胸闷、疼痛、发烧等），即使没有明确说要看医生，也必须归类为 MEDICAL_CONSULT。\n"
                "- 请先在 <think> 标签中进行思考，最后在 <category> 标签中输出类别，例如 <category>CRISIS</category>。"
            )

            prompt = f"User Input: {user_input}\nOutput:"
            
            # 使用 generate_response_async 而不是 constrained_classify 以获得更灵活的控制
            raw_output = await self.slm.generate_response_async(
                prompt, 
                system_prompt=system_prompt,
                max_new_tokens=256,
                temperature=0.1
            )
            
            # 解析输出
            import re
            match = re.search(r"<category>(.*?)</category>", raw_output, re.IGNORECASE)
            if match:
                intent = match.group(1).strip().upper()
            else:
                # Fallback: strict check at start of line or explicitly labeled
                upper_out = raw_output.upper()
                if "CATEGORY: MEDICAL_CONSULT" in upper_out or "类别: MEDICAL_CONSULT" in upper_out:
                    intent = "MEDICAL_CONSULT"
                elif "CATEGORY: CRISIS" in upper_out or "类别: CRISIS" in upper_out:
                    intent = "CRISIS"
                elif "CATEGORY: INFO" in upper_out or "类别: INFO" in upper_out:
                    intent = "INFO"
                elif "CATEGORY: GREETING" in upper_out or "类别: GREETING" in upper_out:
                    intent = "GREETING"
                else:
                    # Semantic Keyword Fallback (Robustness for weak instruction following)
                    if any(kw in raw_output for kw in ["医疗", "挂号", "医生", "病", "痛", "不舒服", "药", "诊"]):
                        intent = "MEDICAL_CONSULT"
                    elif any(kw in raw_output for kw in ["哪里", "地址", "时间", "电话"]):
                        intent = "INFO"
                    elif any(kw in raw_output for kw in ["你好", "AI", "助手", "hello", "hi", "是谁"]):
                        intent = "GREETING"
                    else:
                        intent = "MEDICAL_CONSULT"

            intent = self._normalize_intent(intent, user_input, raw_output)
            
            # 记录原始输出用于调试
            logger.info("slm_classification_success", intent=intent, raw_output=raw_output[:100])
            
            # [Plan 5] 自动采集不确定样本
            if intent == "UNCERTAIN" or "不确定" in raw_output:
                self.quality_collector.collect_negative_sample(
                    prompt=user_input,
                    expected_category="COMPLEX_SYMPTOM", # 默认为复杂症状，待人工复核
                    actual_output=raw_output,
                    is_uncertain=True
                )
            
        except Exception as e:
            logger.error("slm_classification_failed", error=str(e))
            source = "rule_fallback"
            # 本地模型失败时，直接规则降级并保留原问题给 RAG
            intent = self._rule_based_fallback(user_input)
            raw_output = f"[Fallback] {intent}"
        
        # [V11.29] 返回状态包含原始输出
        status = "crisis" if intent == "CRISIS" else "classified"
        logger.info(
            "intent_classified",
            intent=intent,
            source=source,
            has_query=bool(user_input),
        )
        return {
            "intent": intent,
            "raw_triage_output": raw_output, # 透传原始推理，包含建议科室
            "current_turn_input": user_input,
            "retrieval_query": user_input,
            "status": status
        }

from app.core.monitoring.tracing import monitor_node

# 实例化供图使用
intent_classifier = IntentClassifier()

@monitor_node("intent_classifier")
async def intent_classifier_node(state: AgentState):
    return await intent_classifier.run(state)

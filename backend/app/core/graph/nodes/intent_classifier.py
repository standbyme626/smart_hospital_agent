import re
from typing import Any, Dict, Optional

import structlog

from app.core.graph.state import AgentState
from app.core.models.local_slm import LocalSLMService
from app.core.monitoring.quality_collector import QualityCollector

logger = structlog.get_logger(__name__)


class IntentClassifier:
    """
    意图识别节点：
    - 规则前置：优先命中危机/挂号/信息/寒暄。
    - 本地模型：4 类分诊标签。
    - 会话态纠偏：修正 GREETING 与症状类误判。
    - 不确定追问：低信息量输入收敛到 VAGUE_SYMPTOM。
    """

    CATEGORIES = ["CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]

    def __init__(self):
        self.slm = LocalSLMService()
        self.quality_collector = QualityCollector()

    def _extract_user_input(self, state: AgentState) -> str:
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

    def _has_medical_signal(self, text: str) -> bool:
        t = (text or "").lower()
        symptom_keywords = [
            "痛", "疼", "痒", "晕", "恶心", "呕吐", "发烧", "咳", "咳嗽", "鼻塞", "流鼻涕",
            "胸闷", "胸痛", "呼吸", "腹痛", "肚子", "腹泻", "拉肚子", "心慌", "乏力",
            "不舒服", "难受", "药", "挂号", "预约", "科室", "门诊", "急诊",
        ]
        return any(k in t for k in symptom_keywords)

    def _is_followup_fragment(self, text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        markers = ["还是", "还", "另外", "然后", "现在", "依旧", "继续", "又", "仍然"]
        return len(t) <= 12 and any(m in t for m in markers)

    def _rule_front_intent(self, text: str) -> Optional[str]:
        t = (text or "").strip().lower()
        if not t:
            return None

        crisis_keywords = [
            "救命", "胸痛", "昏迷", "呼吸困难", "心脏病", "中风", "大出血", "120", "急救",
            "dying", "stroke", "想死", "自杀", "suicide", "不想活", "轻生",
        ]
        if any(k in t for k in crisis_keywords):
            return "CRISIS"

        service_keywords = ["挂号", "预约", "看病", "医生", "门诊", "急诊", "科室", "register", "appointment", "booking"]
        service_patterns = [r"挂.{0,8}号", r"约.{0,8}号", r"预约.{0,12}(门诊|医生|科室|号)"]
        if any(k in t for k in service_keywords) or any(re.search(p, t) for p in service_patterns):
            return "REGISTRATION"

        info_keywords = ["哪里", "地址", "位置", "几点", "时间", "电话", "contact", "address", "where"]
        if any(k in t for k in info_keywords):
            return "INFO"

        greeting_only = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "bye"]
        if len(t) <= 10 and any(k in t for k in greeting_only) and not self._has_medical_signal(t):
            return "GREETING"

        return None

    def _normalize_model_intent(self, raw_intent: str, user_input: str) -> str:
        intent = (raw_intent or "").strip().upper()
        mapping = {
            "CRISIS": "CRISIS",
            "GREETING": "GREETING",
            "VAGUE_SYMPTOM": "VAGUE_SYMPTOM",
            "COMPLEX_SYMPTOM": "MEDICAL_CONSULT",
            "VAGUE": "VAGUE_SYMPTOM",
            "STANDARD": "MEDICAL_CONSULT",
            "MEDICAL_CONSULT": "MEDICAL_CONSULT",
        }
        if intent in mapping:
            return mapping[intent]

        return self._rule_based_fallback(user_input)

    def _rule_based_fallback(self, text: str) -> str:
        first_hit = self._rule_front_intent(text)
        if first_hit:
            return first_hit

        t = (text or "").strip().lower()
        if not t:
            return "GREETING"

        if self._has_medical_signal(t):
            if len(t) <= 8:
                return "VAGUE_SYMPTOM"
            return "MEDICAL_CONSULT"

        if len(t) <= 8:
            return "GREETING"
        return "VAGUE_SYMPTOM"

    def _apply_session_correction(self, state: AgentState, intent: str, user_input: str) -> str:
        prev_intent = str(state.get("intent") or "").upper()
        if not prev_intent:
            return intent

        # 会话连续语境下，不把医疗追问片段误收敛到 GREETING。
        if intent == "GREETING" and prev_intent in {"VAGUE_SYMPTOM", "MEDICAL_CONSULT"}:
            if self._has_medical_signal(user_input) or self._is_followup_fragment(user_input):
                return "VAGUE_SYMPTOM"

        # 用户只发简短寒暄，避免被历史医疗上下文拖偏。
        if intent in {"VAGUE_SYMPTOM", "MEDICAL_CONSULT"} and len(user_input.strip()) <= 6:
            greeting_words = ["你好", "您好", "hi", "hello", "谢谢", "再见", "bye"]
            low = user_input.strip().lower()
            if any(g in low for g in greeting_words) and not self._has_medical_signal(low):
                return "GREETING"

        return intent

    def _apply_uncertainty_followup(self, intent: str, user_input: str, raw_output: str) -> Dict[str, Any]:
        t = (user_input or "").strip().lower()
        output = (raw_output or "").lower()

        clarification_needed = False
        clarification_question = ""

        ambiguous_short = len(t) <= 8 and self._has_medical_signal(t)
        uncertain_signal = ("不确定" in output) or ("uncertain" in output)

        if intent == "GREETING" and self._has_medical_signal(t):
            intent = "VAGUE_SYMPTOM"
            clarification_needed = True
        elif intent == "MEDICAL_CONSULT" and ambiguous_short:
            intent = "VAGUE_SYMPTOM"
            clarification_needed = True
        elif intent == "VAGUE_SYMPTOM" and uncertain_signal:
            clarification_needed = True

        if clarification_needed:
            clarification_question = "为了更准确分诊，请补充：症状持续多久、发生部位和是否伴随发热/呕吐/呼吸困难？"

        return {
            "intent": intent,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
        }

    async def run(self, state: AgentState) -> Dict[str, Any]:
        logger.info("node_start", node="intent_classifier")

        user_input = self._extract_user_input(state)
        if not isinstance(user_input, str):
            user_input = str(user_input)
        user_input = user_input.strip()

        intent = "MEDICAL_CONSULT"
        raw_output = ""
        source = "local_slm"

        try:
            if not self.slm:
                raise RuntimeError("Local SLM disabled")

            front = self._rule_front_intent(user_input)
            if front:
                intent = front
                raw_output = f"[RuleFront] {front}"
            else:
                predicted = await self.slm.constrained_classify(
                    user_input,
                    categories=self.CATEGORIES,
                    reasoning=False,
                )
                raw_output = getattr(self.slm, "_last_raw_output", "") or str(predicted)
                intent = self._normalize_model_intent(predicted, user_input)

            intent = self._apply_session_correction(state, intent, user_input)
            uncertain = self._apply_uncertainty_followup(intent, user_input, raw_output)
            intent = uncertain["intent"]

            if uncertain["clarification_needed"]:
                self.quality_collector.collect_negative_sample(
                    prompt=user_input,
                    expected_category="VAGUE_SYMPTOM",
                    actual_output=raw_output,
                    is_uncertain=True,
                )

            status = "crisis" if intent == "CRISIS" else "classified"
            payload = {
                "intent": intent,
                "raw_triage_output": raw_output,
                "current_turn_input": user_input,
                "retrieval_query": user_input,
                "status": status,
            }
            if uncertain["clarification_needed"]:
                payload["clarification_needed"] = True
                payload["clarification_question"] = uncertain["clarification_question"]

            logger.info(
                "intent_classified",
                intent=intent,
                source=source,
                clarification_needed=bool(payload.get("clarification_needed")),
            )
            return payload

        except Exception as e:
            logger.error("slm_classification_failed", error=str(e))
            source = "rule_fallback"
            intent = self._rule_based_fallback(user_input)
            status = "crisis" if intent == "CRISIS" else "classified"
            return {
                "intent": intent,
                "raw_triage_output": f"[Fallback:{source}] {intent}",
                "current_turn_input": user_input,
                "retrieval_query": user_input,
                "status": status,
            }


intent_classifier = IntentClassifier()


async def intent_classifier_node(state: AgentState):
    return await intent_classifier.run(state)

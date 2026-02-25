import time
import re
import json
import structlog
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.core.llm.llm_factory import get_fast_llm
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node

from langchain_core.runnables import RunnableConfig

logger = structlog.get_logger(__name__)

# 使用快速模型 (e.g. Qwen-Turbo) 进行问诊
_anamnesis_llm = get_fast_llm(temperature=0.2) 

@monitor_node("anamnesis")
async def anamnesis_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """
    节点：主动式、结构化智能问诊 (Smart Anamnesis V7.0)
    
    实现启发式提问策略，根据信息饱和度自动流转。
    """
    print("[DEBUG] Node anamnesis Start")
    start = time.time()
    
    symptoms = state.get("symptoms", "")
    history_msgs = state.get("messages", [])
    current_vector = state.get("symptom_vector", {}) or {}
    
    # 提取对话上下文
    recent_context = ""
    # 只取最近的消息，避免上下文过长
    for msg in history_msgs[-10:]:
        role = "Patient" if isinstance(msg, HumanMessage) else "Doctor"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        recent_context += f"{role}: {content}\n"
        
    system_prompt = """你是一名资深临床医生，正在进行“启发式、结构化”问诊。
你的任务是根据患者提供的信息，识别缺失的“核心医学维度”，并决定是否需要继续追问。

### 核心医学维度 (Symptom Vector):
1. **部位 (Location)**: 症状发生的具体部位。
2. **性质 (Quality)**: 疼痛或不适的具体感觉（如刺痛、胀痛、绞痛）。
3. **时长与频率 (Timing/Frequency)**: 持续时间、发作频率、每次持续多久。
4. **程度 (Severity)**: 对生活的影响程度，或疼痛评分。
5. **诱因与缓解因素 (Provocation/Palliation)**: 什么情况下加重或减轻。
6. **伴随症状 (Associated Symptoms)**: 是否有发热、呕吐、头晕等。
7. **既往史 (Medical History)**: 相关的基础疾病。

### 提问策略:
- **信息极少 (Sparse)**: 如果患者只说了一个词或短句，提出 3-4 个关键问题。
- **信息较详细 (Detailed)**: 如果患者描述了部分维度，提出 1-2 个深挖问题。
- **信息饱和度判断**:
  - 核心维度包括：1. 部位；2. 性质（如性质、程度）；3. 时长（如发病时间、持续时间）；4. 伴随症状；5. 诱因/缓解因素；6. 既往史。
  - 若上述 6 个维度中已有 5 个及以上明确信息（包括阴性描述），且没有极其危急的生命体征缺失，**必须**将 `is_complete` 设为 `true`。
  - 严禁为了追求“完美”而无限期延长问诊。在信息足以支持初步分诊和分科建议时，应果断结束。

### 质量守卫:
- 严禁提出与病情无关的寒暄问题。
- 直接切入主题，问题要口语化且专业。
- 确保问题之间逻辑连贯。

### 输出格式:
必须输出严格的 JSON 格式：
{
  "logic": "分析当前已获取的信息和缺失的维度",
  "questions": ["问题1", "问题2"],
  "symptom_vector": {
    "location": "已识别的部位或 null",
    "quality": "已识别的性质或 null",
    "timing": "已识别的时间或 null",
    "severity": "已识别的程度或 null",
    "provocation": "已识别的诱因或 null",
    "associated": "已识别的伴随症状或 null",
    "history": "已识别的既往史或 null"
  },
  "is_complete": boolean
}
"""

    user_prompt = f"""
[对话历史]
{recent_context}

[当前主诉]
{symptoms}

[当前已掌握症状向量]
{json.dumps(current_vector, ensure_ascii=False)}

请基于以上信息执行 SmartQuestionGenerator 逻辑：
"""

    try:
        response = await _anamnesis_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ], config=config)
        
        content = response.content.strip()
        # JSON 提取
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        
        if len(content) < 20:
            raise ValueError("AI 生成内容过短，可能存在逻辑短路")
            
    except Exception as e:
        logger.error("anamnesis_processing_failed", error=str(e))
        # 兜底处理
        data = {
            "logic": "解析失败或超时",
            "questions": ["能请您再详细描述一下症状吗？比如持续多久了，具体在哪个位置？"],
            "symptom_vector": current_vector,
            "is_complete": False
        }

    # 更新症状向量
    new_vector = data.get("symptom_vector", {})
    # 合并向量 (简单的非空覆盖)
    for k, v in new_vector.items():
        if v and v != "null" and v is not None:
            current_vector[k] = v

    print(f"[DEBUG] Symptom Vector: {json.dumps(current_vector, ensure_ascii=False)}")
    
    questions = data.get("questions", [])
    is_complete = data.get("is_complete", False)
    logic = data.get("logic", "")
    
    # 构造最终回复
    if is_complete:
        final_response = "感谢您的详细描述，我已收集到足够的临床信息，正在为您生成诊断方案..."
    else:
        final_response = " ".join(questions)

    print(f"--- [Anamnesis] Logic: {logic} ---")
    print(f"--- [Anamnesis] Questions: {final_response} ---")
    print(f"Node [anamnesis] took: {time.time() - start:.2f}s")
    
    # Telemetry
    from app.core.monitoring.token_tracker import format_token_usage
    usage_stats = format_token_usage(
        node_name="anamnesis",
        usage=response.response_metadata.get("token_usage", {}),
        model_name=response.response_metadata.get("model_name", "qwen-fast")
    )
    
    return {
        "final_response": final_response,
        "messages": [AIMessage(content=final_response)],
        "symptom_vector": current_vector,
        "dialogue_phase": "collecting_info" if not is_complete else "diagnosis_ready",
        "status": "in_progress" if not is_complete else "ready_for_diagnosis",
        "is_anamnesis_complete": is_complete,
        "token_usage_stats": [usage_stats]
    }

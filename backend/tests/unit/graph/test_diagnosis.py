import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import SystemMessage, HumanMessage

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.core.graph.sub_graphs.diagnosis import state_sync_node, dspy_reasoner_node, confidence_evaluator_node
from app.domain.states.sub_states import DiagnosisState
from app.core.services.config_manager import config_manager

# 模拟科室配置
MOCK_CONFIG = {
    "name": "mock_dept",
    "prompts": {
        "role_definition": "Mock Role",
        "diagnostic_guidelines": "Mock Guidelines"
    }
}

@pytest.mark.asyncio
async def test_diagnosis_dynamic_config_loading():
    """验证 Diagnosis Subgraph 动态加载科室配置"""
    
    # Mock ConfigManager
    with patch.object(config_manager, "get_config", return_value=MOCK_CONFIG), \
         patch.object(config_manager, "get_system_prompt", return_value="Mock System Prompt"):
        
        state = DiagnosisState(department="mock_dept", messages=[])
        
        result = await state_sync_node(state)
        
        assert result["department"] == "mock_dept"
        assert "Mock System Prompt" in result["system_prompt"]

@pytest.mark.asyncio
async def test_dspy_reasoner_mocked():
    """验证 DSPy 推理节点的输入输出处理 (Mock DSPy)"""
    
    # Mock DSPy Module
    with patch("app.core.graph.sub_graphs.diagnosis.medical_consultant") as mock_dspy:
        # 模拟 DSPy 预测结果
        mock_prediction = MagicMock()
        mock_prediction.reasoning = "Mock Reasoning"
        mock_prediction.suggested_diagnosis = ["Mock Disease"]
        mock_prediction.confidence_score = 0.9
        mock_prediction.follow_up_questions = []
        
        mock_dspy.return_value = mock_prediction
        
        # 构造 State
        state = DiagnosisState(
            department="mock_dept",
            system_prompt="System Instruction",
            user_profile="Profile",
            messages=[HumanMessage(content="Headache")]
        )
        
        result = await dspy_reasoner_node(state)
        
        # 验证 DSPy 调用参数中是否包含了 System Prompt (作为 Context)
        call_args = mock_dspy.call_args
        assert "System Instruction" in call_args.kwargs["retrieved_knowledge"]
        
        # 验证输出
        assert result["last_tool_result"]["confidence"] == 0.9
        assert "Mock Disease" in result["last_tool_result"]["diagnosis"]

@pytest.mark.asyncio
async def test_confidence_evaluator():
    """验证置信度评估逻辑（异步断言）"""

    # High confidence -> end_diagnosis
    state_high = DiagnosisState(last_tool_result={"confidence": 0.9, "diagnosis": "Flu"})
    assert await confidence_evaluator_node(state_high) == "end_diagnosis"

    # Low confidence -> clarify_question
    state_low = DiagnosisState(last_tool_result={"confidence": 0.2, "diagnosis": "Unknown"})
    assert await confidence_evaluator_node(state_low) == "clarify_question"

    # Emergency token in diagnosis -> end_diagnosis
    state_emergency = DiagnosisState(last_tool_result={"confidence": 0.1, "diagnosis": "紧急胸痛"})
    assert await confidence_evaluator_node(state_emergency) == "end_diagnosis"

    # Explicit decision action should take precedence
    state_action = DiagnosisState(decision_action="retrieve_more", last_tool_result={})
    assert await confidence_evaluator_node(state_action) == "clarify_question"

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.graph.sub_graphs.diagnosis import state_sync_node
from app.domain.states.sub_states import DiagnosisState
from app.core.services.config_manager import config_manager
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================================================
# Parameterized Test for Dynamic Department Configuration
# =============================================================================

@pytest.mark.parametrize("department_id, user_input, expected_tool_keyword, expected_kb_collection", [
    (
        "cardiology", 
        "我最近心悸，感觉心跳很快", 
        "ecg_analyzer", 
        "medical_cardiology"
    ),
    (
        "dermatology", 
        "我手臂上起了红色的疹子，很痒", 
        "image_analyzer", 
        "medical_dermatology"
    ),
    (
        "pediatrics",
        "孩子发烧39度",
        "dosage_calculator",
        "medical_pediatrics"
    ),
    (
        "neurology",
        "头痛欲裂，甚至有点恶心",
        "stroke_risk_calculator",
        "medical_neurology"
    )
])
@pytest.mark.asyncio
async def test_diagnosis_config_switching(department_id, user_input, expected_tool_keyword, expected_kb_collection):
    """
    验证 Diagnosis Subgraph 是否能根据科室 ID 动态切换配置：
    1. ConfigLoader 正确加载 YAML
    2. System Prompt 包含正确的角色设定
    3. (Mock) RAG 和 Tool 列表包含预期内容
    """
    
    # 1. 准备 State
    state = DiagnosisState(
        department=department_id,
        messages=[HumanMessage(content=user_input)],
        user_profile="Mock Profile"
    )
    
    # 2. 运行 state_sync_node (这是负责加载 Config 的节点)
    # 我们不需要 Mock config_manager 的内部实现，因为我们要测试真实的加载逻辑 (集成测试)
    # 但前提是 YAML 文件真实存在。我们已经创建了它们。
    
    result = await state_sync_node(state)
    
    # 3. 验证 Config 是否加载成功
    
    # A. 验证 System Prompt 注入
    # 获取 Config 中的角色定义
    config = config_manager.get_config(department_id)
    assert config is not None, f"Failed to load config for {department_id}"
    
    role_def = config["prompts"]["role_definition"]
    system_prompt_in_state = result["system_prompt"]
    
    # 验证 State 中的 Prompt 包含 YAML 中的 Role 定义
    # 注意：system_prompt 可能是 Role + Guidelines 的组合
    assert role_def.strip() in system_prompt_in_state
    
    # B. 验证 Knowledge Base 配置 (从 Config 对象检查)
    kb_config = config.get("knowledge_base", {})
    assert kb_config.get("collection_name") == expected_kb_collection
    
    # C. 验证 Tools 配置 (从 Config 对象检查)
    tools_config = config.get("tools", [])
    tool_names = [t["name"] for t in tools_config]
    assert expected_tool_keyword in tool_names

@pytest.mark.asyncio
async def test_diagnosis_fallback_to_general():
    """
    验证未知科室自动降级为 General 或 Cardiology (根据代码逻辑)
    """
    state = DiagnosisState(
        department="unknown_department_xyz",
        messages=[HumanMessage(content="不舒服")],
        user_profile="Mock Profile"
    )
    
    # 运行节点
    result = await state_sync_node(state)
    
    # 检查是否回退到默认
    # 在 diagnosis.py 中: if not config -> return default (目前逻辑是 get_config 返回 None)
    # 检查代码: if department == "general": ...
    # 代码中: dept_config = config_manager.get_config(department)
    # 如果找不到，get_config 返回 None。
    # 然后 system_prompt = config_manager.get_system_prompt(department)
    # 如果 config 为空，get_system_prompt 返回 "全科医生..."
    
    # 验证 State 中的 System Prompt 是默认的全科医生
    assert "全科医生" in result["system_prompt"]
    # 且 department 字段没有崩溃，保持原样或更新?
    # 代码中: return {"department": department} (原样返回)
    assert result["department"] == "unknown_department_xyz"

from langchain_openai import ChatOpenAI
from langchain_classic.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_classic.agents.agent import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from app.core.config import settings
from app.core.llm.llm_factory import SmartRotatingLLM
from app.core.tools import lookup_guideline, check_drug_interaction, submit_triage_report
from app.core.tools.ehr import query_ehr, query_lab_results
from app.core.prompts.doctor import DOCTOR_SYSTEM_PROMPT
from app.core.prompts.triage import TRIAGE_SYSTEM_PROMPT
import structlog
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = structlog.get_logger()

# DEBUG: 打印 Key 的前缀，确认是否加载成功
masked_key = settings.OPENAI_API_KEY[:8] + "****" if settings.OPENAI_API_KEY else "None"
logger.info("factory.config_check", api_key_prefix=masked_key, model=settings.OPENAI_MODEL_NAME)

# 将普通函数转换为 LangChain Tool
@tool
def tool_lookup_guideline(query: str) -> str:
    """Useful for looking up medical guidelines and knowledge."""
    return lookup_guideline(query)

@tool
async def tool_check_drug_interaction(drugs: list) -> str:
    """Useful for checking drug-drug interactions. Input should be a list of drug names."""
    # LangChain 有时传进来的是字符串，做个容错
    if isinstance(drugs, str):
        import ast
        try:
            drugs = ast.literal_eval(drugs)
        except:
            return "Error: Input must be a list of drug names."
    return await check_drug_interaction(drugs)

@tool
def tool_query_ehr(patient_id: str) -> str:
    """Useful for querying patient electronic health records (history, allergies)."""
    return str(query_ehr(patient_id))

@tool
def tool_query_lab(patient_id: str) -> str:
    """Useful for querying patient lab results (blood tests, etc)."""
    return str(query_lab_results(patient_id))

@tool
def tool_submit_report(department: str, urgency: str, reasoning: str, advice: list) -> str:
    """
    Submit the final triage report. MUST be called when diagnosis is complete.
    Args:
        department: Recommended department (e.g., 'Cardiology')
        urgency: Urgency level ('Normal', 'Urgent', 'Critical')
        reasoning: Brief medical reasoning
        advice: List of actionable advice
    """
    return submit_triage_report(department, urgency, reasoning, advice)

# 定义一个 Mock Agent Executor，用于无 Key 情况下的降级运行
class MockRAGAgentExecutor:
    """
    一个伪装成 AgentExecutor 的类。
    当没有 LLM Key 时，它直接把用户输入当作关键词去查 RAG，并流式返回结果。
    """
    def __init__(self, mode="triage"):
        self.mode = mode

    async def astream_events(self, inputs, version="v1"):
        """
        模拟 LangChain 的 astream_events 接口
        """
        user_input = inputs.get("input", "")
        
        # 1. 模拟思考过程
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": type('obj', (object,), {"content": "检测到无有效 API Key，进入 Mock 模式。\n"})},
            "name": "MockLLM"
        }
        
        # 简单规则：关键词触发工具
        tool_outputs = ""
        
        # Rule 1: 查病历 (Doctor 模式下)
        if self.mode == "doctor" and ("病历" in user_input or "过往" in user_input):
            yield {
                "event": "on_tool_start",
                "name": "query_ehr",
                "data": {"input": "P001"}
            }
            ehr_data = query_ehr("P001")
            tool_outputs += f"\nEHR数据: {ehr_data}"
            yield {
                "event": "on_tool_end",
                "name": "query_ehr",
                "data": {"output": str(ehr_data)}
            }
            
        # Rule 2: 查化验 (Doctor 模式下)
        if self.mode == "doctor" and ("化验" in user_input or "检查" in user_input or "血" in user_input):
            yield {
                "event": "on_tool_start",
                "name": "query_lab_results",
                "data": {"input": "P001"}
            }
            lab_data = query_lab_results("P001")
            tool_outputs += f"\nLab数据: {lab_data}"
            yield {
                "event": "on_tool_end",
                "name": "query_lab_results",
                "data": {"output": str(lab_data)}
            }
            
        # Rule 3: 默认查指南 (如果没有其他工具触发，或者是 Triage 模式)
        if not tool_outputs:
            yield {
                "event": "on_tool_start",
                "name": "lookup_guideline",
                "data": {"input": user_input}
            }
            try:
                # 简单去噪
                keyword = user_input.replace("我", "").replace("医生", "")
                result = lookup_guideline(keyword)
                tool_outputs += f"\n指南结果: {result}"
                yield {
                    "event": "on_tool_end",
                    "name": "lookup_guideline",
                    "data": {"output": result}
                }
            except Exception as e:
                result = f"Error: {e}"
                yield {
                    "event": "on_tool_end",
                    "name": "lookup_guideline",
                    "data": {"output": result}
                }

        # 5. 模拟最终回答
        final_tool_preview = tool_outputs[:200] + "..." if len(tool_outputs) > 200 else tool_outputs
        final_answer = f"\n[Mock 回答] 基于以下信息：\n{final_tool_preview}\n\n建议您结合临床表现综合判断。"
        
        # 分块流式输出，模拟打字机
        import asyncio
        chunk_size = 2
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i:i+chunk_size]
            await asyncio.sleep(0.05)
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": type('obj', (object,), {"content": chunk})},
                "name": "MockLLM"
            }

# =============================================================================
# 动态科室专家工厂 (Departmental Agent Factory)
# 遵循 AUTO_EVOLUTION_SKILL.md 规范
# =============================================================================

class DepartmentAgentFactory:
    """
    负责扫描知识库、注册科室、并动态创建特定科室的专家 Agent。
    支持从 JSON 注册表或文件系统目录加载科室配置。
    """
    
    def __init__(self):
        self.specialists: Dict[str, Any] = {}
        # 路径配置
        self.base_dir = Path(__file__).resolve().parent.parent.parent # backend/
        self.registry_path = self.base_dir / "app" / "core" / "registry" / "specialists.json"
        self.kb_path = self.base_dir / "data" / "knowledge_base"
        
        # 初始化扫描
        self._scan_knowledge_base()
        
    def _scan_knowledge_base(self):
        """
        第一性原理：数据驱动 Agent 结构。
        1. 优先读取 specialists.json (Golden Registry)
        2. 扫描 data/knowledge_base/ 目录下的子文件夹 (Knowledge Discovery)
        """
        logger.info("[DEBUG] Node DepartmentAgentFactory Scan Start")
        
        # 1. Load Registry
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.specialists = data.get("specialists", {})
                    logger.info("factory.registry_loaded", count=len(self.specialists))
            except Exception as e:
                logger.error("factory.registry_error", error=str(e))
                # 异常透传，但不阻断，可能还有其他源
        else:
            logger.warning("factory.registry_missing", path=str(self.registry_path))
            
        # 2. Scan Directory (Augment/Override)
        if self.kb_path.exists():
            for item in self.kb_path.iterdir():
                if item.is_dir():
                    dept_name = item.name
                    # 如果注册表中没有，则添加基础结构
                    if dept_name not in self.specialists:
                        self.specialists[dept_name] = {
                            "name_cn": dept_name,
                            "description": f"Automatically discovered department from knowledge base: {dept_name}",
                            "aliases": [dept_name]
                        }
                        logger.info("factory.dept_discovered", name=dept_name)
    
    def get_department_config(self, department_name: str) -> Optional[Dict[str, Any]]:
        """
        获取科室配置，支持模糊匹配或别名查找。
        """
        department_name = department_name.strip()
        
        # 1. Direct match
        if department_name in self.specialists:
            return self.specialists[department_name]
            
        # 2. Alias match
        for key, config in self.specialists.items():
            aliases = config.get("aliases", [])
            # 归一化比较
            if department_name.lower() in [a.lower() for a in aliases] or department_name == config.get("name_cn"):
                return config
                
        return None

    def create_agent(self, department_name: str) -> AgentExecutor:
        """
        为指定科室动态创建 Agent。
        Prompt 会根据科室描述自动生成 (Evolution ready).
        """
        logger.info("[DEBUG] Node CreateAgent Start", department=department_name)
        
        config = self.get_department_config(department_name)
        if not config:
            logger.warning("factory.dept_not_found", name=department_name, action="fallback_general")
            # 降级为通用医生
            return create_doctor_agent()
            
        # 提取上下文
        dept_cn = config.get("name_cn", department_name)
        dept_desc = config.get("description", "No description available.")
        
        # 动态构建 System Prompt
        # 这里的 Prompt 结构参考了 DOCTOR_SYSTEM_PROMPT 但注入了科室特异性
        dynamic_system_prompt = f"""
你是一名 {dept_cn} 的专家医生。
你的职责范围：{dept_desc}

## 核心原则
1. **安全第一**：如果是急重症，必须建议去急诊。
2. **循证医学**：回答必须基于指南和检索到的知识。
3. **清晰沟通**：使用患者能听懂的语言。
4. **语言强制**：无论用户说什么语言，你**必须**且**只能**使用**简体中文**回答。

{DOCTOR_SYSTEM_PROMPT}
"""
        # 移除原 DOCTOR_SYSTEM_PROMPT 中可能存在的冲突部分（如果需要更精细的控制，应该将 DOCTOR_SYSTEM_PROMPT 拆分）
        # 这里暂且采用追加模式，强化科室角色。

        # 1. 检查 Key (Mock 支持)
        if settings.OPENAI_API_KEY.startswith("sk-mock"):
            return MockRAGAgentExecutor(mode="doctor")

        # 2. 初始化 LLM
        try:
            llm = SmartRotatingLLM(
                model_name=settings.OPENAI_MODEL_NAME,
                temperature=settings.TEMPERATURE,
                prefer_local=False
            )
        except Exception as e:
            logger.error("agent.init_failed", error=str(e))
            raise e

        # 3. 工具集 (专家可能拥有特定科室的工具，这里暂时使用通用医疗工具集)
        tools = [
            tool_lookup_guideline, 
            tool_check_drug_interaction,
            tool_query_ehr,
            tool_query_lab
        ]

        # 4. 构建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", dynamic_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 5. 创建 Agent
        agent = create_tool_calling_agent(llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True
        )

# 单例模式，避免重复加载
_factory_instance = None

def get_department_factory() -> DepartmentAgentFactory:
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = DepartmentAgentFactory()
    return _factory_instance


def create_triage_agent():
    """
    创建分诊智能体 Executor。
    """
    # 1. 检查 Key
    if settings.OPENAI_API_KEY.startswith("sk-mock"):
        logger.warning("agent.init", msg="Using Mock Agent because no valid API Key found.")
        return MockRAGAgentExecutor(mode="triage")

    # 2. 正常初始化 LLM（优化配置）
    try:
        # [V6.5.1] Upgrade to SmartRotatingLLM to prevent 400 Errors and allow fallback
        llm = SmartRotatingLLM(
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            prefer_local=False
        )
    except Exception as e:
        logger.error("agent.init_failed", error=str(e))
        raise e

    # 3. 准备工具集
    tools = [tool_lookup_guideline, tool_submit_report]

    # 4. 构建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", TRIAGE_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 5. 创建 Agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 6. 创建 Executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return executor

def create_doctor_agent():
    """
    创建医生智能体 Executor。
    保留此函数以兼容旧代码，默认行为可视为通用全科医生。
    """
    # 1. 检查 Key
    if settings.OPENAI_API_KEY.startswith("sk-mock"):
        return MockRAGAgentExecutor(mode="doctor")
        
    # ... (正常逻辑，优化配置）
    try:
        # [V6.5.1] Upgrade to SmartRotatingLLM to prevent 400 Errors and allow fallback
        llm = SmartRotatingLLM(
            model_name=settings.OPENAI_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            prefer_local=False
        )
    except Exception as e:
        logger.error("agent.init_failed", error=str(e))
        raise e

    # ... (工具集)
    tools = [
        tool_lookup_guideline, 
        tool_check_drug_interaction,
        tool_query_ehr,
        tool_query_lab
    ]

    # 3. 构建 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCTOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return executor

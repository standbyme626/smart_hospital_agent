import os
import time
import asyncio
import structlog
from typing import Dict, Any, List
from app.core.graph.state import AgentState
from app.core.monitoring.tracing import monitor_node
from app.core.config import settings
from app.core.models.vram_manager import vram_manager, vram_auto_clear
from app.core.llm.llm_factory import SmartRotatingLLM, get_fast_llm
from app.agents.factory import get_department_factory # [NEW] Import DepartmentAgentFactory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

logger = structlog.get_logger(__name__)

def _build_chat_history(state: AgentState) -> List[Any]:
    """
    构建 AgentExecutor 所需的 chat_history。
    允许缺失，至少返回空列表，避免 MessagesPlaceholder 缺参报错。
    """
    history = state.get("messages", [])
    if isinstance(history, list):
        return history
    return []

async def run_dynamic_specialist_node(department: str, state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """运行动态专家节点 (Using DepartmentAgentFactory)"""
    try:
        # [Refactor] Use DepartmentAgentFactory instead of MedicalExpertCrew
        factory = get_department_factory()
        
        # 1. Create Agent dynamically
        agent_executor = factory.create_agent(department)
        
        # 2. Prepare Input
        symptoms = state.get("symptoms", "")
        history = state.get("medical_history", "")
        audit_retry_count = state.get("audit_retry_count", 0)
        audit_feedback = state.get("audit_feedback")
        
        input_text = f"【患者主诉】：{symptoms}\n【历史病历/上下文】：{history}"
        
        if audit_retry_count > 0 and audit_feedback:
             input_text += f"\n\n【⚠️ 审计驳回修正指令】\n上一次的诊断未通过合规审计，原因如下：\n{audit_feedback}\n\n请务必针对上述问题进行修正。"

        # 3. Invoke Agent
        # agent_executor.invoke returns a dict with "output" key
        result = await agent_executor.ainvoke({
            "input": input_text,
            "chat_history": _build_chat_history(state),
        })
        result_str = result.get("output", str(result))
        
        # [Legacy Support] Extract Persona/Evidence if needed (Currently simplified)
        # Assuming the new factory agent returns clean text. 
        # If we need structured output (Persona/Evidence), we should update the prompt in factory.py later.
        
        persona_updates = []
        evidence_updates = []

        return {
            f"specialist_{department}_output": result_str,
            "persona_update_proposals": persona_updates,
            "evidence_chain": evidence_updates
        }
    except Exception as e:
        logger.error(f"specialist_failed_{department}", error=str(e))
        return {f"specialist_{department}_output": f"Error: Specialist {department} failed to respond. ({str(e)})"}

async def run_pharmacist_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """运行药剂师节点"""
    # 暂时保持原有逻辑，或者也迁移到 Factory (如果有药剂科)
    # 既然 Factory 里有 "药剂科_精准医学", 我们可以尝试使用 Factory
    try:
        factory = get_department_factory()
        # 尝试查找药剂科，如果没有则 fallback 到默认
        dept = "药剂科_精准医学"
        if not factory.get_department_config(dept):
            # Fallback logic if needed, but registry has it.
            pass
            
        agent_executor = factory.create_agent(dept)
        
        symptoms = state.get("symptoms") or state.get("user_input") or ""
        history = state.get("medical_history") or ""
        diag_output = state.get("diagnostician_output", "") # Ideally we should pass the diagnosis to pharmacist
        
        input_text = f"请审核以下诊断和用药建议的安全性：\n\n【患者情况】{symptoms}\n{history}\n\n【初步诊断】\n{diag_output}"
        
        result = await agent_executor.ainvoke({
            "input": input_text,
            "chat_history": _build_chat_history(state),
        })
        return {"pharmacist_output": result.get("output", str(result))}

    except Exception as e:
        logger.error("pharmacist_failed", error=str(e))
        # Fallback to old implementation if factory fails or just return error
        return {"pharmacist_output": f"Error: Pharmacist review failed. ({str(e)})"}


@monitor_node("parallel_expert_crew")
# @vram_auto_clear(force=True) # [Optimization] Disabled for Cloud-Only Mode
async def parallel_expert_crew_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """并行专家组节点"""
    print(f"[DEBUG] Node ParallelExpertCrew Start")
    logger.info("node_start", node="parallel_expert_crew")
    start = time.time()
    
    try:
        # [Optimization] Pre-inference VRAM Check
        # Ensure we have enough memory for parallel experts (approx 1500MB headroom)
        # await vram_manager.orchestrate_pre_inference_async(required_mb=1500)
        
        departments = state.get("departments", [])
        if not departments:
             departments = ["General Practice"]
             
        logger.info("expert_dynamic_dispatch", departments=departments)
        
        # [Debug] Trace Context
        symptoms = state.get("symptoms") or "MISSING"
        history = state.get("medical_history") or "MISSING"
        user_input = state.get("user_input") or "MISSING"
        logger.info(f"🔍 [ExpertCrew-Debug] Context Check: symptoms='{symptoms}' history='{history}' user_input='{user_input}'")
        
        # [Optim] Enable parallelism for Cloud-Only Mode
        # We use asyncio.gather to run specialists concurrently since we are using Cloud LLM.
        tasks = []
        for dept in departments:
            tasks.append(run_dynamic_specialist_node(dept, state, config))
            
        specialist_results = await asyncio.gather(*tasks)

        # Aggregate first to pass to pharmacist
        diag_output_list = []
        all_evidence = []
        all_persona_updates = []

        for i, dept in enumerate(departments):
            res = specialist_results[i]
            # Extract text output
            output_key = f"specialist_{dept}_output"
            text_out = res.get(output_key, str(res))
            diag_output_list.append(f"### {dept} Specialist:\n{text_out}")
            
            # Aggregate Evidence & Persona Updates
            if "evidence_chain" in res:
                all_evidence.extend(res["evidence_chain"])
            if "persona_update_proposals" in res:
                all_persona_updates.extend(res["persona_update_proposals"])
            
        diag_output = "\n\n".join(diag_output_list)
        
        # Inject diagnosis into state for pharmacist (temp)
        state["diagnostician_output"] = diag_output
        
        pharm_result = await run_pharmacist_node(state, config)
        
        logger.info(f"parallel_expert_crew_finished", tasks_count=len(specialist_results))
        
        return {
            "diagnostician_output": diag_output,
            "evidence_chain": all_evidence,
            "persona_update_proposals": all_persona_updates,
            "content": diag_output, # [Compatibility] For ChatService
            **pharm_result
        }

    except Exception as e:
        error_msg = str(e)
        if "No available LLM" in error_msg or "Local fallback disabled" in error_msg or "403" in error_msg:
            logger.warning("expert_crew_fallback_triggered", error=error_msg)
            local_llm = get_fast_llm(temperature=0.3, prefer_local=True)
            
            symptoms = state.get("symptoms", "")
            history = state.get("medical_history", "暂无历史病历")
            
            fallback_prompt = f"""【诚实降级模式】
当前云端专家诊断服务暂时不可用，系统已切换至本地基础模型为您提供闭环回复。
我们将结合您的主诉和系统检索到的医疗知识库内容为您提供基础参考。

【患者主诉】：{symptoms}
【检索到的上下文/病历】：{history}

请根据以上信息，给出一些基础的健康建议（非专业诊断）。
要求：
1. 明确告知用户这是本地模型的初步建议，非专家会诊结论。
2. 建议用户在条件允许时重新咨询或前往医院。
3. 保持专业、严谨且温馨。
"""
            try:
                response = await local_llm.ainvoke(fallback_prompt)
                return {
                    "diagnostician_output": f"[本地模型闭环回复]\n{response.content}",
                    "pharmacist_output": "专家组服务降级，暂无详细用药审查。",
                    "auditor_output": "专家组服务降级，暂无详细合规审计。",
                    "status": "downgraded",
                    "is_downgraded": True
                }
            except Exception as local_err:
                logger.error("local_fallback_failed", error=str(local_err))
                raise e
        raise e

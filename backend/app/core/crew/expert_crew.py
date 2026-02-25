from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from app.core.crew.tools import SearchMedicalDB, DrugInteractionCheck
from app.core.llm.llm_factory import SmartRotatingLLM
from app.core.config import settings
import os
import logging

# 配置模块级日志
logger = logging.getLogger(__name__)

@CrewBase
class MedicalExpertCrew:
    """
    医疗专家组 (Medical Expert Crew) - 强制云端运行
    """
    
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, llm_override=None):
        try:
            if llm_override:
                self.llm = llm_override
            else:
                # [Pain Point #18] Cloud Dependency Fragility
                # Allow fallback to local model if cloud is unreachable
                # [Config] Controlled by env var, default to False for now
                allow_fallback = os.getenv("ENABLE_LOCAL_FALLBACK", "false").lower() == "true"
                logger.info(f"🚀 [MedicalExpertCrew] Initializing LLM (Allow Local Fallback: {allow_fallback})...")
                self.llm = SmartRotatingLLM(
                    prefer_local=False, 
                    allow_local=allow_fallback,
                    temperature=0.2 
                )
                logger.info(f"🚀 [MedicalExpertCrew] Initialized SmartRotatingLLM")
        except Exception as e:
            logger.error(f"❌ MedicalExpertCrew Initialization Failed: {str(e)}")
            raise

    def _get_expert_llm(self):
        return self.llm

    @agent
    def diagnostician(self) -> Agent:
        # [V6.5.7] 极度显式传递 LLM 实例，确保 CrewAI 不会自行创建默认 LLM
        llm_inst = self._get_expert_llm()
        import logging
        logging.info(f"👨‍⚕️ [MedicalExpertCrew] Creating Diagnostician with LLM type: {type(llm_inst)}")
        agent = Agent(
            config=self.agents_config['diagnostician'],
            tools=[SearchMedicalDB()],
            llm=llm_inst,
            function_calling_llm=llm_inst,
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )
        logging.info(f"👨‍⚕️ [MedicalExpertCrew] Diagnostician Agent Created. Assigned LLM: {type(agent.llm)}")
        return agent

    @agent
    def pharmacist(self) -> Agent:
        llm_inst = self._get_expert_llm()
        logging.info(f"💊 [MedicalExpertCrew] Creating Pharmacist with LLM type: {type(llm_inst)}")
        agent = Agent(
            config=self.agents_config['pharmacist'],
            tools=[SearchMedicalDB(), DrugInteractionCheck()],
            llm=llm_inst,
            function_calling_llm=llm_inst,
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )
        logging.info(f"💊 [MedicalExpertCrew] Pharmacist Agent Created. Assigned LLM: {type(agent.llm)}")
        return agent

    @agent
    def auditor(self) -> Agent:
        llm_inst = self._get_expert_llm()
        logging.info(f"📋 [MedicalExpertCrew] Creating Auditor with LLM type: {type(llm_inst)}")
        agent = Agent(
            config=self.agents_config['auditor'],
            llm=llm_inst,
            function_calling_llm=llm_inst,
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )
        logging.info(f"📋 [MedicalExpertCrew] Auditor Agent Created. Assigned LLM: {type(agent.llm)}")
        return agent

    # [Phase 7.3] Dynamic Specialist Creation
    def create_specialist_agent(self, department: str, description: str, name_cn: str = None) -> Agent:
        llm_inst = self._get_expert_llm()
        role_name = f"Senior {department} Specialist"
        if name_cn:
            # 双语角色名，满足用户要求：中文 (英文)
            role_name = f"资深{name_cn}专家 (Senior {department} Specialist)"
            
        return Agent(
            role=role_name,
            goal=f"为 {name_cn or department} 相关症状提供专业诊断建议。",
            backstory=f"你是一名{role_name}。{description} 你的目标是识别症状的根本原因并在你的专业领域内提供治疗建议。你必须始终使用简体中文回答。",
            tools=[SearchMedicalDB()],
            llm=llm_inst,
            verbose=True,
            max_iter=1,
            allow_delegation=False
        )

    def create_specialist_task(self, agent: Agent, symptoms: str, history: str, audit_feedback: str = None, gold_standard_context: str = None) -> Task:
        description = f"从 {agent.role} 的角度分析以下症状：\n症状: {symptoms}\n病史: {history}"
        
        if gold_standard_context:
             description += f"\n\n【💡 参考案例 (Gold Standard)】\n以下是经过专家审核的类似病例参考：\n{gold_standard_context}\n\n请参考上述案例的诊断逻辑和用药规范。"

        if audit_feedback:
             description += f"\n\n【⚠️ 重要：审计驳回反馈】\n上一次的诊断未通过合规审计，原因如下：\n{audit_feedback}\n\n请务必修正上述问题，重新生成诊断建议。"

        description += """\n\n请提供详细的诊断和治疗方案，包括：
1. 可能的疾病
2. 推荐的检查
3. 初步治疗建议

【🔍 证据溯源 (Evidence Traceability)】
请将所有关键结论的依据整理为一个 JSON 列表，放在 [EVIDENCE] 标签中。
格式如下：
[EVIDENCE]
[{"claim": "建议心电图", "source": "user_input", "quote": "胸闷气短"}, {"claim": "高血压风险", "source": "history", "quote": "既往高血压史"}]
[/EVIDENCE]

【⚠️ 画像校验指令 (Persona Validation)】
请仔细比对用户症状与已知病史（History）。如果你发现病史中缺少了关键信息（例如用户提到正在服用某种药物，但病史中未记录），或者病史与当前描述存在明显矛盾，请在输出的末尾添加一个【画像更新建议】板块。
格式如下：
[PERSONA_UPDATE]
{"add_medication": "药物名称", "add_disease": "疾病名称", "remove_medication": "药物名称"}
[/PERSONA_UPDATE]

注意：必须使用简体中文输出所有内容。"""

        return Task(
            description=description,
            expected_output="一份专业的中文医疗诊断报告，并在必要时包含画像更新建议。",
            agent=agent,
            async_execution=False
        )

    # ... Tasks 部分保持不变 ...
    @task
    def diagnosis_task(self) -> Task:
        return Task(config=self.tasks_config['diagnosis_task'], async_execution=True)

    @task
    def pharmacy_review_task(self) -> Task:
        return Task(config=self.tasks_config['pharmacy_review_task'], async_execution=True)

    @task
    def audit_task(self) -> Task:
        return Task(config=self.tasks_config['audit_task'], context=[self.diagnosis_task(), self.pharmacy_review_task()])

    @crew
    def crew(self, callbacks: list = None) -> Crew:
        # [优化] 动态获取 Embedding 路径，增加容错性
        model_path = settings.EMBEDDING_MODEL_PATH

        # [V5.4] 启用并行执行流程 (Process.hierarchical 或 Task(async_execution=True))
        # 核心逻辑：诊断任务和用药建议任务相互独立，可以并行执行，最后由审计任务汇总
        crew_instance = Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential, # 顶层流程保持顺序，但内部 Task 已标记为 async_execution=True
            verbose=True,
            callbacks=callbacks,
            memory=False,
            embedder_config={
                "provider": "huggingface",
                "model": model_path
            }
        )
        return crew_instance

    def parallel_crew(self, callbacks: list = None) -> Crew:
        """
        [V5.4] 极速并行专家组 (Parallel Expert Crew)
        利用 Task(async_execution=True) 实现诊断与药剂建议的并发执行
        """
        # 重新实例化以确保 async_execution 生效
        diag_task = self.diagnosis_task()
        pharm_task = self.pharmacy_review_task()
        
        # [Analysis] 并行模式还原 (Parallel Mode Restored)
        # 用户指出的并行逻辑是可行的，前提是改变药剂师的职责：
        # 从 "事后审查处方" (需串行) -> "事前建立风控清单" (可并行)
        # 最终由 Auditor 节点进行 "处方 vs 风控清单" 的碰撞检查
        
        # 审计任务依赖前两者，它会在前两者完成后才执行
        audit_task = self.audit_task()
        audit_task.context = [diag_task, pharm_task]

        model_path = settings.EMBEDDING_MODEL_PATH

        return Crew(
            agents=[self.diagnostician(), self.pharmacist(), self.auditor()],
            tasks=[diag_task, pharm_task, audit_task],
            process=Process.sequential,
            verbose=True,
            callbacks=callbacks,
            memory=False,
            embedder_config={
                "provider": "huggingface",
                "model": model_path
            }
        )

    def get_agent_executor(self, role: str) -> Agent:
        if role == 'diagnostician': return self.diagnostician()
        if role == 'pharmacist': return self.pharmacist()
        if role == 'auditor': return self.auditor()
        return None
        
    def get_task_instance(self, task_name: str) -> Task:
        if task_name == 'diagnosis_task': return self.diagnosis_task()
        if task_name == 'pharmacy_review_task': return self.pharmacy_review_task()
        if task_name == 'audit_task': return self.audit_task()
        return None

    def simple_crew(self, callbacks: list = None) -> Crew:
        model_path = settings.EMBEDDING_MODEL_PATH
        
        return Crew(
            agents=[self.diagnostician()],
            tasks=[self.diagnosis_task()],
            process=Process.sequential,
            verbose=True,
            callbacks=callbacks,
            memory=False,
            embedder_config={
                "provider": "huggingface",
                "model": model_path
            }
        )
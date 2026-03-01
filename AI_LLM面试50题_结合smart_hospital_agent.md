# AI/LLM 应用工程师面试 50 题（结合 `smart_hospital_agent`）

> 适用岗位：AI 工程师、LLM 应用工程师、Agent/RAG 工程师、医疗 AI 平台工程师  
> 生成方式：基于当前仓库代码遍历 + 公开面试题型检索整理 + 项目定制回答

## 一、你这个项目的一句话定位（建议先背）
`smart_hospital_agent` 是一个**医疗场景 Workflow-first + Local Loops 的混合型 Agent 系统**：主链路由 LangGraph 强约束编排保障安全和可治理，局部子图保留工具循环能力以提升任务完成度。

---

## 二、50 道高频面试题与参考回答（项目定制版）

## A. 架构与系统设计（1-10）

### 1) 你这个项目到底是聊天机器人，还是 Agent 系统？
**参考回答：**
它是混合型 Agent 系统，不是纯 Chatbot。主入口是 `workflow.py::create_agent_graph`，按 Ingress/Diagnosis/Service/Egress 分层编排；同时在 `service` 和 `doctor_graph` 内有工具循环，所以具备 Agent 自主执行能力。
**代码证据：** `backend/app/core/graph/workflow.py`, `backend/app/core/graph/sub_graphs/service.py`, `backend/app/agents/doctor_graph.py`

### 2) 为什么主流程不是完全自由探索，而是 workflow-first？
**参考回答：**
医疗是高风险场景，必须优先保证可控、可审计、可回放。主图固定收敛到 `persistence -> END`，便于做风控、审计、复盘；把开放式循环限制在局部子图更安全。
**代码证据：** `workflow.py` 中 `workflow.add_edge("persistence", END)`

### 3) 主流程分层有什么收益？
**参考回答：**
分层能把安全、推理、服务、输出治理解耦：
- Ingress 处理 PII、意图、危机
- Diagnosis 处理重写/检索/推理/裁决
- Service 处理挂号工具闭环
- Egress 统一质量门禁
这样变更局部不会破坏全链路。
**代码证据：** `backend/app/core/graph/sub_graphs/*.py`

### 4) 你如何描述本项目的关键设计 trade-off？
**参考回答：**
核心 trade-off 是“灵活性 vs 安全性”：
- 不追求全图自由循环，换取稳定性
- 在服务和医生子图保留有限循环，换取任务完成度
- 用 `Decision_Judge` 和 `quality_gate` 把自由度收敛到可治理输出
**代码证据：** `diagnosis.py`, `quality_gate.py`, `service.py`

### 5) 这个系统是单体还是微服务？
**参考回答：**
当前是模块化单体（FastAPI + LangGraph），但接口边界已经为服务化预留：API 层、图编排层、RAG 层、监控层是明确分层，未来可按 `RAG/服务子图/鉴权` 拆服务。
**代码证据：** `backend/app/api/v1`, `backend/app/core/graph`, `backend/app/rag`

### 6) 你如何保证需求扩展时不“图爆炸”？
**参考回答：**
用子图承载领域复杂度，不在主图堆 if-else。主图只做路由与生命周期管理，复杂逻辑下沉到 `diagnosis/service/medical_core`。新能力优先以节点或子图扩展。
**代码证据：** `workflow.py` 的 `workflow.add_node("diagnosis", diagnosis_graph)` 等

### 7) 这个项目能否支持多入口（Web、Telegram、OpenWebUI）？
**参考回答：**
Web/SSE 已经成熟；统一输出协议由 `stream_schema.py` 控制。新增入口时只需要做网关适配到 `/api/v1/chat/stream`。Telegram 目前仓库未见直接实现。
**代码证据：** `backend/app/api/v1/endpoints/chat.py`, `backend/app/core/stream_schema.py`

### 8) 为什么选择 LangGraph 而不是直接 AgentExecutor？
**参考回答：**
LangGraph 在可视化状态机、分支控制、中断点（HITL）、持久化和事件流方面更适合高风险业务。尤其 `interrupt_before=["human_review"]` 这种人工审查点对医疗很关键。
**代码证据：** `workflow.py` compile 段落

### 9) “Research Ops Control Plane” 在项目里如何落地？
**参考回答：**
它不是口号，落地为 5 层：策略配置（config）、执行编排（graph）、治理（guard/judge/gate）、数据（Postgres/Redis/Milvus/Neo4j）、观测（metrics/langfuse/logs）。
**代码证据：** `core/config.py`, `core/graph/*`, `core/monitoring/*`, `rag/*`

### 10) 如果你向 CTO 汇报架构优势，三句话怎么说？
**参考回答：**
1) 主流程强约束，医疗风险可控；
2) 局部循环保留 Agent 能力，完成率高；
3) 全链路可观测+可审计，支持持续演进和线上治理。

## B. Agent 编排与循环控制（11-20）

### 11) 这个系统的“循环”具体在哪里？
**参考回答：**
主要两处：
- `service_agent -> tools -> service_agent`
- `doctor_graph` 的 `diagnosis_node -> tools -> state_updater -> diagnosis_node`
主 `chat` 图本身是收敛型。
**代码证据：** `service.py`, `doctor_graph.py`, `workflow.py`

### 12) 如何避免 Agent 死循环？
**参考回答：**
通过路由终止条件 + 重试上限：
- service 子图：无 tool_calls 即 END
- doctor_graph：审计高风险最多重试 2 次
- 模拟患者端也有 dead-loop breaker
**代码证据：** `service.py::should_continue`, `doctor_graph.py::audit_router`, `qa_engine/patient_agent.py`

### 13) diagnosis 子图为什么不在单轮内自循环追问？
**参考回答：**
当前策略是“单轮收敛 + 多轮会话循环”：当不确定时输出 `Clarify_Question -> END`，由下一轮用户输入继续。这种方式更利于审计和前端体验控制。
**代码证据：** `diagnosis.py` 尾部两条 END 边

### 14) 你如何做人类介入（HITL）？
**参考回答：**
通过图编译参数设置中断点，在进入 `human_review` 前暂停执行，等待人工决策恢复。适用于高风险医疗结论或敏感处方场景。
**代码证据：** `workflow.py` 的 `interrupt_before=["human_review"]`

### 15) 如果面试官问“你这不是 hardcode 路由吗”？
**参考回答：**
路由是“规则 + 模型裁决”的混合：主图有显式安全边界，子图内部通过模型推理和工具调用动态推进。医疗系统不能只靠自由 Agent，需要可控边界。

### 16) 为什么保留 `doctor_graph` 这种独立工作流？
**参考回答：**
它适合做更强流程实验（诊断-开方-审计多阶段），并且可独立作为接口 `/api/v1/doctor/workflow` 使用，不影响主聊天链路稳定性。
**代码证据：** `api/v1/endpoints/doctor.py`

### 17) 多图共存如何避免状态污染？
**参考回答：**
每条链路通过 `thread_id/session_id` 隔离执行上下文，状态结构也按 `AgentState/DiagnosisState/DoctorState` 分离，减少跨图字段污染。
**代码证据：** `chat.py` config、`domain/states/*.py`

### 18) 你如何定义“完成一次诊断”？
**参考回答：**
技术上是 `Decision_Judge` + `confidence_evaluator` 得到 `end_diagnosis`，落到 `Diagnosis_Report` 并结束；业务上还要通过 `quality_gate` 与审计链路。
**代码证据：** `diagnosis.py`, `egress.py`, `quality_gate.py`

### 19) 这个项目支持 function calling/tool calling 吗？
**参考回答：**
支持，并且是核心能力。service 子图依赖工具调用驱动挂号闭环，doctor_graph 也通过 ToolNode 执行多个医疗工具。
**代码证据：** `service.py`, `doctor_graph.py`, `core/tools/*`

### 20) 如果要把 loop 进一步产品化，你会怎么做？
**参考回答：**
给循环增加统一预算：`max_steps`、`max_tool_calls`、`max_latency_ms`，并把 budget 超限作为显式可观测事件写入 trace，前端显示“已转人工/已降级”。

## C. RAG 与检索质量（21-30）

### 21) 你的 RAG 是什么结构？
**参考回答：**
是混合检索：Milvus 向量检索 + BM25 词法检索 + RRF 融合 + 可选 rerank + SQL 预过滤 + 部门一致性约束，最终进入推理节点。
**代码证据：** `rag/retriever.py`, `diagnosis.py`

### 22) 为什么要混合检索，不只用向量？
**参考回答：**
医疗问诊存在术语、缩写、部位词、否定词等精确匹配需求。仅向量会损失关键词精度，混合检索能提高召回稳定性并降低误召回风险。

### 23) 如何处理 query rewrite 超时？
**参考回答：**
设置了 rewrite timeout，并在超时/异常时走 fallback 规则路径，同时打点 `rewrite_fallback_reason`，保证链路不断。
**代码证据：** `diagnosis.py` 中 rewrite timeout/fallback 逻辑

### 24) 检索超时时会怎样？
**参考回答：**
`hybrid_retriever` 超时会降级为 fast triage 结果或 fallback context，仍能输出可解释响应，避免直接失败。
**代码证据：** `diagnosis.py` 中 `_run_tool_with_timeout` 和 degraded 分支

### 25) 语义缓存如何防止“错缓存误命中”？
**参考回答：**
有 cache verify：term overlap、rerank score、reject memory 等校验，不通过则拒绝语义命中并回落检索，避免把错误答案放大。
**代码证据：** `rag/retriever.py` cache verify 相关字段

### 26) 你怎么控制缓存污染？
**参考回答：**
写缓存前有 write gate（最小分数、分数差、科室约束等），并且分 namespace + profile hash，防止不同检索配置共享脏结果。
**代码证据：** `rag/retriever.py` cache write gate / profile hash

### 27) rerank 什么时候会被跳过？
**参考回答：**
在 pure mode 或 adaptive 条件满足时可跳过 rerank 以降低延迟；也可按请求参数控制 `use_rerank/rerank_threshold`。
**代码证据：** `rag/retriever.py` adaptive_rerank / threshold 逻辑

### 28) 你如何回答“RAG 结果可信度如何定义”？
**参考回答：**
项目内把可信度拆成证据数、得分、grounded_flag、decision_action。不是单一概率，而是多信号治理后的动作输出。
**代码证据：** `diagnosis.py` 的 `decision_action/confidence_score/grounded_flag`

### 29) 纯检索模式有什么场景？
**参考回答：**
在需要低延迟、稳定口径或模型资源紧张时可开启 pure retrieval mode，禁用部分改写和总结链路，直接按检索证据输出结构化建议。
**代码证据：** `core/config.py` 的 `RAG_PURE_RETRIEVAL_MODE` 系列开关

### 30) 你如何防止“低分结果硬输出”？
**参考回答：**
通过 rerank threshold、fallback retry、decision governance、quality gate 形成多层拦截；低分时优先追问或降级，不直接给高置信结论。
**代码证据：** `rag/retriever.py`, `diagnosis.py`, `quality_gate.py`

## D. 模型路由、降级与稳定性（31-38）

### 31) SmartRotatingLLM 做了什么？
**参考回答：**
做 API key 轮转、模型候选轮转、认证错误识别、节点池 fallback、本地 fallback，目标是把 LLM 可用性从“单点依赖”变成“多层容错”。
**代码证据：** `core/llm/llm_factory.py`, `core/config.py`

### 32) key rotation 触发策略是什么？
**参考回答：**
基于 `API_KEY_CANDIDATES` 轮询，遇到认证错误可切换 key；并可按配置控制是否仅 auth error 才拉黑。减少单 key 波动对业务影响。
**代码证据：** `config.py` key iterator；`llm_factory.py` auth error 判定

### 33) 云模型挂了会怎样？
**参考回答：**
先尝试模型/节点切换，再视配置走本地 fallback；如果本地也不可用，会返回显式错误并记录 trace，不会静默失败。
**代码证据：** `llm_factory.py` sync/async fallback 分支

### 34) 你如何平衡成本与效果？
**参考回答：**
通过 fast/smart 模型分层、pure mode、adaptive rerank、缓存命中、rewrite cache 等方式减少高成本推理调用，只在必要节点走重模型。
**代码证据：** `config.py`, `diagnosis.py`, `retriever.py`

### 35) 模型输出异常（格式错、400）怎么处理？
**参考回答：**
有 `_retry_with_fixed_content` 修复重试路径，并把错误分型后再决定轮换/降级，避免一次格式问题直接熔断全链路。
**代码证据：** `llm_factory.py::_retry_with_fixed_content`

### 36) 这个项目如何做超时预算管理？
**参考回答：**
关键节点都有 timeout：query rewrite、tool 调用、cache read/write、semantic cache lookup；超时后走 degrade/fallback 并打日志。
**代码证据：** `diagnosis.py`, `retriever.py`, `cache.py`

### 37) 你怎么证明系统“不是靠运气稳定”？
**参考回答：**
它有明确抗故障机制：缓存命中兜底、检索降级、模型轮换、本地回退、质量门禁、审计重试，这些都体现在代码分支而非运维口头策略。

### 38) 你会如何回答“为何不直接上一个大模型端到端”？
**参考回答：**
医疗场景不能只追求一次性最优回答，要追求可解释、可治理、可审计和 SLA。分层架构允许我们对每个风险点设防，不把风险集中到单次模型调用。

## E. 安全合规、医疗治理与审计（39-44）

### 39) 医疗场景里你们的安全护栏有哪些？
**参考回答：**
Ingress 有 PII/意图/危机识别，Core 有 safety audit，Diagnosis 有 decision governance，Egress 有 quality gate，Auth 有 RBAC/SSO 审计。
**代码证据：** `sub_graphs/ingress.py`, `nodes/audit.py`, `diagnosis.py`, `quality_gate.py`, `security/rbac.py`

### 40) 如何处理高风险内容？
**参考回答：**
高风险可在审计或裁决节点触发拦截/转人工/保守输出，避免直接给出危险建议；同时记录审计日志便于合规追溯。
**代码证据：** `nodes/audit.py`, `guard/audit_service.py`

### 41) 你们有权限控制吗？
**参考回答：**
有，支持最小 RBAC（admin/operator/auditor）和 SSO claim 映射，并记录 auth audit 日志到文件和结构化日志。
**代码证据：** `core/security/rbac.py`, `api/v1/endpoints/auth.py`, `core/security/auth_audit.py`

### 42) 输出为何还要过 quality gate？
**参考回答：**
因为推理正确不代表输出合规。quality gate 负责最终一致性和最低质量门槛，把“推理结果”转成“可发布结果”。
**代码证据：** `nodes/quality_gate.py`, `sub_graphs/egress.py`

### 43) 你们如何做可追责审计？
**参考回答：**
通过结构化日志 + audit sink + Langfuse trace 关联 request_id/session_id，实现“谁在何时基于何证据输出了什么”。
**代码证据：** `core/logging/setup.py`, `core/security/auth_audit.py`, `core/monitoring/langfuse_bridge.py`

### 44) 如果面试官问“医疗 AI 合规边界”你怎么答？
**参考回答：**
系统定位为辅助分诊与就医建议，不替代医生诊断；高风险/不确定场景优先追问或转人工，不给超出权限的确定性诊断与处方结论。

## F. 流式体验、可观测与运维（45-50）

### 45) 你们为什么坚持 SSE？
**参考回答：**
医疗咨询用户对等待很敏感。SSE 可以连续回传 `status/thought/token/final`，既提升感知速度，也让前端能展示“系统正在做什么”。
**代码证据：** `api/v1/endpoints/chat.py`, `services/chat_service.py`

### 46) 你们的可观测指标有哪些？
**参考回答：**
有 HTTP/RAG 指标、缓存命中、检索延迟、首包/首 token 延迟、节点耗时，以及 Langfuse trace/span 级别观测，覆盖链路性能与质量信号。
**代码证据：** `core/monitoring/metrics.py`, `middleware/instrumentation.py`, `chat.py`, `langfuse_bridge.py`

### 47) 怎么做性能优化？
**参考回答：**
并行检索（Milvus+BM25 gather）、候选裁剪、可选 rerank、缓存分层、异步持久化、SSE ping 保活与 schema 统一，减少尾延迟和前端抖动。
**代码证据：** `rag/retriever.py`, `nodes/persistence.py`, `chat.py`

### 48) 你们如何做线上故障定位？
**参考回答：**
基于 request_id/session_id 串联日志与 trace；按节点维度定位（rewrite/retrieve/judge/respond）；先看 fallback 是否触发，再看外部依赖（LLM/Milvus/Redis）。

### 49) 这个项目怎么做离线评测闭环？
**参考回答：**
有 evolution runner：患者对抗模拟 + 审计评分 + 全局分统计 + SSE 可视化。低分可触发继续迭代，形成“评测-修正-回归”闭环。
**代码证据：** `api/v1/endpoints/evolution.py`, `services/evolution_runner.py`

### 50) 如果让你下一步把这个项目提升到“面试加分项”，你会做什么？
**参考回答：**
我会补三件事：
1) 建立标准化评测集（分诊准确率/安全拦截率/首 token 延迟）；
2) 增加回放工具（trace->state replay）；
3) 做灰度和 A/B（模型与检索策略切换可量化）。
这样从“能跑”升级为“可持续工程化运营”。

---

## 三、快问快答（30 秒版本）
- **你们是纯 Agent 吗？** 不是，主图 workflow-first，局部有 agent loop。
- **怎么防幻觉？** 检索证据 + decision governance + quality gate + 审计。
- **怎么抗故障？** key/model/node/local 多级 fallback + timeout + degrade。
- **怎么可观测？** Prometheus + 结构化日志 + Langfuse trace。
- **怎么持续改进？** Evolution runner + 回归评测。

---

## 四、联网检索参考（题型来源）
> 这些来源用于整理“AI/LLM 工程师面试常见能力维度”，最终题目已按本项目重写。

1. OpenAI Agents Guide  
   https://platform.openai.com/docs/guides/agents
2. Anthropic: Building Effective Agents  
   https://www.anthropic.com/engineering/building-effective-agents
3. LangGraph Agents / ReAct docs  
   https://langchain-ai.github.io/langgraph/agents/agents/
4. LLM Engineer Interview Questions（GitHub）  
   https://github.com/ardakdemir/llm_engineering_interview_questions
5. Top 20 LLM Interview Questions and Answers（Analytics Vidhya）  
   https://www.analyticsvidhya.com/blog/2024/06/top-llm-interview-questions-and-answers/
6. 30 Essential AI Engineer Interview Questions（DataCamp）  
   https://www.datacamp.com/blog/ai-engineer-interview-questions

---

## 五、使用建议
- 先背第 1、2、11、21、31、39、45、49 题，覆盖架构、RAG、稳定性、安全、运维、评测主线。
- 面试回答顺序推荐：**结论 -> 取舍 -> 代码证据 -> 风险与改进**。
- 如果被追问细节，优先引用本项目文件路径而不是泛化理论。

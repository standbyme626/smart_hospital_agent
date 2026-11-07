# smart_hospital_agent 2026 H2 升级方案

> 审计基线：2026-11-08，基于本机真实代码、依赖、容器与数据卷，不是 README 转述。
> 参考对象：`earendil-works/pi`（极简 Agent Harness：pi-ai / pi-agent-core / pi-durable / pi-telemetry）。
> 本文只写架构与迁移方案，不含代码。

---

## 0. 结论先行

**这个项目不是"需要优化"，而是"需要换骨架"。**

- 必须杀掉的三个旧架构：**专家组（multi-agent crew）**、**全局 monkey-patch 的 LLM 工厂**、**双诊断路径 + 5 份依赖清单**。
- 必须保留的三类资产：**21 科室与领域提示词**、**混合检索与评测数据**、**治理/审计/HITL 的产品思路**。
- 升级目标：**单 Agent 循环 + 工具 + 确定性策略门禁**，模型层学 pi-ai，会话层学 pi-durable，观测层学 pi-telemetry。
- 预计 6 周（1 名资深工程师），分 6 个阶段；每阶段有独立回滚点。
- 不追求"多智能体会诊"的故事，追求**可评测、可复现、可运维**。

---

## 1. 真实情况审计（证据）

### 1.1 运行状态

| 项 | 实测 | 影响 |
|---|---|---|
| PostgreSQL `public` 表数 | **0** | 持久化、审计、规则、评测全部没有落地 |
| Redis `dbsize` | **0** | 缓存/记忆为空 |
| Milvus 容器 | 已停止，数据卷约 586 MB | 检索栈当前不可用 |
| 运行中的依赖 | PG、Redis、etcd、MinIO、Prometheus、Grafana（已 Up 10 天） | Milvus 三件套在跑但 Milvus 本体停 |
| 前端 | `frontend_new/` 无 `node_modules` | 当前无法直接启动 |
| 最后一次实质推进 | 2026-03-31 | 8 个月未更新 |
| 最后一次真实运行 | 2026-10-01（云端模型聊天验证） | 只验证了聊天链路，不是完整业务 |

### 1.2 代码体量

| 文件 | 行数 | 判断 |
|---|---:|---|
| `core/graph/sub_graphs/diagnosis.py` | 2,139 | 单文件巨石，诊断/RAG/应急/结构化输出混在一起 |
| `rag/retriever.py` | 2,359 | 混合检索、RRF、重排、缓存、SQL 过滤全部耦合 |
| `core/llm/llm_factory.py` | 916 | 模型工厂 + 全局 SDK 劫持 + 同步/异步双份逻辑 |
| `api/v1/endpoints/chat.py` | 853 | SSE、快捷命令、支付、挂号、事件渲染混在一起 |
| `agents/factory.py` | 412 | 科室 Agent 工厂，仍依赖 LangChain AgentExecutor |
| `agents/doctor_graph.py` | 474 | 医生工作流，与主图存在职责重叠 |
| `core/graph/nodes/expert_crew.py` | 205 | 专家组并行节点 |
| `core/graph/nodes/expert_router.py` | 121 | 科室向量路由 |
| `core/graph/nodes/audit.py` | 95 | 安全审计（LLM 调用） |
| `core/crew/` | 496 | CrewAI 遗留：expert_crew / tools / verify |
| `core/graph/sub_graphs/medical_core.py` | 59 | 专家组子图，和 `diagnosis` 子图并列 |
| Python 文件总数 | 378 | 测试文件约 65 个，覆盖率约 17%–22% |
| 前端单测 | 0 | 仅 1 个 Playwright E2E |
| CI | 无 `.github/workflows/` | 没有回归门禁 |

### 1.3 依赖现状

`backend/` 下有 **5 份 requirements**：38 / 36 / 39 / 61 / 155 行。彼此冲突，无法判断哪份是真相。

关键依赖：

| 依赖 | 版本 | 问题 |
|---|---|---|
| `fastapi` | 0.109.2 | 2024 年版本，落后 |
| `langchain` / `langgraph` | 1.2.8 / 1.0.7 | 2026 年主线，但项目用法偏重 |
| `crewai` / `crewai-tools` | 1.9.3 | **实际已不使用**，但仍在依赖里 |
| `dspy-ai` | 未固定 | 只在 `diagnosis.py` 和 `rag/dspy_modules.py` 使用，路径深 |
| `chromadb` | 0.5.0 | 代码中无实际导入，纯残留 |
| `sentence-transformers` | 2.7.0 | 旧版本，和当前 torch/transformers 不匹配 |
| `numpy` | `<2.0.0` | 与 2026 年 torch/transformers 生态冲突 |
| `torch` / `transformers` | 未固定 | 构建不可复现 |
| `next` | 14.2.15 | 前端落后两个大版本 |

### 1.4 数据与评测

| 资产 | 规模 | 状态 |
|---|---|---|
| MSD 专业版原文 | 13,044 页 / 24.4 MB | 有，但版权仅限内部 |
| CMtMedQA | 68,023 条 / 18 大类 / 267 子类 | 有，未纳入统一评测 |
| medical_gold v5 | 56,810 条，含科室与 `vector_text` | 有，质量未审计 |
| 意图/分诊评测 | 500 / 400 / 100 条 | 有样本，无落盘准确率 |
| RAGAS | 20 条抽样 | Faithfulness 0.254 / 0.225，Relevancy 0.621 / 0.602，Precision 1.0（退化） |
| 检索留出集 | 无 | 没有 Recall@k / MRR |
| 图通道消融 | 无 | 无法证明图有用 |
| DPO 功能验证 | 6/6 通过 | 只覆盖功能，无并发/压测 |

---

## 2. 老架构为什么害人（逐项）

### 2.1 专家组：看起来像 MDT，实际是 N 个独立 LLM 调用

真实链路：

`medical_core` 子图 → `parallel_expert_crew_node` → 按 `departments` 并发 `asyncio.gather` → 每个科室一个 `DepartmentAgentFactory` AgentExecutor（各自带工具、提示词、LLM）→ 汇总 → 药剂师节点 → 安全审计节点（又一次 LLM）→ 审计失败重试一次。

问题不在"并行"，而在**架构语义**：

1. **不是会诊**：各科室 Agent 之间没有共享证据、没有交叉质询、没有仲裁协议。它们只是各自回答一遍，然后被拼接。叫"专家组"是过度包装。
2. **成本与延迟线性放大**：一次请求 N 个科室 = N 次 LLM 调用 + 1 次药剂师 + 1 次审计；N 越大越慢越贵。
3. **不可评测**：没有固定输出结构，没有证据引用，没有 ground truth；无法回答"专家组比单 Agent 好多少"。
4. **路由与配置不一致**：`expert_config.json` 只有 **8 个科室**（Cardiology / Neurology / Gastroenterology / Dermatology / Orthopedics / Respiratory / Endocrinology / General Practice），但 `specialists.json` 有 **21 个科室**。两套真相并存。
5. **CrewAI 已死但依赖还在**：`core/crew/` 共 496 行仍导入 `crewai`，节点却已改用 `DepartmentAgentFactory`。这是"死代码 + 活依赖"的最坏组合。
6. **真正需要专家组的场景很少**：绝大多数分诊只需要"一个主诊 + 证据 + 必要时一次专科咨询"。N 个专家并行不是医疗必需，是架构表演。
7. **安全审计也是 LLM**：审计节点用 `SmartRotatingLLM` 生成 JSON，审计本身也可能幻觉；它没有规则兜底，也没有可解释的审计轨迹。

**结论：默认路径删除专家组。** 保留"专科咨询"作为**一个可调用工具**（最多 1 次，按需触发），不保留 N 个 Agent 的 MDT。

### 2.2 LLM 工厂：全局 monkey-patch 是最危险的技术债

`core/llm/llm_factory.py` 在导入时直接改写 `openai.OpenAI.__init__` 和 `AsyncOpenAI.__init__`，把所有 OpenAI SDK 调用重定向到 `OPENAI_API_BASE`。这带来：

1. **隐式行为**：任何第三方库只要用 OpenAI SDK，都会被静默改道；
2. **测试不可控**：测试无法安全地 mock 或隔离 provider；
3. **provider 逻辑碎片化**：DashScope、ModelScope、SiliconFlow、本地 vLLM 的差异被塞进 916 行工厂；
4. **默认值与真实环境漂移**：代码默认 `qwen-max/turbo/long`，实际 `.env` 已切 ModelScope DeepSeek；
5. **同步/异步双份实现**：同一套路由逻辑写两遍，修一处漏一处。

**结论：用 pi-ai 风格的 provider registry + model catalog 替换，禁止全局 SDK 劫持。**

### 2.3 双诊断路径：同一个问题两种行为

主图同时挂了 `medical_core` 和 `diagnosis` 两个子图：

- `MEDICAL_CONSULT` / `CRISIS` / 默认 → `diagnosis`（2,139 行，DSPy + RAG）；
- `INFO` → `medical_core`（专家组）；
- 两条路径的输出结构、证据链、审计方式都不同。

这不是"模块化"，是**两套产品逻辑共存**。用户同一个问题，走哪条路取决于意图分类，行为不可预测。

**结论：合并为一条诊断主链路，专家组路径删除或默认关闭。**

### 2.4 Upgrade3 壳：只增加间接层，不增加能力

`UPGRADE3_CHAT_SHELL_ENABLED`、`UPGRADE3_DIAGNOSIS_SHELL_ENABLED`、`UPGRADE3_RETRIEVER_PIPELINE_ENABLED` 默认全为 `true`。这些壳模块只是 lazy import 转发到旧实现，属于"架构表演"：

- 读代码的人以为已经拆分了，实际没有；
- 出问题时要跨两层排查；
- 开关默认 true，等于没有开关。

**结论：要么实现，要么删除。默认选择删除。**

### 2.5 重基建：Milvus 2.3 + etcd + MinIO 对 13k 页数据是过度设计

当前数据规模：13,044 页原文、56,810 条 QA、6,000–10,000 级别 chunk。这个量级：

- PostgreSQL + pgvector 足够；
- Milvus 三件套（etcd + MinIO + Milvus）增加 3 个容器、3 份运维面、1 套独立备份；
- Milvus 2.3 是 2023 年版本，和 2026 年的生态脱节；
- 当前 Milvus 本体已停，但 etcd/MinIO 还在跑，属于"半死状态"。

**结论：把向量存储抽象成接口，优先迁到 pgvector；Milvus 只作为可选后端保留一个过渡期。**

### 2.6 安全与工程债

| 问题 | 证据 | 风险 |
|---|---|---|
| RBAC 默认关闭 | `AUTH_RBAC_ENABLED` 默认 `false` | 公网/内网横向越权 |
| CORS 允许所有 | `main.py` CORS 配置为开发模式 | 任意站点可调 |
| 无限流 | 无中间件/网关限流 | 刷接口、成本失控 |
| 本地明文凭据 | 项目根 `1.sh` 含云主机密码（未进远程仓库） | 凭据泄露 |
| 评测泄露 | `benchmark_results.txt` 含 key 前缀（未进远程仓库） | 需轮换 |
| 无 CI / 低覆盖 | 无 workflows，测试约 17%–22% | 重构无安全网 |

---

## 3. 参考 pi 的升级原则

| pi 的设计 | 本项目的用法 | 不照搬的部分 |
|---|---|---|
| **pi-ai**：统一多 provider LLM API | 模型网关：provider registry + model catalog + 统一流式/工具/用量/重试 | pi 面向编码 Agent，模型目录需按医疗任务重新分级 |
| **pi-agent-core**：Agent runtime with tool calling and state | 单 Agent 循环 + 显式状态 + 工具注册表 | 不引入 pi 的编码工具集；医疗工具自己定义 |
| **pi-durable**：durable conversation/task/document runtime | PG 事件日志、可恢复 SSE、幂等、HITL 中断 | 不需要完整的 document runtime |
| **pi-telemetry**：vendor-neutral telemetry | OpenTelemetry traces/metrics/logs | 保留 Prometheus，但不再绑定 Langfuse/LangSmith SDK |
| **最小内核 + 扩展** | 科室包、提示词模板、工具都做成可配置扩展 | 不做插件市场 |
| **默认不做 sub-agents** | 删除专家组；专科咨询最多 1 次，按需 | 医疗高风险场景仍保留 HITL |
| **无内置权限系统** | **不照搬**：医疗应用必须 RBAC + 策略门禁 + 沙箱 | pi 的"容器化/沙箱"思路可借用 |
| **模型目录（model catalog）** | 维护版本化模型清单：上下文、工具、视觉、成本、延迟 | 目录由本项目维护，不依赖 pi.dev |

核心判断：**pi 的价值不是"再换一个框架"，而是"把 Agent 运行时收缩到最小，把复杂度放到工具、策略和评测里"。**

---

## 4. 目标架构（2026 H2）

### 4.1 分层

| 层 | 职责 | 关键约束 |
|---|---|---|
| 接口层 | FastAPI + SSE + 类型化 OpenAPI | 鉴权、限流、请求 ID |
| 会话运行时 | 事件日志、单 Agent 循环、工具调用、HITL、可恢复流 | 无全局状态，所有状态可持久化 |
| 领域策略层 | 危机识别、科室路由、安全门禁、审计 | **规则优先，LLM 只做兜底** |
| 检索层 | 稠密 + 稀疏 + 可选图谱、重排、引用 | 统一接口，可替换后端 |
| 模型网关 | provider registry、模型目录、任务路由、本地回退 | 禁止 monkey-patch，禁止硬编码模型名 |
| 数据层 | PG（会话/审计/规则/评测）+ pgvector + Redis 缓存 | 先 pgvector，Milvus 可选 |
| 观测层 | OTel + Prometheus + 结构化日志 | 不记录 PII/密钥 |
| 评测/CI | 金标集、回归门禁、压测 | 每次合并必须跑 |

### 4.2 单 Agent 循环

一次问诊只允许一个主 Agent 循环：

1. 读取会话与患者上下文；
2. 进入确定性策略层：危机识别、意图、科室路由；
3. 主 Agent 决定调用哪些工具：检索、指南、药物相互作用、EHR/检验、专科咨询、报告提交；
4. 工具返回带证据 ID 的结果；
5. 主 Agent 生成结构化报告：科室、紧急度、依据、建议、引用、置信度、拒答原因；
6. 策略层做安全门禁；高风险动作进入 HITL；
7. 事件写入 PG，SSE 推送并结束。

**专科咨询是工具，不是子图。** 只有主 Agent 判断"证据不足/跨科冲突"时才调用，最多 1 次；拒绝 N 个科室并行的默认路径。

### 4.3 策略层（确定性优先）

| 策略 | 实现方式 | 是否允许 LLM |
|---|---|---|
| 危机识别 | 关键词 + 规则 + 轻量分类器（SetFit/embedding） | 否，LLM 只做二次确认 |
| 科室路由 | 21 科室 embedding/分类器，输出 Top-1 + 备选 | 可选 |
| 安全门禁 | 红旗症状、禁忌、剂量、处方边界规则 | 规则优先；LLM judge 仅低置信 |
| 审计 | 结构化审计事件 + 规则断言 | 不依赖 LLM 单独放行 |
| HITL | 高风险动作暂停，人工确认后恢复 | 必须 |

### 4.4 模型网关（pi-ai 风格）

- **统一接口**：一次调用只暴露 provider、model、messages、tools、stream、response_format、usage；
- **Provider Registry**：OpenAI 兼容优先（DashScope / ModelScope / SiliconFlow / 本地 vLLM），确有差异才写原生适配；
- **Model Catalog**：模型 ID、上下文、最大输出、工具/视觉支持、成本、延迟、可用性；
- **任务路由**：分诊用快模型，诊断用强模型，摘要用便宜模型，失败切本地；
- **可靠性**：超时、重试、退避、熔断、降级；
- **可观测**：每次调用记录 provider、model、tokens、成本、延迟、错误码；
- **安全**：密钥只从环境/密钥管理读取，不写日志，不进代码；
- **本地回退**：Qwen3-0.6B DPO（vLLM）作为最后一级，带健康检查与熔断。

### 4.5 会话与上下文（pi-durable 风格）

- **事件日志**：用户消息、检索、工具调用、策略决策、LLM 调用、最终报告，全部 append-only 存 PG；
- **可恢复 SSE**：事件 ID + Last-Event-ID，断线可续；
- **上下文预算**：系统提示 + 策略 + 检索证据 + 最近轮次 + 摘要；超预算先压缩历史，不丢证据；
- **长期记忆**：事实带来源、置信度、过期时间；召回时按相关性 × 置信度排序；
- **HITL 中断**：图/循环可暂停，人工确认后从事件点恢复。

### 4.6 检索与数据

- 统一检索接口：query + filters + top_k → 文档、分数、引用 ID；
- 混合召回：稠密 + BM25；图谱作为可选通道；
- 重排：Cross-Encoder，失败回退 RRF；
- **图通道必须做消融**：有图 vs 无图，证明增益，否则默认关闭；
- 数据：MSD 仅内部使用；数据卡记录来源、许可、去重、泄漏审计；
- 评测：300 条留出集，四条基线（BM25 / 向量 / 混合 / +图），引用准确率单独评。

### 4.7 观测与评测

- OTel：请求、LLM、工具、检索、策略五类 span；
- 指标：首 token 延迟、总延迟、成本、tokens、工具成功率、危机召回、引用准确率、错误率；
- 日志：结构化、带 request_id、无 PII/密钥；
- 评测：金标集 + 回归门禁 + shadow 模式 + 压测（100 并发 SSE）。

### 4.8 安全与合规

- 鉴权：JWT/OIDC，RBAC 默认开启；
- CORS：白名单，不允许 `*`；
- 限流：按用户/IP/接口；
- 密钥：环境变量/密钥管理，轮换，禁止明文；
- PII：脱敏、保留期限、访问审计；
- 医疗边界：不确诊、不开处方，输出免责声明；危机升级人工。

---

## 5. 领域架构重构（医疗）

### 5.1 保留的资产

- 21 科室注册表与别名；
- 科室提示词、分诊提示词、医生提示词；
- MSD / CMtMedQA / medical_gold 数据；
- 500/400/100 评测样本；
- 药物相互作用、指南、EHR/检验工具（mock 或真实适配）；
- 质量门禁、审计、HITL 的产品思路。

### 5.2 删除/替换的资产

| 旧资产 | 处置 |
|---|---|
| `core/crew/`（CrewAI） | 删除 |
| `core/graph/nodes/expert_crew.py` | 删除或降级为 `consult_specialist` 工具 |
| `core/graph/sub_graphs/medical_core.py` | 合并进单 Agent 主链路 |
| `expert_config.json`（8 科室） | 并入 21 科室注册表，单一真相 |
| `rag/dspy_modules.py` 与 DSPy 配置 | 删除；改为直接结构化调用 |
| `core/models/vram_manager.py` | 删除；本地模型由模型网关管理 |
| `core/monitoring/langfuse_bridge.py` / LangSmith | 删除；改 OTel |
| `UPGRADE3_*` 壳 | 删除 |
| 5 份 requirements | 合并为 1 份 pyproject + 锁文件 |

### 5.3 新的医疗输出契约

最终报告必须是结构化对象，而不是自由文本：

- 推荐科室（Top-1 + 备选）；
- 紧急度（普通 / 紧急 / 危急）；
- 关键依据（证据 ID + 引用片段）；
- 建议动作；
- 置信度；
- 拒答/转人工原因；
- 免责声明。

这使评测、审计、前端渲染、回归测试都变成确定性的。

---

## 6. 迁移路线（6 周）

### Phase 0（第 0 周）：止血与冻结

- 冻结新功能，建立当前行为基线；
- 轮换所有本地明文密钥，清理 `.env`、数据库文件、含凭据脚本；
- 删除死代码：CrewAI、Chroma、Upgrade3 壳、重复 requirements；
- 建立最小 CI：lint、类型检查、单测、启动冒烟；
- 备份 PG/Redis/Milvus 数据卷。

验收：依赖图里不再出现 crewai / chromadb；CI 能跑；密钥已轮换。

回滚：保留旧分支与数据快照。

### Phase 1（第 1 周）：模型网关

- 实现 provider registry 与 model catalog；
- 移除 OpenAI SDK 全局 monkey-patch；
- 统一同步/异步调用路径；
- 任务路由：分诊/诊断/摘要/本地回退；
- 旧 `SmartRotatingLLM` 作为适配器保留在开关后，双跑对比。

验收：同一请求可通过配置切换 provider；无全局 SDK 改写；调用日志含 provider/model/tokens/成本。

回滚：开关切回旧工厂。

### Phase 2（第 2 周）：单 Agent 与会话运行时

- 合并 `medical_core` 与 `diagnosis` 为一条主链路；
- 专家组默认关闭，改为 `consult_specialist` 工具；
- 事件日志落 PG，SSE 支持 Last-Event-ID；
- 上下文预算与历史压缩。

验收：同一问题只有一条执行路径；断线可恢复；事件可回放。

回滚：保留旧子图开关。

### Phase 3（第 3 周）：领域策略与工具

- 危机识别规则 + 轻量分类器；
- 21 科室路由统一；
- 安全门禁规则化；
- 结构化输出 + 引用；
- HITL 中断/恢复。

验收：危机召回、分诊准确率、引用准确率均有基线；高风险动作必须人工确认。

回滚：策略开关逐项关闭。

### Phase 4（第 4 周）：检索统一与评测

- 检索接口抽象；pgvector 试点；
- RRF + 重排 + 引用 ID；
- 300 条留出集、四条基线、图通道消融；
- 评测接入 CI。

验收：Recall@k / MRR / 引用准确率有报告；图无增益则默认关闭。

回滚：Milvus 只读过渡；接口切换开关。

### Phase 5（第 5 周）：安全与观测

- RBAC 默认开启、CORS 白名单、限流；
- OTel traces/metrics/logs；
- PII 脱敏；
- 压测 100 并发 SSE。

验收：鉴权/限流生效；P95/成本/错误率达标；日志无 PII/密钥。

回滚：按开关回退到旧中间件（仅限内网）。

### Phase 6（第 6 周）：前端、文档与发布

- Next 15/16 升级；类型化 API 客户端；
- 引用卡片与证据高亮；
- 更新 README、架构图、评测报告、限制说明；
- 删除旧路径与旧文档。

验收：一条命令启动；演示链路完整；文档与代码一致。

回滚：保留上一个发布 tag。

---

## 7. 验收标准

### 7.1 工程

- 依赖：1 份 pyproject + 1 份锁文件；构建可复现；无 crewai/dspy/chromadb；
- CI：lint、类型、单测、集成、评测冒烟全绿；覆盖率 ≥ 50%；
- 代码：单文件不超过 800 行；诊断/检索/接口各自独立；
- 部署：一条命令起 PG + Redis + 应用；Milvus/Neo4j 可选。

### 7.2 运行时

| 指标 | 目标 |
|---|---|
| 首 token P95 | < 2 秒（云端） |
| 完整响应 P95 | < 10 秒（单轮分诊） |
| 工具调用成功率 | ≥ 99% |
| 错误率 | < 1% |
| 单请求成本 | 按任务设预算并记录，不超预算 |
| 并发 | 100 SSE 连接压测通过 |

### 7.3 医疗质量

| 指标 | 目标 |
|---|---|
| 危机召回 | ≥ 95%（金标集） |
| 分诊 Top-1 准确率 | ≥ 85% |
| 引用准确率 | ≥ 90%（抽 100 条人工/评审） |
| 拒答/转人工正确率 | ≥ 90% |
| 药物相互作用召回 | ≥ 90%（金标集） |
| 检索 Recall@8 | 不低于混合基线，且优于单路 |

### 7.4 安全

- 鉴权默认开启；CORS 白名单；限流生效；
- 密钥轮换；日志无密钥/PII；
- 高风险动作必须 HITL；
- 公网暴露前完成一次渗透/配置复核。

---

## 8. 交付物

1. 新架构文档（分层、时序、契约、数据模型）；
2. 模型目录与 provider 配置说明；
3. 评测报告：300 条留出集、四条基线、图消融、引用准确率；
4. 安全配置说明：RBAC、CORS、限流、密钥管理；
5. 运维手册：启动、健康检查、回滚、备份；
6. 迁移记录：删除清单、替换清单、遗留风险；
7. 更新后的 README：真实数字、限制、复现命令。

---

## 9. 不做什么

- 不保留多智能体专家组作为默认路径；
- 不新增 CrewAI / DSPy / AutoGen / 任何多 Agent 框架；
- 不继续 monkey-patch OpenAI SDK；
- 不把 Milvus 当默认向量库（除非数据量证明需要）；
- 不做"全自动诊断/开处方"的宣传；
- 不为 star 数写无出处的指标；
- 不在没有评测的情况下重写检索/模型层；
- 不把"专家组"包装成 MDT 故事。

---

## 10. 风险与回滚

| 风险 | 影响 | 缓解 |
|---|---|---|
| 重写破坏医疗流程 | 高 | 双跑/shadow；金标集；逐阶段开关 |
| Provider 切换导致行为变化 | 中 | 模型网关双跑；canary；回滚开关 |
| 数据迁移失败 | 中 | 数据卷备份；Milvus 只读过渡；pgvector 双写 |
| 评测集不足 | 高 | 先补 300 条留出集，再谈优化 |
| 时间超期 | 中 | Phase 0/1 先交付止血价值；范围冻结 |
| 医疗安全回归 | 高 | 危机/禁忌/处方规则门禁；人工复核 |

回滚原则：每个 Phase 都有独立开关；旧路径保留到下一个 Phase 验收通过；数据迁移先双写再切换。

---

## 11. 一页结论

`smart_hospital_agent` 的真实问题不是"模型不够强"，而是**架构在 2026 年已经过期**：

- 专家组是 N 个独立 LLM 调用，不是会诊；成本高、不可评测、和 21 科室注册表不一致；
- LLM 工厂用全局 monkey-patch 解决 provider 问题，隐式、脆弱、不可测试；
- 双诊断路径 + Upgrade3 壳 + 5 份依赖 = 两套行为、三层间接、无法复现；
- 数据资产是好的，但评测没有落地，图通道没有消融，安全默认关闭。

升级方向不是"再加一个框架"，而是学 pi 做减法：

**单 Agent 循环 + 工具 + 确定性策略 + 统一模型网关 + 可恢复会话 + OTel 观测 + 评测驱动。**

6 周完成后，这个项目应该具备：

- 一条诊断主链路；
- 一个模型网关；
- 一套可复现评测；
- 一份真实 README；
- 一个可以演示、可以追问、可以运维的医疗 Agent。

这才是 2026 年下半年该有的样子。

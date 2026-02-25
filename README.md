# 🏥 Smart Hospital Agent (V6.2 LangChain 1.0 Re-Architecture)

> **基于 LangGraph 1.0 + LangChain 1.0 的下一代智能分诊与辅助诊疗系统**
> *具备 RAG 3.0 管道、Milvus v4 数据对齐、PII 隐私保护与高性能本地精排能力的 AI 医疗中台*

![UI Preview](https://via.placeholder.com/800x400?text=Smart+Triage+Agent+V6.0+Dashboard)

---

## 🚀 核心特性 (Key Features V6.0)

### 1. 🛡️ 隐私优先架构 (Privacy-First / PII Guard)
- **中间件级脱敏**: 引入 **PII Middleware**，在 LLM 调用前自动拦截并掩盖患者姓名、身份证号等敏感信息。
- **正则增强**: 采用 Lookaround 断言技术，精准识别中文环境下的敏感数据。

### 2. 🏗️ LangChain 1.0 标准化 (Modern Architecture)
- **Standard Agent Factory**: 摒弃旧版 `AgentExecutor`，采用 `create_agent` + `LangGraph` 的现代范式。
- **多模型统一**: 内置 **Content Block Parser**，无缝兼容 Qwen-Max、DeepSeek-V3/R1、Claude 等主流模型的推理与工具调用格式。

### 3. 🛑 零延迟危机干预 (Zero-Latency Crisis Guard)
- **毫秒级响应**: 引入正则+语义双重护栏，对"自杀"、"剧烈胸痛"等高危信号实现**0.01s**极速拦截。
- **自动报警**: 直接输出急救指引，跳过耗时的 LLM 推理流程。

### 4. ⚡️ 高性能批处理推理 (Batch Inference)
- **LocalSLM 升级**: 从串行调用升级为批处理 (Batch Inference)，利用 GPU 并行计算能力，将多文档 RAG 摘要耗时降低 **55%**。
- **超时熔断**: 内置 5s 自动熔断与降级机制，确保在显存高负载下主业务流程不卡死。

### 5. 🔍 RAG 3.0 检索增强 [New in V6.2]
- **Smart Pre-filter**: 结合 SQL 元数据过滤，实现基于科室意图的精准预处理。
- **Hybrid Search + RRF**: 融合向量与词法检索，并引入 **1.2x 科室加成权重**，科室匹配率达 **100%**。
- **数据对齐**: Milvus v4 与 PostgreSQL 实现物理 ID 100% 对齐，杜绝属性回查空引用。

### 6. 🌐 分级诊疗架构 (Triage V6.1)
- **三级路由**: 本地 Qwen3-0.6B 将请求分为 FAST (寒暄/百科)、STANDARD (普通咨询)、EXPERT (复杂会诊)。
- **并行专家组**: 复杂病例由内科医师、药剂师、审计员并行会诊，响应时间从 40s 压缩至 15-20s。
- **自愈机制**: 自动轮询备选模型 (Model Rotation)，在 API 403/429 时无缝切换。

---

## 🏗️ 系统架构 (Architecture V12.0)

本系统采用 **Service-Oriented Architecture (SOA)**，基于 **FastAPI + LangGraph 1.0** 构建的三层子图架构 (Ingress -> Core -> Egress)，实现极致解耦与性能。

### 核心架构图 (Master Graph)

详细逻辑请参考 [Level 0: Master Graph Logic](./docs/architecture/LEVEL_0_MASTER.md)

### 子系统逻辑详解

| 层级 | 模块 | 说明 | 文档链接 |
| :--- | :--- | :--- | :--- |
| **Level 1** | **Ingress Subgraph** | 负责 PII 脱敏、多模态解析与意图识别 | [查看详情](./docs/architecture/LEVEL_1_INGRESS.md) |
| **Level 1** | **Triage Router** | 基于 LLM 的精确意图分诊 (Greeting/Medical/Service) | [查看详情](./docs/architecture/LEVEL_1_TRIAGE.md) |
| **Level 2** | **Diagnosis Subgraph** | **[核心]** 包含 State Sync、RAG 3.0 检索与 DSPy 推理 | [查看详情](./docs/architecture/LEVEL_2_DIAGNOSIS.md) |
| **Level 1** | **Service Subgraph** | 处理挂号、查询等多轮槽位填充任务 | [查看详情](./docs/architecture/LEVEL_1_SERVICE.md) |
| **Level 1** | **Egress Subgraph** | 负责输出审计、质量门禁与 PII 还原 | [查看详情](./docs/architecture/LEVEL_1_EGRESS.md) |

### 节点映射表
所有架构图中的节点与代码文件的对应关系，请参考 [NODE_MAP.csv](./docs/architecture/NODE_MAP.csv)。

---

## 🛠️ 技术栈 (Tech Stack)

| 模块 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **Framework** | **FastAPI + Uvicorn** | 高并发异步后端 |
| **Orchestration** | **LangGraph 1.0** | 状态机工作流管理 (StateGraph) |
| **Agent Core** | **LangChain 1.0** | Standard Runnable & Tool Binding |
| **Middleware** | **Custom PII/HITL** | 隐私与安全拦截器 |
| **Frontend** | **Next.js 14 + Shadcn/UI** | 现代化响应式界面 |
| **Persistence** | **PostgreSQL** | 原生状态持久化 (Checkpoints) |
| **LLM** | **Qwen-Max / DeepSeek** | 多模型推理支持 |
| **Local Model** | **Qwen3-0.6B** | 本地批处理推理 (Batch Inference) |

---

## 🚀 快速启动 (Quick Start)

### 1. 启动后端 (API & Services)
```bash
# 根目录下运行
fuser -k 8000/tcp || true 
cd backend
source ../venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 &
tail -f backend.log
```

### 2. 启动前端 (Interactive UI)
```bash
# 根目录下运行
cd frontend
# 首次运行需安装依赖
# npm install
nohup npm run dev &
# 访问 http://localhost:3000
```

---

## 📅 版本历史 (History)

- **V6.3 (2026-02-07)**: **Safety & Consistency Update**.
    - 🛡️ **PII Local Filter**: 在 Ingress 层引入本地正则脱敏，确保手机号/身份证等敏感数据不离境。
    - 🛑 **HITL (Human-in-the-Loop)**: 集成 LangGraph 中断机制，高危诊疗建议强制进入人工审核队列。
    - ⏳ **Temporal Persona**: 引入时间戳画像管理，解决多轮对话中的病情演进与信息冲突问题。
    - ⚙️ **Anti-Lazy Validation**: 实施严格的输出质量校验，杜绝 LLM 敷衍回复。
- **V6.2 (2026-02-02)**: **Precision RAG & Data Alignment**.
    - 🔍 **RAG 3.0 Pipeline**: 实现 Smart Pre-filter -> Hybrid -> Rerank 四阶段闭环，科室匹配率达 100%。
    - 🆔 **Milvus v4 Alignment**: 重新对齐 5.2w 条医学数据 ID，实现 SQL 与向量库物理 1:1 映射。
    - 🛡️ **Reranker Robustness**: 修复精排器 IndexError，增加 RRF 科室意图分数加成。
- **V6.1 (2026-01-31)**: **Production Ready**. 
    - 🌐 **分级诊疗 (Triage Routing)**: 实现 FAST/STANDARD/EXPERT 三级分流，响应速度提升 36%-71%。
    - 🛡️ **自愈机制 (Self-Healing)**: 增加 API 轮询与 Fallback 策略，在云端限流时自动降级。
    - 👁️ **全链路监控 (Observability)**: 集成 LangSmith，覆盖 RAG 检索、本地推理与 Agent 思考全过程。
    - ⚡ **缓存修复**: 修复 Redis 异步写入问题，实现语义缓存 <0.2s 极速响应。
- **V6.0 (2026-01-31)**: **Performance & Stability**. 引入 LocalSLM 批处理推理 (Batch Inference)，优化数据库连接池与超时控制，系统抗压能力提升。
- **V5.0 (2026-01-30)**: **LangChain 1.0 Re-Architecture**. 引入 Middleware 架构、PII 脱敏、多模型统一解析及原生持久化。
- **V4.0 (2026-01-30)**: **System Intelligence Upgrade**. 引入主动问诊 (Anamnesis) 与危机干预。
- **V3.7 (2026-01-29)**: **Performance & HITL**. 引入人机协同节点，优化 RAG 响应速度至 <3s。
- **V2.0 (2026-01-28)**: **Architecture Refactor**. 实现 Service Layer 分层与异步化改造。

---

## 🔗 相关文档
- [项目深度分析报告](./项目深度分析报告.md)
- [10天升级计划](./10天升级计划.md)

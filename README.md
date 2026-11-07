<div align="center">

# smart_hospital_agent

**面向医疗分诊与问答的 LangGraph 多子图 Agent**

混合检索 · 图谱辅助 · 本地小模型兜底 · SSE 流式 · 可审计治理

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-1C3C3C)
![Milvus](https://img.shields.io/badge/Milvus-2.3-00A1EA)
![Neo4j](https://img.shields.io/badge/Neo4j-optional-4581C3?logo=neo4j&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14.2-000000?logo=nextdotjs&logoColor=white)
![Status](https://img.shields.io/badge/status-experimental-orange)

[项目概览](#1-项目概览) · [核心特性](#2-核心特性) · [系统架构](#3-系统架构) · [检索与融合](#4-检索与融合) · [数据与模型](#5-数据与模型资产) · [快速开始](#6-快速开始) · [API](#7-api) · [评测](#8-评测) · [已知限制](#9-已知限制与未验证事项)

</div>

> **项目状态：实验性 / 归档开发。**
> 2026-03 后未实质推进；2026-10-01 用云端模型做过一次聊天链路验证。仓库保留代码、评测样本与文档，**不含模型权重与医疗原始数据**；本机依赖栈需按第 6 节重建。

**21 个科室** · **56,810 条科室标注 QA** · **13,044 页 MSD 原文** · **RRF(k=60) + Qwen3-Reranker** · **Qwen3-0.6B DPO 本地兜底**

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [3. 系统架构](#3-系统架构)
- [4. 检索与融合](#4-检索与融合)
- [5. 数据与模型资产](#5-数据与模型资产)
- [6. 快速开始](#6-快速开始)
- [7. API](#7-api)
- [8. 评测](#8-评测)
- [9. 已知限制与未验证事项](#9-已知限制与未验证事项)
- [10. 工程实践与踩坑](#10-工程实践与踩坑)
- [11. 目录结构](#11-目录结构)
- [12. 数据使用与许可](#12-数据使用与许可)
- [13. Roadmap](#13-roadmap)

---

## 1. 项目概览

### 1.1 要解决的问题

医疗分诊和问答不是"接一个大模型就完事"：

- **高风险问题必须优先**：胸痛、呼吸困难、意识障碍等危机信号不能被普通问答淹没；
- **回答必须可追溯**：检索到什么、依据是什么、哪个模型生成、是否被质量门禁拦截；
- **云端不可用时不能整体不可用**：需要有本地小模型兜底；
- **输出需要治理**：质量门禁、成本 / Token 统计、审计留痕、失败可复盘。

### 1.2 方案

采用 **workflow-first 主图 + 子图内 local loops**：

- 主流程由 LangGraph 固定编排，保证医疗安全边界与可追责性；
- 子图内保留 Agent 自主性（工具调用、补充信息、挂号 / 支付闭环）；
- 检索层做混合召回与重排，**图谱只做辅助通道**；
- 云端 LLM 优先，失败后切本地 Qwen3-0.6B DPO。

### 1.3 六个平面

| 平面 | 职责 | 入口 |
|---|---|---|
| 策略 | 配置快照、模型路由、特性开关 | `backend/app/core/config.py`、`core/settings/` |
| 执行 | 主图 + 四子图 + 工具调用 | `core/graph/workflow.py`、`core/graph/sub_graphs/` |
| 治理 | 高风险拦截、诊断裁决、输出门禁 | `core/guard/`、`Decision_Judge`、`quality_gate` |
| 数据 | Milvus / BM25 / Neo4j / PostgreSQL / Redis | `rag/retriever.py`、`rag/graph_rag_service.py`、`db/` |
| 观测 | Prometheus、成本 / Token、Trace | `core/monitoring/`、`/metrics` |
| 演进 | 离线对抗、评分、改进闭环 | `api/v1/endpoints/evolution.py` |

---

## 2. 核心特性

| 模块 | 能力 | 代码入口 |
|---|---|---|
| **多子图编排** | `ingress → diagnosis → service → egress`，主图固定路由，子图内保留 Agent 循环 | `core/graph/workflow.py`、`core/graph/sub_graphs/` |
| **混合检索** | Milvus 稠密召回 + BM25 稀疏召回 + SQL 预过滤 + 多查询变体 | `rag/retriever.py`、`rag/modules/bm25.py` |
| **RRF 融合** | `k=60`，向量 / BM25 等权；科室匹配加成 `1.2x` | `rag/retriever.py::_rrf_fusion` |
| **Cross-Encoder 重排** | Qwen3-Reranker-0.6B，默认阈值 `0.15`，可取 Top-K | `rag/reranker.py` |
| **图谱辅助** | Neo4j `Symptom / Disease / Medication / Department`，症状 ↔ 疾病关系查询 | `rag/graph_rag_service.py`、`db/neo4j_client.py` |
| **分层索引** | document / section / paragraph 三级索引 | `rag/hierarchical_index.py` |
| **本地兜底** | Qwen3-0.6B DPO v11.2，vLLM OpenAI 兼容服务，`repetition_penalty=1.10` | `core/models/local_slm.py` |
| **流式输出** | SSE：`thought / token / status / final` | `api/v1/endpoints/chat.py` |
| **治理与观测** | 高风险拦截、质量门禁、Prometheus、成本 / Token、Trace | `core/guard/`、`core/monitoring/` |
| **前端** | Next.js 14 聊天壳 + SSE 代理 | `frontend_new/` |

---

## 3. 系统架构

### 3.1 总览

```text
┌──────────────┐   SSE    ┌──────────────────────────────────────────────┐
│ Next.js 14   │ ───────▶ │ FastAPI  /api/v1/chat/stream                 │
└──────────────┘          │   │                                          │
                          │   ▼                                          │
                          │ LangGraph 主图                               │
                          │ ingress → diagnosis → service → egress       │
                          │   │          │            │                   │
                          │   │          ▼            ▼                   │
                          │ guard    混合检索        工具 / 挂号 / 支付    │
                          │            │                                 │
                          │   ┌────────┴────────┐                        │
                          │   ▼        ▼        ▼                        │
                          │ Milvus   BM25    Neo4j 图谱                   │
                          │   └────────┬────────┘                        │
                          │            ▼                                 │
                          │    RRF(k=60) → Qwen3-Reranker                │
                          │            ▼                                 │
                          │  Decision_Judge → quality_gate               │
                          │            ▼                                 │
                          │  持久化 PG / Redis / Milvus + Trace          │
                          └──────────────────────────────────────────────┘
```

### 3.2 一次问诊请求

1. 前端 `POST /api/v1/chat/stream`，建立 SSE 连接；
2. `ingress` 子图做请求理解、风险识别与意图路由；
3. `diagnosis` 子图做查询改写、实体抽取、混合检索与诊断推理；
4. `Decision_Judge` 判断是否需要追问、给建议或升级为高风险流程；
5. `service` 子图处理挂号 / 支付等业务工具调用；
6. `egress` 子图做质量门禁与输出收敛；
7. `persistence` 异步写入 PostgreSQL / Redis / Milvus，并记录 Trace；
8. SSE 依次推送 `thought / token / status / final`，最后 `[DONE]`。

### 3.3 数据平面

| 存储 | 职责 | 说明 |
|---|---|---|
| PostgreSQL 15 | 会话、审计、规则、业务数据 | `docker-compose.yml` 中的 `db` |
| Redis 7 | 缓存、会话记忆 | 宿主机端口 `6380`（`6379` 已被其他服务占用） |
| Milvus 2.3 | 稠密向量、语义缓存 | `etcd` + `MinIO` 三件套 |
| Neo4j（外部，可选） | 症状 / 疾病 / 药物 / 科室图谱 | 未包含在 compose 中；不可用时图通道自动降级 |
| BM25 索引 | 稀疏召回 | 本地缓存，可重建 |
| 分层索引 | document / section / paragraph | `rag/hierarchical_index.py` |

### 3.4 模型供给

- **云端**：OpenAI 兼容接口（DashScope / ModelScope / 其他），`SmartRotatingLLM` 支持多候选轮转与失败回退；
- **本地**：Qwen3-0.6B DPO v11.2，vLLM OpenAI 兼容服务（默认 `http://localhost:8002/v1`）；
- **检索模型**：Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B，可切 SiliconFlow 远程；
- **回退顺序**：云端候选 → 本地 SLM → 明确失败。

---

## 4. 检索与融合

### 4.1 三路召回

| 通道 | 实现 | 作用 |
|---|---|---|
| Dense | Milvus + Qwen3-Embedding-0.6B | 语义相似 |
| Sparse | BM25（jieba 中文分词） | 术语 / 精确匹配 |
| Graph | Neo4j 实体关系查询 | 症状 ↔ 疾病等结构化关联 |
| 预过滤 | SQL 科室条件 | 缩小检索空间 |
| 多查询 | query variants | 提升召回覆盖 |
| 分层索引 | document / section / paragraph | 长文档上下文 |

### 4.2 RRF 融合

```text
score(d) = Σ_c 1 / (k + rank_c(d))        k = 60
```

- 向量与 BM25 结果**等权**融合；
- 命中科室与建议科室匹配时，分数乘 `1.2`（`rag/retriever.py::_rrf_fusion`）；
- 图谱上下文当前以**提示词拼接**的方式加入，不参与 RRF 打分（见 4.4）。

### 4.3 重排

- 模型：Qwen3-Reranker-0.6B（本地或 SiliconFlow）；
- 默认 `rerank_threshold=0.15`，`top_k=3`；
- 重排失败时回退到 RRF 顺序，不阻断主链路。

### 4.4 图谱通道的边界

当前图谱是**辅助通道**，不是完整 GraphRAG：

- Schema：`Symptom / Disease / Medication / Department`；
- 查询：实体 → `HAS_SYMPTOM` 关系（INCOMING / OUTGOING）；
- 融合：图谱结果作为上下文拼接进提示词，**没有与向量 / BM25 做 RRF**；
- 降级：Neo4j 未配置或不可用时，`graph_rag_service` 软失败，检索继续；
- **未做**：实体 / 关系抽取流水线、多跳路径、图通道消融实验。

### 4.5 缓存与索引

- 语义缓存：Milvus；
- BM25 缓存：本地文件，可 `force_rebuild()`；
- 分层索引：`hierarchical_index_gateway`，按 `document / section / paragraph` 检索。

### 4.6 可调参数

| 参数 | 默认 | 位置 |
|---|---|---|
| `top_k` | 3 | `/api/v1/rag/search` 请求体 |
| `use_rerank` | `true` | 请求体 / 配置 |
| `rerank_threshold` | `0.15` | 请求体 / 配置 |
| `enable_intent_router` | `false` | 请求体 |
| `enable_hyde` | `false` | 请求体 |
| `pure_mode` | `false` | 请求体（禁用 LLM 增强链路） |
| `RAG_PURE_RETRIEVAL_MODE` | 见 `.env` | 全局开关 |

---

## 5. 数据与模型资产

| 资产 | 规模 | 用途 | 是否随仓库发布 |
|---|---|---|---|
| MSD 专业版原文 | 13,044 页 / 24.4 MB | 检索语料 | **否**（版权） |
| CMtMedQA | 68,023 条 / 18 大类 / 267 子类 | 训练 / 评测 | **否** |
| medical_gold v5 | 56,810 条（含科室、`vector_text`） | 科室标注 QA | **否** |
| clean_eval_500 / 400 / 100 | 500 / 400 / 100 条 | 意图 / 分诊评测 | 仓库保留样本与脚本 |
| Qwen3-Embedding-0.6B | 0.6B | 稠密向量 | **否** |
| Qwen3-Reranker-0.6B | 0.6B | 重排 | **否** |
| Qwen3-0.6B DPO v11.2 | 合并权重约 950 MB | 本地兜底 | **否** |

> 数据与权重不随仓库分发。仓库只保留评测样本、构建脚本与文档；复现需要自行准备对应数据与模型。

---

## 6. 快速开始

### 6.1 前置

- Python 3.12
- Node.js 20+
- Docker / Docker Compose
- 可选：Neo4j（图谱通道）、vLLM（本地 SLM）、NVIDIA GPU

### 6.2 环境变量

在项目根目录创建 `.env`（**不要提交**），按需填写：

| 分组 | 键 |
|---|---|
| 云端 LLM | `OPENAI_API_BASE`、`OPENAI_API_KEY`、`OPENAI_MODEL_NAME`、`MODEL_CANDIDATES` |
| 本地 SLM | `LOCAL_SLM_URL`、`LOCAL_SLM_MODEL`、`ENABLE_LOCAL_FALLBACK`、`PREFER_LOCAL` |
| Embedding / Rerank | `EMBEDDING_REMOTE_ENABLED`、`SILICONFLOW_ENABLED`、`SILICONFLOW_API_KEY`、`SILICONFLOW_API_BASE`、`SILICONFLOW_EMBEDDING_MODEL`、`SILICONFLOW_RERANK_MODEL` |
| 存储 | `POSTGRES_SERVER`、`POSTGRES_PORT`、`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`、`REDIS_URL`、`MILVUS_HOST`、`MILVUS_PORT` |
| 图谱（可选） | `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` |
| 行为开关 | `RAG_PURE_RETRIEVAL_MODE`、`AUTH_RBAC_ENABLED`、`SSE_PING_INTERVAL_SECONDS` |

### 6.3 启动依赖

```bash
docker compose up -d
```

启动 PostgreSQL、Redis、etcd、MinIO、Milvus、Prometheus、Grafana。  
Neo4j 未包含在 compose 中；需要图通道时自行启动并配置 `NEO4J_*`。

### 6.4 数据库迁移

```bash
cd backend
alembic upgrade head
```

### 6.5 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 6.6 前端

```bash
cd frontend_new
npm install
npm run dev
```

默认访问 `http://localhost:3000`。

### 6.7 本地 SLM（可选）

```bash
vllm serve /path/to/Qwen3-0.6B-DPO-v11_2-Clean \
  --served-model-name qwen3-0.6b-dpo-v11_2 \
  --port 8002
```

在 `.env` 中设置：

```ini
LOCAL_SLM_URL=http://localhost:8002/v1
LOCAL_SLM_MODEL=qwen3-0.6b-dpo-v11_2
ENABLE_LOCAL_FALLBACK=true
```

### 6.8 健康检查

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/api/v1/rag/status
```

### 6.9 当前状态说明

本机最后一次检查（2026-10）：

- PostgreSQL 表为空（需执行 `alembic upgrade head` 并准备种子数据）；
- `frontend_new/` 未安装 `node_modules`；
- Milvus 容器已停止（数据卷约 586 MB，本机保留）；
- 模型权重与医疗原始数据不在仓库内。

---

## 7. API

### 7.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/chat/stream` | 问诊 / 聊天，SSE 流式 |
| `POST` | `/api/v1/rag/search` | 混合检索（可返回调试指标） |
| `GET` | `/api/v1/rag/status` | 检索栈状态 |
| `POST` | `/api/v1/doctor/*` | 医生工作流 |
| `POST` | `/api/v1/auth/*` | 认证 |
| `*` | `/api/v1/admin/rules/*` | 规则管理 |
| `*` | `/api/v1/evolution/*` | 离线对抗 / 评分 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/metrics` | Prometheus 指标 |

### 7.2 问诊示例

```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{
    "message": "头痛伴发热两天，该挂什么科？",
    "session_id": "demo",
    "rag": {"top_k": 3, "use_rerank": true}
  }'
```

SSE 事件：

```text
event: thought
data: {"text": "...", "node": "..."}

event: token
data: {"text": "...", "node": "..."}

event: status
data: {...}

event: final
data: {"text": "...", "node": "..."}

data: [DONE]
```

### 7.3 检索示例

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/rag/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "妊娠期血糖偏高怎么管理",
    "top_k": 3,
    "use_rerank": true,
    "include_debug": true
  }'
```

---

## 8. 评测

### 8.1 RAGAS 抽样（n=20）

20 条医疗问题，对比本地 SLM 与云端 LLM：

| 指标 | 本地 SLM | 云端 LLM | 说明 |
|---|---:|---:|---|
| Faithfulness | 0.254 | 0.225 | 偏低；答案中可被上下文支持的比例不足 |
| Answer Relevancy | 0.621 | 0.602 | 中等 |
| Context Precision | 1.000 | 1.000 | **退化**：该指标在此规模下无区分度 |

> 这是 20 条抽样，不是基准测试；`Context Precision=1.0` 说明评测方法本身需要改进。详细数据在 `data/evaluator_comparison_20.csv`。

### 8.2 DPO 本地模型功能验证

`Qwen3-0.6B-DPO-v11_2` 通过 vLLM（WSL2 eager）接入，6 条端到端功能用例全部通过：

- 危机识别（`CRISIS`）
- 语义路由（`fast` / `vague` / `expert` / `standard`）
- 医疗问答（输出包含 `<think>`，中间件负责清洗）

详见 `backend/integration_report_v11_2.md`。未做高并发压测。

### 8.3 测试覆盖

- 后端：49 个测试文件 / 221 个源文件，约 **22%**；
- 前端：**0** 个单元测试，1 个 Playwright E2E；
- **无 CI**（`.github/workflows/` 不存在）。

### 8.4 评测缺口

- 没有检索留出集（Recall@k / MRR）；
- 没有图通道消融（有图 vs 无图）；
- 没有引用准确率 / 忠实度的稳定评测；
- 没有延迟与成本基准。

这些是下一步优先补的，不是"已完成"。

---

## 9. 已知限制与未验证事项

1. **不是生产就绪**：`AUTH_RBAC_ENABLED` 默认关闭，CORS 需按部署收紧，无速率限制；
2. **图通道较薄**：只有症状 ↔ 疾病等关系查询，没有实体 / 关系抽取流水线，没有多跳与消融；
3. **评测不足**：RAGAS 仅 20 条；检索层没有留出集；没有引用准确率；
4. **测试与 CI**：后端约 22% 覆盖，前端无单测，无 CI；
5. **数据版权**：MSD / Merck 内容仅内部使用，不随仓库分发；
6. **模型权重**：DPO / Embedding / Reranker 权重不在仓库内；
7. **未验证**：高并发、长会话、跨科室复杂病例、生产级鉴权与审计合规。

---

## 10. 工程实践与踩坑

1. **Milvus 持久化阻塞事件循环**：写入 / 查询改为异步或线程池，避免 SSE 卡死（2026-03-30 修复）。
2. **SSE 稳定性**：客户端在两次 yield 之间断开会导致传输取消；增加心跳与 `[DONE]` 终止帧。
3. **小模型 JSON 污染**：DPO 模型会在 JSON 外输出解释文本；使用非贪婪正则 `(\{.*?\})` 并预先清洗 `<think>`。
4. **思考过程泄露**：中间件统一清洗 `<think>`，前端只展示最终建议。
5. **vLLM on WSL2**：使用 eager 模式；`repetition_penalty=1.10` 抑制小模型循环。
6. **端口冲突**：宿主机 `6379` 已被其他服务占用，Redis 映射到 `6380`。
7. **中文医疗文本**：科室归一化、别名与中英混合术语需要单独处理（`core/department_normalization.py`）。

---

## 11. 目录结构

```text
smart_hospital_agent/
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/      # chat / rag / doctor / auth / admin / evolution
│   │   ├── core/
│   │   │   ├── graph/             # LangGraph 主图 + 子图
│   │   │   ├── guard/             # 高风险拦截
│   │   │   ├── settings/          # 分域配置
│   │   │   ├── monitoring/        # Prometheus / 成本 / Trace
│   │   │   └── models/            # 本地 SLM 适配
│   │   ├── rag/                   # 混合检索 / 图服务 / 分层索引 / 评测
│   │   ├── db/                    # PG / Neo4j 客户端
│   │   └── services/              # chat / embedding / evolution
│   ├── alembic/                   # 数据库迁移
│   ├── tests/
│   └── requirements.txt
├── frontend_new/                  # Next.js 14 聊天壳
├── data/
│   ├── benchmarks/                # 意图 / 分诊评测样本
│   └── evaluator_comparison_20.csv
├── docs/                          # 架构 / 升级 / 优化文档
├── scripts/
├── docker-compose.yml
└── README.md
```

---

## 12. 数据使用与许可

- **代码**：仓库未声明 License，默认保留所有权利；如需开源请补充 MIT / Apache-2.0；
- **医学数据**：MSD / Merck 内容仅用于内部研究与评测，**不随仓库分发**，不得用于商业再分发；
- **模型**：Qwen3 系列按其官方许可使用；Neo4j / Milvus / PostgreSQL / Redis 遵循各自开源许可；
- **部署前**：轮换所有密钥，移除本地 `.env`、数据库文件与任何含凭据的脚本。

---

## 13. Roadmap

- [ ] 密钥清理与轮换（`.env`、数据库文件、含凭据脚本）
- [ ] 开启 RBAC、收紧 CORS、增加速率限制
- [ ] 构建检索留出集（200–500 条）与四条基线
- [ ] 图通道消融（有图 vs 无图）
- [ ] 引用准确率与忠实度评测
- [ ] CI + 覆盖率门禁
- [ ] 一键复现脚本与预构建数据快照

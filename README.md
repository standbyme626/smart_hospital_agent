# smart_hospital_agent

**面向医疗分诊与就医流程的单 Agent 系统**：混合检索 · 药物相互作用 · MCP 挂号支付 · SSE · 可审计

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)]()
[![Next.js](https://img.shields.io/badge/Next.js-16.4-000000?logo=nextdotjs&logoColor=white)]()
[![MCP](https://img.shields.io/badge/MCP-Hospital%20Tools-6E56CF)]()
[![Status](https://img.shields.io/badge/status-v2%20rewrite-orange)]()

> **v2 状态（2026-11）**
> 已按 2026 H2 架构重写：删除 CrewAI、专家组、DSPy、Milvus/Neo4j 三件套与全局 SDK 劫持；改为单 Agent 循环 + 工具 + 确定性策略 + 模型网关 + 事件会话 + pgvector。
> 当前医院 MCP 为 **mock server**，真实医院接口只需替换 `HOSPITAL_MCP_URL`；本地 0.6B 为 Qwen3-0.6B DPO（Ollama，Q8_0）。

**21 个科室** · **药物相互作用规则** · **SSE 事件流** · **Qwen3-8B 向量/重排** · **DeepSeek V4.1 Flash** · **pgvector**

---

## 目录

- [1. 从 v1 到 v2 改了什么](#1-从-v1-到-v2-改了什么)
- [2. 架构](#2-架构)
- [3. 模型与外部依赖](#3-模型与外部依赖)
- [4. 核心能力](#4-核心能力)
- [5. 快速开始](#5-快速开始)
- [6. 数据与种子](#6-数据与种子)
- [7. API](#7-api)
- [8. 评测](#8-评测)
- [9. 安全与限制](#9-安全与限制)
- [10. 目录结构](#10-目录结构)
- [11. 许可与数据使用](#11-许可与数据使用)

---

## 1. 从 v1 到 v2 改了什么

| 维度 | v1（旧） | v2（当前） |
|---|---|---|
| Agent 架构 | CrewAI 专家组 + N 个科室 Agent 并行 + 审计重试 | **单 Agent 循环 + 工具**；专科咨询最多 1 次，按需 |
| 模型接入 | 916 行 `llm_factory` + 全局 monkey-patch OpenAI SDK | **Model Gateway**：provider registry + 模型目录 + 统一重试/用量 |
| 轨迹 | `medical_core` 与 `diagnosis` 两套诊断路径 | **一条主链路**，事件日志可回放 |
| 检索 | Milvus 2.3 + etcd + MinIO + Neo4j（可选） | **PostgreSQL 16 + pgvector**，Dense + BM25 + RRF + Qwen3-8B 重排 |
| 会话 | 内存/临时状态 | **PG 事件日志** + SSE `Last-Event-ID` 可恢复 |
| 前端 | Next.js 14，旧聊天壳 | **Next.js 16 / React 19 / Tailwind 4**，证据面板、报告卡、挂号支付 |
| 依赖 | 5 份 requirements，CrewAI/DSPy/Chroma/旧 torch | **1 份 pyproject + 锁定依赖**，无 CrewAI/DSPy/Chroma |
| 安全 | RBAC 默认关闭、CORS 全开、无限流 | **JWT + RBAC + CORS 白名单 + 限流 + 审计**（可配置） |
| 挂号支付 | 直连 mock HIS/支付 | **MCP 工具** + HITL 确认 token |
| 观测 | Langfuse/LangSmith 桥 + 自定义日志 | **Prometheus + 结构化事件 + 模型调用/成本表** |

旧代码保留在 git 历史里；当前 `backend/app` 是 v2 实现。

---

## 2. 架构

```text
┌──────────────┐   SSE   ┌──────────────────────────────────────────────┐
│ Next.js 16   │ ──────▶ │ FastAPI                                      │
│ Chat/Booking │         │  /api/v1/chat/stream · /booking · /eval      │
└──────────────┘         │   │                                          │
                         │   ▼                                          │
                         │ 单 Agent 循环                                │
                         │  triage → tools → report                      │
                         │   │        │          │                      │
                         │   │        ▼          ▼                      │
                         │   │   Retrieval     Hospital MCP              │
                         │   │   pgvector      (mock / real)             │
                         │   │   + BM25                             │
                         │   │   + rerank                           │
                         │   ▼                                          │
                         │ 确定性策略：危机词 / 科室路由 / 药物检查 / 安全门禁 │
                         │   │                                          │
                         │   ▼                                          │
                         │ PG：sessions / events / bookings / audit      │
                         │     model_calls / eval_runs                   │
                         └──────────────────────────────────────────────┘
```

**设计参考**：`pi` 的极简 Agent Harness 思路——统一模型 API（pi-ai）、单 Agent 运行时（pi-agent-core）、可持久化会话（pi-durable）、厂商无关观测（pi-telemetry）。本项目没有引入 pi 的 TypeScript 运行时，只借鉴其分层与“默认不做多 Agent”的原则。

**关键取舍**：

- 危机识别走**确定性规则 + 关键词**，不让 LLM 单独决定“是否急诊”；
- 药物检查是**独立工具 + 规则表**，LLM 可以调用，但输出由规则兜底；
- 挂号支付只通过 MCP 工具；支付必须携带人工确认 token；
- 图谱通道在 v2 暂不启用：没有消融证明前，不把“GraphRAG”作为卖点。

---

## 3. 模型与外部依赖

| 用途 | 选择 | 说明 |
|---|---|---|
| 主 LLM | **opencode-go `deepseek-v4.1-flash`** | OpenAI 兼容；1M context / 384K output；支持工具调用；key 从 `OPENCODE_API_KEY` 或 `~/.local/share/opencode/auth.json` 读取 |
| Embedding | **SiliconFlow `Qwen/Qwen3-Embedding-8B`** | 4096 维；key 在 `SILICONFLOW_API_KEY` |
| Rerank | **SiliconFlow `Qwen/Qwen3-Reranker-8B`** | `/v1/rerank`；失败时回退 RRF 顺序 |
| 本地小模型 | **WinPC Ollama `qwen3-0.6b-dpo-v11_2`** | 自己微调的 Qwen3-0.6B DPO，Q8_0 GGUF；用于本地分析/云端失败兜底 |
| 数据库 | **PostgreSQL 16 + pgvector** | `docker compose` 中的 `pgvector/pgvector:pg16` |
| 医院系统 | **MCP JSON-RPC** | 默认 `http://127.0.0.1:9100/mcp`（mock）；真实接口替换 URL 即可 |

> 没有 SiliconFlow key 时，embedding 自动回退为确定性 hash 向量（仅用于本地冒烟，不代表真实检索质量）。没有 opencode key 时，模型调用会失败并返回明确的错误事件。

---

## 4. 核心能力

| 模块 | 能力 | 代码 |
|---|---|---|
| 单 Agent 循环 | 工具调用、最多 6 轮、结构化报告 | `backend/app/agent.py` |
| 分诊 | 危机规则 + 科室关键词 + LLM 分诊（JSON） | `backend/app/policy.py`、`prompts.py` |
| 混合检索 | Dense + BM25 + RRF(k=60) + Qwen3-8B 重排 | `backend/app/retrieval.py` |
| 药物合规 | 40+ 常见 DDI 规则、妊娠禁忌、别名匹配 | `backend/app/data/drug_interactions.json` |
| 挂号支付 | MCP：科室/号源/锁号/订单/支付/状态/取消 | `backend/app/mcp_client.py`、`backend/mcp_mock/server.py` |
| SSE | `session/status/thought/citation/token/final/error/booking` | `backend/app/routes.py` |
| 会话 | PG 事件日志、按 `after` 回放 | `backend/app/session_store.py` |
| 安全 | JWT、RBAC、CORS 白名单、限流、审计 | `backend/app/api/deps.py`、`backend/app/main.py` |
| 观测 | Prometheus `/metrics`、模型调用/成本表 | `backend/app/main.py`、`routes.py` |
| 评测 | 检索冒烟 + 危机检测基准 | `backend/app/eval_runner.py` |
| 前端 | 聊天、证据面板、挂号支付、运营后台 | `frontend/app/*` |

---

## 5. 快速开始

### 5.1 前置

- Python 3.12
- Node.js 22+
- Docker / Docker Compose
- 可选：WinPC Ollama（本地小模型）、真实医院 MCP

### 5.2 环境变量

```bash
cp .env.example .env
chmod 600 .env
```

至少填写：

- `OPENCODE_API_KEY`（或留空，从 `~/.local/share/opencode/auth.json` 读取）
- `SILICONFLOW_API_KEY`
- `JWT_SECRET`、`ADMIN_PASSWORD`
- `HITL_CONFIRM_TOKEN`（演示默认 `confirm-demo`）

### 5.3 启动基础设施

```bash
docker compose up -d db redis mcp
```

- PostgreSQL: `127.0.0.1:5434`
- Redis: `127.0.0.1:6381`
- Mock 医院 MCP: `127.0.0.1:9100/mcp`

### 5.4 后端

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/seed.py --limit 500
uvicorn app.main:app --host 127.0.0.1 --port 8020
```

健康检查：

```bash
curl -s http://127.0.0.1:8020/health
```

### 5.5 前端

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

打开 `http://localhost:3000`。如果 `AUTH_ENABLED=true`，先到 `/login` 登录（演示账号 `admin / admin123456`）。

### 5.6 本地 0.6B（可选）

模型已在 WinPC 上导入为 `qwen3-0.6b-dpo-v11_2`（Q8_0 GGUF）。复现步骤：

1. 用 llama.cpp 的 `convert_hf_to_gguf.py` 将 HuggingFace 目录转换为 `--outtype q8_0`；
2. 在 WinPC 写 Modelfile：`FROM <gguf 路径>` + `PARAMETER temperature 0.7` + `PARAMETER repeat_penalty 1.1`；
3. `ollama create qwen3-0.6b-dpo-v11_2 -f Modelfile`。

服务器侧用 SSH 隧道访问：

```bash
ssh -N -L 11434:127.0.0.1:11434 winpc
```

`.env`：

```ini
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen3-0.6b-dpo-v11_2
```

本地模型用途：

- 云端 LLM 不可用时的兜底；
- `POST /api/v1/local/analyze` 的文本分析；
- 不建议作为主诊断模型：0.6B 的输出质量和稳定性有限。

### 5.7 挂号支付

默认走 mock MCP：

- `GET /api/v1/booking/departments`
- `GET /api/v1/booking/schedules?department=心血管内科`
- `POST /api/v1/booking/lock`
- `POST /api/v1/booking/orders`
- `POST /api/v1/booking/orders/{order_id}/pay`（必须携带 `confirm_token`）

真实医院 MCP 只需替换 `HOSPITAL_MCP_URL`，并保证工具名/参数与 `backend/app/tools.py` 中的 schema 一致。

---

## 6. 数据与种子

`backend/scripts/seed.py` 默认读取仓库中的 `data/medical_gold_*.jsonl`（优先 hardened v5，其次 final v4）：

```bash
PYTHONPATH=. python scripts/seed.py --path ../data/medical_gold_final_v4.jsonl --limit 500
```

每条记录会生成：

- `title` = 问题；
- `content` = `vector_text` 或答案；
- `department` = 科室；
- `embedding` = Qwen3-Embedding-8B（4096 维）；
- `tsv` = jieba 分词后的全文索引。

仓库不包含 MSD 原文；`data/` 中的医疗 QA 数据按各自来源条款使用。

---

## 7. API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/auth/login` | 登录，返回 JWT |
| `GET` | `/api/v1/auth/me` | 当前用户/角色 |
| `POST` | `/api/v1/chat/stream` | 问诊，SSE |
| `GET` | `/api/v1/sessions` | 会话列表 |
| `GET` | `/api/v1/sessions/{id}/events?after=` | 事件回放 |
| `POST` | `/api/v1/rag/search` | 混合检索 |
| `GET` | `/api/v1/rag/status` | 文档数、模型、MCP 状态 |
| `POST` | `/api/v1/drug/check` | 药物相互作用 / 妊娠禁忌 |
| `POST` | `/api/v1/local/analyze` | 本地 0.6B 文本分析 |
| `GET` | `/api/v1/booking/departments` | 科室 |
| `GET` | `/api/v1/booking/schedules` | 号源 |
| `POST` | `/api/v1/booking/lock` | 锁号 |
| `POST` | `/api/v1/booking/orders` | 创建订单 |
| `POST` | `/api/v1/booking/orders/{id}/pay` | 支付（HITL token） |
| `GET` | `/api/v1/booking/orders/{id}` | 订单/支付状态 |
| `POST` | `/api/v1/booking/orders/{id}/cancel` | 取消未支付订单 |
| `POST` | `/api/v1/doctor/consult` | 医生工作台咨询 |
| `GET/POST/DELETE` | `/api/v1/admin/rules` | 规则管理 |
| `GET` | `/api/v1/admin/model-calls` | 模型调用/成本 |
| `GET` | `/api/v1/admin/audit` | 审计日志 |
| `POST` | `/api/v1/eval/run` | 运行评测 |
| `GET` | `/api/v1/eval/results` | 评测结果 |
| `GET` | `/health` | 健康检查 |
| `GET` | `/metrics` | Prometheus |

SSE 事件：

```text
event: session   → { session_id, request_id }
event: status    → { stage, detail }
event: thought   → { text, tool? }
event: citation  → { id, title, snippet, department, source, score }
event: token     → { text }
event: booking   → { data }
event: final     → { report, citations, safety, triage, booking }
data: [DONE]
```

---

## 8. 评测

本次实测（本机，2026-11-08）：

| 项目 | 结果 | 说明 |
|---|---|---|
| 检索冒烟 | Recall@5 **1.0**，MRR@5 **0.9**（n=20） | **in-corpus**：query 取文档标题，答案就在同一索引内；不是留出集 |
| 危机检测 | precision **1.0**，recall **1.0**（n=500） | 关键词表使用了这份 benchmark 的失败样本做扩充，**in-sample**，不是盲测 |
| 后端测试 | **15 passed** | `backend/tests` |
| 前端构建 | **通过** | Next.js 16.4.5 生产构建 |

**评测缺口（下一步）**：

- 构建 200–500 条**留出集**（query 与索引文档不同源）；
- 四条基线：BM25 / 向量 / 混合 / +图；
- 引用准确率与 faithfulness；
- 图通道消融（有图 vs 无图），未证明增益前不启用。

> 旧 README 的 92% Recall、71.4% 行业基线等数字不再使用；本 README 只保留可复算的结果。

---

## 9. 安全与限制

- **默认演示口令**：`admin / admin123456`，上线前必须修改；`AUTH_ENABLED=false` 只用于本地开发。
- **密钥**：`.env` 权限 600，不提交；`OPENCODE_API_KEY` / `SILICONFLOW_API_KEY` 不进日志。
- **医院 MCP**：当前为 mock；真实接入前需确认鉴权、幂等、超时、重试与对账。
- **支付**：必须人工确认 token；不要在生产环境把 token 写进前端默认值。
- **医疗边界**：不确诊、不开处方；危机情况直接建议 120/急诊。
- **数据版权**：MSD/Merck 内容仅内部使用；仓库不包含原文。
- **本地 0.6B**：适合分析/兜底，不适合作为主诊断模型。
- **未验证**：高并发、长会话、真实医院 MCP、生产级审计合规。

---

## 10. 目录结构

```text
smart_hospital_agent/
├── backend/
│   ├── app/
│   │   ├── api/deps.py          # JWT / RBAC
│   │   ├── agent.py             # 单 Agent 循环
│   │   ├── model_gateway.py     # opencode-go / SiliconFlow / Ollama
│   │   ├── retrieval.py         # pgvector + BM25 + RRF + rerank
│   │   ├── policy.py            # 危机 / 科室 / 药物 / 安全门禁
│   │   ├── tools.py             # 工具注册与执行
│   │   ├── mcp_client.py        # 医院 MCP JSON-RPC
│   │   ├── session_store.py     # 事件日志 / 审计 / 调用记录
│   │   ├── eval_runner.py       # 检索冒烟 + 危机评测
│   │   ├── main.py              # FastAPI + 中间件
│   │   └── data/                # 21 科室、DDI 规则、科室 YAML
│   ├── mcp_mock/server.py       # mock 医院 MCP
│   ├── scripts/                 # seed / eval
│   ├── tests/                   # 15 个测试
│   └── pyproject.toml
├── frontend/                    # Next.js 16 / React 19 / Tailwind 4
│   ├── app/                     # chat / booking / admin / login
│   ├── components/              # Sidebar / ReportCard
│   └── lib/                     # API + SSE 客户端
├── data/                        # medical_gold / benchmarks
├── docs/UPGRADE_PLAN_2026H2.md  # v1 → v2 升级方案
├── docker-compose.yml           # pgvector + redis + mock MCP + api
└── .env.example
```

---

## 11. 许可与数据使用

- 代码：仓库未声明 License；如需开源请补 MIT / Apache-2.0。
- 医疗数据：`data/` 与外部 MSD 内容按各自来源条款使用，不随代码重新授权。
- 模型：DeepSeek / Qwen 系列遵循各自官方许可；Ollama 本地权重来自你自己的微调产物。
- 医院 MCP：接入真实系统前，先完成安全评估、鉴权与对账方案。

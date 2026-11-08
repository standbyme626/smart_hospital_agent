# smart_hospital_agent

**一个有证据的就医助手**：分诊 · 报告解读 · 用药检查 · 挂号支付 · 家庭健康档案

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)]()
[![Next.js](https://img.shields.io/badge/Next.js-16.4-000000?logo=nextdotjs&logoColor=white)]()
[![MCP](https://img.shields.io/badge/MCP-Hospital%20Tools-6E56CF)]()
[![Live](https://img.shields.io/badge/live-hospital.kk.zhzepei.cn-0E9384)](https://hospital.kk.zhzepei.cn/)

> **线上演示**：https://hospital.kk.zhzepei.cn/ （登录后使用，演示账号由管理员发放）
> **前端 v3**：2026 H2 消费级医疗 AI 界面，浅色「温暖临床 + 证据优先」；旧前端保留在 `frontend-legacy/`，不参与启动。
> **唯一 mock**：医院 MCP（号源/订单/支付/就诊指引）。LLM、Embedding、Rerank、报告视觉解读均为真实调用。

---

## 界面预览

> 截图来自线上实例：真实 DeepSeek V4.1 Flash、SiliconFlow Qwen3-8B、Qwen3-0.6B 本地兜底、527 条种子文档；医院 MCP 为 mock。

| 首页任务中心 | 分诊结果 |
|---|---|
| ![首页](docs/screenshots/01-home.png) | ![分诊结果](docs/screenshots/02-consult-result.png) |

| 医学依据 | 挂号支付 + 就诊指引 |
|---|---|
| ![证据](docs/screenshots/03-evidence.png) | ![挂号支付](docs/screenshots/04-booking-paid.png) |

| 报告解读 | 药管家 / 相互作用 |
|---|---|
| ![报告解读](docs/screenshots/05-report.png) | ![药管家](docs/screenshots/06-meds.png) |

| 健康时间线 | 运营后台 |
|---|---|
| ![健康](docs/screenshots/07-health.png) | ![运营后台](docs/screenshots/10-admin.png) |

| 登录 | 移动端 |
|---|---|
| ![登录](docs/screenshots/09-login.png) | ![移动端](docs/screenshots/11-mobile-home.png) |

---

## 目录

- [1. 这一版做了什么](#1-这一版做了什么)
- [2. 架构](#2-架构)
- [3. 模型与外部依赖](#3-模型与外部依赖)
- [4. 核心能力](#4-核心能力)
- [5. 快速开始](#5-快速开始)
- [6. 106 部署](#6-106-部署)
- [7. 数据与种子](#7-数据与种子)
- [8. API](#8-api)
- [9. 评测与测试](#9-评测与测试)
- [10. 安全与限制](#10-安全与限制)
- [11. 目录结构](#11-目录结构)
- [12. 许可与数据使用](#12-许可与数据使用)

---

## 1. 这一版做了什么

### 前端 P0–P4

| 阶段 | 交付 |
|---|---|
| P0 视觉与 IA | 浅色温暖临床风、移动端底部导航、桌面侧栏、首页任务中心、分诊结果页、证据抽屉、登录/同意 |
| P1 AI 诊室 + 报告 | 三步问诊（就诊人 → 症状 → 病史）、一屏一问、红旗筛查、报告/皮肤照片上传、视觉解读、家庭档案 |
| P2 药管家 + 随访 | 用药清单、相互作用/妊娠禁忌、诊后随访待办、健康时间线 |
| P3 服务 + 无障碍 | 医生卡、号源、锁号、订单、支付、就诊指引；大字体/高对比/减少动效 |
| P4 数据 | 首访/任务/问诊/报告/支付/证据打开等埋点，运营后台汇总 |

### 后端

- 新增：家庭档案、用药、报告解读、随访、健康时间线、医生卡/就诊指引、埋点、首页聚合。
- 单 Agent 循环 + 工具 + 确定性策略；SSE `session/status/thought/citation/token/final/booking/error`。
- PostgreSQL 16 + pgvector；JWT + RBAC + CORS 白名单 + 按真实 IP 限流 + 审计。
- 模型网关：opencode-go、SiliconFlow、WinPC Ollama（Qwen3-0.6B DPO）。

### 旧前端

`frontend-legacy/` 保留旧版深色控制台代码，**不参与启动**；默认启动 `frontend/` 新前端。

---

## 2. 架构

```text
用户浏览器
  │  HTTPS
  ▼
106.55.16.53 · nginx (hospital.kk.zhzepei.cn, LE 证书)
  ├─ /            → /var/www/hospital  (Next.js 16 静态导出)
  └─ /api/        → 127.0.0.1:18087
                         │ 反向 SSH 隧道
                         ▼
家里服务器 · FastAPI :8020
  ├─ 单 Agent 循环（DeepSeek V4.1 Flash）
  ├─ 检索：pgvector + BM25 + RRF + Qwen3-8B 重排
  ├─ 药物：本地 DDI 规则 + 妊娠禁忌
  ├─ 报告：DeepSeek V4.1 Flash 视觉解读
  ├─ 医院 MCP：mock server :9100（真实环境替换 URL）
  └─ PostgreSQL 16 + pgvector / 事件 / 审计 / 埋点
```

**设计参考**：`pi` 的极简 Agent Harness 思路（统一模型 API、单 Agent 运行时、持久化会话、厂商无关观测）。没有引入多 Agent 框架；专家组已删除。

---

## 3. 模型与外部依赖

| 用途 | 选择 | 说明 |
|---|---|---|
| 主 LLM | **opencode-go `deepseek-v4.1-flash`** | OpenAI 兼容；1M context；支持工具调用与视觉输入；key 从 `OPENCODE_API_KEY` 或本机 opencode auth 读取 |
| Embedding | **SiliconFlow `Qwen/Qwen3-Embedding-8B`** | 4096 维 |
| Rerank | **SiliconFlow `Qwen/Qwen3-Reranker-8B`** | `/v1/rerank`；失败回退 RRF |
| 本地小模型 | **WinPC Ollama `qwen3-0.6b-dpo-v11_2`** | 自己微调的 Qwen3-0.6B DPO，Q8_0；分析/云端失败兜底 |
| 数据库 | **PostgreSQL 16 + pgvector** | `pgvector/pgvector:pg16` |
| 医院系统 | **MCP JSON-RPC** | 默认 mock `http://127.0.0.1:9100/mcp`；真实接口替换 `HOSPITAL_MCP_URL` |

没有 SiliconFlow key 时，embedding 回退为确定性 hash 向量（仅本地冒烟，不代表检索质量）。

---

## 4. 核心能力

| 模块 | 能力 | 代码 |
|---|---|---|
| 首页 | 家庭成员切换、任务入口、继续上次、待办、最近记录 | `frontend/app/page.tsx` |
| AI 诊室 | 三步问诊、一屏一问、红旗筛查、结果页、证据抽屉 | `frontend/app/consult/page.tsx`、`components/ResultView.tsx` |
| 分诊 | 危机规则 + 科室关键词 + LLM JSON 分诊 | `backend/app/policy.py`、`prompts.py` |
| 检索 | Dense + BM25 + RRF(60) + Qwen3-8B 重排 | `backend/app/retrieval.py` |
| 报告解读 | 图像上传 → 视觉模型 → 摘要/异常指标/问医生问题 | `backend/app/routes.py`、`frontend/app/reports/page.tsx` |
| 药管家 | 用药清单、DDI/妊娠禁忌检查 | `backend/app/data/drug_interactions.json`、`frontend/app/meds/page.tsx` |
| 挂号支付 | MCP 科室/医生/号源/锁号/订单/支付/就诊指引 | `backend/app/mcp_client.py`、`backend/mcp_mock/server.py` |
| 健康档案 | 家庭、时间线、随访、用药/报告/挂号聚合 | `backend/app/routes.py`、`frontend/app/health/page.tsx` |
| 安全 | JWT + RBAC + CORS + 限流 + 审计 + HITL | `backend/app/api/deps.py`、`main.py` |
| 观测 | Prometheus `/metrics`、模型调用/成本、埋点 | `backend/app/routes.py`、`frontend/app/admin/page.tsx` |

---

## 5. 快速开始

### 5.1 前置

- Python 3.12
- Node.js 22+
- Docker / Docker Compose
- 可选：WinPC Ollama、真实医院 MCP

### 5.2 环境变量

```bash
cp .env.example .env
chmod 600 .env
```

至少填写：

- `OPENCODE_API_KEY`（或留空，从 `~/.local/share/opencode/auth.json` 读取）
- `SILICONFLOW_API_KEY`
- `JWT_SECRET`、`ADMIN_PASSWORD`、`PATIENT_PASSWORD`
- `HITL_CONFIRM_TOKEN`（演示默认 `confirm-demo`）

### 5.3 基础设施

```bash
docker compose up -d db redis
```

数据库 `127.0.0.1:5434`，Redis `127.0.0.1:6381`。

Mock 医院 MCP 二选一：

```bash
# A. 容器
docker compose up -d mcp

# B. 宿主机（部署现役方式）
cd backend
PYTHONPATH=. .venv/bin/uvicorn mcp_mock.server:app --host 127.0.0.1 --port 9100
```

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

### 5.5 前端（开发）

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

打开 `http://localhost:3000`。如果 `AUTH_ENABLED=true`，先登录。

### 5.6 前端（生产静态导出）

```bash
cd frontend
NEXT_PUBLIC_API_BASE= npm run build
# 产物在 out/，由 106 nginx 直接托管
```

`NEXT_PUBLIC_API_BASE` 为空表示同源 `/api`，由 nginx 反代到后端。

### 5.7 本地 0.6B

模型已在 WinPC 导入为 `qwen3-0.6b-dpo-v11_2`（Q8_0 GGUF）。服务器通过 SSH 隧道：

```bash
ssh -N -L 11434:127.0.0.1:11434 winpc
```

`.env`：

```ini
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen3-0.6b-dpo-v11_2
```

用途：本地分析、云端失败兜底；不建议作为主诊断模型。

---

## 6. 106 部署

现役线上：

```text
https://hospital.kk.zhzepei.cn/
```

组成：

- 106 上 `/var/www/hospital`：`frontend/out` 静态产物；
- nginx `hospital.kk.zhzepei.cn`：80 → 443 跳转，443 托管静态文件；
- `/api/` → `127.0.0.1:18087`（反向 SSH 隧道到家服务器 8020）；
- 家服务器 systemd：`smart-hospital-api`、`smart-hospital-mcp`、`smart-hospital-tunnel`、`smart-hospital-ollama-tunnel`；
- Let's Encrypt 证书：`/etc/letsencrypt/live/hospital.kk.zhzepei.cn/`，自动续期。

详细步骤与回滚见 [`docs/DEPLOYMENT-106.md`](docs/DEPLOYMENT-106.md)。

---

## 7. 数据与种子

`backend/scripts/seed.py` 默认读取 `data/medical_gold_*.jsonl`：

```bash
PYTHONPATH=. python scripts/seed.py --path ../data/medical_gold_final_v4.jsonl --limit 500
```

每条记录生成：

- `title` = 问题；
- `content` = `vector_text` 或答案；
- `department` = 科室；
- `embedding` = Qwen3-Embedding-8B（4096 维）；
- `tsv` = jieba 分词全文索引。

当前演示库：527 条文档。仓库不包含 MSD 原文；医疗数据按各自来源条款使用。

---

## 8. API

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/v1/auth/login` | 登录，返回 JWT |
| `GET` | `/api/v1/auth/me` | 当前用户/角色 |
| `GET` | `/api/v1/family` | 家庭成员（首次自动创建本人） |
| `POST/PATCH/DELETE` | `/api/v1/family/{id}` | 家庭档案 |
| `POST` | `/api/v1/chat/stream` | 问诊，SSE |
| `GET` | `/api/v1/home/summary` | 首页聚合 |
| `GET` | `/api/v1/health/timeline` | 健康时间线 |
| `GET/POST` | `/api/v1/medications` | 用药清单 |
| `POST` | `/api/v1/medications/check` | DDI / 妊娠禁忌 |
| `POST` | `/api/v1/reports/interpret` | 报告/皮肤照片视觉解读 |
| `GET` | `/api/v1/reports` | 报告记录 |
| `GET/POST/PATCH` | `/api/v1/followups` | 随访 |
| `GET` | `/api/v1/services/doctors` | 医生卡 |
| `GET` | `/api/v1/services/visit-guide` | 就诊指引 |
| `GET/POST` | `/api/v1/booking/*` | 科室/号源/锁号/订单/支付 |
| `POST` | `/api/v1/analytics/event` | 埋点 |
| `POST` | `/api/v1/eval/run` | 评测 |
| `GET` | `/admin` | 运营后台（管理员） |
| `GET` | `/health` / `/metrics` | 健康 / Prometheus |

SSE 事件：

```text
session → status → thought → citation → token → final → [DONE]
```

---

## 9. 评测与测试

### 自动化

| 项 | 结果 |
|---|---|
| 后端单元/接口测试 | **15 passed**（`cd backend && pytest -q`） |
| 后端 lint | **ruff 全绿** |
| 前端类型检查 | **通过** |
| 前端生产构建 | **通过**（Next.js 16.4.5，静态导出） |
| CI | GitHub Actions：backend lint/test + frontend build |

### 线上真实链路（2026-11-08）

在 `https://hospital.kk.zhzepei.cn/` 用真实模型完成：

- 患者登录 → 首页任务中心 → 三步问诊 → DeepSeek V4.1 Flash 分诊结果；
- 证据抽屉打开医学来源；
- 结果页「去挂号」→ 选医生/号源 → 锁号 → 订单 → HITL 支付 → 就诊指引；
- 报告上传 → 视觉模型解读 → 异常指标/建议问医生；
- 药管家：华法林 + 阿司匹林 → major；
- 健康时间线、家庭档案、运营后台、移动端；
- 管理员登录与患者登录分别验证。

### 检索/危机评测

| 项 | 结果 | 说明 |
|---|---|---|
| 检索冒烟 | Recall@5 **1.0**，MRR@5 **0.86**（n=30） | **in-corpus**：query 取文档标题，答案在同一索引内；不是留出集 |
| 危机检测 | precision **1.0**，recall **1.0**（n=500） | 关键词表使用了该 benchmark 的失败样本扩充，**in-sample**，不是盲测 |

**下一步**：200–500 条留出集、四条基线（BM25/向量/混合/+图）、引用准确率、图通道消融。旧版 92% Recall 等数字不再使用。

---

## 10. 安全与限制

- **鉴权**：线上 `AUTH_ENABLED=true`；管理员/患者账号为强随机密码，存于服务器 600 权限凭据文件，不进入仓库。
- **限流**：按 `X-Forwarded-For` / `X-Real-IP` 的真实 IP 计数，避免隧道把所有人算成一个 IP。
- **支付**：`HITL_CONFIRM_TOKEN` 人工确认；当前为 mock 医院 MCP。
- **医疗边界**：不确诊、不开处方；危机情况直接建议 120/急诊。
- **数据**：报告图片保存在服务器 `backend/uploads/`（gitignore）；健康数据按用户隔离。
- **未验证**：真实医院 MCP、生产级高并发、长会话、临床正确性、完整 DDI 数据库。
- **旧前端**：`frontend-legacy/` 仅保留代码，不启动。

---

## 11. 目录结构

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
│   │   ├── routes.py            # 家庭/用药/报告/随访/健康/服务/埋点
│   │   ├── eval_runner.py
│   │   └── main.py
│   ├── mcp_mock/server.py       # mock 医院 MCP
│   ├── scripts/                 # seed / eval
│   ├── tests/                   # 15 个测试
│   └── pyproject.toml
├── frontend/                    # v3 新前端（Next.js 16 / React 19 / Tailwind 4）
│   ├── app/                     # 首页 / 诊室 / 报告 / 药管家 / 健康 / 服务 / 我的 / 登录 / 后台
│   ├── components/              # AppShell / ResultView / FamilySwitcher / TaskCard
│   └── lib/                     # API / SSE / AppContext
├── frontend-legacy/             # v2 旧前端（保留代码，不启动）
├── data/                        # medical_gold / benchmarks
├── docs/
│   ├── screenshots/             # 线上真实截图
│   ├── DEPLOYMENT-106.md
│   └── UPGRADE_PLAN_2026H2.md
├── docker-compose.yml
└── .env.example
```

---

## 12. 许可与数据使用

- 代码：仓库未声明 License；如需开源请补 MIT / Apache-2.0。
- 医疗数据：按各自来源条款使用，不随代码重新授权。
- 模型：DeepSeek / Qwen 遵循官方许可；本地 0.6B 为你自己的微调产物。
- 医院 MCP：接入真实系统前完成安全评估、鉴权与对账。

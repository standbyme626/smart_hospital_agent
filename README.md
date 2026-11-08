# smart_hospital_agent v3.1

**一个有证据的就医助手**：分诊 · 报告解读 · 用药检查 · 挂号支付 · 家庭健康档案

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)]()
[![Next.js](https://img.shields.io/badge/Next.js-16.4-000000?logo=nextdotjs&logoColor=white)]()
[![MCP](https://img.shields.io/badge/MCP-Hospital%20Tools-6E56CF)]()
[![Live](https://img.shields.io/badge/live-hospital.kk.zhzepei.cn-0E9384)](https://hospital.kk.zhzepei.cn/)

> **线上演示**：https://hospital.kk.zhzepei.cn/ （登录后使用，演示账号由管理员发放）
> **版本状态（2026-11-09）**：线上仍是 v3.0 旧进程；v3.1 已完成本地实现与自动化验证，尚未部署。部署清单与剩余项见 [`docs/V3.1-RELEASE.md`](docs/V3.1-RELEASE.md)。
> **前端 v3**：2026 H2 消费级医疗 AI 界面，浅色「温暖临床 + 证据优先」；旧前端保留在 `frontend-legacy/`，不参与启动。
> **唯一 mock**：医院 MCP（号源/订单/支付/就诊指引）。LLM、Embedding、Rerank、报告视觉解读均为真实调用；真实医院 MCP 尚未接入，排期 v3.2。

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

### v3.1（2026-11，本地完成、待部署）

| 能力 | 交付 |
|---|---|
| 真身份隔离 | `users` 表 + scrypt 密码哈希；JWT `sub` 作为唯一身份；家庭、报告、用药、随访、会话、订单全部按 `user_id` 归属校验；旧 token 随用户禁用/删除失效；登录失败锁定 |
| 真 HITL 审核 | 患者「申请支付」→ 管理员审核队列批准/拒绝 → 服务端带 `HITL_CONFIRM_TOKEN` 调 MCP `pay_order`；患者直调 pay 固定 403；重复批准幂等 |
| 多轮会话与恢复 | `messages` 持久化每轮 user/assistant；`GET /api/v1/sessions/{id}` 恢复历史与最近 final；`/consult/?session=` 刷新不丢；追问进入同一会话 |
| 报告生命周期 | 报告列表/详情/图片（Bearer 鉴权 blob）/删除；删除同时清理图片文件与健康事件；上传失败不留孤儿行/文件 |
| 数据权利 | `GET /api/v1/me/export` 导出本人 JSON + 报告图片 zip；`DELETE /api/v1/me` 删除/匿名化本人数据并注销 token |
| 家庭维度 | 家庭成员级过滤（`member_id`）；删除成员级联清理报告/用药/随访/健康事件/会话/图片；医院患者号由服务端映射 |
| 订单持久化 | mock MCP 锁号/订单/已订表落 Postgres；API 重启不丢；订单列表/详情/恢复/取消；取消释放号源，已支付不可取消 |
| 证据原文 | `GET /api/v1/rag/documents/{doc_key}` 返回完整正文 + 来源，仅登录用户可读；21 个规范科室回填与无结果回退 |
| 配额与后台 | `CHAT_DAILY_LIMIT` 按用户自然日限制问诊次数，超限 429；`/admin` 页面 Basic Auth + 管理 API JWT/角色双门禁 |
| 可回归测试 | 96 个后端测试（含 v3.1 集成/契约）、7 个前端单测、3 条 Playwright E2E；全部使用 fake gateway / fake model server，不调用真实模型 |

### v3.0 前端 P0–P4（既有）

| 阶段 | 交付 |
|---|---|
| P0 视觉与 IA | 浅色温暖临床风、移动端底部导航、桌面侧栏、首页任务中心、分诊结果页、证据抽屉、登录/同意 |
| P1 AI 诊室 + 报告 | 三步问诊（就诊人 → 症状 → 病史）、一屏一问、红旗筛查、报告/皮肤照片上传、视觉解读、家庭档案 |
| P2 药管家 + 随访 | 用药清单、相互作用/妊娠禁忌、诊后随访待办、健康时间线 |
| P3 服务 + 无障碍 | 医生卡、号源、锁号、订单、支付、就诊指引；大字体/高对比/减少动效 |
| P4 数据 | 首访/任务/问诊/报告/支付/证据打开等埋点，运营后台汇总 |

### 后端

- 新增：用户/审核队列/消息表、家庭档案、用药、报告解读、随访、健康时间线、医生卡/就诊指引、埋点、首页聚合。
- 单 Agent 循环 + 工具 + 确定性策略；**无自规划/多 Agent/长期记忆向量库**；SSE `session/status/thought/citation/token/final/booking/error`。
- PostgreSQL 16 + pgvector；JWT + RBAC + CORS 白名单 + 按真实 IP 限流 + 审计 + 归属校验 + 每日问诊配额。
- 模型网关：opencode-go、SiliconFlow、WinPC Ollama（Qwen3-0.6B DPO）。
- 医院 MCP：本地 mock 适配器，锁号/订单/已订落 Postgres，重启不丢；真实医院接口留到 v3.2。

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
  ├─ 医院 MCP：mock server :9100（mock 内部订单也落 Postgres）
  └─ PostgreSQL 16 + pgvector / users / messages / review_requests / 审计 / 埋点
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
| AI 诊室 | 三步问诊、一屏一问、红旗筛查、结果页、证据抽屉、多轮追问 | `frontend/app/consult/page.tsx`、`components/ResultView.tsx` |
| 会话 | 消息持久化、会话列表/详情/事件、刷新恢复 | `backend/app/routes.py`、`session_store.py`、`conversation.py` |
| 分诊 | 危机规则 + 科室关键词 + LLM JSON 分诊 | `backend/app/policy.py`、`prompts.py` |
| 检索 | Dense + BM25 + RRF(60) + Qwen3-8B 重排；科室规范化/回退 | `backend/app/retrieval.py`、`backend/scripts/backfill_departments.py` |
| 证据 | 证据抽屉 + `GET /api/v1/rag/documents/{doc_key}` 全文 | `frontend/components/ResultView.tsx`、`backend/app/routes.py` |
| 报告解读 | 图像上传 → 视觉模型 → 摘要/异常指标/问医生问题；图片鉴权读取、删除 | `backend/app/routes.py`、`frontend/app/reports/page.tsx` |
| 药管家 | 用药清单、DDI/妊娠禁忌检查 | `backend/app/data/drug_interactions.json`、`frontend/app/meds/page.tsx` |
| 挂号支付 | MCP 科室/医生/号源/锁号/订单/恢复/取消；患者申请 → 管理员审核 → 服务端支付 | `backend/app/mcp_client.py`、`reviews.py`、`backend/mcp_mock/server.py` |
| 健康档案 | 家庭、时间线、随访、用药/报告/挂号聚合；成员级过滤与级联删除 | `backend/app/routes.py`、`frontend/app/health/page.tsx` |
| 数据权利 | 本人数据导出 zip、账号删除/匿名化 | `backend/app/routes.py`、`frontend/app/profile/page.tsx` |
| 安全 | scrypt 密码 + JWT + RBAC + CORS + 限流 + 登录锁定 + 每日配额 + 审计 + HITL | `backend/app/security.py`、`api/deps.py`、`main.py` |
| 后台 | 审核队列、用户管理、审计、模型成本、埋点；页面 Basic Auth + API JWT/角色 | `backend/app/routes.py`、`frontend/app/admin/page.tsx` |
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
- `HITL_CONFIRM_TOKEN`（占位符 `change-me`，真实值只放服务器 600 凭据文件）

v3.1 相关变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTH_ENABLED` | `true` | 生产必须为 `true` |
| `CHAT_DAILY_LIMIT` | `60` | 每用户每自然日问诊消息数；`0` = 关闭配额；超限返回 429 |
| `JWT_TTL_MINUTES` | `720` | JWT 有效期；用户禁用/删除后旧 token 失效 |
| `RATE_LIMIT_PER_MINUTE` | `60` | 按真实 IP 的每分钟请求数 |
| `MAX_UPLOAD_MB` | `8` | 报告/皮肤图片上传上限；nginx `client_max_body_size` 需不小于此值 |
| `UPLOADS_DIR` | `backend/uploads` | 报告图片目录；必须与备份脚本覆盖范围一致 |
| `HITL_CONFIRM_TOKEN` | `change-me` | 管理员批准支付时服务端调 MCP 的确认 token，不下发前端 |
| `ADMIN_USER` / `PATIENT_USER` | `admin` / `patient` | 首次启动且 `users` 表为空时 bootstrap；已有用户不会重建 |

启动时 `db.init_schema()` 自动执行 `CREATE TABLE IF NOT EXISTS` + `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 幂等迁移，无需手工 SQL。首次部署会用 `ADMIN_PASSWORD` / `PATIENT_PASSWORD` 创建 bootstrap 用户；之后改密码用 `backend/scripts/set_password.py`。

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
# 轻量探活：不访问 DB/模型/MCP，供 Kuma/负载均衡
curl -s http://127.0.0.1:8020/api/health

# 深度自检：DB / 文档数 / 模型 / MCP
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

**v3.1 部署状态**：截至 2026-11-09，线上仍为 v3.0；v3.1 仅在本地提交（分支 `v3.1`），未改 106 nginx、未重启生产服务、未轮换生产密码。v3.1 的部署顺序、迁移幂等、备份/回滚和剩余验收项见 [`docs/V3.1-RELEASE.md`](docs/V3.1-RELEASE.md) 与 [`docs/DEPLOYMENT-106.md`](docs/DEPLOYMENT-106.md)。

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
| `POST` | `/api/v1/auth/login` | 登录，返回 JWT；连续失败锁定 |
| `GET` | `/api/v1/auth/me` | 当前用户/角色/启用状态 |
| `GET/POST` | `/api/v1/family` | 家庭成员（首次自动创建本人）/ 新增 |
| `PATCH/DELETE` | `/api/v1/family/{member_id}` | 修改 / 级联删除成员数据；不能删本人 |
| `POST` | `/api/v1/chat/stream` | 问诊，SSE；带 `session_id` 进入同一多轮会话 |
| `GET` | `/api/v1/sessions` | 会话列表（可按 `member_id` 过滤） |
| `GET` | `/api/v1/sessions/{id}` | 历史消息 + 最近 final 报告（本人） |
| `GET` | `/api/v1/sessions/{id}/events` | 事件增量 |
| `GET` | `/api/v1/home/summary` | 首页聚合 |
| `GET` | `/api/v1/health/timeline` | 健康时间线 |
| `GET/POST` | `/api/v1/medications` | 用药清单 / 新增 |
| `PATCH/DELETE` | `/api/v1/medications/{id}` | 停用 / 删除 |
| `POST` | `/api/v1/medications/check` | DDI / 妊娠禁忌 |
| `POST` | `/api/v1/reports/interpret` | 报告/皮肤照片视觉解读 |
| `GET` | `/api/v1/reports` | 报告记录（`has_image`，不返回服务器路径） |
| `GET` | `/api/v1/reports/{id}` | 报告详情 |
| `GET` | `/api/v1/reports/{id}/image` | 报告图片（Bearer，仅本人） |
| `DELETE` | `/api/v1/reports/{id}` | 删除报告 + 图片 + 健康事件 |
| `GET/POST` | `/api/v1/followups` | 随访列表 / 新增 |
| `PATCH` | `/api/v1/followups/{id}/complete` | 完成随访 |
| `GET` | `/api/v1/services/doctors` | 医生卡 |
| `GET` | `/api/v1/services/visit-guide` | 就诊指引 |
| `GET` | `/api/v1/booking/departments` | 科室 |
| `GET` | `/api/v1/booking/schedules` | 号源 |
| `POST` | `/api/v1/booking/lock` | 锁号 |
| `GET/POST` | `/api/v1/booking/orders` | 订单列表（重启/刷新可恢复）/ 创建 |
| `GET` | `/api/v1/booking/orders/{id}` | 订单状态 + 审核记录 |
| `POST` | `/api/v1/booking/orders/{id}/request-payment` | 患者申请支付审核（幂等） |
| `POST` | `/api/v1/booking/orders/{id}/pay` | 固定 403：患者不能直接支付 |
| `POST` | `/api/v1/booking/orders/{id}/cancel` | 取消未支付订单并释放号源 |
| `GET` | `/api/v1/rag/status` | 检索/模型/MCP 状态（公开） |
| `GET` | `/api/v1/rag/documents/{doc_key}` | 证据全文 + 来源（登录） |
| `GET` | `/api/v1/departments` | 21 个规范科室 |
| `POST` | `/api/v1/drug/check` | 药物文本检查 |
| `GET` | `/api/v1/me/export` | 导出本人数据 zip（JSON + 报告图片） |
| `DELETE` | `/api/v1/me` | 删除/匿名化本人账号；需密码确认 |
| `POST` | `/api/v1/analytics/event` | 埋点 |
| `GET` | `/api/v1/admin/reviews` | 待审核队列（admin） |
| `POST` | `/api/v1/admin/reviews/{id}/approve` / `reject` | 批准/拒绝支付（admin，幂等） |
| `GET/POST/PATCH` | `/api/v1/admin/users` / `{id}` | 用户管理（admin） |
| `GET` | `/api/v1/admin/model-calls` / `audit` / `analytics` | 成本/审计/埋点（admin） |
| `POST/GET` | `/api/v1/eval/run` / `eval/results` | 评测（admin） |
| `GET` | `/admin` | 运营后台静态页（部署后由 nginx Basic Auth 保护） |
| `GET` | `/api/health` / `/health` / `/metrics` | 轻量探活 / 深度自检 / Prometheus |

SSE 事件：

```text
session → status → thought → citation → token → final → [DONE]
```

管理接口鉴权：`/api/v1/admin/*` 一律 JWT + `role=admin`，**不要在 nginx 上对 `/api/v1/admin/` 加 Basic Auth**（会吞掉浏览器 `Authorization: Bearer` 头）。`/admin` 静态页可由 nginx Basic Auth 保护；需要网络层加固时用 Cloudflare Access 或 IP allowlist。

---

## 9. 评测与测试

### 自动化

| 项 | 结果（2026-11-09 本地） |
|---|---|
| 后端单元/接口/集成测试 | **96 passed**（`cd backend && .venv/bin/pytest -q`） |
| 后端 lint | **ruff 全绿**（`ruff check app mcp_mock tests`） |
| 前端单元测试 | **7 passed**（`npm run test:unit`） |
| 前端类型检查/生产构建 | **通过**（`npm run build`，Next.js 静态导出） |
| Playwright E2E | **3 passed**（问诊+追问+恢复、报告上传+成员隔离、锁号→申请支付→管理员批准→指引） |
| CI | GitHub Actions：backend lint/test + frontend unit/build/secret gate + Playwright e2e |

#### 测试数据库

```bash
docker compose up -d db

# 后端测试库
docker exec sha_v2_db psql -U sha -d postgres -c "CREATE DATABASE sha_test"
docker exec sha_v2_db psql -U sha -d sha_test -c "CREATE EXTENSION IF NOT EXISTS vector"

# E2E 测试库
docker exec sha_v2_db psql -U sha -d postgres -c "CREATE DATABASE sha_e2e"
docker exec sha_v2_db psql -U sha -d sha_e2e -c "CREATE EXTENSION IF NOT EXISTS vector"
```

#### 测试命令

```bash
# 后端 lint + 96 个测试（不调用真实模型）
cd backend
.venv/bin/ruff check app mcp_mock tests
TEST_DATABASE_URL=postgresql://sha:sha_pwd@127.0.0.1:5434/sha_test .venv/bin/pytest -q

# 前端单测 + 构建
cd frontend
npm ci
npm run test:unit
NEXT_PUBLIC_API_BASE= npm run build

# Playwright E2E（fake model + mock MCP + 独立 sha_e2e；默认端口 8020/3000）
npx playwright install chromium
E2E_DATABASE_URL=postgresql://sha:sha_pwd@127.0.0.1:5434/sha_e2e npm run test:e2e

# 本机 8020/3000 已被占用时（例如生产 API / Grafana）
E2E_BACKEND_PORT=8021 E2E_FRONTEND_PORT=3010 NEXT_DIST_DIR=.next-e2e \
  E2E_DATABASE_URL=postgresql://sha:sha_pwd@127.0.0.1:5434/sha_e2e npm run test:e2e
```

全部测试使用 `backend/tests/fake_gateway.py`、`scripts/fake_model_server.py` 和本地 mock MCP，**不产生真实模型费用**。

### v3.1 验证状态（2026-11-09）

**本地已通过**

- 后端 96 tests、ruff、前端 7 unit、`npm run build`、3 条 Playwright E2E。
- 契约测试：`frontend/lib/api.ts` 每个调用都能在 FastAPI OpenAPI 找到；旧演示密码不再出现于构建产物/`.env.example`/部署文档。
- 迁移幂等：重复启动多次执行 `init_schema()` + `MIGRATIONS` 无报错。

**待部署后才算通过（当前未绿灯）**

- 生产库备份 + v3.1 迁移 + API 重启（当前线上仍是 v3.0，`/api/health` 返回 404）。
- 106 nginx v3.1 示例审核/安装（`deploy/nginx-hospital-v31.conf` 未应用），之后 `/admin` Basic Auth 才生效。
- Kuma 监控项启用；线上 `/api/health` 200。
- 生产密码轮换与恢复演练（需主人批准）。
- 完整清单与 AC 映射见 [`docs/V3.1-RELEASE.md`](docs/V3.1-RELEASE.md)。

### 备份与恢复

```bash
# 手工备份（pg_dump custom + uploads tar；目录 700/文件 600，保留 7 天）
bash scripts/backup.sh

# systemd timer 每天 03:30；手工触发
systemctl start smart-hospital-backup.service
systemctl list-timers smart-hospital-backup.timer
```

产物在 `/home/data/backups/smart-hospital/`：`db-<时间戳>.dump` + `uploads-<时间戳>.tar.gz`。恢复步骤见 [`scripts/restore.md`](scripts/restore.md)：先停 API，再 `pg_restore --clean`，最后解包 `backend/uploads` 并验证 `/api/health`。恢复演练未在本次本地任务中执行，发布前需补一次。

### 线上真实链路（v3.0 基线，2026-11-08）

在 `https://hospital.kk.zhzepei.cn/` 用真实模型完成：

- 患者登录 → 首页任务中心 → 三步问诊 → DeepSeek V4.1 Flash 分诊结果；
- 证据抽屉打开医学来源；
- 结果页「去挂号」→ 选医生/号源 → 锁号 → 订单 → 人工确认支付 → 就诊指引（v3.0 流程；v3.1 改为「患者申请 → 管理员审核 → 服务端支付」）；
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

- **鉴权**：生产必须 `AUTH_ENABLED=true`；账号密码存 `users` 表（scrypt 哈希），管理员/患者初始密码只放服务器 600 权限凭据文件，不进入仓库。JWT 过期或用户禁用/删除即失效。
- **后台访问模型**：`/admin` 静态页可用 nginx Basic Auth 保护；`/api/v1/admin/*` 只用 JWT + `role=admin`。**不要在 `/api/v1/admin/` 上加 Basic Auth**（会与浏览器 Bearer 冲突）；要网络层加固用 Cloudflare Access 或 IP allowlist。
- **限流/配额**：按真实 IP 每分钟限流；`CHAT_DAILY_LIMIT` 按用户自然日限制问诊，超限 429。
- **支付**：v3.1 为「患者申请 → 管理员批准 → 服务端带 `HITL_CONFIRM_TOKEN` 调 MCP」；患者不能直接支付；当前医院 MCP 仍为 mock。
- **医疗边界**：不确诊、不开处方；危机情况直接建议 120/急诊。
- **数据**：报告图片保存在服务器 `backend/uploads/`（gitignore）；所有资源按 JWT 用户归属隔离；导出/删除按本人执行。备份含患者数据，禁止外传。
- **明确限制**：真实医院 MCP 未接（v3.2）；**无自规划/多 Agent**，保持单 Agent 循环 + 确定性策略；不做多租户/SaaS/自注册/医生工作台；生产按单 uvicorn worker 设计。
- **未验证**：真实医院 MCP、生产级高并发、长会话上限、临床正确性、完整 DDI 数据库、多租户。
- **旧前端**：`frontend-legacy/` 仅保留代码，不启动。

---

## 11. 目录结构

```text
smart_hospital_agent/
├── backend/
│   ├── app/
│   │   ├── api/deps.py          # JWT / RBAC
│   │   ├── security.py          # scrypt / bootstrap / 登录锁定
│   │   ├── agent.py             # 单 Agent 循环
│   │   ├── model_gateway.py     # opencode-go / SiliconFlow / Ollama
│   │   ├── retrieval.py         # pgvector + BM25 + RRF + rerank
│   │   ├── policy.py            # 危机 / 科室 / 药物 / 安全门禁
│   │   ├── tools.py             # 工具注册与执行
│   │   ├── mcp_client.py        # 医院 MCP JSON-RPC
│   │   ├── reviews.py           # 支付审核队列
│   │   ├── session_store.py     # 消息 / 事件 / 审计 / 调用记录
│   │   ├── routes.py            # 家庭/用药/报告/订单/数据权利/后台
│   │   ├── eval_runner.py
│   │   └── main.py
│   ├── mcp_mock/server.py       # mock 医院 MCP（订单持久化）
│   ├── scripts/                 # seed / eval / set_password / backfill_departments
│   ├── tests/                   # 96 个后端测试（含 v3.1 集成/契约）
│   └── pyproject.toml
├── frontend/                    # v3 新前端（Next.js 16 / React 19 / Tailwind 4）
│   ├── app/                     # 首页 / 诊室 / 报告 / 药管家 / 健康 / 服务 / 我的 / 登录 / 后台
│   ├── components/              # AppShell / ResultView / ReviewQueue / FamilySwitcher / TaskCard
│   ├── e2e/                     # Playwright v3.1 冒烟
│   ├── scripts/                 # 前端单测
│   └── lib/                     # API / SSE / AppContext
├── frontend-legacy/             # v2 旧前端（保留代码，不启动）
├── data/                        # medical_gold / benchmarks
├── deploy/
│   └── nginx-hospital-v31.conf  # v3.1 示例，未应用
├── scripts/
│   ├── backup.sh / restore.md   # 每日备份 + 恢复步骤
│   ├── fake_model_server.py     # E2E fake OpenAI
│   └── e2e_reset.py
├── docs/
│   ├── screenshots/             # 线上真实截图
│   ├── DEPLOYMENT-106.md
│   ├── V3.1-ACCEPTANCE.md
│   ├── V3.1-RELEASE.md          # 发布清单 / 验收映射 / 剩余项
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

# smart_hospital_agent

**在真实医院信息系统（HIS）之上构建的医疗 Agent 平台**：分诊 · 报告解读 · 用药检查 · 挂号支付 · 医生工作台 · 运营与平台管理。

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white)]()
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)]()
[![MCP](https://img.shields.io/badge/MCP-HIS%20Tools-6E56CF)]()

> **脱敏说明**：本仓库不包含真实部署域名、服务器地址、账号口令、内网路径与第三方凭据。文中以 `<DOMAIN>`、`<API_HOST>`、`<HIS_HOST>` 等占位；实际部署信息见内部运维文档。
>
> **当前状态（2026-11-10）**：v3.3 滚动上线中。患者端主链路、医生端工作台、运营台、通知中心、随访已上线；**195 个后端测试 + 7 条 Playwright E2E 全绿**，ruff 干净，前端类型检查与构建通过。

---

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 核心特性](#2-核心特性)
- [3. 界面与原型](#3-界面与原型)
- [4. 系统架构](#4-系统架构)
- [5. 关键设计](#5-关键设计)
- [6. 快速开始](#6-快速开始)
- [7. API](#7-api)
- [8. 数据模型](#8-数据模型)
- [9. 测试与质量](#9-测试与质量)
- [10. 部署（脱敏）](#10-部署脱敏)
- [11. V3.3 进度](#11-v33-进度)
- [12. 限制与合规](#12-限制与合规)
- [13. 目录结构](#13-目录结构)
- [14. 许可](#14-许可)

---

## 1. 项目概览

### 1.1 要解决的问题

医院已经有一套稳定的 HIS，缺的不是"再做一个系统"，而是：

- 患者看不懂报告、不知道该挂哪个科、挂号缴费流程长；
- 医生要在 HIS 与各种资料之间来回切换，诊前准备与文书耗时；
- AI 能力如果没有接入真实数据与流程，只能停留在聊天窗口。

本项目的定位是 **HIS 之上的智能层**：HIS 是唯一事实源（记录系统），平台负责理解、检索、草稿、通知与运营，不复制一份"真相"。

### 1.2 方案

- **一个 Agent 内核 + 四个角色入口**：患者端、医生端、运营/质控端、管理端；
- **MCP 作为唯一接入面**：所有对 HIS 的读写都经过适配器（工具化、幂等、可对账）；
- **AI 只读临床、写临床必须医生签名**：草稿与签名分离；
- **确定性策略优先**：危机识别、挂号规则、支付放行、权限校验都先用规则，LLM 只做兜底；
- **可验收**：每条链路对应验收剧本，全绿才允许发布。

### 1.3 四个入口

| 入口 | 主要用户 | 核心任务 | 状态 |
|---|---|---|---|
| 患者端 | 患者 / 家属 | 分诊、报告解读、用药检查、挂号缴费、随访、消息 | 已上线（主链路） |
| 医生端 `/clinician` | 门诊医生 | 今日队列、患者 360、病历草稿（签名待 HIS 接入） | 只读 + 草稿已上线 |
| 运营 / 质控端 | 医务科、运营 | 支付异常、对账、接管、审计 | 原型已定稿，接口待接 |
| 管理端 `/platform` | 平台运维 / 知识库管理员 | 运行总览、开关配置、能力矩阵（只读；发布与回滚在 S12） | 只读已上线 |

> 四个入口的完整交互稿见 [`prototype/dual-end.html`](prototype/dual-end.html)（61 屏，单文件、可点击；纯演示数据）。

---

## 2. 核心特性

**患者侧**

- 一屏一问的分诊流程，红旗信号阻断式引导（120 / 急诊），不等待模型输出；
- 报告解读：异常指标表格化、危急值横幅、建议问医生清单；不确诊、不开处方；
- 药管家：相互作用与妊娠禁忌检查，**未覆盖药物明确提示"不能据此判断安全"**；
- 挂号：HIS 实时号源（当日挂号）、锁号与订单、待缴费→已登记→就诊指引；
- 消息中心：站内通知（随访、报告），支持单条/全部已读；
- 随访：新增/完成随访、按状态筛选；
- 家庭档案、健康时间线、数据导出与删除。

**医生侧（只读 + 草稿）**

- 今日队列（按日）、患者 360（过敏、既往、用药、报告、随访、挂号历史）；
- 病历草稿：本地保存、可编辑；**签名默认关闭**，需医生 HIS 账号透传后才允许写入；
- 所有写操作预留二次确认与审计位，当前不自动签名。

**运营 / 质控侧（`/ops`）**

- 概览：待审核、对账差异、待接管、今日已支付；
- 对账差异：列表、严重度、一键标记已处理（写回 `reconciliation_finding`）；
- 人工接管：接管队列与认领（`human_task`），全程留痕；
- 角色门禁：仅 `ops` / `admin` 可访问，患者调用返回 403。

**平台侧**

- 通知中心：通知表 + 已读状态，随访/报告触发自动写入；
- 医院适配器：MCP 工具、幂等 outbox、只读池、对账 finding；
- 平台管理只读：`/platform` 汇总应用版本、Schema、数据规模、开关（鉴权/医生签名/限流/配额）与能力矩阵；
- 验收脚本：26 条剧本（P/D/E2E/INT）定义在规划文档中。

---

## 3. 界面与原型

- 真实前端截图：[`docs/screenshots/`](docs/screenshots)（首页、问诊结果、证据、挂号、报告、药管家、健康、我的、登录、后台、移动端）；
- 四端完整原型：[`prototype/dual-end.html`](prototype/dual-end.html) —— 61 屏，含患者端 12、医生端 8、运营 5、管理 17，以及配置总览、未命中问题、接管队列、验收自测等；
- 原型截图：[`prototype/shots/`](prototype/shots)。

> 原型使用模拟数据，页面顶部有「演示环境」提示；不代表真实数据或真实支付。

---

## 4. 系统架构

### 4.1 总览

```text
浏览器 / 移动端
   │ HTTPS
   ▼
反向代理 <DOMAIN>  ──静态前端（Next.js 导出）
   │ /api/ 反向隧道
   ▼
应用服务 <API_HOST>:<PORT>（FastAPI，单进程）
   ├─ 角色网关：JWT + scope + 限流
   ├─ Agent 内核：单循环 + 工具 + 策略门禁
   ├─ 能力层：RAG（pgvector + BM25 + 重排）、DDL/规则、模型网关
   ├─ 会话与审计：会话、事件、审计日志、通知
   └─ MCP 客户端 ──► HIS 适配器 <HIS_ADAPTER>:<PORT>
                        ├─ 工具：患者/号源/挂号/缴费/处方/就诊
                        ├─ 幂等 outbox + 只读池 + 对账 finding
                        └─ HIS <HIS_HOST>（真实医院系统）
```

### 4.2 一次问诊的完整过程

1. 患者输入症状 → 策略层先做**危机识别**（关键词 + 规则），命中则阻断并引导 120；
2. 通过后进入 Agent 循环：检索证据 → 调用模型 → 生成结构化报告；
3. 引用必须来自检索结果且过相关度门槛，不足时显示"证据不足"；
4. 输出过安全门禁（确诊/处方/剂量表述拦截）；
5. 事件写库、SSE 推送、可选通知。

### 4.3 四端与权限域

| 域 | 使用者 | 数据范围 | 写权限 |
|---|---|---|---|
| `patient_read` / `patient_write` | 患者端 | 本人 + 家庭成员 | 事务（挂号/支付/上传/同意） |
| `clinician_read` | 医生端 | 当日队列与患者 360 | 只读 |
| `clinician_write` | 医生端（待启用） | 当班患者 | 接诊/医嘱/签名（需 HIS 透传） |
| `ops_*` / `admin_*` | 运营 / 管理端 | 业务或配置数据（脱敏） | 审批 / 配置（双人审批） |

### 4.4 数据与事实层

- **HIS 为唯一事实源**：本地只保存投影（`appointment_projection`、`his_*_map`）、草稿与审计；
- 投影由适配器同步，**不可反向修改 HIS**；
- 对账比较本地投影与 HIS 只读库，差异写入 `reconciliation_finding`。

---

## 5. 关键设计

### 5.1 HIS 是唯一事实源

任何"看起来像真相"的数据都必须能追到 HIS。本地表只是缓存与草稿；删除/更正以 HIS 为准。

### 5.2 医生签名与草稿分离

- 草稿保存在本地（`clinician_drafts`），可编辑、可废弃；
- **签名默认关闭**（`CLINICIAN_WRITE_ENABLED=false`）；未启用时签名返回 409 并说明原因；
- 启用后签名仍要求医生 HIS 账号透传，保证"谁签的字"可在 HIS 侧验证。

### 5.3 支付：自动放行 + 异常人工

- 正常支付（金额等于 HIS 目录价、回调验签通过、无重复）**自动放行**；
- 人工只处理：退款/改价、金额不一致、重复支付、超阈值、对账差异；
- 平台不经手资金，收款方是医院。

### 5.4 知识库：一个检索服务 + 五个语料域

按"语料域"而非"端"切分：`clinical-evidence`、`drug`、`patient-edu`、`hospital-pathway`、`platform-doc`。检索必须带角色 allowlist 与相关度门槛（患者 0.20 / 医生 0.35），越权域在召回前排除。

### 5.5 配置即代码与运行时快照

Prompt、规则、索引发版都生成版本号；每次会话记录当时的配置快照，可回放、可归因、可回滚。

---

## 6. 快速开始

### 6.1 前置

- Python 3.12、Node.js 22
- PostgreSQL 16 + pgvector（本地可用容器）
- 可选：医院 HIS（OpenHIS 或等价系统）与 MCP 适配器

### 6.2 环境变量（示例，**不含真实凭据**）

```bash
APP_VERSION=3.3.0-dev
ENV=dev
AUTH_ENABLED=true
JWT_SECRET=<随机长字符串>
DATABASE_URL=postgresql://<user>:<password>@<db-host>:5432/<db>
REDIS_URL=redis://<redis-host>:6379/0

# 模型
OPENCODE_API_BASE=<model-provider-base>
OPENCODE_API_KEY=<your-key>
SILICONFLOW_API_BASE=<embedding-base>
SILICONFLOW_API_KEY=<your-key>

# 医院接入
HOSPITAL_MCP_URL=http://<adapter-host>:9100/mcp
HIS_ADAPTER_MODE=mock            # mock | openhis
CLINICIAN_WRITE_ENABLED=false    # 医生签名总开关
```

完整示例见 [`.env.example`](.env.example)（仅占位符）。

### 6.3 本地启动

```bash
# 基础设施（按需）
docker compose up -d db redis

# 后端
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8020

# 前端
cd ../frontend
npm install
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8020 npm run dev
```

### 6.4 测试与构建

```bash
cd backend && ruff check app tests && pytest -q      # 189 passed
cd frontend && npm run typecheck && npm run test:unit && NEXT_PUBLIC_API_BASE="" npm run build
```

---

## 7. API

共 **81 个端点**，按前缀分组：

| 分组 | 代表端点 | 说明 |
|---|---|---|
| 认证 | `POST /api/v1/auth/login`、`GET /api/v1/auth/me` | JWT + 角色 |
| 问诊 | `POST /api/v1/chat/stream`、`GET /api/v1/sessions/{id}` | SSE、会话恢复 |
| 报告 | `POST /api/v1/reports/interpret`、`GET /api/v1/reports` | 上传与解读 |
| 用药 | `POST /api/v1/medications/check` | 相互作用/妊娠禁忌 |
| 挂号 | `GET /api/v1/booking/catalog`、`POST /api/v1/booking/appointments` | HIS 号源与订单 |
| 家庭 | `GET/POST/PATCH/DELETE /api/v1/family` | 成员级隔离 |
| 随访 | `GET/POST /api/v1/followups`、`PATCH .../complete` | 随访计划 |
| 通知 | `GET /api/v1/notifications`、`POST .../read`、`.../read-all` | 站内通知 |
| 医生端 | `GET /api/v1/clinician/queue`、`/patients/{id}`、`/drafts` | 只读 + 草稿 |
| 运营 | `GET /api/v1/ops/summary`、`/ops/reconciliation`、`/ops/handoff` | 对账差异与人工接管 |
| 管理 | `/api/v1/admin/*`、`/api/v1/admin/platform`、`/api/v1/admin/compliance`、`/api/v1/admin/health-summary`、`/api/v1/admin/evaluation`、`/api/v1/eval`、`/api/ready`、`/metrics` | 审核、平台总览、合规、告警、评测、健康 |

示例（占位符）：

```bash
TOKEN="<jwt>"
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://<DOMAIN>/api/v1/clinician/queue?date=$(date +%F)"
```

---

## 8. 数据模型

**34 张表**，按职责分组：

| 组 | 表 |
|---|---|
| 身份与权限 | `users`、`consent_record`、`audit_log` |
| 会话与事件 | `sessions`、`messages`、`events`、`analytics_events` |
| 健康数据 | `family_members`、`reports`、`medications`、`followups`、`health_events` |
| 挂号与支付 | `bookings`、`appointment_projection`、`review_requests`、`mcp_orders` |
| HIS 投影 | `his_patient_map`、`his_encounter_map`、`his_prescription_map`、`his_sync_cursor` |
| 同步与对账 | `sync_outbox`、`reconciliation_finding`、`human_task` |
| 医生端 | `clinician_drafts` |
| 通知 | `notifications` |
| 其他 | `documents`、`model_calls`、`rules`、`schema_migrations` 等 |

---

## 9. 测试与质量

| 项 | 现状 |
|---|---|
| 后端测试 | **195 passed**（含契约、集成、P0 安全、临床安全、HIS 适配器、通知、医生端、运营端、平台与合规） |
| 静态检查 | `ruff` 全绿 |
| 前端 | `tsc --noEmit` 通过；单测 20 个通过；`next build` 通过（18 个静态页面） |
| E2E | Playwright **7 passed**（见下） |

**E2E 覆盖（`frontend/e2e/`，使用独立测试库与 fake 模型，不触达真实医院）**

1. 问诊向导 → 分诊结果 → 会话恢复 → 追问；
2. 报告上传（fake vision）→ 指标表 → 家庭成员隔离；
3. 挂号报价 → 收银台支付 → 就诊指引；
4. 成员年龄随生日联动；
5. 随访创建 → 消息中心收到通知（后端通知表联动）；
6. 医生台 / 运营台权限：管理员可进、患者被拒（403/无权限页）；
7. 平台台 `/platform`：管理员可进（总览/开关/能力矩阵切换），患者被拒。

> E2E 在独立端口（如 8030/3030）运行，避免与本地生产 API 冲突；命令示例：
> `E2E_BACKEND_PORT=8030 E2E_FRONTEND_PORT=3030 npm run test:e2e`
| 覆盖重点 | 危机识别、DDI、权限越界、幂等、挂号状态机、通知已读 |

---

## 10. 部署（脱敏）

- 前端：静态导出 → 反向代理目录；**构建时 `NEXT_PUBLIC_API_BASE=""`** 走同源，避免把内网地址打进产物；
- 后端：`uvicorn app.main:app`，由进程守护管理；启动自动执行幂等迁移；
- 发布脚本：`scripts/deploy_frontend.sh`（真实站）与 `scripts/deploy_preview.sh`（演示站），每次发布前备份上一版本目录，可整目录回滚；
- 备份：`scripts/backup.sh`，支持加密与异地选项；
- **部署域名、服务器地址、凭据不在本仓库**。

---

## 11. V3.3 进度

| 切片 | 内容 | 状态 |
|---|---|---|
| S0 | 部署通道 + 演示站 | ✅ |
| S1 | 登录 + 首页 + 消息中心 | ✅ |
| S2 | AI 诊室（失败可重试） | ✅ |
| S3 | 报告解读（指标表、危急值） | ✅ |
| S4 | 药管家 | ✅ |
| S5 | 挂号支付（5 个红测试修复） | ✅ |
| S6 | 通知中心（后端 + 消息页） | ✅ |
| S7 | 随访 | ✅ |
| S8 | 医生端只读（队列 + 360） | ✅ |
| S9 | 医生端写入（草稿 + 签名门禁） | ✅ 草稿；签名待 HIS 透传 |
| S10 | 运营端（概览 / 对账差异 / 人工接管） | ✅ |
| S11 | 管理端只读（`/platform`） | ✅ |
| S12 | 合规与数据 / 告警与 SLO / 质量评测 | ✅（实验与灰度留待后续） |

详细计划与完成定义见 [`docs/V3.3-ROLLOUT-PLAN.md`](docs/V3.3-ROLLOUT-PLAN.md)。

---

## 12. 限制与合规

- **不确诊、不开处方**；危机场景直接引导 120 / 急诊；
- 医生端签名默认关闭，需 HIS 账号透传后才启用；
- 真实医院对接需院方接口、鉴权与数据合规评估；
- 医保、互联网诊疗等资质不在本项目范围内；
- 演示站数据为模拟，不得用于真实就诊。

---

## 13. 目录结构

```text
backend/            FastAPI 应用、策略、检索、适配器、测试
  app/              API、Agent、RAG、策略、数据访问
  his_adapter/      HIS MCP 适配器（工具、幂等、对账）
  tests/            189 个后端测试
frontend/           Next.js 前端（患者端 + /clinician）
prototype/          四端 61 屏原型（单文件 HTML）与截图
docs/               架构、计划、部署、验收与审计文档
scripts/            备份、发布、回归与演示脚本
```

---

## 14. 许可

代码许可待定（如开源请补充 MIT / Apache-2.0）。医疗数据与第三方文档按各自来源条款使用，不随代码重新授权。

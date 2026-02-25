# Smart Hospital Agent

面向医疗问答/分诊场景的后端工程，核心由 `FastAPI + LangGraph + RAG` 组成，支持：

- 流式对话接口（SSE）
- 意图分流（问候/挂号/医疗/危机场景）
- Milvus + BM25 混合检索与重排
- PostgreSQL/Redis 持久化与缓存
- 本地与云端模型混合策略（配置驱动）

## 当前仓库说明

这是发布版代码快照（`main` 首次提交），重点保留了：

- 后端核心代码（`backend/app`）
- 数据库初始化与持久化脚本（`database/`、`backend/alembic`）
- 全链路回归脚本（`scripts/e2e_fullchain_logger.py`）
- 可公开数据集（`data/`）

已剔除日志、模型大文件、测试杂项等非核心产物。

## 目录结构

```text
.
├── backend/                 # FastAPI + LangGraph + RAG 后端
│   ├── app/
│   │   ├── api/v1/endpoints/chat.py
│   │   ├── core/graph/      # 工作流与节点
│   │   ├── rag/             # 检索、重排、评估相关
│   │   └── services/        # Embedding 等服务封装
│   ├── alembic/             # 数据库迁移
│   └── requirements.txt
├── database/                # 初始化 SQL
├── scripts/
│   ├── e2e_fullchain_logger.py
│   └── e2e_cases_multiturn.json
├── data/                    # 训练/评估/知识数据
└── docker-compose.yml       # 基础设施编排（可选）
```

## 运行前提

- OS: Linux（推荐）
- Python: 3.10+（建议 3.11/3.12）
- 关键依赖：`FastAPI`、`LangGraph`、`pymilvus`、`redis`、`transformers`、`torch`
- 外部组件（按需）：
  - PostgreSQL
  - Redis
  - Milvus（含 etcd/minio）

## 配置规则（重要）

项目当前采用**根目录 `.env` 单一真源**：

- 后端配置固定读取：`PROJECT_ROOT/.env`
- `backend/.env` 不作为运行配置来源

最少应设置：

```bash
OPENAI_MODEL_NAME=qwen-max
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=your_key

POSTGRES_SERVER=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=smart_triage

REDIS_URL=redis://127.0.0.1:6379/0
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530

# 设备策略：auto/cuda/cpu
EMBEDDING_DEVICE=auto
RERANKER_DEVICE=auto
```

## 启动方式（推荐：代码本地运行）

### 1) 准备依赖

```bash
cd /path/to/smart_hospital_agent
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) 准备基础设施

可选 A：使用现有本机服务（PostgreSQL/Redis/Milvus）。

可选 B：仅用 Docker 起基础设施（后端代码仍在本地运行）：

```bash
docker compose up -d db redis etcd minio milvus-standalone
```

### 3) 启动后端

```bash
cd backend
source ../.venv/bin/activate
export PYTHONPATH=$(pwd)
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

健康检查：

```bash
curl http://127.0.0.1:8001/health
```

## 核心接口

- `GET /health`：服务存活探针
- `POST /api/v1/chat/stream`：SSE 流式聊天

示例：

```bash
curl -N -X POST "http://127.0.0.1:8001/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message":"我最近头痛恶心三天","session_id":"demo-001"}'
```

## 全链路回归（E2E）

```bash
python scripts/e2e_fullchain_logger.py \
  --project-root . \
  --base-url http://127.0.0.1:8001 \
  --cases-file scripts/e2e_cases_multiturn.json \
  --backend-log-file logs/backend.log
```

输出默认在 `logs/e2e_fullchain/<timestamp>/`，包含：

- `summary.json`
- `report.md`
- `cases.jsonl`

## 模型与资源策略

- 云端模型由 `OPENAI_MODEL_NAME`/`OPENAI_API_BASE` 驱动
- `ENABLE_LOCAL_FALLBACK=true` 时允许本地兜底路径参与
- Embedding/Reranker 可通过 `EMBEDDING_DEVICE`、`RERANKER_DEVICE` 独立控制在 CPU/GPU 上运行
- 生命周期中包含模型池与显存回收逻辑（`backend/app/core/infra.py`）

## 已知问题（当前快照）

1. 若未设置 `OPENAI_MODEL_NAME`，启动会触发配置校验错误。
2. 当 Milvus 不可达时，检索路径会降级或失败，需先确认 `MILVUS_HOST/MILVUS_PORT`。
3. 当前公开快照不包含 `backend/app/core/models/` 目录，涉及本地 SLM 的导入路径会报：
   - `ModuleNotFoundError: No module named 'app.core.models'`
   - 需要从完整私有工程补齐该目录后再启用本地 SLM 全功能链路。

## 诊断建议

- 后端日志：建议重定向到 `logs/backend.log`
- 先看 `/health`，再看 `/api/v1/chat/stream` 是否持续返回 token
- 使用 `scripts/e2e_fullchain_logger.py` 做多意图回归，优先排查：
  - 路由断言失败
  - 检索信号缺失
  - `stall_timeout` / `case_timeout`

## 安全说明

- 禁止提交 `.env`、密钥、模型权重、运行日志
- 线上部署前请更换默认数据库口令与 API Key

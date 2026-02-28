# Smart Hospital Agent

基于 FastAPI + LangGraph 的智能分诊与医疗问答系统仓库。  
当前仓库同时包含后端服务、前端界面、数据与模型资产、以及运行期脚本。

## 1. 仓库目标

- 提供可流式返回的医疗问答接口。
- 支持分诊、问诊、服务流程等子图编排。
- 支持 RAG 检索、评估与发布门禁脚本。

## 2. 快速启动

### 2.1 后端（本地）

```bash
cd /home/kkk/Project/smart_hospital_agent
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

export PYTHONPATH=/home/kkk/Project/smart_hospital_agent/backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --app-dir backend
```

健康检查：

```bash
curl http://127.0.0.1:8001/health
```

### 2.2 基础设施（可选，Docker）

```bash
cd /home/kkk/Project/smart_hospital_agent
docker compose up -d db redis etcd minio milvus-standalone
```

### 2.3 前端（Next.js）

```bash
cd /home/kkk/Project/smart_hospital_agent/frontend_new
npm install
npm run dev
```

默认地址：`http://127.0.0.1:3000`

## 3. 最小验证命令

### 3.1 后端最小验证

```bash
cd /home/kkk/Project/smart_hospital_agent
python -m compileall backend/app
```

### 3.2 前端最小验证

```bash
cd /home/kkk/Project/smart_hospital_agent/frontend_new
npm run typecheck
```

## 4. 目录说明（治理后约定）

详细约定见：`docs/structure.md`

- `backend/`：后端源码与测试（主 source）。
- `frontend_new/`：前端源码（主 source）。
- `scripts/`：跨模块脚本（发布门禁、回放、评估）。
- `docs/`：文档与治理报告。
- `data/`：可版本化的数据与样本（非运行时挂载）。
- `third_party/`：明确纳管的第三方代码。
- `logs/`、`data_persist/`、`backups_runtime/`、`cache/`、`volumes/`：运行产物目录，不应入库。

## 5. 工程卫生规则（执行摘要）

- 冲突标记禁止入库（`<<<<<<<`, `=======`, `>>>>>>>`）。
- 密钥与环境变量文件禁止入库（`.env`、`*.env`）。
- 数据库/日志/缓存/模型权重禁止入库。
- 大体积第三方源码应放在 `third_party/` 并明确来源；本地临时克隆不入库。

## 6. 常用开发命令

后端：

```bash
cd /home/kkk/Project/smart_hospital_agent
export PYTHONPATH=/home/kkk/Project/smart_hospital_agent/backend
pytest backend/tests -q
```

前端：

```bash
cd /home/kkk/Project/smart_hospital_agent/frontend_new
npm run build
```

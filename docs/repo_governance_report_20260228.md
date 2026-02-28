# Repository Governance Report

日期：2026-02-28  
范围：结构治理（不涉及业务功能开发）

## 1. 发现阶段问题清单

### P0（立即处理）

- 冲突标记残留：`README.md`、`backend/requirements.txt`、`.gitignore`。
- 入口文档存在双版本混写，信息冲突，无法作为可信启动入口。

### P1（本次处理）

- Git 噪音高：日志、缓存、数据库、模型权重、压缩包、运行时目录大量未受控。
- 目录职责边界不清晰，根目录文档与实验资产堆积。

### P2（后续治理）

- 大体积本地第三方克隆与实验目录未归档到 `third_party/` 或外部制品库。
- 数据资产缺乏“可入库/不可入库”分层策略。

## 2. 清单与决策

### 保留（Keep）

- `backend/`：后端核心源码与测试。
- `frontend_new/`：前端核心源码。
- `scripts/`、`backend/scripts/`：工程脚本。
- `data/`、`backend/data/`：可复现数据与知识库资源（需继续瘦身）。
- `docs/`：架构与治理文档。
- `third_party/`：纳管的第三方代码目录。

### 移动（Move）

本轮实际执行：无高风险移动（避免影响运行路径）。  
建议下一轮低风险迁移：

- 根目录专题文档（如 `升级1.md`、`升级2.md`、`项目架构蓝图_*.md`）迁入 `docs/blueprints/`。
- 一次性审计报告迁入 `docs/reports/`。
- 本地第三方克隆统一迁入 `third_party/local/` 或外部目录（按需保留）。

### 忽略（Ignore）

已通过 `.gitignore` 收敛：

- 运行产物：`logs/`、`cache/`、`data_persist/`、`backups_runtime/`、`volumes/`。
- 数据库与检查点：`*.db`、`*.sqlite*`、`checkpoints.sqlite`。
- 模型与权重：`models/`、`*.gguf`、`*.safetensors`、`*.pt`、`*.onnx` 等。
- 前端构建产物：`frontend_new/node_modules/`、`frontend_new/.next/`。
- 大压缩包与中间数据：`*.zip`、`*.tar.gz`、`data/**/*.arrow`、`data/**/*.lock`。

### 待确认删除（Confirm Before Delete）

以下为高风险目录，仅列出，不在本轮删除：

- `data/`（约 4.9G）：包含训练集与压缩包，需先做白名单再删。
- `models/`（约 11G）：模型资产，需确认是否迁至制品库。
- `data_persist/`（约 1.1G）：数据库与向量库持久化目录，删除前需备份策略。
- `backups_runtime/`（约 339M）：运行快照，需确认保留周期。
- `venv/`（约 13G）：历史虚拟环境，应由本地重建替代。
- `llama.cpp/`、`GPTQModel/`、`local_llama_factory/` 等本地克隆目录：需确认是否纳管。

## 3. 本次变更结果

- 修复冲突：`README.md`、`backend/requirements.txt`、`.gitignore`。
- 重写入口文档：`README.md`（单一可信版本）。
- 新增结构约定：`docs/structure.md`。
- 完成治理报告：本文件。

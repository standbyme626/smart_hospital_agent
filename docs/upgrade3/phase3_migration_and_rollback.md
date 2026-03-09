# Upgrade3 Phase 3 迁移与回退说明

## 1) 回退开关（Feature Flags）

| 开关 | 默认值 | 环境变量名 | 作用 |
| --- | --- | --- | --- |
| Chat Shell | `true` | `UPGRADE3_CHAT_SHELL_ENABLED` | 控制 `chat.py` 是否走 Upgrade3 的 SSE renderer + UX policy 逻辑 |
| Diagnosis Shell | `true` | `UPGRADE3_DIAGNOSIS_SHELL_ENABLED` | 控制 `build_diagnosis_graph` 是否绑定 `shell_*` 实现 |
| Retriever Pipeline | `true` | `UPGRADE3_RETRIEVER_PIPELINE_ENABLED` | 控制 `retriever.py`/`rag.pipeline` 是否走 pipeline 包装路径 |

### 回退步骤（最小化）

1. 在部署环境设置目标开关为 `false`（支持逐项回退）。
2. 重启后端进程使配置生效。
3. 运行以下契约测试确认对外接口未破坏：
   - `backend/tests/unit/api/test_chat_stream_contract.py`
   - `backend/tests/unit/rag/test_retriever_contract.py`
   - `backend/tests/unit/graph/test_diagnosis_state_contract.py`
4. 检查 Langfuse trace metadata 是否包含 `upgrade3_flags`，确认开关状态被记录。

## 2) 兼容入口保持不变清单

- `POST /api/v1/chat/stream`：SSE 事件契约保持 `type/content/seq/request_id` 主字段不变。
- `MedicalRetriever.search_sync/search`：方法签名保持不变。
- `app.rag.pipeline.search_sync/search`：方法签名和返回类型（`List[Dict[str, Any]]`）不变。
- `build_diagnosis_graph()`：图节点名与关键路由映射保持不变。

## 3) 测试结果与已知风险

本轮执行（2026-03-09）：

- `pytest -q backend/tests/unit/api/test_chat_stream_contract.py backend/tests/unit/rag/test_retriever_contract.py backend/tests/unit/graph/test_diagnosis_state_contract.py`
  - 结果：`9 passed`
- `pytest -q backend/tests/unit/graph/test_diagnosis_state_contract.py`（新增图契约测试复跑）
  - 结果：`2 passed`

已知风险：
- `backend/tests/unit/graph/test_diagnosis.py::test_confidence_evaluator` 存在历史异步断言问题（已有 follow-up：`smart_hospital_agent-txt`），不属于本轮新增改动。

## 4) 废弃调用清理清单（本轮实际）

- `backend/app/core/graph/sub_graphs/diagnosis.py`
  - 移除 `build_diagnosis_graph` 内无实际用途的本地 `llm` 初始化语句（不影响行为）。
  - 移除未使用的 `START` import（无行为变化）。

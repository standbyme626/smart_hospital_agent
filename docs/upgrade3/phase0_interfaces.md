# 升级3 Phase0 接口基线

## 范围与目标
- 范围锁定：`core/config.py`、`core/graph/sub_graphs/diagnosis.py`、`rag/retriever.py`、`api/v1/endpoints/chat.py`、`services/chat_service.py`。
- 目标：冻结现有接口，不改行为；为 Phase1 的“拆壳不改心”提供边界清单。

## 接口清单（source-verified）

### 1) HTTP / SSE 接口
- `POST /api/v1/chat/stream`（`backend/app/api/v1/endpoints/chat.py`）
  - 入参模型 `ChatRequest`：
    - `message: str`
    - `session_id: str = "default"`
    - `rag.top_k/use_rerank/rerank_threshold`
    - `request_id`
    - `debug_include_nodes`
    - `rewrite_timeout`
    - `crisis_fastlane`
  - 返回：`StreamingResponse`，`media_type=text/event-stream`。
  - SSE 结束标记：`data: [DONE]\n\n`。

### 2) Chat 流式生成接口
- `event_generator(...) -> AsyncGenerator[str, None]`（`chat.py`）
  - 输入：用户消息 + 会话 + RAG runtime override。
  - 对 LangGraph 调用：`graph_app.astream_events(inputs, config, version="v2")`。
  - 输出事件类型（SSE payload）：`status/thought/token/final/error/ping/department_result/doctor_slots/payment_required/...`。
  - 事件协议构造：`build_stream_payload(...)`。

### 3) 诊断子图构建接口
- `build_diagnosis_graph()`（`diagnosis.py`）
  - 入口节点：`State_Sync`
  - 主要节点：`Query_Rewrite`、`Quick_Triage`、`Hybrid_Retriever`、`DSPy_Reasoner`、`Decision_Judge(可选)`、`Diagnosis_Report`、`Clarify_Question`
  - 返回：`workflow.compile()`

### 4) RAG 检索接口（对外稳定面）
- `MedicalRetriever.search_rag30(...)`（`retriever.py`）
  - 支持参数：`top_k/use_rerank/rerank_threshold/skip_intent_router/skip_hyde/pure_mode/return_debug/...`
  - 返回：
    - 默认：`List[Dict[str, Any]]`
    - `return_debug=True`：`(results, metrics)`
- 兼容入口：
  - `search_sync(...)`
  - `search(...)`（legacy）
  - `get_retriever()`（singleton）

### 5) 配置接口（单例稳定面）
- `settings = Settings()`（`config.py`）
- 兼容导入目标：`from app.core.config import settings`
- 关键 runtime 开关（当前链路相关）：
  - `QUERY_REWRITE_TIMEOUT_SECONDS`
  - `CRISIS_FASTLANE_ENABLED`
  - `SSE_PING_INTERVAL_SECONDS`
  - `TRIAGE_FAST_PATH_ENABLED`

### 6) legacy 服务接口
- `ChatService.stream_events(message, session_id)`（`chat_service.py`）
  - 仍可用作兼容层。
  - 当前主入口不在此（见 `升级3.md` 项目定制修订）。

## 接口边界结论
- `chat.py` 是当前主链路入口；`chat_service.py` 是 legacy 兼容层。
- `retriever.py` 的 `search_rag30/get_retriever` 必须在 Phase1/2 保持签名稳定。
- `config.py` 的 `settings` 单例和初始化副作用（key rotation/env export）必须先冻结时序。

## summary-inference
- Chat 事件最小集合（`status/token/final/error/[DONE]`）已具独立契约测试，更多业务事件（如 `doctor_slots/payment_required/booking_*`）仍需扩展用例矩阵。

## P0 完成判定（source-verified）
- 接口基线文档已落地并与当前主入口对齐：`chat.py::stream_chat/event_generator`、`diagnosis.py::build_diagnosis_graph`、`retriever.py::search_rag30/get_retriever`、`config.py::settings`。
- 最小契约测试已落地：
  - `backend/tests/unit/api/test_chat_stream_contract.py`
  - `backend/tests/unit/rag/test_retriever_contract.py`

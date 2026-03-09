# 升级3 Phase0 主要调用链基线

## 1) 线上主链路：`/chat/stream`（source-verified）
1. `chat.py::stream_chat` 接收请求并生成 `resolved_request_id`
2. 进入 `chat.py::event_generator`
3. 组装 `inputs`（含 retrieval/runtime override 字段）
4. 调用 `graph_app.astream_events(inputs, config, version="v2")`
5. `workflow.py` 主图路由（`triage_router`）将医疗咨询导向 `diagnosis` 子图
6. `diagnosis.py::build_diagnosis_graph` 执行节点链
7. `event_generator` 将图事件转换为 SSE payload（`build_stream_payload`）
8. 输出 `final`、`stream_closed`、`[DONE]`

## 2) diagnosis 子图内部链（source-verified）
固定骨架（决策治理开关前提下略有分支）：

1. `State_Sync`
2. `Query_Rewrite`
3. `Quick_Triage`
4. 条件分支：
   - `fast_exit -> Diagnosis_Report`
   - `deep_diagnosis -> Hybrid_Retriever`
5. `Hybrid_Retriever` 后分支：
   - `pure_report -> Diagnosis_Report`
   - `dspy_reasoner -> DSPy_Reasoner`
6. `DSPy_Reasoner` 后：
   - `ENABLE_DECISION_GOVERNANCE=true`：`Decision_Judge -> Confidence_Evaluator`
   - 否则：`Confidence_Evaluator`
7. `end_diagnosis -> Diagnosis_Report -> END`
8. `clarify_question -> Clarify_Question -> END`

## 3) retriever 对外调用链（source-verified）
1. `get_retriever()` 返回单例 `MedicalRetriever`
2. 上层调用 `search_rag30(...)`（或 `search/search_sync`）
3. `search_rag30` 内部阶段：
   - cache namespace/profile 计算
   - 语义缓存命中/校验
   - intent router（可跳过）
   - SQL prefilter（科室）
   - 向量+BM25 并行召回 + RRF
   - rerank（可跳过）
   - 动态阈值与 fallback rewrite
   - summarize（可跳过）
   - 部门后处理
   - cache write gate + 写回
4. 返回 `results` 或 `(results, metrics)`

## 4) legacy 兼容链：`ChatService`（source-verified）
1. `ChatService.stream_events`
2. `build_medical_graph()`（`workflow.py` 别名）
3. `graph.astream_events(...)`
4. 转为业务事件字典：`status/thought/token/error/done`

## 5) 当前边界结论
- `source-verified`：当前生产主链在 `chat.py::event_generator`，不是 `chat_service.py`。
- `summary-inference`：Phase1 建议采用“chat.py-first 拆分”，`chat_service.py` 保留兼容 wrapper（与 `升级3.md` 修订版一致）。

## 6) P0 完成判定（source-verified）
- 主链路、子图链路、retriever 链路与 legacy 链路均已在代码中完成复核并冻结。
- 对应最小契约覆盖已存在：
  - chat 流式顺序与可选事件：`backend/tests/unit/api/test_chat_stream_contract.py`
  - retriever 对外契约：`backend/tests/unit/rag/test_retriever_contract.py`

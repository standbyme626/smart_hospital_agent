# 升级3 Phase0 关键状态字段基线

## 目标
- 冻结诊断链路关键状态字段，避免 Phase1 拆分时发生隐式字段漂移。

## DiagnosisState 契约字段（source-verified）
来源：`backend/app/domain/states/sub_states.py::DiagnosisState`

### A. 输入与会话
- `messages`
- `patient_id`
- `user_profile`
- `dialogue_history`
- `current_turn_input`
- `user_input`
- `event`
- `request_id`
- `session_id`

### B. 检索规划/运行时控制
- `retrieval_query`
- `retrieval_query_variants`
- `retrieval_top_k`
- `retrieval_top_k_override`
- `retrieval_use_rerank`
- `retrieval_rerank_threshold`
- `query_rewrite_timeout_override_s`
- `crisis_fastlane_override`
- `runtime_config_requested`
- `runtime_config_effective`
- `retrieval_plan`
- `retrieval_index_scope`
- `variant_hits_map`
- `topk_source_ratio`
- `fusion_method`
- `context_pack`

### C. 诊断与治理输出
- `department_top1`
- `department_top3`
- `recommended_department`
- `diagnosis_output`
- `validated`
- `validation_error`
- `repair_attempted`
- `current_hypothesis`
- `guideline_matches`
- `differential_diagnosis`
- `confirmed_diagnosis`
- `confidence`
- `last_tool_result`
- `decision_action`
- `decision_reason`
- `confidence_score`
- `grounded_flag`
- `is_diagnosis_confirmed`
- `loop_count`
- `intent`
- `step`
- `error`
- `debug_include_nodes`
- `debug_snapshots`

## 运行中高频写入字段（source-verified）
来源：`diagnosis.py` 与 `chat.py` 的 `inputs` 注入/节点返回

- `state_sync_node`：
  - 写入：`profile_text/department/system_prompt/rag_pure_mode`
- `query_rewrite_node`：
  - 写入：`retrieval_query/retrieval_query_variants/retrieval_top_k/retrieval_plan/...`
  - 写入：`runtime_config_effective`
- `quick_triage_node`：
  - 写入：`triage_fast_result/triage_fast_ready`
  - 条件写入：`department_top1/department_top3/recommended_department/confidence`
- `hybrid_retriever_node`：
  - 写入：`messages(context)/context_pack/variant_hits_map/topk_source_ratio/fusion_method/triage_tool_trace`
  - pure 模式写入：`pure_retrieval_result/recommended_department/department_top1/department_top3`
- `decision_judge_node`：
  - 写入：`last_tool_result + decision_* + confidence_score + grounded_flag`
- `generate_report_node`：
  - 写入：`clinical_report/messages/is_diagnosis_confirmed/diagnosis_output` 及科室字段

## Chat 入口注入字段（source-verified）
来源：`chat.py::event_generator`

- 必注入：`symptoms/user_input/current_turn_input/retrieval_query/messages/patient_id/session_id/event`
- runtime override 注入：
  - `retrieval_top_k_override`
  - `retrieval_use_rerank`
  - `retrieval_rerank_threshold`
  - `query_rewrite_timeout_override_s`
  - `crisis_fastlane_override`
  - `runtime_config_requested/effective`

## Phase0 风险标记
- `source-verified`：`department_top1/department_top3/recommended_department` 已纳入 `DiagnosisState`，但仍存在运行态“超契约字段”（如 `triage_fast_result/pure_retrieval_result/profile_text/department/system_prompt/rag_pure_mode`）。
- `summary-inference`：Phase1/2 若继续拆分 diagnosis 节点，需优先收敛剩余“超契约字段”到 contract 层，避免静态类型与运行态漂移。

## P0 完成判定（source-verified）
- 字段冻结文档已覆盖输入/会话、检索控制、诊断输出、治理调试四簇关键字段。
- 关键字段已具最小契约测试：`backend/tests/unit/graph/test_diagnosis_state_contract.py::test_diagnosis_state_contract_covers_phase1_critical_fields`。

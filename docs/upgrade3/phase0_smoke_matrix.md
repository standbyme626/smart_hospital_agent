# 升级3 Phase0 最小 Smoke / 回归样本矩阵

## 目标
- 用最小样本冻结“接口不变 + 关键字段不破坏 + 基本链路不断流”。
- 本矩阵仅定义样本与断言，不涉及 Phase1/2/3 重构。

## 样本矩阵（source-verified + summary-inference）

| Case ID | 场景 | 输入样本 | 关键断言 | 当前状态 |
|---|---|---|---|---|
| SMK-01 | 普通医疗咨询 | `最近头痛三天，伴恶心` | `/api/v1/chat/stream` 可持续输出；事件序列至少包含 `status -> token/thought -> final -> status(stream_closed) -> [DONE]` | 已定义；最小顺序契约已自动化覆盖 |
| SMK-02 | 危机场景快车道 | `胸痛并呼吸困难，感觉快晕倒` + `crisis_fastlane=true` | `Query_Rewrite` 路径可产生 `rewrite_path`/runtime config 元信息；流不断 | 已定义；`rewrite_path` 触发语义已自动化覆盖 |
| SMK-03 | RAG runtime 覆盖 | `message=胃痛` + `rag.top_k=5,use_rerank=true,rerank_threshold=0.2` | 输入注入字段生效：`retrieval_top_k_override/retrieval_use_rerank/retrieval_rerank_threshold` | 已定义（待扩展自动化） |
| SMK-04 | 预约快捷命令 | `BOOK:slot_demo_001` | 走 `booking_shortcut` 分支，不进入诊断主链；出现 `booking_preview/payment_required/final/[DONE]` | 已定义（待扩展自动化） |
| SMK-05 | 支付确认快捷命令 | `PAY:order_demo_001` | 走 `booking_shortcut`；出现 `booking_confirmed` 或 `booking_error`，最终都输出 `final/[DONE]` | 已定义（待扩展自动化） |
| SMK-06 | 诊断澄清分支 | 模糊输入：`最近不太舒服` | 可出现 `Clarify_Question` 路径；输出可读追问文本，不中断 | 已定义（待扩展自动化） |
| SMK-07 | retriever 兼容接口 | 直接调用 `get_retriever().search_rag30("咳嗽", top_k=3)` | 返回类型保持：默认 list，`return_debug=True` 返回 `(results, metrics)` | 已自动化覆盖 |
| SMK-08 | legacy chat service | `ChatService().stream_events("头晕", "s1")` | 仍能产出 `status/token/error/done` 事件字典流 | 已定义（待扩展自动化） |

## 回归样本最小集合（建议固定）
- 医疗咨询：
  - `头痛+恶心`
  - `胸痛+呼吸困难`
  - `胃痛+反酸`
- 服务命令：
  - `BOOK:slot_demo_001`
  - `PAY:order_demo_001`
- 模糊/澄清：
  - `不舒服，不知道挂什么科`

## 最低检查项
- 接口兼容：`POST /api/v1/chat/stream` 路径与请求体字段保持。
- 状态兼容：`department_top1/top3/confidence/diagnosis_output` 字段不丢。
- 检索兼容：`search_rag30/get_retriever` 签名不变。
- 流式兼容：`[DONE]` 收尾稳定存在。

## 标注
- `source-verified`：事件类型、路由分支、接口签名来自当前代码。
- `summary-inference`：样本文本与“建议固定集合”为 Phase0 回归抽样建议，后续会在 CI 中继续扩展自动化覆盖。

## P0 完成判定（source-verified）
- Phase0 所需 smoke 矩阵已完整落地（SMK-01 ~ SMK-08）。
- 已落地自动化最小子集：
  - chat 事件顺序与可选事件：`backend/tests/unit/api/test_chat_stream_contract.py`
  - retriever 契约：`backend/tests/unit/rag/test_retriever_contract.py`

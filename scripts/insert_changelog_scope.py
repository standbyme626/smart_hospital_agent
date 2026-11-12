import pathlib

entry = '''### 2026-11-12 11:10 · fix · 待提交 · 面板按账号过滤 + 内容宽两档收敛
- 屏：四端全部（右栏与内容宽度）
- 改动：(1) `/api/v1/agent/panel` 增加 `scope`：平台端按**本院区**聚合，其余端按**本账号**过滤（`model_calls.user_id`），
  患者/医生不会再看到别人的调用；前端摘要显示「本账号 / 本院区」；
  (2) 内容宽统一为两档：常规页铺满主区（1120 上限），阅读类页面用 `readable` 收到 **768**；
  删掉 13 个页面里各自为政的 `mx-auto max-w-3xl/4xl` 包装。
- 文件：`backend/app/routes_panel.py`、`backend/tests/test_agent_panel_scope.py`、`frontend/components/AgentPanel.tsx`、`frontend/lib/api.ts`、`frontend/app/globals.css`、`frontend/app/**/page.tsx`（13 页）
- 验证：后端 **298 passed**；E2E **26 passed**；`tsc` 0 错误；生产实测：常规页内容 852（主区 904 − padding）、阅读页 768；
  患者登录取面板 `scope=self`，平台端 `scope=tenant`。
- 风险/未做：`model_calls.user_id` 为空的历史调用不计入「本账号」统计（属预期）。

'''

p = pathlib.Path("docs/V3.4-CHANGELOG.md")
text = p.read_text(encoding="utf-8")
anchor = "### 2026-11-12 00:55 · feat · 68a9a74 · 全端对齐演示站三栏页框"
assert anchor in text
p.write_text(text.replace(anchor, entry + anchor, 1), encoding="utf-8")
print("changelog inserted")

import pathlib

entry = '''### 2026-11-12 00:55 · feat · 待提交 · 全端对齐演示站三栏页框
- 屏：四端 61 屏全部（患者 16 / 医生 7 路由 / 运营 7 / 管理 6）
- 改动：(1) 新增 `AppFrame`（顶栏 64 + 左栏 208 + 主区 1fr + 右栏 328）与 `ConsoleShell`，患者端 `AppShell` 复用同一页框；
  (2) 左栏改为分组导航（常用/就医/我的、诊疗/协同、监控/治理、概览/接入/建设），选中项实心；
  (3) 新增 `GET /api/v1/agent/panel?end=`：按当前端返回能力矩阵、`model_calls` 近 7 天调用与延迟、边界说明，右栏真实数据；
  (4) 页头统一 21/12.5，每页恰好一个 H1（删掉 6 处移动端重复标题），卡片圆角 16px、底色 `#f5f8f6` 对齐演示站；
  (5) 响应式：<1024 左栏抽屉、<1280 右栏抽屉、患者端底部 6 项导航，390 宽横向溢出 0；
  (6) 演示站自身问题一并修：移动端、真实数据、可分享路由、六态、scope 标识、skip link / aria-current / Esc。
- 文件：`frontend/components/AppFrame.tsx`、`AgentPanel.tsx`、`ConsoleShell.tsx`、`AppShell.tsx`、`frontend/lib/nav.ts`、`frontend/app/globals.css`、`frontend/app/{clinician,ops,platform}/**/page.tsx`（20 页）、`backend/app/routes_panel.py`、`backend/app/clock.py`、`backend/tests/test_agent_panel.py`
- 验证：后端 **297 passed**；E2E **26 passed**；`tsc` 0 错误；生产实测四端 64 / 208 / 904 / 328、H1 21px；右栏按端显示；移动端溢出 0、底栏 6 项。
- 风险/未做：右栏「最近调用」按租户聚合，未按单个医生/患者过滤；抽屉打开时未锁定背景滚动（下一步补）。

'''

p = pathlib.Path("docs/V3.4-CHANGELOG.md")
text = p.read_text(encoding="utf-8")
anchor = "### 2026-11-11 23:15 · fix · 4448957 · 登录弹框 / 反复登录 + 前端体验修复"
assert anchor in text
p.write_text(text.replace(anchor, entry + anchor, 1), encoding="utf-8")
print("changelog inserted")

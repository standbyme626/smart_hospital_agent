#!/usr/bin/env python3
"""生成 V3.4 统一布局草图（SVG）。坐标全部居中计算，避免重叠。"""
import pathlib

W, H = 1240, 1660
INK, MUTED, LINE, FILL = "#1F2937", "#6B7280", "#D9E1E8", "#F6F9F8"
ACCENT, ACCENT_SOFT = "#0E9384", "#E6F4F1"

parts = []
def rect(x, y, w, h, fill="none", stroke=LINE, rx=10, dash=None, sw=1.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def text(x, y, s, size=12, fill=INK, anchor="start", weight=400):
    parts.append(
        f'<text x="{x}" y="{y}" font-family="PingFang SC, Noto Sans CJK SC, Microsoft YaHei, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'
    )

def bar(x, y, w, h, label, fill=FILL, color=MUTED):
    rect(x, y, w, h, fill=fill)
    text(x + 12, y + h / 2 + 4, label, 11, color)

def panel(y, title, note):
    text(80, y, title, 17, INK, weight=600)
    text(80, y + 20, note, 11, MUTED)

parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
parts.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# ---------------- 面板 1：统一页框 · 控制台端 ----------------
panel(48, "① 统一页框 · 控制台端（医生 / 运营 / 管理）", "一条品牌条 + 一行页签 + 一个内容宽（1120）；子页复用同一页框，不再另造小页头")
fx, fy, fw, fh = 80, 100, 1080, 360
rect(fx, fy, fw, fh, FILL, LINE, 14)
bar(fx, fy, fw, 56, "品牌条 56px：入口名 · 当前角色 · 账号 / 退出（右侧）", ACCENT_SOFT, ACCENT)
rect(fx, fy + 56, fw, 44, FILL)
text(fx + 16, fy + 84, "页签 44px：工作台 · 文书打印 · 诊毕随访 · 药师审方 …（当前项实心）", 11, MUTED)
rect(fx + 16, fy + 112, fw - 32, 44, "none", LINE, 8)
text(fx + 32, fy + 140, "H1 页面标题 20/600  ·  副标题 12 muted  ·  右侧操作区（刷新 / 主操作）", 12, INK)
rect(fx + 16, fy + 170, 300, 32, "none", LINE, 8)
text(fx + 32, fy + 190, "筛选 / 分段控件（44 高）", 11, MUTED)
for i in range(3):
    cx = fx + 16 + i * 352
    rect(cx, fy + 216, 336, 112, "none", LINE, 10)
    text(cx + 16, fy + 244, "卡片 H 112 · padding 20 · radius 16", 11, INK)
    text(cx + 16, fy + 268, "内容行 1（13/21）", 10, MUTED)
    text(cx + 16, fy + 288, "内容行 2（11/18）", 10, MUTED)
text(80, fy + fh + 24, "页宽 1120 · 页面 padding 32 · 区块间距 16 · 行高 44（列表）/ 56（可点行）", 11, MUTED)

# ---------------- 面板 2：患者端外壳 ----------------
panel(520, "② 患者端外壳（/ 与 /messages 等）", "左侧 240 导航 + 顶部 64 页头 + 内容 1120；移动端换成底部 6 项导航")
py0 = 572
rect(80, py0, 1080, 300, FILL, LINE, 14)
rect(80, py0, 240, 300, "none", LINE, 14)
text(96, py0 + 30, "侧栏 240", 11, INK, weight=600)
for i, label in enumerate(["首页", "消息", "随访", "AI 诊室", "报告解读", "药管家"]):
    text(96, py0 + 58 + i * 26, label, 11, MUTED if i else ACCENT)
rect(320, py0, 840, 64, "none", LINE, 10)
text(336, py0 + 28, "页头 64：H1 17/600 + 副标题 12 muted", 11, INK)
text(336, py0 + 48, "右侧：消息 · 家庭成员切换 · 个人中心", 11, MUTED)
rect(320, py0 + 80, 840, 200, "none", LINE, 10)
text(336, py0 + 108, "内容区 1120（阅读类页面收窄到 768）", 11, INK)
rect(336, py0 + 124, 808, 64, "none", LINE, 8)
text(352, py0 + 162, "首屏主操作卡：标题 + 输入 + 主按钮（高度固定 64）", 10, MUTED)
rect(336, py0 + 200, 392, 64, "none", LINE, 8)
rect(752, py0 + 200, 392, 64, "none", LINE, 8)
text(352, py0 + 238, "六宫格入口（3 列，卡高 118）", 10, MUTED)
text(768, py0 + 238, "入口卡 118 · 圆角 16 · 同一主色", 10, MUTED)
text(80, py0 + 322, "统一：页头 64 / 侧栏 240 / 内容 1120 / 卡圆角 16 / 主色 #0E9384（唯一强调）", 11, MUTED)

# ---------------- 面板 3：现状 vs 目标 ----------------
panel(960, "③ 布局不一致的根因 → 目标规则", "现状 4 种内容宽 + 5 种页头；目标只保留 2 种外壳、2 档宽度")
lx, ly, lw, lh = 80, 1012, 520, 300
rect(lx, ly, lw, lh, FILL, LINE, 14)
text(lx + 16, ly + 26, "现状（实测，1440 宽）", 12, INK, weight=600)
rows = [
    ("患者端 /messages", "内容 1152 · 页头 69 · H1 15"),
    ("医生端 /clinician", "内容 1152 · 页头 64 · 无 H1"),
    ("医生端 /clinician/print", "内容 1024 · 页头 52 · H1 20"),
    ("运营端 /ops/reviews", "内容 1024 · 页头 106 · 无 H1"),
    ("管理端 /platform/config", "内容 1152 · 无页头 · H1 20"),
    ("统一入口 /entries", "内容 1440 · 页头 93 · H1 26"),
]
for i, (a, b) in enumerate(rows):
    yy = ly + 56 + i * 36
    rect(lx + 16, yy - 14, lw - 32, 28, "none", LINE, 6)
    text(lx + 26, yy + 4, a, 10, INK)
    text(lx + lw - 26, yy + 4, b, 10, MUTED, anchor="end")

rx0 = 640
rect(rx0, ly, lw, lh, ACCENT_SOFT, ACCENT, 14)
text(rx0 + 16, ly + 26, "目标（统一规则）", 12, ACCENT, weight=600)
rules = [
    "内容宽只有 2 档：1120（常规）· 768（阅读/表单）",
    "页头只有 2 种：患者端 64 · 控制台 56+44",
    "每页必有 H1：20/600；副标题 12；不再用 p 代替标题",
    "页签位置固定：页头下、内容上，左对齐同宽",
    "子页复用父页框，禁止另造窄页头",
    "卡片：radius 16 / padding 20 / 边框 1px line",
    "栅格：12 列 · 间距 16 · 页面 padding 32（移动 16）",
    "主色唯一 #0E9384；渐变仅用于 logo",
]
for i, rule in enumerate(rules):
    text(rx0 + 20, ly + 60 + i * 30, "· " + rule, 10.5, INK)

# ---------------- 面板 4：移动端 ----------------
panel(1350, "④ 移动端（≤ 640）", "顶栏 52 带页面标题；底部 6 项导航；列表行高 44，内容单列")
mx, my, mw, mh = 80, 1402, 300, 190
rect(mx, my, mw, mh, FILL, LINE, 20)
rect(mx, my, mw, 52, "none", LINE, 20)
text(mx + 16, my + 32, "标题 + 返回/操作", 11, INK)
rect(mx + 12, my + 64, mw - 24, 44, "none", LINE, 8)
text(mx + 24, my + 92, "筛选行 44", 10, MUTED)
rect(mx + 12, my + 116, mw - 24, 30, "none", LINE, 8)
text(mx + 24, my + 136, "列表行 44", 10, MUTED)
rect(mx + 12, my + 150, mw - 24, 28, "none", LINE, 8)
text(mx + 24, my + 169, "列表行 44", 10, MUTED)
text(400, my + 30, "底栏 56：首页 · 消息 · 问诊 · 服务 · 健康 · 我的（6 项均分）", 11, MUTED)
text(400, my + 56, "安全区：padding-bottom: env(safe-area-inset-bottom)", 11, MUTED)
text(400, my + 82, "所有列表横向 padding 16，禁止横向滚动（实测溢出 0）", 11, MUTED)

parts.append("</svg>")
out = pathlib.Path("docs/assets/v34-layout-sketch.svg")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(parts), encoding="utf-8")
print("written:", out, out.stat().st_size, "bytes")

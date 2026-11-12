import pathlib

api = pathlib.Path("lib/api.ts")
t = api.read_text(encoding="utf-8")
t = t.replace('  boundaries: string[];\n  starter: string;\n};', '  boundaries: string[];\n  starter: string;\n  scope: "self" | "tenant";\n};')
api.write_text(t, encoding="utf-8")

p = pathlib.Path("components/AgentPanel.tsx")
s = p.read_text(encoding="utf-8")
old = '7 天 {data.summary.calls_7d} 次 · 均值 {data.summary.avg_latency_ms}ms'
new = '{data.scope === "tenant" ? "本院区" : "本账号"} · 7 天 {data.summary.calls_7d} 次 · 均值 {data.summary.avg_latency_ms}ms'
assert old in s, "summary anchor missing"
p.write_text(s.replace(old, new), encoding="utf-8")

# 内容宽统一：阅读类 768，其余铺满 1120
for path in pathlib.Path("app").rglob("page.tsx"):
    if path.parts[1] in {"entries", "privacy", "terms", "login"}:
        continue
    text = path.read_text(encoding="utf-8")
    updated = text.replace('className="mx-auto max-w-3xl space-y-4"', 'className="readable space-y-4"')
    updated = updated.replace('className="mx-auto max-w-4xl space-y-4"', 'className="space-y-4"')
    if updated != text:
        path.write_text(updated, encoding="utf-8")
        print("width:", path)
print("done")

import pathlib

api = pathlib.Path("lib/api.ts")
t = api.read_text(encoding="utf-8")
old_api = 'export async function getAgentPanel(): Promise<AgentPanelData> {\n  return apiFetch<AgentPanelData>("/api/v1/agent/panel");\n}'
assert old_api in t, "api anchor missing"
t = t.replace(old_api, 'export async function getAgentPanel(end?: "patient" | "clinician" | "ops" | "platform"): Promise<AgentPanelData> {\n  const query = end ? "?end=" + end : "";\n  return apiFetch<AgentPanelData>("/api/v1/agent/panel" + query);\n}')
api.write_text(t, encoding="utf-8")

p = pathlib.Path("components/AgentPanel.tsx")
s = p.read_text(encoding="utf-8")
old_sig = 'export function AgentPanel({ className = "" }: { className?: string }) {'
assert old_sig in s, "signature anchor missing"
s = s.replace(old_sig, 'const LABEL: Record<AgentPanelData["end"], string> = {\n  patient: "患者端 Agent",\n  clinician: "医生端 Agent",\n  ops: "运营 Agent",\n  platform: "平台 Agent",\n};\n\nexport function AgentPanel({\n  end,\n  className = "",\n}: {\n  end?: "patient" | "clinician" | "ops" | "platform";\n  className?: string;\n}) {')
s = s.replace("    getAgentPanel()", "    getAgentPanel(end)")
s = s.replace("  }, []);", "  }, [end]);")
s = s.replace('{data ? data.end === "patient" ? "患者端 Agent" : data.end === "clinician" ? "医生端 Agent" : data.end === "ops" ? "运营 Agent" : "平台 Agent" : "Agent"}', '{data ? LABEL[data.end] : "Agent"}')
p.write_text(s, encoding="utf-8")

f = pathlib.Path("components/AppFrame.tsx")
u = f.read_text(encoding="utf-8")
old_frame = '<AgentPanel className={agentOpen ? "open" : ""} />'
assert old_frame in u, "frame anchor missing"
u = u.replace(old_frame, '<AgentPanel end={end} className={agentOpen ? "open" : ""} />')
f.write_text(u, encoding="utf-8")
print("all patched")

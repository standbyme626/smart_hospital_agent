import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"


def run_py(code: str, timeout_sec: int):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND)
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return {
            "ok": proc.returncode == 0,
            "elapsed_sec": round(time.time() - started, 2),
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "elapsed_sec": timeout_sec,
            "stdout": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            "stderr": (e.stderr or "")[-2000:] if isinstance(e.stderr, str) else "",
            "error": "timeout",
        }


REAL_IMPORT_CODE = r'''
import json, time
start=time.time()
import app.core.graph.workflow  # noqa
print(json.dumps({"import_sec": round(time.time()-start, 2)}))
'''

STUB_GRAPH_CODE = r'''
import asyncio, importlib, json, sys, types
from langchain_core.messages import HumanMessage

INTENT = "__INTENT__"

async def fake_cache_lookup(_state):
    return {"cache_hit": False}

async def fake_persistence(_state):
    return {"status": "persisted"}

async def fake_fast_reply(_state):
    return {"final_output": "hello", "status": "completed"}

async def fake_anamnesis(_state):
    return {"final_output": "need more info", "status": "in_progress"}

async def fake_human_review(_state):
    return {"status": "approved"}

for module_name, fn_name, fn in [
    ("app.core.graph.nodes.cache", "cache_lookup_node", fake_cache_lookup),
    ("app.core.graph.nodes.persistence", "persistence_node", fake_persistence),
    ("app.core.graph.nodes.fast_reply", "fast_reply_node", fake_fast_reply),
    ("app.core.graph.anamnesis_node", "anamnesis_node", fake_anamnesis),
    ("app.core.graph.human_node", "human_review_node", fake_human_review),
]:
    m = types.ModuleType(module_name)
    setattr(m, fn_name, fn)
    sys.modules[module_name] = m

async def ingress(_state):
    return {"intent": INTENT, "status": "ok"}

async def medical_core(_state):
    return {"status": "approved"}

async def diagnosis(_state):
    return {"clinical_report": "dx ok", "status": "approved"}

async def egress(state):
    return {"final_output": state.get("clinical_report", "egress ok"), "status": "approved"}

for module_name, fn_name, node_fn in [
    ("app.core.graph.sub_graphs.ingress", "build_ingress_graph", ingress),
    ("app.core.graph.sub_graphs.medical_core", "build_medical_core_graph", medical_core),
    ("app.core.graph.sub_graphs.diagnosis", "build_diagnosis_graph", diagnosis),
    ("app.core.graph.sub_graphs.egress", "build_egress_graph", egress),
]:
    m = types.ModuleType(module_name)
    setattr(m, fn_name, lambda fn=node_fn: fn)
    sys.modules[module_name] = m

sys.modules.pop("app.core.graph.workflow", None)
workflow_mod = importlib.import_module("app.core.graph.workflow")
app = workflow_mod.create_agent_graph(checkpointer=None)

initial_state = {
    "messages": [HumanMessage(content="probe")],
    "symptoms": "probe",
    "event": {
        "event_type": "GREETING" if INTENT == "GREETING" else "SYMPTOM_DESCRIPTION",
        "payload": {},
        "raw_input": "probe",
        "timestamp": 1.0,
    },
}

async def run():
    steps = []
    async def collect():
        async for event in app.astream(initial_state, config={"configurable": {"thread_id": f"probe-{INTENT}"}, "recursion_limit": 20}):
            for k in event.keys():
                steps.append(k)
    await asyncio.wait_for(collect(), timeout=5.0)
    print(json.dumps({"intent": INTENT, "steps": steps}, ensure_ascii=False))

asyncio.run(run())
'''


def main():
    results = []

    r1 = run_py(REAL_IMPORT_CODE, timeout_sec=60)
    results.append({"probe": "real_import", **r1})

    for intent in ["GREETING", "MEDICAL_CONSULT"]:
        code = STUB_GRAPH_CODE.replace("__INTENT__", intent)
        res = run_py(code, timeout_sec=20)
        results.append({"probe": f"stub_graph_{intent.lower()}", **res})

    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

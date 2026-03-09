#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_stage(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "cmd": cmd,
        "cwd": str(cwd),
        "timeout_s": timeout_s,
        "timed_out": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
    }
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        result["returncode"] = proc.returncode
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
    except subprocess.TimeoutExpired as exc:
        result["timed_out"] = True
        result["returncode"] = 124
        result["stdout"] = exc.stdout or ""
        result["stderr"] = exc.stderr or ""
    return result


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    out_dir = project_root / "logs" / "regression" / _now_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PYTHONPATH"] = f"{project_root / 'backend'}:{base_env.get('PYTHONPATH', '')}".rstrip(":")

    stages: list[dict[str, Any]] = []
    stages.append(
        _run_stage(
            name="python_unit_api",
            cmd=["pytest", "-q", "backend/tests/unit/api"],
            cwd=project_root,
            timeout_s=args.python_timeout_s,
            env=base_env,
        )
    )
    stages.append(
        _run_stage(
            name="python_unit_rag",
            cmd=["pytest", "-q", "backend/tests/unit/rag"],
            cwd=project_root,
            timeout_s=args.python_timeout_s,
            env=base_env,
        )
    )
    stages.append(
        _run_stage(
            name="python_unit_graph_mainline",
            cmd=[
                "pytest",
                "-q",
                "backend/tests/unit/graph/test_audit.py",
                "backend/tests/unit/graph/test_diagnosis.py",
                "backend/tests/unit/graph/test_diagnosis_config.py",
                "backend/tests/unit/graph/test_diagnosis_stage_b_integration.py",
                "backend/tests/unit/graph/test_diagnosis_stage_c_router_adapter.py",
                "backend/tests/unit/graph/test_diagnosis_state_contract.py",
                "backend/tests/unit/graph/test_triage.py",
                # Legacy graph API compatibility tests kept out of mainline gate:
                # backend/tests/unit/graph/test_ingress.py (imports removed route_ingress)
                # backend/tests/unit/graph/test_service.py (calls async node synchronously)
                # backend/tests/unit/graph/test_service_integration.py (calls async node synchronously)
            ],
            cwd=project_root,
            timeout_s=args.python_timeout_s,
            env=base_env,
        )
    )
    stages.append(
        _run_stage(
            name="python_contract_gate",
            cmd=[
                "pytest",
                "-q",
                "backend/tests/unit/api/test_chat_stream_contract.py",
                "backend/tests/unit/rag/test_retriever_contract.py",
                "backend/tests/unit/graph/test_diagnosis_state_contract.py",
                "backend/tests/unit/rag/test_upgrade3_compat_bridges.py",
            ],
            cwd=project_root,
            timeout_s=args.python_timeout_s,
            env=base_env,
        )
    )

    if args.include_legacy:
        stages.append(
            _run_stage(
                name="python_legacy_compat",
                cmd=[
                    "pytest",
                    "-q",
                    "backend/tests/unit/graph/test_ingress.py",
                    "backend/tests/unit/graph/test_service.py",
                    "backend/tests/unit/graph/test_service_integration.py",
                    "backend/tests/test_core_business_logic.py",
                    "backend/tests/test_llm_rotation_logic.py",
                ],
                cwd=project_root,
                timeout_s=args.python_timeout_s,
                env=base_env,
            )
        )

    if not args.skip_frontend:
        stages.append(
            _run_stage(
                name="frontend_typecheck",
                cmd=[
                    "npm",
                    "exec",
                    "--",
                    "tsc",
                    "-p",
                    "tsconfig.regression.json",
                    "--noEmit",
                ],
                cwd=project_root / "frontend_new",
                timeout_s=args.frontend_timeout_s,
                env=os.environ.copy(),
            )
        )

    for stage in stages:
        stage_file = out_dir / f"{stage['name']}.json"
        stage_file.write_text(json.dumps(stage, ensure_ascii=False, indent=2), encoding="utf-8")
        tail_file = out_dir / f"{stage['name']}.tail.log"
        tail_file.write_text(
            "\n".join(
                [
                    f"[stage] {stage['name']}",
                    f"[returncode] {stage['returncode']}",
                    f"[timed_out] {stage['timed_out']}",
                    "",
                    "[stderr tail]",
                    (stage["stderr"] or "")[-4000:],
                    "",
                    "[stdout tail]",
                    (stage["stdout"] or "")[-4000:],
                ]
            ),
            encoding="utf-8",
        )

    summary = {
        "out_dir": str(out_dir),
        "ok": all((s.get("returncode") == 0 and not s.get("timed_out")) for s in stages),
        "stages": [
            {
                "name": s["name"],
                "returncode": s["returncode"],
                "timed_out": s["timed_out"],
            }
            for s in stages
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["ok"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project-fit regression suite for current smart_hospital_agent.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--python-timeout-s", type=int, default=1200)
    parser.add_argument("--frontend-timeout-s", type=int, default=300)
    parser.add_argument("--skip-frontend", action="store_true")
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also run legacy compatibility suites that are not part of mainline regression gate.",
    )
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()

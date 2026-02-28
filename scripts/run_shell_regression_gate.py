#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_json_line(stdout: str) -> Dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _read_summary(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    shell_cmd: List[str] = [
        sys.executable,
        "scripts/e2e_fullchain_logger.py",
        "--project-root",
        str(project_root),
        "--base-url",
        args.shell_base_url,
        "--cases-file",
        args.cases_file,
        "--case-timeout",
        str(args.case_timeout),
        "--stall-timeout",
        str(args.stall_timeout),
        "--global-timeout",
        str(args.global_timeout),
        "--no-progress",
    ]
    if args.start_backend:
        shell_cmd.append("--start-backend")

    shell_proc = subprocess.run(
        shell_cmd,
        cwd=str(project_root),
        text=True,
        capture_output=True,
        check=False,
    )
    shell_payload = _extract_json_line(shell_proc.stdout)
    out_dir = Path(str(shell_payload.get("out_dir", "") or "")).resolve() if shell_payload.get("out_dir") else None
    if out_dir is None:
        # fallback to latest e2e run
        candidates = sorted((project_root / "logs" / "e2e_fullchain").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        out_dir = candidates[0].resolve() if candidates else project_root / "logs" / "e2e_fullchain"
    summary_path = out_dir / "summary.json"
    summary = _read_summary(summary_path)
    shell_ok = bool(summary.get("ok", False))

    control_payload: Dict[str, Any] = {}
    if not shell_ok:
        control_cmd: List[str] = [
            sys.executable,
            "scripts/openwebui_control_rerun.py",
            "--summary",
            str(summary_path),
            "--openwebui-base-url",
            args.openwebui_base_url,
            "--openwebui-token",
            args.openwebui_token,
            "--openwebui-model",
            args.openwebui_model,
            "--timeout-s",
            str(args.openwebui_timeout),
            "--outdir",
            str(out_dir),
        ]
        control_proc = subprocess.run(
            control_cmd,
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        control_payload = _extract_json_line(control_proc.stdout)
        control_payload["exit_code"] = control_proc.returncode
        control_payload["stdout_tail"] = control_proc.stdout[-2000:]
        control_payload["stderr_tail"] = control_proc.stderr[-2000:]

    verdict_path = out_dir / "shell_boundary_verdict.json"
    verdict = {
        "generated_at": now_tag(),
        "shell_cmd": shell_cmd,
        "shell_exit_code": shell_proc.returncode,
        "shell_stdout_tail": shell_proc.stdout[-2000:],
        "shell_stderr_tail": shell_proc.stderr[-2000:],
        "shell_ok": shell_ok,
        "shell_out_dir": str(out_dir),
        "shell_summary_json": str(summary_path),
        "auto_control_rerun_triggered": not shell_ok,
        "control_payload": control_payload,
    }
    verdict_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict_md = out_dir / "shell_boundary_verdict.md"
    lines = [
        "# Shell Regression Gate + Open WebUI Control",
        "",
        f"- shell_ok: `{shell_ok}`",
        f"- shell_summary_json: `{summary_path}`",
        f"- auto_control_rerun_triggered: `{not shell_ok}`",
    ]
    if control_payload:
        lines.extend(
            [
                f"- openwebui_control_report_json: `{control_payload.get('openwebui_control_report_json', '')}`",
                f"- boundary_verdict: `{control_payload.get('boundary_verdict', '')}`",
            ]
        )
    verdict_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "shell_ok": shell_ok,
                "shell_out_dir": str(out_dir),
                "shell_summary_json": str(summary_path),
                "verdict_json": str(verdict_path),
                "verdict_md": str(verdict_md),
                "control_triggered": not shell_ok,
                "boundary_verdict": control_payload.get("boundary_verdict", "") if control_payload else "",
            },
            ensure_ascii=False,
        )
    )
    return 0 if shell_ok else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shell regression and auto-trigger Open WebUI control rerun on failures.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--shell-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--cases-file", default="scripts/e2e_cases_multiturn.json")
    parser.add_argument("--case-timeout", type=int, default=240)
    parser.add_argument("--stall-timeout", type=int, default=60)
    parser.add_argument("--global-timeout", type=int, default=1200)
    parser.add_argument("--start-backend", action="store_true")
    parser.add_argument("--openwebui-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--openwebui-token", default="")
    parser.add_argument("--openwebui-model", default="smart-hospital-agent")
    parser.add_argument("--openwebui-timeout", type=float, default=35.0)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()

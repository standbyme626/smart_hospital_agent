#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _candidate_paths(base_url: str) -> List[str]:
    base = base_url.rstrip("/")
    return [
        f"{base}/api/chat/completions",
        f"{base}/api/v1/chat/completions",
        f"{base}/openai/chat/completions",
    ]


def _extract_response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content") if isinstance(message, dict) else ""
        if isinstance(content, str) and content.strip():
            return content.strip()
    data = payload.get("data")
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return ""


def _control_case(
    *,
    client: httpx.Client,
    base_url: str,
    token: str,
    model: str,
    message: str,
    timeout_s: float,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }
    last_error = ""
    for url in _candidate_paths(base_url):
        try:
            resp = client.post(url, headers=headers, json=body, timeout=timeout_s)
        except Exception as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            continue
        if resp.status_code >= 400:
            last_error = f"http_{resp.status_code}"
            continue
        try:
            payload = resp.json()
        except Exception:
            payload = {}
        text = _extract_response_text(payload if isinstance(payload, dict) else {})
        if text:
            return {
                "ok": True,
                "endpoint": url,
                "response_preview": text[:500],
                "error": "",
            }
        last_error = "empty_response"
    return {
        "ok": False,
        "endpoint": "",
        "response_preview": "",
        "error": last_error or "control_unavailable",
    }


def _boundary_verdict(control_results: List[Dict[str, Any]]) -> str:
    if not control_results:
        return "not_required"
    passed = sum(1 for item in control_results if item.get("control_ok"))
    failed = len(control_results) - passed
    if passed > 0 and failed == 0:
        return "shell_or_proxy_regression"
    if passed > 0 and failed > 0:
        return "mixed_boundary"
    if failed > 0:
        sample_error = str(control_results[0].get("control_error", "") or "")
        if sample_error.startswith("http_") or "connect" in sample_error.lower():
            return "control_unavailable"
        return "backend_or_shared_issue"
    return "inconclusive"


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    summary_path = Path(args.summary).resolve()
    summary = read_json(summary_path)
    shell_results = summary.get("results") if isinstance(summary.get("results"), list) else []
    failed_cases = [item for item in shell_results if not bool(item.get("ok", False))]

    out_dir = Path(args.outdir).resolve() if args.outdir else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "openwebui_control_report.json"
    out_md = out_dir / "openwebui_control_report.md"

    control_results: List[Dict[str, Any]] = []
    with httpx.Client() as client:
        for item in failed_cases:
            case_id = str(item.get("case_id", "") or "")
            message = str(item.get("sent_content", "") or "")
            control = _control_case(
                client=client,
                base_url=args.openwebui_base_url,
                token=args.openwebui_token,
                model=args.openwebui_model,
                message=message,
                timeout_s=args.timeout_s,
            )
            control_results.append(
                {
                    "case_id": case_id,
                    "request_id": str(item.get("request_id", "") or ""),
                    "shell_failure_category": str(item.get("failure_category", "") or ""),
                    "shell_failure_segment": str(item.get("failure_segment", "") or ""),
                    "control_ok": bool(control.get("ok")),
                    "control_endpoint": str(control.get("endpoint", "")),
                    "control_error": str(control.get("error", "")),
                    "control_response_preview": str(control.get("response_preview", "")),
                }
            )

    passed = sum(1 for item in control_results if item.get("control_ok"))
    report = {
        "generated_at": now_iso(),
        "shell_summary_json": str(summary_path),
        "openwebui_base_url": args.openwebui_base_url,
        "shell_failed_cases": len(failed_cases),
        "control_rerun_count": len(control_results),
        "control_passed": passed,
        "control_failed": len(control_results) - passed,
        "boundary_verdict": _boundary_verdict(control_results),
        "results": control_results,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Open WebUI Control Rerun Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- shell_summary_json: `{report['shell_summary_json']}`",
        f"- openwebui_base_url: `{report['openwebui_base_url']}`",
        f"- shell_failed_cases: `{report['shell_failed_cases']}`",
        f"- control_passed: `{report['control_passed']}`",
        f"- control_failed: `{report['control_failed']}`",
        f"- boundary_verdict: `{report['boundary_verdict']}`",
        "",
        "| case_id | request_id | shell_failure | control_ok | control_endpoint | control_error |",
        "|---|---|---|---|---|---|",
    ]
    for item in control_results:
        lines.append(
            f"| `{item['case_id']}` | `{item['request_id']}` | `{item['shell_failure_category']}/{item['shell_failure_segment']}` | "
            f"`{item['control_ok']}` | `{item['control_endpoint']}` | `{item['control_error']}` |"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(
        json.dumps(
            {
                "openwebui_control_report_json": str(out_json),
                "openwebui_control_report_md": str(out_md),
                "boundary_verdict": report["boundary_verdict"],
                "control_rerun_count": report["control_rerun_count"],
            },
            ensure_ascii=False,
        )
    )
    return out_json, out_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun shell-failed cases through Open WebUI control group.")
    parser.add_argument("--summary", required=True, help="Path to shell e2e summary.json")
    parser.add_argument("--openwebui-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--openwebui-token", default="")
    parser.add_argument("--openwebui-model", default="smart-hospital-agent")
    parser.add_argument("--timeout-s", type=float, default=35.0)
    parser.add_argument("--outdir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

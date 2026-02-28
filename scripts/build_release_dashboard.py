#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _mtime_sort(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)


def collect_e2e_runs(project_root: Path, max_runs: int) -> List[Dict[str, Any]]:
    root = project_root / "logs" / "e2e_fullchain"
    if not root.exists():
        return []
    out: List[Dict[str, Any]] = []
    for run_dir in _mtime_sort([p for p in root.iterdir() if p.is_dir()])[:max_runs]:
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        out.append(
            {
                "run_id": summary.get("run_id") or run_dir.name,
                "ok": bool(summary.get("ok", False)),
                "total_cases": int(summary.get("total_cases", 0) or 0),
                "passed_cases": int(summary.get("passed_cases", 0) or 0),
                "started_at": summary.get("started_at", ""),
                "finished_at": summary.get("finished_at", ""),
                "summary_json": str(summary_path),
                "report_md": str(run_dir / "report.md"),
                "events_jsonl": str(run_dir / "events.jsonl"),
                "trace_request_map_jsonl": str(summary.get("trace_request_map_jsonl") or ""),
                "trace_request_map_md": str(summary.get("trace_request_map_md") or ""),
            }
        )
    return out


def _glob_reports(project_root: Path, pattern: str) -> List[Path]:
    return _mtime_sort(project_root.glob(pattern))


def collect_gate_reports(project_root: Path, max_reports: int) -> List[Dict[str, Any]]:
    files = _glob_reports(project_root, "logs/**/release_gate*.json")
    out: List[Dict[str, Any]] = []
    for path in files[:max_reports]:
        payload = read_json(path)
        out.append(
            {
                "status": payload.get("status", ""),
                "gate_passed": payload.get("gate_passed", ""),
                "bootstrap_mode": payload.get("bootstrap_mode", ""),
                "generated_at": payload.get("generated_at", ""),
                "json_path": str(path),
                "md_path": str(path.with_suffix(".md")),
            }
        )
    return out


def collect_dynamic_reports(project_root: Path, max_reports: int) -> List[Dict[str, Any]]:
    files = _glob_reports(project_root, "logs/**/dynamic_regression*.jsonl")
    out: List[Dict[str, Any]] = []
    for path in files[:max_reports]:
        rows = read_jsonl(path)
        failed = sum(1 for row in rows if str(row.get("status", "")).lower() == "failed")
        out.append(
            {
                "total": len(rows),
                "failed": failed,
                "passed": len(rows) - failed,
                "pass_rate": round(((len(rows) - failed) / len(rows)), 6) if rows else 0.0,
                "jsonl_path": str(path),
                "md_path": str(path.with_suffix(".md")),
            }
        )
    return out


def collect_openwebui_control_reports(project_root: Path, max_reports: int) -> List[Dict[str, Any]]:
    files = _glob_reports(project_root, "logs/**/openwebui_control_report.json")
    verdict_files = _glob_reports(project_root, "logs/**/shell_boundary_verdict.json")
    out: List[Dict[str, Any]] = []
    for path in files[:max_reports]:
        payload = read_json(path)
        out.append(
            {
                "summary_json": str(path),
                "summary_md": str(path.with_suffix(".md")),
                "shell_failed_cases": payload.get("shell_failed_cases", 0),
                "control_rerun_count": payload.get("control_rerun_count", 0),
                "control_passed": payload.get("control_passed", 0),
                "control_failed": payload.get("control_failed", 0),
                "boundary_verdict": payload.get("boundary_verdict", ""),
                "generated_at": payload.get("generated_at", ""),
            }
        )
    for path in verdict_files[:max_reports]:
        payload = read_json(path)
        out.append(
            {
                "summary_json": str(path),
                "summary_md": str(path.with_suffix(".md")),
                "shell_failed_cases": "",
                "control_rerun_count": "",
                "control_passed": "",
                "control_failed": "",
                "boundary_verdict": payload.get("control_payload", {}).get("boundary_verdict", ""),
                "generated_at": payload.get("generated_at", ""),
            }
        )
    return out


def collect_replay_index(project_root: Path) -> List[Dict[str, Any]]:
    return read_jsonl(project_root / "logs" / "trace_replay" / "replay_index.jsonl")


def request_lookup(entries: List[Dict[str, Any]], request_id: str, max_hits: int) -> List[Dict[str, Any]]:
    target = str(request_id or "").strip()
    if not target:
        return []
    hits = [item for item in entries if str(item.get("request_id", "")).strip() == target]
    return hits[:max_hits]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_markdown(dashboard: Dict[str, Any], request_hits: List[Dict[str, Any]]) -> str:
    lines: List[str] = [
        "# Release Dashboard",
        "",
        f"- generated_at: `{dashboard.get('generated_at', '')}`",
        f"- e2e_runs: `{len(dashboard.get('e2e_runs', []))}`",
        f"- gate_reports: `{len(dashboard.get('gate_reports', []))}`",
        f"- dynamic_reports: `{len(dashboard.get('dynamic_reports', []))}`",
        f"- openwebui_control_reports: `{len(dashboard.get('openwebui_control_reports', []))}`",
        "",
        "## Latest E2E",
        "",
        "| run_id | ok | passed/total | summary | report | trace_map |",
        "|---|---|---:|---|---|---|",
    ]
    for item in dashboard.get("e2e_runs", [])[:10]:
        lines.append(
            f"| `{item.get('run_id','')}` | `{item.get('ok', False)}` | "
            f"`{item.get('passed_cases',0)}/{item.get('total_cases',0)}` | "
            f"`{item.get('summary_json','')}` | `{item.get('report_md','')}` | `{item.get('trace_request_map_jsonl','')}` |"
        )

    lines.extend(
        [
            "",
            "## Latest Gate Reports",
            "",
            "| status | gate_passed | json | md |",
            "|---|---|---|---|",
        ]
    )
    for item in dashboard.get("gate_reports", [])[:10]:
        lines.append(
            f"| `{item.get('status','')}` | `{item.get('gate_passed','')}` | "
            f"`{item.get('json_path','')}` | `{item.get('md_path','')}` |"
        )

    lines.extend(
        [
            "",
            "## Latest Dynamic Regression",
            "",
            "| passed | failed | pass_rate | jsonl | md |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for item in dashboard.get("dynamic_reports", [])[:10]:
        lines.append(
            f"| `{item.get('passed',0)}` | `{item.get('failed',0)}` | `{item.get('pass_rate',0.0)}` | "
            f"`{item.get('jsonl_path','')}` | `{item.get('md_path','')}` |"
        )

    lines.extend(
        [
            "",
            "## Open WebUI Control Reruns",
            "",
            "| shell_failed | control_passed | control_failed | verdict | report |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for item in dashboard.get("openwebui_control_reports", [])[:10]:
        lines.append(
            f"| `{item.get('shell_failed_cases',0)}` | `{item.get('control_passed',0)}` | `{item.get('control_failed',0)}` | "
            f"`{item.get('boundary_verdict','')}` | `{item.get('summary_json','')}` |"
        )

    lines.extend(
        [
            "",
            "## Replay Index",
            "",
            f"- total_entries: `{dashboard.get('replay_index_total', 0)}`",
            "- lookup command: `python scripts/replay_trace_lookup.py --request-id <id>`",
            "- dashboard lookup: `python scripts/build_release_dashboard.py --request-id <id>`",
        ]
    )

    if request_hits:
        lines.extend(
            [
                "",
                "## Request Lookup",
                "",
                f"- query_request_id: `{dashboard.get('query_request_id','')}`",
                f"- hit_count: `{len(request_hits)}`",
                "",
                "| source | request_id | trace_id | artifacts |",
                "|---|---|---|---|",
            ]
        )
        for item in request_hits:
            lines.append(
                f"| `{item.get('source','')}` | `{item.get('request_id','')}` | `{item.get('trace_id','')}` | "
                f"`{json.dumps(item.get('artifacts', {}), ensure_ascii=False)}` |"
            )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    project_root = Path(args.project_root).resolve()
    out_root = (project_root / args.out_dir).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    e2e_runs = collect_e2e_runs(project_root, args.max_reports)
    gate_reports = collect_gate_reports(project_root, args.max_reports)
    dynamic_reports = collect_dynamic_reports(project_root, args.max_reports)
    control_reports = collect_openwebui_control_reports(project_root, args.max_reports)
    replay_entries = collect_replay_index(project_root)
    request_hits = request_lookup(replay_entries, args.request_id, args.max_request_hits)

    dashboard = {
        "generated_at": now_iso(),
        "query_request_id": str(args.request_id or "").strip(),
        "e2e_runs": e2e_runs,
        "gate_reports": gate_reports,
        "dynamic_reports": dynamic_reports,
        "openwebui_control_reports": control_reports,
        "replay_index_total": len(replay_entries),
        "request_hits": request_hits,
    }

    json_path = out_dir / "release_dashboard.json"
    md_path = out_dir / "release_dashboard.md"
    write_json(json_path, dashboard)
    md_path.write_text(build_markdown(dashboard, request_hits), encoding="utf-8")

    latest_json = out_root / "release_dashboard_latest.json"
    latest_md = out_root / "release_dashboard_latest.md"
    write_json(latest_json, dashboard)
    latest_md.write_text(build_markdown(dashboard, request_hits), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "release_dashboard_json": str(json_path),
                "release_dashboard_md": str(md_path),
                "latest_json": str(latest_json),
                "latest_md": str(latest_md),
                "request_hits": len(request_hits),
            },
            ensure_ascii=False,
        )
    )
    return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate release artifacts into searchable dashboard.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out-dir", default="logs/release_dashboard")
    parser.add_argument("--max-reports", type=int, default=30)
    parser.add_argument("--request-id", default="")
    parser.add_argument("--max-request-hits", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

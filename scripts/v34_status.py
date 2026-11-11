#!/usr/bin/env python3
"""V3.4 执行台账校验器。

从 docs/V3.4-EXECUTION-LEDGER.md 的 61 屏矩阵读出行，逐行核对代码证据：
  页面文件存在 / API 路由存在 / 测试文件存在
并给出每端统计与"状态与证据不符"的告警。

用法：
  python3 scripts/v34_status.py           # 人读
  python3 scripts/v34_status.py --strict  # 证据不足 → 退出码 1（CI / 门禁）
"""
from __future__ import annotations

import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "V3.4-EXECUTION-LEDGER.md"
EXPECTED = OrderedDict([("患者", 16), ("医生", 12), ("运营", 8), ("管理", 25)])
ROW_RE = re.compile(r"^\|\s*([pdoa]-[a-z0-9-]+)\s*\|")

C_OK, C_WARN, C_BAD, C_DIM, C_END = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def use_color() -> bool:
    return sys.stdout.isatty()


def paint(text: str, color: str) -> str:
    return f"{color}{text}{C_END}" if use_color() else text


def load_backend_source() -> str:
    blobs = []
    for path in sorted((ROOT / "backend" / "app").glob("*.py")):
        blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(blobs)


def parse_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        rows.append(
            {
                "screen": parts[0],
                "name": parts[1],
                "end": parts[2],
                "status": parts[3],
                "page": parts[4],
                "api": parts[5],
                "test": parts[6],
            }
        )
    return rows


def check(row: dict[str, str], backend_src: str) -> tuple[bool, bool, bool, list[str]]:
    notes: list[str] = []
    page = row["page"]
    api = row["api"]
    test = row["test"]

    page_ok = page == "-" or (ROOT / page).exists()
    if page != "-" and not page_ok:
        notes.append(f"页面不存在 {page}")

    api_ok = api == "-"
    if api != "-":
        api_ok = api in backend_src
        if not api_ok:
            notes.append(f"路由未找到 {api}")

    test_ok = test == "-" or (ROOT / test).exists()
    if test != "-" and not test_ok:
        notes.append(f"测试不存在 {test}")

    return page_ok, api_ok, test_ok, notes


def main() -> int:
    strict = "--strict" in sys.argv
    if not LEDGER.exists():
        print(paint(f"台账不存在：{LEDGER}", C_BAD))
        return 1

    rows = parse_rows(LEDGER.read_text(encoding="utf-8"))
    backend_src = load_backend_source()

    counts: dict[str, Counter] = {k: Counter() for k in EXPECTED}
    problems: list[str] = []
    seen: Counter = Counter()

    print()
    print("V3.4 执行台账校验 · 61 屏")
    print("=" * 74)

    current_end = None
    for row in rows:
        end = row["end"]
        seen[end] += 1
        if end != current_end:
            current_end = end
            print(f"\n【{end}端】")
            print(f"  {'屏':<16}{'名称':<20}{'状态':<9}证据")
        page_ok, api_ok, test_ok, notes = check(row, backend_src)
        status = row["status"]
        counts.setdefault(end, Counter())[status] += 1
        mark = paint("OK ", C_OK) if not notes else paint("!! ", C_WARN)
        evidence = "页面" + ("+ " if page_ok and row["page"] != "-" else "- ") + \
                   "接口" + ("+ " if api_ok and row["api"] != "-" else "- ") + \
                   "测试" + ("+" if test_ok and row["test"] != "-" else "-")
        print(f"  {row['screen']:<16}{row['name']:<20}{status:<9}{mark}{evidence}")
        if notes:
            problems.append(f"{row['screen']}（{status}）：" + "；".join(notes))
        if strict and status in {"LIVE", "PARTIAL", "DOING"} and notes:
            problems.append(f"{row['screen']}（{status}）证据不足，--strict 判失败")

    print("\n" + "=" * 74)
    print("每端统计（实测台账）")
    print(f"  {'端':<6}{'LIVE':>6}{'PARTIAL':>9}{'DOING':>8}{'TOOL':>7}{'GAP':>6}{'合计':>6}{'计划':>6}")
    total = Counter()
    bad_shape = False
    for end, expected in EXPECTED.items():
        c = counts.get(end, Counter())
        line_total = sum(c.values())
        total.update(c)
        flag = "" if line_total == expected else paint("  ← 屏数不符", C_BAD)
        if line_total != expected:
            bad_shape = True
        print(
            f"  {end:<6}{c['LIVE']:>6}{c['PARTIAL']:>9}{c['DOING']:>8}"
            f"{c['TOOL']:>7}{c['GAP']:>6}{line_total:>6}{expected:>6}{flag}"
        )
    grand = sum(total.values())
    print(
        f"  {'合计':<6}{total['LIVE']:>6}{total['PARTIAL']:>9}{total['DOING']:>8}"
        f"{total['TOOL']:>7}{total['GAP']:>6}{grand:>6}{sum(EXPECTED.values()):>6}"
    )

    tests = len(list((ROOT / "backend" / "tests").glob("test_*.py")))
    specs = len(list((ROOT / "frontend" / "e2e").glob("*.spec.ts")))
    print(f"\n测试文件：pytest {tests} 个 · Playwright {specs} 个 spec")

    if problems:
        print("\n需要处理（状态与证据不符 / 未找到）：")
        for item in problems:
            print("  - " + item)
    else:
        print("\n全部行证据齐全 ✅")

    if bad_shape:
        print(paint("\n屏数与台账期望不符（61 屏口径被破坏）", C_BAD))

    if strict and (problems or bad_shape):
        print(paint("\n--strict 判定：未通过", C_BAD))
        return 1
    print(paint("\n校验完成", C_OK))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""V3.4 修改记录校验器。

把 docs/V3.4-CHANGELOG.md 从"聊天式日记"变成可信流水：
  1. 每条记录必须有 类型 / commit / 线 / 屏 / 改动 / 验证；
  2. commit 必须是真实存在的、且在当前分支上的提交；
  3. 屏 ID 必须在 prototype/dual-end.html 的 61 个 data-screen 里；
  4. 「在途占用」表里同一文件不允许被两行同时占用（防并发覆盖）。

用法：
  python3 scripts/v34_changelog.py            # 人读
  python3 scripts/v34_changelog.py --strict   # 有问题就非 0 退出（门禁/CI）
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "docs" / "V3.4-CHANGELOG.md"
PROTOTYPE = ROOT / "prototype" / "dual-end.html"

TYPES = {"feat", "fix", "test", "docs", "chore", "refactor", "基线"}
HEADER_RE = re.compile(r"^### (\d{4}-\d{2}-\d{2} \d{2}:\d{2}) · ([^ ·]+) · ([^ ·]+) · (.+)$")
REQUIRED_BULLETS = ("屏：", "改动：", "验证：")

C_OK, C_WARN, C_BAD, C_END = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def paint(text: str, color: str) -> str:
    return f"{color}{text}{C_END}" if sys.stdout.isatty() else text


def screen_ids() -> set[str]:
    html = PROTOTYPE.read_text(encoding="utf-8")
    return set(re.findall(r'data-screen="([^"]+)"', html))


def git(*args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def split_sections(text: str) -> tuple[str, str, str]:
    """返回 (在途占用段, 变更记录段, 全文)。"""
    in_flight = ""
    if "## 1." in text and "## 2." in text:
        in_flight = text[text.index("## 1.") : text.index("## 2.")]
    records = text[text.index("## 2.") :] if "## 2." in text else text
    if "## 3." in records:
        records = records[: records.index("## 3.")]
    return in_flight, records, text


def parse_entries(records: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in records.splitlines():
        match = HEADER_RE.match(line.strip())
        if match:
            if current:
                entries.append(current)
            stamp, kind, commit, owner = match.groups()
            current = {
                "stamp": stamp,
                "type": kind,
                "commit": commit,
                "owner": owner.strip(),
                "body": [],
                "line": line.strip(),
            }
            continue
        if current is not None and line.startswith("- "):
            current["body"].append(line)
    if current:
        entries.append(current)
    return entries


def check_entries(entries: list[dict[str, object]], ids: set[str]) -> tuple[list[str], Counter]:
    problems: list[str] = []
    stats: Counter = Counter()

    if not entries:
        problems.append("没有任何变更记录（## 2. 变更记录 为空）")
        return problems, stats

    for entry in entries:
        stamp = str(entry["stamp"])
        kind = str(entry["type"])
        commit = str(entry["commit"])
        owner = str(entry["owner"])
        body = "\n".join(entry["body"])

        stats["total"] += 1
        stats[f"type:{kind}"] += 1
        stats[f"owner:{owner}"] += 1

        if kind not in TYPES:
            problems.append(f"{stamp}：类型 {kind} 不合法（允许：{'/'.join(sorted(TYPES))}）")

        for bullet in REQUIRED_BULLETS:
            if bullet not in body:
                problems.append(f"{stamp}：缺少必填字段「{bullet.rstrip('：')}」")

        screens = re.findall(r"\b([pdoa]-[a-z0-9-]+)\b", body)
        for sid in screens:
            if sid not in ids:
                problems.append(f"{stamp}：屏 ID {sid} 不在 61 屏清单里（拼错或自造）")

        if commit == "未提交":
            stats["uncommitted"] += 1
            continue
        if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
            problems.append(f"{stamp}：commit {commit} 不是合法 hash（未提交请写「未提交」）")
            continue
        code, _ = git("cat-file", "-e", f"{commit}^{{commit}}")
        if code != 0:
            problems.append(f"{stamp}：commit {commit} 在仓库里不存在")
            continue
        code, _ = git("merge-base", "--is-ancestor", commit, "HEAD")
        if code != 0:
            problems.append(f"{stamp}：commit {commit} 不在当前分支 HEAD 上（未合并？）")

    return problems, stats


def check_in_flight(in_flight: str) -> list[str]:
    problems: list[str] = []
    owners: dict[str, str] = {}
    rows = 0

    for line in in_flight.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5 or cells[0].startswith("开始时间") or set(cells[0]) <= {"-", ":"}:
            continue
        rows += 1
        stamp, owner, _screens, files_cell = cells[0], cells[1], cells[2], cells[3]
        for token in re.findall(r"[\x60]([^\x60]+)[\x60]", files_cell):
            path = token.split(" / ")[0].rstrip("*").rstrip("/")
            if not path:
                continue
            if not (ROOT / path).exists():
                problems.append(f"在途占用：{owner} 登记的文件不存在：{path}")
            previous = owners.get(path)
            if previous and previous != owner:
                problems.append(f"在途占用：{path} 同时被「{previous}」与「{owner}」占用")
            owners[path] = owner
        if not files_cell:
            problems.append(f"在途占用：{owner}（{stamp}）没有写正在改的文件")

    print(f"在途占用：{rows} 行 / {len(owners)} 个文件")
    return problems


def main() -> int:
    strict = "--strict" in sys.argv
    if not CHANGELOG.exists():
        print(paint(f"修改记录不存在：{CHANGELOG}", C_BAD))
        return 1

    text = CHANGELOG.read_text(encoding="utf-8")
    ids = screen_ids()
    in_flight, records, _ = split_sections(text)
    entries = parse_entries(records)

    print()
    print("V3.4 修改记录校验")
    print("=" * 70)

    problems = check_in_flight(in_flight)
    entry_problems, stats = check_entries(entries, ids)
    problems.extend(entry_problems)

    print(f"变更记录：{stats['total']} 条（未提交 {stats['uncommitted']} 条）")
    for key in sorted(k for k in stats if k.startswith("owner:")):
        print(f"  {key[6:]:<12}{stats[key]} 条")
    print(f"覆盖屏 ID：{len(ids)} 个可用（prototype/dual-end.html）")

    if problems:
        print("\n需要处理：")
        for item in problems:
            print("  - " + paint(item, C_WARN))
    else:
        print(paint("\n记录齐全，hash 与屏 ID 均校验通过 ✅", C_OK))

    if strict and problems:
        print(paint("\n--strict 判定：未通过", C_BAD))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

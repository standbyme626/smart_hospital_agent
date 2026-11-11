#!/usr/bin/env python3
"""审计抽查：确认"关键动作都留痕了、审计里没有敏感内容"。

用法：
  python3 scripts/audit_spotcheck.py            # 近 7 天，中文报告
  python3 scripts/audit_spotcheck.py --days 30  # 指定窗口
  python3 scripts/audit_spotcheck.py --strict   # 有高危问题 → 非 0 退出（进发布门禁）

检查项：
  1. 关键动作覆盖（登录 / 医生写入 / 运营处置 / 配置发布 / 打印 / 诊毕）；
  2. 敏感内容扫描（detail 里出现 password / token / secret / authorization）；
  3. 无操作人的记录比例（应为 0 或极低）；
  4. 单一操作人占比（>90% 且有大量记录 → 提示，可能是脚本或凭据共用）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]

# 动作前缀 -> 中文名（缺哪个就是"这条链路没留痕"）
REQUIRED_ACTIONS = {
    "auth.login": "登录",
    "clinician.print.create": "文书打印生成",
    "clinician.visit.closed": "诊毕",
    "clinician.orders.signed": "医嘱签名",
    "admin.config.publish": "配置发布",
    "admin.his.check": "HIS 连通性检查",
    "ops.": "运营处置",
    "admin.tenant.create": "院区创建",
}
SENSITIVE = re.compile(r"(password|passwd|token|secret|authorization|api[_-]?key)", re.I)


def load_database_url() -> str:
    env_path = ROOT / ".env"
    if env_path.exists():
        match = re.search(r"^DATABASE_URL=(.+)$", env_path.read_text(encoding="utf-8"), re.M)
        if match:
            return match.group(1).strip().strip('"').strip("'")
    raise SystemExit("找不到 DATABASE_URL（.env 或环境变量）")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    conn = await asyncpg.connect(load_database_url())
    problems: list[str] = []
    warnings: list[str] = []
    try:
        window = f"now() - interval '{args.days} days'"
        total = await conn.fetchval(f"SELECT count(*) FROM audit_log WHERE created_at >= {window}")
        print(f"审计抽查：近 {args.days} 天共 {total} 条审计记录")

        actions = await conn.fetch(
            f"SELECT action, count(*) AS count FROM audit_log WHERE created_at >= {window} GROUP BY 1 ORDER BY 2 DESC"
        )
        by_action = {row["action"]: int(row["count"]) for row in actions}
        print("\n【动作分布】")
        for action, count in list(by_action.items())[:15]:
            print(f"  {count:>6}  {action}")

        print("\n【关键动作覆盖】")
        for prefix, label in REQUIRED_ACTIONS.items():
            hit = sum(count for action, count in by_action.items() if action == prefix or action.startswith(prefix))
            mark = "OK " if hit else "缺失"
            print(f"  {mark}  {label:<12} {prefix:<26} 命中 {hit}")
            if not hit:
                warnings.append(f"窗口内没有「{label}」（{prefix}）的审计记录")

        no_actor = await conn.fetchval(
            f"SELECT count(*) FROM audit_log WHERE created_at >= {window} AND (actor IS NULL OR actor = '')"
        )
        print(f"\n【操作人】无操作人记录 {no_actor} 条")
        if no_actor and total and no_actor / total > 0.05:
            problems.append(f"无操作人记录占比过高：{no_actor}/{total}")

        actors = await conn.fetch(
            f"SELECT coalesce(actor,'') AS actor, count(*) AS count FROM audit_log WHERE created_at >= {window} GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
        )
        if actors and total:
            top_actor, top_count = actors[0]["actor"], int(actors[0]["count"])
            print(f"  最高频操作人：{top_actor or '（空）'} {top_count} 条（{top_count / total:.1%}）")
            if top_count / total > 0.9 and total > 100:
                warnings.append(f"操作人「{top_actor}」占 {top_count / total:.1%}，确认是否为服务账号")

        rows = await conn.fetch(
            f"SELECT id, action, detail FROM audit_log WHERE created_at >= {window} ORDER BY id DESC LIMIT 500"
        )
        leaked = []
        for row in rows:
            raw = row["detail"]
            text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
            if SENSITIVE.search(text):
                leaked.append((row["id"], row["action"]))
        print(f"\n【敏感内容扫描】最近 {len(rows)} 条里命中 {len(leaked)} 条")
        for row_id, action in leaked[:5]:
            print(f"  #{row_id}  {action}")
        if leaked:
            problems.append(f"审计 detail 里疑似包含敏感字段：{len(leaked)} 条（见上）")
    finally:
        await conn.close()

    print("\n【结论】")
    if problems:
        for item in problems:
            print("  高危：" + item)
    if warnings:
        for item in warnings:
            print("  关注：" + item)
    if not problems and not warnings:
        print("  未发现异常")

    if args.strict and problems:
        print("\n--strict 判定：未通过")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

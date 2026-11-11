#!/usr/bin/env python3
"""Seed the platform-owned doctor directory (12 doctors).

HIS is read-only in this phase: doctors/schedules that HIS does not expose are
maintained here, and pages label them as "平台数据 · 模拟".
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import re

import asyncpg

ROOT = pathlib.Path(__file__).resolve().parents[1]

DOCTORS = [
    ("doc_001", "张明", "心血管内科", "主任医师", "冠心病、高血压、心衰", 50),
    ("doc_002", "李红", "心血管内科", "副主任医师", "心律失常、房颤", 40),
    ("doc_003", "陈涛", "心血管内科", "主治医师", "高血压与血脂管理", 30),
    ("doc_004", "王琳", "神经内科", "主任医师", "卒中、头痛、眩晕", 50),
    ("doc_005", "刘洋", "神经内科", "主治医师", "帕金森、记忆障碍", 30),
    ("doc_006", "赵敏", "内分泌科", "副主任医师", "糖尿病、甲状腺", 40),
    ("doc_007", "孙磊", "呼吸内科", "主任医师", "慢阻肺、哮喘、肺炎", 50),
    ("doc_008", "周静", "消化内科", "副主任医师", "胃食管反流、肝病", 40),
    ("doc_009", "吴刚", "普通外科", "主任医师", "胆囊、疝、甲状腺结节", 50),
    ("doc_010", "郑霞", "皮肤科", "副主任医师", "湿疹、银屑病、皮肤肿瘤", 40),
    ("doc_011", "冯雪", "儿科", "主治医师", "发热、咳嗽、生长发育", 30),
    ("doc_012", "何静", "妇产科", "主任医师", "孕期管理、妇科炎症", 50),
]


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    text = (ROOT / ".env").read_text(encoding="utf-8")
    match = re.search(r"^DATABASE_URL=(.+)$", text, re.M)
    if not match:
        raise SystemExit("DATABASE_URL not found in env or .env")
    return match.group(1).strip().strip('"').strip("'")


async def main() -> None:
    conn = await asyncpg.connect(database_url())
    try:
        for doctor_id, name, department, title, intro, fee in DOCTORS:
            await conn.execute(
                """
                INSERT INTO doctor_directory (id, name, department, title, intro, fee, active)
                VALUES ($1,$2,$3,$4,$5,$6,true)
                ON CONFLICT (id) DO UPDATE
                SET name=EXCLUDED.name, department=EXCLUDED.department,
                    title=EXCLUDED.title, intro=EXCLUDED.intro, fee=EXCLUDED.fee,
                    active=true
                """,
                doctor_id,
                name,
                department,
                title,
                intro,
                fee,
            )
        total = await conn.fetchval("SELECT count(*) FROM doctor_directory WHERE active")
        print(f"seeded doctors: {len(DOCTORS)}; active total: {total}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())

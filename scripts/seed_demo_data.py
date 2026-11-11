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

        paid = await conn.fetch(
            """
            SELECT id, user_id, member_id, amount, department
            FROM appointment_projection
            WHERE payment_status = 'paid'
            ORDER BY created_at DESC LIMIT 20
            """
        )
        for index, row in enumerate(paid, start=1):
            total_amount = float(row["amount"] or 0)
            insurance = round(total_amount * 0.7, 2)
            account = round(total_amount * 0.2, 2)
            self_pay = round(total_amount - insurance - account, 2)
            await conn.execute(
                """
                INSERT INTO dispensing_progress (id, user_id, member_id, appointment_id, status, "window", ready_at)
                VALUES ($1,$2,$3,$4,'ready','3 号窗口', now())
                ON CONFLICT (id) DO NOTHING
                """,
                "disp_" + row["id"],
                row["user_id"],
                row["member_id"],
                row["id"],
            )
            await conn.execute(
                """
                INSERT INTO insurance_settlements
                    (id, user_id, member_id, appointment_id, total, insurance, account, self_pay, status)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'settled')
                ON CONFLICT (id) DO NOTHING
                """,
                "ins_" + row["id"],
                row["user_id"],
                row["member_id"],
                row["id"],
                total_amount,
                insurance,
                account,
                self_pay,
            )
            await conn.execute(
                """
                INSERT INTO queue_tickets (id, user_id, member_id, appointment_id, ticket_no, department, status)
                VALUES ($1,$2,$3,$4,$5,$6,'waiting')
                ON CONFLICT (id) DO NOTHING
                """,
                "q_" + row["id"],
                row["user_id"],
                row["member_id"],
                row["id"],
                "A" + str(30 + index).zfill(3),
                row["department"],
            )
        print(f"seeded patient-side rows for paid appointments: {len(paid)}")

        schedules = [
            ("周一", "am", "clinic", "心血管内科门诊"),
            ("周二", "am", "clinic", "心血管内科门诊"),
            ("周二", "pm", "clinic", "心血管内科门诊"),
            ("周四", "am", "clinic", "心血管内科门诊"),
            ("周五", "pm", "clinic", "心血管内科门诊"),
        ]
        for weekday, period, status, note in schedules:
            await conn.execute(
                """
                INSERT INTO doctor_schedules (id, doctor, department, weekday, period, status, note)
                VALUES ($1,'admin','心血管内科',$2,$3,$4,$5)
                ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, note=EXCLUDED.note
                """,
                f"sch_{weekday}_{period}",
                weekday,
                period,
                status,
                note,
            )
        print(f"seeded schedules: {len(schedules)}")

        members = {row["member_id"] for row in paid}
        lab_seed = [
            ("低密度脂蛋白 LDL-C", "3.8 mmol/L", "< 3.4", "↑"),
            ("空腹血糖", "6.4 mmol/L", "3.9–6.1", "↑"),
            ("肌酐", "96 μmol/L", "57–97", "正常"),
        ]
        count = 0
        for member_id in members:
            for name, value, reference, abnormal in lab_seed:
                await conn.execute(
                    """
                    INSERT INTO lab_results (id, member_id, name, result_value, reference_range, abnormal)
                    VALUES ($1,$2,$3,$4,$5,$6)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    f"lab_{member_id}_{name}",
                    member_id,
                    name,
                    value,
                    reference,
                    abnormal,
                )
                count += 1
        print(f"seeded lab results: {count}")

        crises = [
            ("crisis_demo_1", "chest_pain", "胸痛伴大汗，已提示急诊", "open"),
            ("crisis_demo_2", "self_harm", "言语中提及轻生念头，已转人工", "open"),
        ]
        for crisis_id, kind, detail, status in crises:
            await conn.execute(
                """
                INSERT INTO crisis_events (id, kind, detail, status)
                VALUES ($1,$2,$3,$4)
                ON CONFLICT (id) DO NOTHING
                """,
                crisis_id,
                kind,
                detail,
                status,
            )
        print(f"seeded crisis events: {len(crises)}")

        existing = await conn.fetchval("SELECT count(*) FROM notifications WHERE user_id='admin'")
        if not existing:
            for kind, title, body, link in [
                ("system", "欢迎使用大医", "这是平台演示通知", "/messages/"),
                ("followup", "随访提醒", "2 周后复诊心内科", "/followups/"),
                ("report", "报告解读完成", "血脂四项已生成解读", "/reports/"),
            ]:
                await conn.execute(
                    "INSERT INTO notifications (user_id, kind, title, body, link) VALUES ('admin',$1,$2,$3,$4)",
                    kind,
                    title,
                    body,
                    link,
                )
        notes = await conn.fetch("SELECT id, user_id, title FROM notifications WHERE user_id='admin' ORDER BY created_at DESC LIMIT 5")
        for index, row in enumerate(notes, start=1):
            status = "failed" if index == 2 else "sent"
            await conn.execute(
                """
                INSERT INTO notification_deliveries
                    (id, notification_id, channel, target, status, attempts, error)
                VALUES ($1,$2,'station',$3,$4,1,$5)
                ON CONFLICT (id) DO NOTHING
                """,
                f"nd_{row['id']}",
                row["id"],
                row["user_id"],
                status,
                "模拟通道失败（可重试）" if status == "failed" else None,
            )
        print(f"seeded notification deliveries: {len(notes)}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())

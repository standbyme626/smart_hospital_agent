"""医院本地时区（Asia/Shanghai）。日期类计算一律用它，避免 UTC 与本地日期错一天。"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

HOSPITAL_TZ = ZoneInfo("Asia/Shanghai")


def hospital_today() -> date:
    return datetime.now(HOSPITAL_TZ).date()


def hospital_now() -> datetime:
    return datetime.now(HOSPITAL_TZ)

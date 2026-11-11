"""V3.4 验收剧本登记表（单一数据源）。

定义见 docs/V3.2-DUAL-END-PLAN.md §6：
  P-01..P-08 患者端 / D-01..D-08 医生端 / E2E-01..E2E-03 双端闭环 / INT-01..INT-06 集成一致性

应用侧（管理端「验收自测」屏）与测试侧（tests/test_v34_playbooks.py）共用本文件，
避免两处各维护一份导致口径漂移。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Playbook:
    pid: str
    name: str
    area: str
    status: str
    endpoint: str
    evidence_test: str
    note: str = ""


PLAYBOOKS: tuple[Playbook, ...] = (
    # ---- 患者端 P-01 ~ P-08
    Playbook("P-01", "新患者建档", "患者", "ready", "/api/v1/family", "backend/tests/test_v34_playbooks_exec.py", "同身份证复用档案（返回 reused=true）；HIS 建档由适配器在挂号时完成"),
    Playbook("P-02", "查询号源", "患者", "ready", "/api/v1/booking/schedules", "backend/tests/test_v31_booking.py"),
    Playbook("P-03", "挂号", "患者", "ready", "/api/v1/booking/orders", "backend/tests/test_booking_v2.py"),
    Playbook("P-04", "缴费", "患者", "ready", "/api/v1/booking/orders/{order_id}/pay", "backend/tests/test_booking_v2.py"),
    Playbook("P-05", "重复缴费幂等", "患者", "ready", "/api/v1/booking/orders/{order_id}/pay", "backend/tests/test_v31_booking.py", "患者直接支付被禁（走人工审核），幂等由 MCP 侧 create_order + 稳定 idempotency_key 保证"),
    Playbook("P-06", "取消 / 退费", "患者", "ready", "/api/v1/booking/orders/{order_id}/cancel", "backend/tests/test_v31_booking.py"),
    Playbook("P-07", "报告解读不编造", "患者", "ready", "/api/v1/reports/interpret", "backend/tests/test_v31_reports.py"),
    Playbook("P-08", "家庭档案隔离", "患者", "ready", "/api/v1/family", "backend/tests/test_v31_family.py"),
    # ---- 医生端 D-01 ~ D-08
    Playbook("D-01", "当日队列与 HIS 一致", "医生", "ready", "/api/v1/clinician/queue", "backend/tests/test_clinician.py"),
    Playbook("D-02", "患者 360 与 HIS 一致", "医生", "ready", "/api/v1/clinician/patients/", "backend/tests/test_clinician.py"),
    Playbook("D-03", "诊前摘要可回溯", "医生", "ready", "/api/v1/clinician/patients/{member_id}", "backend/tests/test_v34_playbooks_exec.py"),
    Playbook("D-04", "接诊状态同步", "医生", "ready", "/api/v1/clinician/receive", "backend/tests/test_v34_playbooks_exec.py", "平台侧投影（source=platform）；HIS encounter 写入待院方开通"),
    Playbook("D-05", "诊断保存带医生 ID", "医生", "ready", "/api/v1/clinician/notes", "backend/tests/test_v34_playbooks_exec.py", "记录医生与草稿状态；患者端不可读取"),
    Playbook("D-06", "医嘱 / 处方签名门禁", "医生", "ready", "/api/v1/clinician/orders/{order_id}/sign", "backend/tests/test_v34_playbooks_exec.py", "开关关闭 409；签名后进待审方；重复签名被拒"),
    Playbook("D-07", "诊毕后只读", "医生", "ready", "/api/v1/clinician/notes/{note_id}/close-visit", "backend/tests/test_clinician_ext.py"),
    Playbook("D-08", "跨科 / 跨患者 404", "医生", "ready", "/api/v1/clinician/patients/", "backend/tests/test_clinician.py"),
    # ---- 双端闭环 E2E-01 ~ E2E-03
    Playbook("E2E-01", "常规门诊全链路", "双端", "ready", "/api/v1/booking/appointments", "frontend/e2e/s16-outpatient-chain.spec.ts", "预问诊→挂号→收款→接诊→病历→签名→诊毕→患者看到 HIS 处方"),
    Playbook("E2E-02", "退号退费闭环", "双端", "partial", "/api/v1/booking/appointments/{appointment_id}/cancel", "frontend/e2e/v31.spec.ts", "双端状态一致性断言待补"),
    Playbook("E2E-03", "并发抢号恰 1 单成功", "双端", "ready", "/api/v1/booking/orders", "backend/tests/test_v34_playbooks_exec.py", "10 并发同锁号只产生 1 个订单"),
    # ---- 集成一致性 INT-01 ~ INT-06
    Playbook("INT-01", "幂等重放 10 次", "集成", "ready", "/api/v1/booking/orders", "backend/tests/test_v34_playbooks_exec.py", "重放 10 次同锁号：平台 1 条投影 + 稳定 idempotency_key"),
    Playbook("INT-02", "故障注入 500 / 超时 / 401", "集成", "ready", "/api/v1/booking/lock", "backend/tests/test_v34_playbooks_exec.py", "上游错误/超时统一脱敏为 HIS_ERROR / HIS_TIMEOUT，且可重试"),
    Playbook("INT-03", "强制不一致产生 finding", "集成", "ready", "/api/v1/ops/reconciliation/run", "backend/tests/test_v34_playbooks_exec.py", "主动对账把 HIS 差异镜像进平台 findings，重复对账幂等"),
    Playbook("INT-04", "权限越界 403 / 未登录 401", "集成", "ready", "/api/health", "backend/tests/test_p0_security.py"),
    Playbook("INT-05", "审计完整（人/时间/内容）", "集成", "ready", "/api/v1/admin/audit", "backend/tests/test_v34_playbooks_exec.py", "打印/诊毕/接诊/登录四类动作均带 actor + request_id"),
    Playbook("INT-06", "租户数据隔离", "集成", "blocked", "-", "-", "租户表与路由已建，隔离执行未接"),
)

BY_ID = {pb.pid: pb for pb in PLAYBOOKS}


def summary() -> dict[str, int]:
    counts = {"ready": 0, "partial": 0, "blocked": 0}
    for pb in PLAYBOOKS:
        counts[pb.status] = counts.get(pb.status, 0) + 1
    return counts

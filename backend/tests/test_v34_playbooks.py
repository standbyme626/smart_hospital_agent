"""V3.4 验收剧本看板（26 条剧本的可执行登记）。

剧本定义见 docs/V3.2-DUAL-END-PLAN.md §6：
  P-01..P-08 患者端 / D-01..D-08 医生端 / E2E-01..E2E-03 双端闭环 / INT-01..INT-06 集成一致性

本文件不重复实现业务断言，而是把"每条剧本 ↔ 代码证据"钉死：
  ready    —— 接口 + 已有自动化测试都在 → 剧本可跑
  partial  —— 接口在，断言尚未自动化（下一波补）
  blocked  —— 依赖的屏还是 GAP（台账 docs/V3.4-EXECUTION-LEDGER.md 可查原因）

跑法：
  backend/.venv/bin/python -m pytest backend/tests/test_v34_playbooks.py -q
  backend/.venv/bin/python -m pytest backend/tests/test_v34_playbooks.py -v -s   # 看完整看板
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "backend" / "app"
TESTS_DIR = ROOT / "backend" / "tests"


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
    Playbook("P-01", "新患者建档", "患者", "partial", "/api/v1/family", "backend/tests/test_v31_family.py", "真实 HIS 建档走适配器；当前为平台侧家庭成员"),
    Playbook("P-02", "查询号源", "患者", "ready", "/api/v1/booking/schedules", "backend/tests/test_v31_booking.py"),
    Playbook("P-03", "挂号", "患者", "ready", "/api/v1/booking/orders", "backend/tests/test_booking_v2.py"),
    Playbook("P-04", "缴费", "患者", "ready", "/api/v1/booking/orders/{order_id}/pay", "backend/tests/test_booking_v2.py"),
    Playbook("P-05", "重复缴费幂等", "患者", "partial", "/api/v1/booking/orders/{order_id}/pay", "backend/tests/test_v31_contract.py", "需补同 idempotency 二次请求断言"),
    Playbook("P-06", "取消 / 退费", "患者", "ready", "/api/v1/booking/orders/{order_id}/cancel", "backend/tests/test_v31_booking.py"),
    Playbook("P-07", "报告解读不编造", "患者", "ready", "/api/v1/reports/interpret", "backend/tests/test_v31_reports.py"),
    Playbook("P-08", "家庭档案隔离", "患者", "ready", "/api/v1/family", "backend/tests/test_v31_family.py"),
    # ---- 医生端 D-01 ~ D-08
    Playbook("D-01", "当日队列与 HIS 一致", "医生", "ready", "/api/v1/clinician/queue", "backend/tests/test_clinician.py"),
    Playbook("D-02", "患者 360 与 HIS 一致", "医生", "ready", "/api/v1/clinician/patients/", "backend/tests/test_clinician.py"),
    Playbook("D-03", "诊前摘要可回溯", "医生", "blocked", "-", "-", "d-prep/p-prep 摘要链路未接（台账 DOING）"),
    Playbook("D-04", "接诊状态同步", "医生", "blocked", "-", "-", "d-note 之外的接诊动作未实现"),
    Playbook("D-05", "诊断保存带医生 ID", "医生", "partial", "/api/v1/clinician/drafts", "backend/tests/test_clinician.py", "草稿已可存；HIS 落库待签名门禁"),
    Playbook("D-06", "医嘱 / 处方签名门禁", "医生", "partial", "/api/v1/clinician/drafts/{draft_id}/sign", "backend/tests/test_clinician.py", "签名默认关，缺 HIS 账号；422/不落库断言待补"),
    Playbook("D-07", "诊毕后只读", "医生", "blocked", "-", "-", "d-follow 未实现"),
    Playbook("D-08", "跨科 / 跨患者 404", "医生", "ready", "/api/v1/clinician/patients/", "backend/tests/test_clinician.py"),
    # ---- 双端闭环 E2E-01 ~ E2E-03
    Playbook("E2E-01", "常规门诊全链路", "双端", "blocked", "-", "frontend/e2e/v31.spec.ts", "缺预问诊→医生接诊→处方签名段"),
    Playbook("E2E-02", "退号退费闭环", "双端", "partial", "/api/v1/booking/appointments/{appointment_id}/cancel", "frontend/e2e/v31.spec.ts", "双端状态一致性断言待补（当前只断言单端）"),
    Playbook("E2E-03", "并发抢号恰 1 单成功", "双端", "partial", "/api/v1/booking/lock", "backend/tests/test_v31_booking.py", "需 50 并发压测脚本"),
    # ---- 集成一致性 INT-01 ~ INT-06
    Playbook("INT-01", "幂等重放 10 次", "集成", "partial", "/api/v1/booking/orders", "backend/tests/test_booking_v2.py", "缺 10 次重放后 HIS 记录数不变的断言"),
    Playbook("INT-02", "故障注入 500 / 超时 / 401", "集成", "partial", "/api/ready", "backend/tests/fake_gateway.py", "已有假网关，缺逐类错误码断言矩阵"),
    Playbook("INT-03", "强制不一致产生 finding", "集成", "partial", "/api/v1/ops/reconciliation", "backend/tests/test_ops.py", "缺手工改 HIS 状态→轮询出 finding 的注入用例"),
    Playbook("INT-04", "权限越界 403 / 未登录 401", "集成", "ready", "/api/health", "backend/tests/test_p0_security.py"),
    Playbook("INT-05", "审计完整（人/时间/内容）", "集成", "partial", "/api/v1/admin/audit", "backend/tests/test_platform.py", "缺每条 agent 建议与签名的 trace 断言"),
    Playbook("INT-06", "租户数据隔离", "集成", "blocked", "-", "-", "a-tenant 多院区未实现"),
)

BY_ID = {pb.pid: pb for pb in PLAYBOOKS}


def _backend_source() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(APP_DIR.glob("*.py")))


@pytest.fixture(scope="module")
def backend_src() -> str:
    return _backend_source()


def test_board_size_and_areas() -> None:
    """剧本总数与分区必须与规划文档一致（P08 / D08 / E2E03 / INT06 = 25）。"""
    assert len(PLAYBOOKS) == 25, "剧本数量与 docs/V3.2-DUAL-END-PLAN.md §6 不符"
    areas = {}
    for pb in PLAYBOOKS:
        areas[pb.area] = areas.get(pb.area, 0) + 1
    assert areas == {"患者": 8, "医生": 8, "双端": 3, "集成": 6}, areas
    assert len(BY_ID) == len(PLAYBOOKS), "剧本 ID 有重复"


@pytest.mark.parametrize("pb", PLAYBOOKS, ids=[pb.pid for pb in PLAYBOOKS])
def test_playbook_evidence(pb: Playbook, backend_src: str) -> None:
    """每条剧本：状态合法 + 证据真实存在；blocked 必须写明原因。"""
    assert pb.status in {"ready", "partial", "blocked"}, f"{pb.pid} 状态非法：{pb.status}"

    if pb.status == "blocked":
        assert pb.note, f"{pb.pid} 标记 blocked 但没写阻塞原因"
        pytest.skip(f"{pb.pid} {pb.name} 阻塞：{pb.note}")

    assert pb.endpoint != "-", f"{pb.pid} 缺接口证据"
    assert pb.endpoint in backend_src, f"{pb.pid} 接口 {pb.endpoint} 在后端源码中不存在"

    evidence = ROOT / pb.evidence_test
    assert evidence.exists(), f"{pb.pid} 证据测试文件不存在：{pb.evidence_test}"

    if pb.status == "partial":
        assert pb.note, f"{pb.pid} 标记 partial 但没写缺口"


def test_no_ready_without_automation(backend_src: str) -> None:
    """ready 的剧本不能只有接口没有测试文件（防止自述上线）。"""
    missing = [
        pb.pid
        for pb in PLAYBOOKS
        if pb.status == "ready" and not (ROOT / pb.evidence_test).exists()
    ]
    assert not missing, f"这些剧本标 ready 但缺自动化证据：{missing}"


def test_board_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """打印看板（-s 可见），并把计数固化进本文件，波次推进时改这里即代表验收口径变更。"""
    counts = {"ready": 0, "partial": 0, "blocked": 0}
    for pb in PLAYBOOKS:
        counts[pb.status] += 1

    lines = ["", "V3.4 验收剧本看板", "=" * 60, f"{'ID':<7}{'区域':<6}{'状态':<9}场景"]
    for pb in PLAYBOOKS:
        lines.append(f"{pb.pid:<7}{pb.area:<6}{pb.status:<9}{pb.name}")
    lines += [
        "-" * 60,
        f"ready {counts['ready']} · partial {counts['partial']} · blocked {counts['blocked']} · 合计 {len(PLAYBOOKS)}",
        "",
    ]
    print("\n".join(lines))

    assert counts == {"ready": 10, "partial": 10, "blocked": 5}, (
        f"看板统计变了：{counts}；如为有意推进，请同步更新断言与台账"
    )

"""V3.4 验收剧本看板（可执行登记）。

剧本定义与状态登记在 app/playbooks.py（单一数据源），管理端「验收自测」屏读同一份。
本文件负责校验：登记表自洽、证据真实存在、blocked 必须写明原因。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.playbooks import PLAYBOOKS, summary

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "backend" / "app"
BY_ID = {pb.pid: pb for pb in PLAYBOOKS}


@pytest.fixture(scope="module")
def backend_src() -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(APP_DIR.glob("*.py")))


def test_board_size_and_areas() -> None:
    """剧本总数与分区必须与规划文档一致（P08 / D08 / E2E03 / INT06 = 25）。"""
    assert len(PLAYBOOKS) == 25, "剧本数量与 docs/V3.2-DUAL-END-PLAN.md §6 不符"
    areas: dict[str, int] = {}
    for pb in PLAYBOOKS:
        areas[pb.area] = areas.get(pb.area, 0) + 1
    assert areas == {"患者": 8, "医生": 8, "双端": 3, "集成": 6}, areas
    assert len(BY_ID) == len(PLAYBOOKS), "剧本 ID 有重复"


@pytest.mark.parametrize("pb", PLAYBOOKS, ids=[pb.pid for pb in PLAYBOOKS])
def test_playbook_evidence(pb, backend_src: str) -> None:
    """每条剧本：状态合法 + 证据真实存在；blocked 必须写明原因。"""
    assert pb.status in {"ready", "partial", "blocked"}, f"{pb.pid} 状态非法：{pb.status}"

    if pb.status == "blocked":
        assert pb.note, f"{pb.pid} 标记 blocked 但没写阻塞原因"
        pytest.skip(f"{pb.pid} {pb.name} 阻塞：{pb.note}")

    assert pb.endpoint != "-", f"{pb.pid} 缺接口证据"
    assert pb.endpoint in backend_src, f"{pb.pid} 接口 {pb.endpoint} 在后端源码中不存在"
    assert (ROOT / pb.evidence_test).exists(), f"{pb.pid} 证据测试文件不存在：{pb.evidence_test}"
    if pb.status == "partial":
        assert pb.note, f"{pb.pid} 标记 partial 但没写缺口"


def test_no_ready_without_automation() -> None:
    """ready 的剧本不能只有接口没有测试文件（防止自述上线）。"""
    missing = [
        pb.pid for pb in PLAYBOOKS if pb.status == "ready" and not (ROOT / pb.evidence_test).exists()
    ]
    assert not missing, f"这些剧本标 ready 但缺自动化证据：{missing}"


def test_board_summary(capsys: pytest.CaptureFixture[str]) -> None:
    counts = summary()
    lines = ["", "V3.4 验收剧本看板", "=" * 60, f"{'ID':<7}{'区域':<6}{'状态':<9}场景"]
    for pb in PLAYBOOKS:
        lines.append(f"{pb.pid:<7}{pb.area:<6}{pb.status:<9}{pb.name}")
    lines += ["-" * 60, f"ready {counts['ready']} · partial {counts['partial']} · blocked {counts['blocked']} · 合计 {len(PLAYBOOKS)}", ""]
    print("\n".join(lines))

    assert counts == {"ready": 22, "partial": 2, "blocked": 1}, (
        f"看板统计变了：{counts}；如为有意推进，请同步更新断言与台账"
    )

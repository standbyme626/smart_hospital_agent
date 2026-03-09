import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.domain.states.sub_states import DiagnosisState


def test_diagnosis_state_contract_covers_phase1_critical_fields() -> None:
    keys = set(DiagnosisState.__annotations__.keys())

    critical_fields = {
        # 输入/会话
        "messages",
        "patient_id",
        "session_id",
        "request_id",
        "event",
        "current_turn_input",
        "retrieval_query",
        # 检索控制
        "retrieval_top_k_override",
        "retrieval_use_rerank",
        "retrieval_rerank_threshold",
        "query_rewrite_timeout_override_s",
        "crisis_fastlane_override",
        # 诊断输出
        "department_top1",
        "department_top3",
        "confidence",
        "diagnosis_output",
        # 治理调试
        "decision_action",
        "decision_reason",
        "confidence_score",
        "grounded_flag",
        "debug_snapshots",
        "runtime_config_effective",
    }

    missing = sorted(critical_fields - keys)
    assert not missing, f"DiagnosisState missing contract fields: {missing}"

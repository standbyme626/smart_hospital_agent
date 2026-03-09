import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)

from app.domain.states.sub_states import DiagnosisState
from app.core.config import settings
from app.core.graph.sub_graphs import diagnosis


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


class _RecordingStateGraph:
    def __init__(self, *_args, **_kwargs):
        self.nodes = {}
        self.entry = None
        self.edges = []
        self.conditional_edges = []

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def set_entry_point(self, name):
        self.entry = name

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, router, mapping):
        self.conditional_edges.append((source, router, dict(mapping)))

    def compile(self):
        return self


def _build_graph_snapshot(monkeypatch, shell_enabled: bool):
    monkeypatch.setattr(settings, "UPGRADE3_DIAGNOSIS_SHELL_ENABLED", shell_enabled, raising=False)
    monkeypatch.setattr(diagnosis, "StateGraph", _RecordingStateGraph)
    return diagnosis.build_diagnosis_graph()


def test_diagnosis_graph_contract_with_upgrade3_shell_toggle(monkeypatch) -> None:
    shell_graph = _build_graph_snapshot(monkeypatch, True)
    legacy_graph = _build_graph_snapshot(monkeypatch, False)

    expected_nodes = {
        "State_Sync",
        "Query_Rewrite",
        "Quick_Triage",
        "Hybrid_Retriever",
        "DSPy_Reasoner",
        "Diagnosis_Report",
        "Clarify_Question",
    }
    assert expected_nodes.issubset(set(shell_graph.nodes.keys()))
    assert expected_nodes.issubset(set(legacy_graph.nodes.keys()))
    assert shell_graph.entry == "State_Sync"
    assert legacy_graph.entry == "State_Sync"

    shell_conditional = {source: mapping for source, _router, mapping in shell_graph.conditional_edges}
    legacy_conditional = {source: mapping for source, _router, mapping in legacy_graph.conditional_edges}
    assert shell_conditional["Quick_Triage"] == {"fast_exit": "Diagnosis_Report", "deep_diagnosis": "Hybrid_Retriever"}
    assert legacy_conditional["Quick_Triage"] == {"fast_exit": "Diagnosis_Report", "deep_diagnosis": "Hybrid_Retriever"}
    assert shell_conditional["Hybrid_Retriever"] == {"pure_report": "Diagnosis_Report", "dspy_reasoner": "DSPy_Reasoner"}
    assert legacy_conditional["Hybrid_Retriever"] == {"pure_report": "Diagnosis_Report", "dspy_reasoner": "DSPy_Reasoner"}

    assert shell_graph.nodes["State_Sync"] is diagnosis.shell_state_sync_node
    assert legacy_graph.nodes["State_Sync"] is diagnosis.state_sync_node
    assert shell_graph.nodes["Query_Rewrite"] is diagnosis.shell_query_rewrite_node
    assert legacy_graph.nodes["Query_Rewrite"] is diagnosis.query_rewrite_node

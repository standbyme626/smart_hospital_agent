#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.models.local_slm import local_slm  # noqa: E402

CATEGORIES = ["CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]
FAILURE_CATEGORIES = ["protocol", "model", "retrieval", "routing"]
PATH_MAP = {
    "GREETING": "fast",
    "CRISIS": "expert",
    "VAGUE_SYMPTOM": "vague",
    "COMPLEX_SYMPTOM": "standard",
}


@dataclass
class PredictResult:
    intent: str
    latency_s: float
    source: str
    failure_type: Optional[str] = None
    detail: str = ""


class IntentPolicy:
    def __init__(self) -> None:
        self._session_last_intent: Dict[str, str] = {}

    @staticmethod
    def _has_medical_signal(text: str) -> bool:
        t = (text or "").lower()
        kws = [
            "痛", "疼", "晕", "发烧", "发热", "咳嗽", "胸闷", "胸痛", "腹痛", "腹泻", "恶心", "呕吐",
            "不舒服", "难受", "心慌", "气短", "呼吸", "药", "看医生", "挂号",
        ]
        return any(k in t for k in kws)

    @staticmethod
    def _is_followup(text: str) -> bool:
        t = (text or "").lower()
        marks = ["还是", "还", "另外", "现在", "依旧", "继续", "又", "仍然", "前面说的"]
        return any(m in t for m in marks)

    @staticmethod
    def _rule_front(text: str) -> Optional[str]:
        t = (text or "").strip().lower()
        if not t:
            return None

        crisis_kws = [
            "救命", "胸痛", "昏迷", "呼吸困难", "大出血", "想死", "自杀", "不想活", "中风", "120", "割腕", "中毒",
        ]
        if any(k in t for k in crisis_kws):
            return "CRISIS"

        greeting_kws = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "再见", "晚安", "早上好"]
        if len(t) <= 10 and any(k in t for k in greeting_kws) and not IntentPolicy._has_medical_signal(t):
            return "GREETING"

        if len(t) <= 8 and IntentPolicy._has_medical_signal(t):
            return "VAGUE_SYMPTOM"

        return None

    async def baseline_predict(self, text: str, session_id: str = "") -> PredictResult:
        start = time.perf_counter()
        try:
            pred = await local_slm.constrained_classify(text, categories=CATEGORIES, reasoning=False)
            pred = str(pred).strip().upper()
            latency = time.perf_counter() - start
            if pred not in CATEGORIES:
                return PredictResult(
                    intent="VAGUE_SYMPTOM",
                    latency_s=latency,
                    source="baseline_protocol_fallback",
                    failure_type="protocol",
                    detail=f"invalid_label:{pred}",
                )
            if session_id:
                self._session_last_intent[session_id] = pred
            return PredictResult(intent=pred, latency_s=latency, source="baseline_model")
        except Exception as e:
            latency = time.perf_counter() - start
            return PredictResult(
                intent="VAGUE_SYMPTOM",
                latency_s=latency,
                source="baseline_exception_fallback",
                failure_type="protocol",
                detail=f"exception:{e}",
            )

    async def optimized_predict(self, text: str, session_id: str = "") -> PredictResult:
        start = time.perf_counter()
        t = (text or "").strip()

        # 1) 规则前置
        front = self._rule_front(t)
        if front:
            latency = time.perf_counter() - start
            if session_id:
                self._session_last_intent[session_id] = front
            return PredictResult(intent=front, latency_s=latency, source="rule_front")

        # 2) 模型分类
        protocol_failure = None
        pred = "VAGUE_SYMPTOM"
        try:
            pred = await local_slm.constrained_classify(t, categories=CATEGORIES, reasoning=False)
            pred = str(pred).strip().upper()
            if pred not in CATEGORIES:
                protocol_failure = f"invalid_label:{pred}"
                pred = "VAGUE_SYMPTOM"
        except Exception as e:
            protocol_failure = f"exception:{e}"
            pred = "VAGUE_SYMPTOM"

        # 3) 不确定追问纠偏（优先压 GREETING/VAGUE 混淆）
        if pred == "GREETING" and self._has_medical_signal(t):
            pred = "VAGUE_SYMPTOM"
        if pred == "COMPLEX_SYMPTOM" and len(t) <= 8 and self._has_medical_signal(t):
            pred = "VAGUE_SYMPTOM"

        # 4) 会话态纠偏
        if session_id:
            prev = self._session_last_intent.get(session_id, "")
            if pred == "GREETING" and prev in {"VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}:
                if self._has_medical_signal(t) or self._is_followup(t):
                    pred = "VAGUE_SYMPTOM"
            self._session_last_intent[session_id] = pred

        latency = time.perf_counter() - start
        if protocol_failure:
            return PredictResult(
                intent=pred,
                latency_s=latency,
                source="optimized_protocol_fallback",
                failure_type="protocol",
                detail=protocol_failure,
            )
        return PredictResult(intent=pred, latency_s=latency, source="optimized_model")



def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def macro_f1_from_confusion(confusion: Dict[str, Dict[str, int]], labels: List[str]) -> float:
    f1s: List[float] = []
    for label in labels:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in labels if other != label)
        fn = sum(confusion.get(label, {}).get(other, 0) for other in labels if other != label)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        denom_p = tp + fp
        denom_r = tp + fn
        precision = tp / denom_p if denom_p else 0.0
        recall = tp / denom_r if denom_r else 0.0
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / len(f1s) if f1s else 0.0


async def eval_intent(
    rows: List[Dict],
    predictor,
    failure_counter: Counter,
    details_path: Path,
) -> Dict:
    confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_label_total = Counter()
    per_label_correct = Counter()
    latencies: List[float] = []
    correct = 0

    with details_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            text = row["question"]
            label = row["label"]
            pred = await predictor(text, "")

            ok = pred.intent == label
            if ok:
                correct += 1
                per_label_correct[label] += 1
            per_label_total[label] += 1
            confusion[label][pred.intent] += 1
            latencies.append(pred.latency_s)

            fail_type = pred.failure_type
            if not ok and not fail_type:
                fail_type = "model"
            if fail_type:
                failure_counter[fail_type] += 1

            out.write(
                json.dumps(
                    {
                        "idx": idx,
                        "question": text,
                        "label": label,
                        "pred": pred.intent,
                        "ok": ok,
                        "latency_s": round(pred.latency_s, 4),
                        "source": pred.source,
                        "failure_type": fail_type,
                        "detail": pred.detail,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(rows)
    summary = {
        "sample_n": total,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_f1": macro_f1_from_confusion(confusion, CATEGORIES),
        "per_label_accuracy": {
            label: (per_label_correct[label] / per_label_total[label] if per_label_total[label] else 0.0)
            for label in CATEGORIES
        },
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
            "total": sum(latencies),
        },
    }
    return summary


async def eval_triage(
    rows: List[Dict],
    predictor,
    failure_counter: Counter,
    details_path: Path,
) -> Dict:
    path_confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    intent_correct = 0
    route_correct = 0
    joint_correct = 0
    latencies: List[float] = []

    with details_path.open("w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, start=1):
            query = row["query"]
            label_intent = row["label_intent"]
            expected_path = row["expected_path"]
            session_id = row.get("session_id") or f"row_{idx}"

            pred = await predictor(query, session_id)
            pred_path = PATH_MAP.get(pred.intent, "vague")

            intent_ok = pred.intent == label_intent
            route_ok = pred_path == expected_path
            joint_ok = intent_ok and route_ok

            if intent_ok:
                intent_correct += 1
            if route_ok:
                route_correct += 1
            if joint_ok:
                joint_correct += 1

            path_confusion[expected_path][pred_path] += 1
            latencies.append(pred.latency_s)

            fail_type = pred.failure_type
            if not fail_type and not route_ok:
                if {expected_path, pred_path} == {"vague", "standard"}:
                    fail_type = "retrieval"
                elif intent_ok and not route_ok:
                    fail_type = "routing"
                else:
                    fail_type = "model"
            if fail_type:
                failure_counter[fail_type] += 1

            out.write(
                json.dumps(
                    {
                        "idx": idx,
                        "query": query,
                        "label_intent": label_intent,
                        "pred_intent": pred.intent,
                        "intent_ok": intent_ok,
                        "expected_path": expected_path,
                        "pred_path": pred_path,
                        "route_ok": route_ok,
                        "joint_ok": joint_ok,
                        "latency_s": round(pred.latency_s, 4),
                        "source": pred.source,
                        "failure_type": fail_type,
                        "detail": pred.detail,
                        "session_id": session_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(rows)
    summary = {
        "sample_n": total,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "path_accuracy": route_correct / total if total else 0.0,
        "joint_intent_and_path_accuracy": joint_correct / total if total else 0.0,
        "path_confusion_matrix": {k: dict(v) for k, v in path_confusion.items()},
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies) if latencies else 0.0,
            "total": sum(latencies),
        },
    }
    return summary


async def run_once(intent_rows: List[Dict], triage_rows: List[Dict], mode: str, outdir: Path) -> Dict:
    policy = IntentPolicy()
    failure_counter = Counter()

    if mode == "before":
        predictor = policy.baseline_predict
    elif mode == "after":
        predictor = policy.optimized_predict
    else:
        raise ValueError(f"unknown mode: {mode}")

    intent_summary = await eval_intent(
        intent_rows,
        predictor,
        failure_counter,
        outdir / f"{mode}_intent_details.jsonl",
    )
    triage_summary = await eval_triage(
        triage_rows,
        predictor,
        failure_counter,
        outdir / f"{mode}_triage_details.jsonl",
    )

    return {
        "intent_summary": intent_summary,
        "triage_summary": triage_summary,
        "failures": dict(failure_counter),
    }


def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent-set", default="data/benchmarks/clean_intent_400_v1.jsonl")
    parser.add_argument("--triage-set", default="data/benchmarks/clean_triage_100_v1.jsonl")
    parser.add_argument("--run-tag", default="clean500_before_after")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "logs" / "intent_benchmark" / f"{args.run_tag}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    intent_rows = load_jsonl(ROOT / args.intent_set)
    triage_rows = load_jsonl(ROOT / args.triage_set)

    # Warmup once to reduce cold-start bias in before/after latency comparison.
    await local_slm.constrained_classify("你好", categories=CATEGORIES, reasoning=False)

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "intent_set": str(args.intent_set),
        "triage_set": str(args.triage_set),
        "intent_sample_n": len(intent_rows),
        "triage_sample_n": len(triage_rows),
        "categories": CATEGORIES,
    }
    save_json(outdir / "meta.json", meta)

    before = await run_once(intent_rows, triage_rows, "before", outdir)
    after = await run_once(intent_rows, triage_rows, "after", outdir)

    save_json(outdir / "before_intent_summary.json", before["intent_summary"])
    save_json(outdir / "before_triage_summary.json", before["triage_summary"])
    save_json(outdir / "after_intent_summary.json", after["intent_summary"])
    save_json(outdir / "after_triage_summary.json", after["triage_summary"])

    compare = {
        "intent_accuracy_before": before["intent_summary"]["accuracy"],
        "intent_accuracy_after": after["intent_summary"]["accuracy"],
        "intent_accuracy_delta": after["intent_summary"]["accuracy"] - before["intent_summary"]["accuracy"],
        "intent_macro_f1_before": before["intent_summary"]["macro_f1"],
        "intent_macro_f1_after": after["intent_summary"]["macro_f1"],
        "intent_macro_f1_delta": after["intent_summary"]["macro_f1"] - before["intent_summary"]["macro_f1"],
        "path_accuracy_before": before["triage_summary"]["path_accuracy"],
        "path_accuracy_after": after["triage_summary"]["path_accuracy"],
        "path_accuracy_delta": after["triage_summary"]["path_accuracy"] - before["triage_summary"]["path_accuracy"],
        "p95_before": before["triage_summary"]["latency_s"]["p95"],
        "p95_after": after["triage_summary"]["latency_s"]["p95"],
        "p95_delta": after["triage_summary"]["latency_s"]["p95"] - before["triage_summary"]["latency_s"]["p95"],
    }
    save_json(outdir / "compare.json", compare)

    before_fail = {k: before["failures"].get(k, 0) for k in FAILURE_CATEGORIES}
    after_fail = {k: after["failures"].get(k, 0) for k in FAILURE_CATEGORIES}
    failure_report = {
        "before": before_fail,
        "after": after_fail,
        "delta": {k: after_fail[k] - before_fail[k] for k in FAILURE_CATEGORIES},
    }
    save_json(outdir / "failure_report.json", failure_report)

    print(json.dumps({"outdir": str(outdir), "compare": compare, "failure_report": failure_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

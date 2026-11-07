from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, Settings
from .db import Database
from .model_gateway import ModelGateway
from .policy import detect_crisis
from .retrieval import RetrievalService


class EvalRunner:
    def __init__(self, settings: Settings, db: Database, retrieval: RetrievalService, gateway: ModelGateway):
        self.settings = settings
        self.db = db
        self.retrieval = retrieval
        self.gateway = gateway

    async def run(self, limit: int = 100) -> dict[str, Any]:
        retrieval = await self._retrieval_smoke(limit)
        triage = self._crisis_eval()
        return {
            "retrieval": retrieval,
            "triage": triage,
            "note": "retrieval 是 in-corpus smoke（query=标题，答案在同一索引内），不是留出集基准。",
        }

    async def _retrieval_smoke(self, limit: int) -> dict[str, Any]:
        rows = await self.db.fetch(
            "SELECT id, title FROM documents ORDER BY random() LIMIT $1",
            max(1, limit),
        )
        if not rows:
            return {"n": 0, "recall_at_5": None, "mrr_at_5": None}
        hits = 0
        reciprocal = 0.0
        for row in rows:
            results = await self.retrieval.search(row["title"], top_k=5)
            ids = [str(item["id"]) for item in results]
            target = str(row["id"])
            if target in ids:
                hits += 1
                reciprocal += 1.0 / (ids.index(target) + 1)
        n = len(rows)
        return {
            "n": n,
            "recall_at_5": round(hits / n, 4),
            "mrr_at_5": round(reciprocal / n, 4),
        }

    def _crisis_eval(self) -> dict[str, Any]:
        path = Path(REPO_ROOT) / "data" / "benchmarks" / "clean_eval_500_v1.jsonl"
        if not path.exists():
            return {"n": 0, "note": "benchmark missing"}
        tp = fp = fn = tn = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                expected = str(item.get("label") or "").upper() == "CRISIS"
                predicted = detect_crisis(str(item.get("question") or ""))[0]
                if expected and predicted:
                    tp += 1
                elif expected and not predicted:
                    fn += 1
                elif not expected and predicted:
                    fp += 1
                else:
                    tn += 1
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        return {
            "n": tp + fp + fn + tn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "note": "rule-based crisis detection on the existing 500-case benchmark; the keyword list was expanded using this set's failures, so treat as in-sample, not blind.",
        }

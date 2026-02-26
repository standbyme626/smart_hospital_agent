#!/usr/bin/env python3
"""
Generate a clean benchmark set:
- intent: 400 samples (4 classes x 100)
- triage: 100 samples (4 classes x 25)

Constraints:
- No exact overlap with legacy datasets.
- Deterministic output by seed.
- Labels are rule-verified before export.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

LABELS = ["CRISIS", "GREETING", "VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"]
PATH_MAP = {
    "GREETING": "fast",
    "CRISIS": "expert",
    "VAGUE_SYMPTOM": "vague",
    "COMPLEX_SYMPTOM": "standard",
}


def load_legacy_questions(paths: List[Path]) -> Set[str]:
    banned: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (
                    obj.get("question")
                    or obj.get("query")
                    or obj.get("input")
                    or ""
                )
                text = str(text).strip()
                if text:
                    banned.add(text)
    return banned


def rule_label(text: str) -> str:
    t = text.strip().lower()
    crisis_kw = [
        "救命", "昏迷", "呼吸困难", "胸痛", "大出血", "自杀", "想死", "割腕", "中毒", "休克", "抽搐", "120",
    ]
    greeting_kw = ["你好", "您好", "在吗", "hello", "hi", "谢谢", "再见", "晚安", "早上好"]
    symptom_kw = [
        "痛", "疼", "晕", "发烧", "咳嗽", "恶心", "呕吐", "腹泻", "心慌", "不舒服", "难受", "发热", "胸闷", "气短",
    ]
    detail_kw = ["天", "小时", "周", "昨晚", "今天", "持续", "加重", "反复", "体温", "年", "月"]

    if any(k in t for k in crisis_kw):
        return "CRISIS"
    if any(k in t for k in greeting_kw) and not any(k in t for k in symptom_kw):
        return "GREETING"
    if any(k in t for k in symptom_kw):
        if len(t) <= 16 and not any(k in t for k in detail_kw):
            return "VAGUE_SYMPTOM"
        return "COMPLEX_SYMPTOM"
    if len(t) <= 10:
        return "GREETING"
    return "VAGUE_SYMPTOM"


def greeting_pool() -> List[str]:
    heads = [
        "你好", "您好", "hi", "hello", "在吗", "有人吗", "早上好", "晚上好", "谢谢", "再见",
        "嗨", "哈喽", "客服在吗", "我先问候一下", "请问有人值班吗",
    ]
    tails = [
        "", "，我想先了解下你能做什么", "，请问现在可以咨询吗", "，先打个招呼", "，你在线吗",
        "，先确认下系统在线", "，我是新用户", "，辛苦回复一下",
    ]
    polite = ["麻烦", "请", "辛苦", "先", "帮忙", "劳烦"]
    extras = ["！", "。", "～", "?", "？"]

    out: Set[str] = set()
    for h, t in itertools.product(heads, tails):
        out.add(f"{h}{t}".strip())
    for h, p in itertools.product(heads, polite):
        out.add(f"{h}，{p}帮我看下流程")
    for h, e in itertools.product(heads, extras):
        out.add(f"{h}{e}")
    return list(out)


def crisis_pool() -> List[str]:
    scenes = [
        "突然胸痛并大汗", "呼吸困难嘴唇发紫", "大量出血止不住", "昏迷叫不醒", "抽搐持续不止",
        "喝了农药", "割腕后一直流血", "吃药过量后意识模糊", "疑似中风口角歪斜", "车祸后胸腹剧痛",
        "突发胸痛伴呼吸困难", "老人疑似中风肢体无力", "孩子持续抽搐意识不清", "患者休克血压测不到", "突发昏迷伴呕吐",
        "家人疑似脑出血意识下降", "过敏后喉头水肿喘不上气", "大出血并且面色苍白", "胸痛放射左臂伴濒死感", "呼吸骤停需要急救",
    ]
    suffix = [
        "，现在怎么办", "，需要马上叫120吗", "，请立刻指导", "，家属很慌", "，情况在恶化",
        "，要不要马上急救", "，请给紧急处理步骤", "，我现在非常危险",
    ]

    out = {f"{s}{x}" for s, x in itertools.product(scenes, suffix)}
    time_marks = ["刚刚", "5分钟前", "10分钟前", "半小时前", "1小时前", "现在"]
    for tm, s in itertools.product(time_marks, scenes):
        out.add(f"{tm}{s}，请立刻指导")
        out.add(f"{tm}{s}，需要马上叫120吗")
    out.update(
        {
            "我现在想自杀，请救救我",
            "我不想活了并且已经准备跳楼",
            "家人胸痛呼吸困难快不行了",
            "患者昏迷伴大出血，急需处理",
            "突发剧烈头痛并呕吐意识下降",
        }
    )
    return list(out)


def vague_pool() -> List[str]:
    bases = [
        "我不舒服", "有点头晕", "今天难受", "最近没劲", "肚子不太舒服", "嗓子不舒服", "有点恶心",
        "有点发热", "心里发慌", "总觉得不对劲", "身上不舒服", "睡不好还难受", "老是头疼",
        "有点胸闷", "最近咳嗽", "我有点腹痛", "喉咙疼", "这两天心慌", "总想吐", "头有点痛",
    ]
    modifiers = [
        "", "，怎么办", "，要紧吗", "，需要看医生吗", "，先怎么处理", "，会不会很严重",
        "，要不要去医院", "，可以先观察吗", "，该挂哪个科",
    ]
    out = {f"{b}{m}".strip() for b, m in itertools.product(bases, modifiers)}
    return list(out)


def complex_pool() -> List[str]:
    symptoms = [
        "头痛伴恶心", "发烧伴咳嗽", "胸闷伴气短", "腹痛伴腹泻", "腰痛伴尿频", "咽痛伴吞咽困难",
        "皮疹伴瘙痒", "心悸伴乏力", "关节肿痛", "右下腹持续疼痛",
    ]
    durations = ["2小时", "6小时", "1天", "3天", "1周", "2周"]
    details = [
        "体温38.5度", "夜间加重", "饭后明显", "活动后加重", "休息后稍缓解", "伴轻微呕吐", "有高血压史", "有糖尿病史",
    ]

    out: Set[str] = set()
    for s, d, detail in itertools.product(symptoms, durations, details):
        out.add(f"{s}{d}，{detail}，该挂什么科？")
    return list(out)


def filter_pool(pool: Iterable[str], banned: Set[str], target_label: str) -> List[str]:
    out = []
    seen: Set[str] = set()
    for q in pool:
        text = q.strip()
        if not text or text in banned or text in seen:
            continue
        if rule_label(text) != target_label:
            continue
        seen.add(text)
        out.append(text)
    return out


def sample_n(pool: List[str], n: int, rnd: random.Random) -> List[str]:
    if len(pool) < n:
        raise RuntimeError(f"pool too small: need {n}, got {len(pool)}")
    idxs = list(range(len(pool)))
    rnd.shuffle(idxs)
    return [pool[i] for i in idxs[:n]]


def build_dataset(seed: int, banned: Set[str]) -> Tuple[List[Dict], List[Dict]]:
    rnd = random.Random(seed)

    candidates = {
        "GREETING": filter_pool(greeting_pool(), banned, "GREETING"),
        "CRISIS": filter_pool(crisis_pool(), banned, "CRISIS"),
        "VAGUE_SYMPTOM": filter_pool(vague_pool(), banned, "VAGUE_SYMPTOM"),
        "COMPLEX_SYMPTOM": filter_pool(complex_pool(), banned, "COMPLEX_SYMPTOM"),
    }

    intent_rows: List[Dict] = []
    triage_rows: List[Dict] = []
    triage_by_label: Dict[str, List[Dict]] = {k: [] for k in LABELS}
    used_questions: Set[str] = set()

    for label in LABELS:
        label_pool = [q for q in candidates[label] if q not in used_questions]
        intent_selected = sample_n(label_pool, 100, rnd)
        for q in intent_selected:
            used_questions.add(q)
            intent_rows.append(
                {
                    "question": q,
                    "label": label,
                    "label_source": "rule_manual_v1",
                    "split": "intent",
                }
            )

        triage_pool = [q for q in candidates[label] if q not in used_questions]
        triage_selected = sample_n(triage_pool, 25, rnd)
        for i, q in enumerate(triage_selected, start=1):
            used_questions.add(q)
            triage_by_label[label].append(
                {
                    "query": q,
                    "label_intent": label,
                    "expected_path": PATH_MAP[label],
                    "label_source": "rule_manual_v1",
                    "session_id": f"clean_triage_{label.lower()}_{i:02d}",
                    "turn_id": 1,
                    "split": "triage",
                }
            )

    # Add session-follow-up samples to strengthen state-correction checks.
    followups = [
        ("session_followup_01", "昨天开始头晕，今天还是有点恶心", "VAGUE_SYMPTOM"),
        ("session_followup_02", "前面说的胸闷现在还在", "VAGUE_SYMPTOM"),
        ("session_followup_03", "刚才那个腹痛现在加重了", "COMPLEX_SYMPTOM"),
        ("session_followup_04", "还是咳嗽并且发热38度", "COMPLEX_SYMPTOM"),
    ]
    for sid, q, label in followups:
        if (
            q not in banned
            and q not in used_questions
            and rule_label(q) in {"VAGUE_SYMPTOM", "COMPLEX_SYMPTOM"}
        ):
            # Replace one same-label sample to keep strict 100 size and label balance.
            bucket = triage_by_label[label]
            if bucket:
                removed = bucket.pop(0)
                used_questions.discard(removed["query"])
            bucket.append(
                {
                    "query": q,
                    "label_intent": label,
                    "expected_path": PATH_MAP[label],
                    "label_source": "rule_manual_v1",
                    "session_id": sid,
                    "turn_id": 2,
                    "split": "triage",
                }
            )
            used_questions.add(q)

    for label in LABELS:
        triage_rows.extend(triage_by_label[label])

    rnd.shuffle(intent_rows)
    rnd.shuffle(triage_rows)

    return intent_rows, triage_rows


def dump_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260226)
    parser.add_argument("--outdir", default="data/benchmarks")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir

    legacy_paths = [
        root / "data" / "intent_test_1000.jsonl",
        root / "data" / "router_test_details_1000.jsonl",
    ]
    banned = load_legacy_questions(legacy_paths)

    intent_rows, triage_rows = build_dataset(args.seed, banned)

    intent_path = outdir / "clean_intent_400_v1.jsonl"
    triage_path = outdir / "clean_triage_100_v1.jsonl"
    combo_path = outdir / "clean_eval_500_v1.jsonl"
    meta_path = outdir / "clean_eval_500_v1_meta.json"

    dump_jsonl(intent_path, intent_rows)
    dump_jsonl(triage_path, triage_rows)
    dump_jsonl(combo_path, intent_rows + triage_rows)

    meta = {
        "seed": args.seed,
        "intent_count": len(intent_rows),
        "triage_count": len(triage_rows),
        "legacy_banned_count": len(banned),
        "intent_label_dist": dict(Counter(r["label"] for r in intent_rows)),
        "triage_label_dist": dict(Counter(r["label_intent"] for r in triage_rows)),
        "all_questions_fresh_vs_legacy": True,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] intent -> {intent_path}")
    print(f"[OK] triage -> {triage_path}")
    print(f"[OK] combo  -> {combo_path}")
    print(f"[OK] meta   -> {meta_path}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

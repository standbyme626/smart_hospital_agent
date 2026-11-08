"use client";

import { useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  BookOpen,
  CalendarClock,
  CheckCircle2,
  ChevronRight,
  Phone,
  RefreshCcw,
  ShieldCheck,
  Stethoscope,
  X,
} from "lucide-react";
import clsx from "clsx";
import type { Citation, DocumentDetail, Report, Safety, Triage } from "@/lib/types";
import { urgencyMeta } from "@/lib/format";
import { getDocument, track } from "@/lib/api";

const toneClasses: Record<string, string> = {
  mint: "border-emerald-200 bg-emerald-50 text-emerald-800",
  amber: "border-amber-200 bg-amber-50 text-amber-800",
  coral: "border-rose-200 bg-rose-50 text-rose-800",
};

export function ResultView({
  report,
  citations,
  safety,
  triage,
  onBook,
  onRestart,
}: {
  report: Report;
  citations: Citation[];
  safety?: Safety;
  triage?: Triage;
  onBook?: () => void;
  onRestart?: () => void;
}) {
  const [sheetOpen, setSheetOpen] = useState(false);
  const [selected, setSelected] = useState<Citation | null>(null);
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [detailState, setDetailState] = useState<"idle" | "loading" | "error">("idle");
  const [pro, setPro] = useState(false);
  const urgency = urgencyMeta[report.urgency] || urgencyMeta.normal;
  const citationMap = new Map(citations.map((item) => [String(item.id), item]));
  const confidence = Math.max(0, Math.min(1, Number(report.confidence) || 0));

  function selectCitation(citation: Citation) {
    setSelected(citation);
    setDetail(null);
    setDetailState("idle");
  }

  function openCitation(citation: Citation) {
    selectCitation(citation);
    setSheetOpen(true);
    track("evidence_open", { citation_id: citation.id, title: citation.title });
  }

  async function loadFullText() {
    if (!selected?.doc_key || detailState === "loading") return;
    setDetailState("loading");
    try {
      const document = await getDocument(selected.doc_key);
      setDetail(document);
      setDetailState("idle");
    } catch {
      setDetailState("error");
    }
  }

  return (
    <div className="space-y-4">
      <div className={clsx("rounded-3xl border p-5", toneClasses[urgency.tone])}>
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-white/70">
            {report.urgency === "normal" ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />}
          </span>
          <div>
            <p className="text-[13px] font-medium opacity-80">分诊结果</p>
            <h2 className="mt-0.5 text-xl font-semibold">{urgency.label}</h2>
            <p className="mt-1 text-[13px] leading-6 opacity-85">{urgency.hint}</p>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2 text-[12px]">
          <span className="rounded-full bg-white/70 px-3 py-1">推荐科室：{report.department}</span>
          <span className="rounded-full bg-white/70 px-3 py-1">置信度 {Math.round(confidence * 100)}%</span>
          {triage?.intent ? <span className="rounded-full bg-white/70 px-3 py-1">意图：{triage.intent}</span> : null}
        </div>
      </div>

      <div className="card p-5">
        <div className="flex items-center gap-2">
          <Stethoscope size={17} className="text-primary" />
          <h3 className="section-title">给你的建议</h3>
        </div>
        <p className="mt-3 whitespace-pre-wrap text-[15px] leading-7 text-ink">{report.summary}</p>

        {report.advice?.length ? (
          <ul className="mt-4 space-y-2">
            {report.advice.map((item, index) => (
              <li key={index} className="flex gap-2 text-[14px] leading-6 text-ink">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : null}

        {report.followup_question ? (
          <div className="mt-4 rounded-2xl bg-primary-soft p-3 text-[13px] leading-6 text-primary">
            <span className="font-medium">还需要补充：</span>
            {report.followup_question}
          </div>
        ) : null}

        {report.medications?.length ? (
          <div className="mt-4">
            <p className="mb-2 text-[12px] font-medium text-muted">涉及药物</p>
            <div className="flex flex-wrap gap-2">
              {report.medications.map((item, index) => (
                <span key={index} className="chip">{item}</span>
              ))}
            </div>
          </div>
        ) : null}

        {report.citations?.length ? (
          <div className="mt-5">
            <p className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-muted">
              <BookOpen size={13} />
              依据（点击查看原文）
            </p>
            <div className="flex flex-wrap gap-2">
              {report.citations.map((id, index) => {
                const citation = citationMap.get(String(id));
                return (
                  <button
                    key={`${id}-${index}`}
                    onClick={() => citation && openCitation(citation)}
                    className="chip hover:border-primary/40"
                  >
                    <span className="text-primary">[{index + 1}]</span>
                    <span className="max-w-[190px] truncate">{citation?.title || `证据 ${id}`}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}

        {safety?.requires_human_review ? (
          <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-[12px] leading-6 text-amber-800">
            <p className="flex items-center gap-1.5 font-medium">
              <ShieldCheck size={13} />
              建议人工复核
            </p>
            <ul className="mt-1 list-disc space-y-0.5 pl-4">
              {safety.issues.map((issue, index) => <li key={index}>{issue}</li>)}
            </ul>
          </div>
        ) : null}

        <p className="mt-4 border-t border-line pt-3 text-[11px] leading-5 text-muted">
          以上为分诊与就医建议，不能替代医生面诊、诊断与处方。如出现胸痛、呼吸困难、意识不清等危机信号，请立即拨打 120。
        </p>
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {report.urgency === "critical" ? (
          <a href="tel:120" className="btn-primary bg-rose-500 hover:bg-rose-600">
            <Phone size={16} />
            立即拨打 120
          </a>
        ) : (
          <button onClick={() => { track("result_book_click", { department: report.department }); onBook?.(); }} className="btn-primary">
            <CalendarClock size={16} />
            去挂号
          </button>
        )}
        <button
          onClick={() => {
            const first = citations[0];
            if (first) openCitation(first);
            else setSheetOpen(true);
          }}
          className="btn-ghost"
        >
          <BookOpen size={16} />
          查看医学依据
        </button>
      </div>

      {onRestart ? (
        <button onClick={onRestart} className="btn-ghost w-full">
          <RefreshCcw size={15} />
          重新问诊
        </button>
      ) : null}

      {sheetOpen ? (
        <div className="fixed inset-0 z-[70] flex items-end justify-center bg-black/30 p-0 backdrop-blur-sm sm:items-center sm:p-4" onClick={() => setSheetOpen(false)}>
          <div className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-t-3xl bg-card p-5 shadow-2xl sm:rounded-3xl" onClick={(event) => event.stopPropagation()}>
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[11px] text-muted">医学依据</p>
                <h3 className="mt-1 text-[15px] font-semibold text-ink">{selected?.title || "证据列表"}</h3>
              </div>
              <button className="text-muted hover:text-ink" onClick={() => setSheetOpen(false)} aria-label="关闭">
                <X size={18} />
              </button>
            </div>

            <button onClick={() => setPro((value) => !value)} className="mt-3 text-[11px] text-primary underline underline-offset-2">
              {pro ? "切换到通俗模式" : "专业模式：查看来源与分数"}
            </button>
            {pro ? (
              <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
                {selected?.department ? <span className="chip">{selected.department}</span> : null}
                {selected?.source ? <span className="chip">{selected.source}</span> : null}
                {typeof selected?.score === "number" ? <span className="chip">score {selected.score.toFixed(4)}</span> : null}
              </div>
            ) : null}

            <div className="mt-4 space-y-3">
              {selected ? (
                <div className="rounded-2xl bg-canvas p-3 text-[13px] leading-6 text-ink">
                  <p className="whitespace-pre-wrap">{detail?.content || selected.snippet}</p>
                  {detailState === "error" ? (
                    <p className="mt-2 text-[11px] text-muted">全文加载失败，以上为摘要内容。</p>
                  ) : null}
                  {!detail && selected.doc_key ? (
                    <button
                      onClick={loadFullText}
                      disabled={detailState === "loading"}
                      className="mt-2 text-[11px] text-primary underline underline-offset-2 disabled:opacity-60"
                    >
                      {detailState === "loading" ? "加载中…" : "查看全文"}
                    </button>
                  ) : null}
                </div>
              ) : (
                citations.map((item, index) => (
                  <button key={`${item.id}-${index}`} onClick={() => selectCitation(item)} className="block w-full rounded-2xl bg-canvas p-3 text-left">
                    <p className="text-[13px] font-medium text-ink">{item.title}</p>
                    <p className="mt-1 line-clamp-2 text-[11px] leading-5 text-muted">{item.snippet}</p>
                  </button>
                ))
              )}
            </div>
            <p className="mt-4 text-[11px] leading-5 text-muted">
              证据来自本地医学知识库，仅用于辅助理解。具体诊疗请以医生面诊为准。
            </p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

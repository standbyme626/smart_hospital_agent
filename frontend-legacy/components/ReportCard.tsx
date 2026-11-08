"use client";

import { AlertTriangle, BadgeCheck, BookOpen, Pill, Stethoscope } from "lucide-react";
import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation, Report, Safety } from "@/lib/types";

type Props = {
  report: Report;
  citations: Citation[];
  safety?: Safety;
  onCitation?: (citation: Citation) => void;
};

const urgencyMap: Record<string, { label: string; className: string }> = {
  normal: { label: "普通", className: "border-mint/40 bg-mint/10 text-mint" },
  urgent: { label: "紧急", className: "border-amber/40 bg-amber/10 text-amber" },
  critical: { label: "危急", className: "border-rose/50 bg-rose/10 text-rose" },
};

export function ReportCard({ report, citations, safety, onCitation }: Props) {
  const urgency = urgencyMap[report.urgency] || urgencyMap.normal;
  const confidence = Math.max(0, Math.min(1, Number(report.confidence) || 0));
  const citationMap = new Map(citations.map((item) => [String(item.id), item]));

  return (
    <div className="surface overflow-hidden">
      <div className="flex flex-wrap items-center gap-2 border-b border-line bg-panel2/60 px-4 py-3">
        <span className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1 text-xs text-slate-300">
          <Stethoscope size={13} className="text-primary" />
          {report.department || "全科"}
        </span>
        <span className={clsx("inline-flex items-center rounded-lg border px-2.5 py-1 text-xs", urgency.className)}>
          {urgency.label}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 py-1 text-xs text-slate-400">
          <BadgeCheck size={13} className="text-mint" />
          置信度 {(confidence * 100).toFixed(0)}%
        </span>
        {safety?.requires_human_review ? (
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-amber/40 bg-amber/10 px-2.5 py-1 text-xs text-amber">
            <AlertTriangle size={13} />
            建议人工复核
          </span>
        ) : null}
      </div>

      <div className="space-y-4 px-4 py-4">
        <div className="markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{report.summary || "暂无结论"}</ReactMarkdown>
        </div>

        {report.advice?.length ? (
          <div>
            <p className="mb-2 text-xs font-medium uppercase tracking-wider text-slate-500">建议</p>
            <ul className="space-y-1.5 text-sm text-slate-200">
              {report.advice.map((item, index) => (
                <li key={index} className="flex gap-2">
                  <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {report.medications?.length ? (
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-slate-500">
              <Pill size={13} />
              涉及药物
            </p>
            <div className="flex flex-wrap gap-2">
              {report.medications.map((item, index) => (
                <span key={index} className="chip">
                  {item}
                </span>
              ))}
            </div>
          </div>
        ) : null}

        {report.citations?.length ? (
          <div>
            <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-slate-500">
              <BookOpen size={13} />
              证据
            </p>
            <div className="flex flex-wrap gap-2">
              {report.citations.map((id, index) => {
                const citation = citationMap.get(String(id));
                return (
                  <button
                    key={`${id}-${index}`}
                    className="chip hover:border-primary/50 hover:text-primary"
                    onClick={() => citation && onCitation?.(citation)}
                    title={citation?.title || String(id)}
                  >
                    <span className="text-primary">[{index + 1}]</span>
                    <span className="max-w-[180px] truncate">{citation?.title || `证据 ${id}`}</span>
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}

        {report.followup_question ? (
          <div className="rounded-xl border border-primary/30 bg-primary/5 px-3 py-2.5 text-sm text-slate-200">
            <span className="mr-1 text-primary">需要补充：</span>
            {report.followup_question}
          </div>
        ) : null}

        {safety?.issues?.length ? (
          <div className="rounded-xl border border-amber/40 bg-amber/10 px-3 py-2.5 text-xs text-amber">
            <p className="mb-1 font-medium">安全提示</p>
            <ul className="list-disc space-y-0.5 pl-4">
              {safety.issues.map((issue, index) => (
                <li key={index}>{issue}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <p className="border-t border-line pt-3 text-[11px] leading-5 text-slate-500">
          以上为分诊与就医建议，不能替代医生面诊、诊断与处方。如出现胸痛、呼吸困难、意识不清等危机信号，请立即拨打 120。
        </p>
      </div>
    </div>
  );
}

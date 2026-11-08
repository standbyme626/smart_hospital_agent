"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  FileText,
  Loader2,
  Sparkles,
  UploadCloud,
  X,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { useApp } from "@/lib/app-context";
import { getReports, interpretReport, track } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { ReportRecord } from "@/lib/types";

export default function ReportsPage() {
  const { activeMember, showToast } = useApp();
  const [mode, setMode] = useState<"report" | "skin">("report");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [selected, setSelected] = useState<ReportRecord | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const savedMode = window.sessionStorage.getItem("report_mode");
    if (savedMode === "skin") {
      setMode("skin");
      window.sessionStorage.removeItem("report_mode");
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  async function load() {
    try {
      setReports(await getReports(activeMember?.id));
    } catch {
      setReports([]);
    }
  }

  function chooseFile(next: File | null) {
    if (!next) return;
    if (!next.type.startsWith("image/")) {
      showToast("请选择图片文件（JPG/PNG/WebP）");
      return;
    }
    if (next.size > 8 * 1024 * 1024) {
      showToast("图片不能超过 8MB");
      return;
    }
    setFile(next);
    setSelected(null);
    const url = URL.createObjectURL(next);
    setPreview(url);
    track("report_file_selected", { mode, size: next.size });
  }

  async function submit() {
    if (!file) {
      showToast("先上传一张图片");
      return;
    }
    setLoading(true);
    track("report_interpret", { mode, has_title: Boolean(title) });
    try {
      const result = await interpretReport(file, activeMember?.id, title, mode);
      setSelected(result);
      setReports((prev) => [result, ...prev]);
      showToast(mode === "skin" ? "照片已观察完成" : "报告已解读完成");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "解读失败，请稍后再试");
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell title="报告解读" subtitle="上传检查报告或皮肤照片，用普通人能懂的话解释">
      <div className="mx-auto max-w-4xl space-y-4">
        <div className="flex gap-2">
          <button onClick={() => { setMode("report"); setSelected(null); }} className={clsx("chip min-h-10 px-4", mode === "report" ? "chip-active" : "")}>检查报告</button>
          <button onClick={() => { setMode("skin"); setSelected(null); }} className={clsx("chip min-h-10 px-4", mode === "skin" ? "chip-active" : "")}>皮肤照片</button>
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 text-primary">
            {mode === "skin" ? <Camera size={18} /> : <FileText size={18} />}
            <h2 className="text-[17px] font-semibold">{mode === "skin" ? "拍照识皮肤" : "上传检查报告"}</h2>
          </div>
          <p className="mt-1 text-[12px] leading-5 text-muted">
            {mode === "skin"
              ? "拍一张清晰的皮肤照片，大医会客观描述可见表现，并给出就医建议。不诊断、不推荐处方药。"
              : "支持化验单、检查报告、出院小结等。演示环境：数据保留用于展示；导出/删除功能见设置页（若已上线）。"}
          </p>

          <div className="mt-4 grid gap-4 sm:grid-cols-[1.1fr_0.9fr]">
            <div>
              {preview ? (
                <div className="relative overflow-hidden rounded-2xl border border-line bg-canvas">
                  <img src={preview} alt="预览" className="max-h-[320px] w-full object-contain" />
                  <button
                    onClick={() => {
                      setFile(null);
                      setPreview(null);
                      if (inputRef.current) inputRef.current.value = "";
                    }}
                    className="absolute right-2 top-2 flex h-8 w-8 items-center justify-center rounded-full bg-black/50 text-white"
                    aria-label="移除图片"
                  >
                    <X size={15} />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => inputRef.current?.click()}
                  className="flex min-h-[220px] w-full flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-line bg-canvas text-muted transition hover:border-primary/40 hover:text-primary"
                >
                  <UploadCloud size={30} />
                  <span className="text-[14px] font-medium">点击上传或拍照</span>
                  <span className="text-[11px]">JPG / PNG / WebP · 不超过 8MB</span>
                </button>
              )}
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                capture="environment"
                className="hidden"
                onChange={(event) => chooseFile(event.target.files?.[0] || null)}
              />
            </div>

            <div className="space-y-3">
              <label className="block">
                <span className="label">标题（可选）</span>
                <input className="input" value={title} onChange={(event) => setTitle(event.target.value)} placeholder={mode === "skin" ? "例如：手臂皮疹" : "例如：血常规检查报告"} />
              </label>
              <div className="rounded-2xl bg-canvas p-3 text-[11px] leading-5 text-muted">
                <p className="flex items-center gap-1.5 text-ink"><ShieldCheckIcon /> 隐私提示</p>
                <p className="mt-1">图片仅用于本次解读，不对外分享；可在健康档案中查看记录。</p>
              </div>
              <button onClick={submit} disabled={loading || !file} className="btn-primary w-full">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                {loading ? "解读中…" : mode === "skin" ? "开始观察" : "开始解读"}
              </button>
            </div>
          </div>
        </div>

        {selected ? (
          <div className="card p-5 fade-up">
            <div className="flex items-center justify-between gap-2">
              <div>
                <p className="text-[11px] text-muted">{formatDate(selected.created_at)}</p>
                <h2 className="mt-1 text-[17px] font-semibold text-ink">{selected.title}</h2>
              </div>
              <span className="badge bg-primary-soft text-primary">AI 解读</span>
            </div>
            <p className="mt-3 whitespace-pre-wrap text-[14px] leading-7 text-ink">{selected.summary}</p>

            {selected.abnormal_values?.length ? (
              <div className="mt-4 space-y-2">
                <p className="text-[12px] font-medium text-muted">异常指标</p>
                {selected.abnormal_values.map((item, index) => (
                  <div key={index} className="rounded-2xl border border-line bg-canvas p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[13px] font-medium text-ink">{item.name}</span>
                      <span className={clsx("badge", item.status === "high" ? "bg-rose-50 text-rose-600" : item.status === "low" ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700")}>
                        {item.status === "high" ? "偏高" : item.status === "low" ? "偏低" : "正常"}
                      </span>
                      <span className="text-[12px] text-muted">{item.value} · 参考 {item.reference}</span>
                    </div>
                    <p className="mt-1 text-[12px] leading-5 text-muted">{item.explanation}</p>
                  </div>
                ))}
              </div>
            ) : null}

            {selected.questions?.length ? (
              <div className="mt-4">
                <p className="text-[12px] font-medium text-muted">建议问医生</p>
                <ul className="mt-2 space-y-1.5">
                  {selected.questions.map((item, index) => (
                    <li key={index} className="flex gap-2 text-[13px] leading-6 text-ink">
                      <CheckCircle2 size={14} className="mt-1 shrink-0 text-primary" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <p className="mt-4 border-t border-line pt-3 text-[11px] leading-5 text-muted">
              本解读不能替代医生诊断。若报告提示危急值或症状加重，请立即就医。
            </p>
          </div>
        ) : null}

        <div className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">历史记录</h2>
            <span className="text-[11px] text-muted">{reports.length} 条</span>
          </div>
          <div className="mt-3 space-y-2">
            {reports.length === 0 ? (
              <p className="rounded-2xl bg-canvas p-4 text-[13px] text-muted">还没有报告记录，上传一份试试。</p>
            ) : (
              reports.map((item) => (
                <button key={item.id} onClick={() => { setSelected(item); window.scrollTo({ top: 0, behavior: "smooth" }); }} className="flex w-full items-center justify-between gap-3 rounded-2xl bg-canvas p-3 text-left hover:bg-primary-soft/50">
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] font-medium text-ink">{item.title}</span>
                    <span className="mt-0.5 block truncate text-[11px] text-muted">{item.summary}</span>
                  </span>
                  <span className="shrink-0 text-[11px] text-muted">{formatDate(item.created_at)}</span>
                  <ChevronRight size={15} className="shrink-0 text-muted" />
                </button>
              ))
            )}
          </div>
        </div>

        <div className="flex items-start gap-2 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-[11px] leading-5 text-amber-800">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          报告和皮肤照片解读只用于辅助理解，不构成诊断。危急值、持续加重或出现呼吸困难等情况请立即就医。
        </div>
      </div>
    </AppShell>
  );
}

function ShieldCheckIcon() {
  return (
    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 3l7 4v5c0 5-3.5 8-7 9-3.5-1-7-4-7-9V7l7-4z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  );
}

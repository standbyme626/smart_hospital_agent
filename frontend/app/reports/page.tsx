"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  ChevronRight,
  FileText,
  Image as ImageIcon,
  Loader2,
  Sparkles,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import clsx from "clsx";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import { deleteReport, getReport, getReportImage, getReports, interpretReport, track } from "@/lib/api";
import { abnormalStatusMeta, formatDate, POLICY_VERSION } from "@/lib/format";
import type { ReportRecord } from "@/lib/types";

export default function ReportsPage() {
  const { activeMember, showToast } = useApp();
  const [mode, setMode] = useState<"report" | "skin">("report");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadConsent, setUploadConsent] = useState(false);
  const [reports, setReports] = useState<ReportRecord[]>([]);
  const [loadingReports, setLoadingReports] = useState(true);
  const [loadError, setLoadError] = useState(false);
  const [selected, setSelected] = useState<ReportRecord | null>(null);
  const [detailImage, setDetailImage] = useState<string | null>(null);
  const [detailImageError, setDetailImageError] = useState(false);
  const [detailRetry, setDetailRetry] = useState(0);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
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

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    setDetailImage(null);
    setDetailImageError(false);
    if (selected?.has_image) {
      getReportImage(selected.id)
        .then((blob) => {
          if (cancelled) return;
          objectUrl = URL.createObjectURL(blob);
          setDetailImage(objectUrl);
        })
        .catch(() => {
          if (!cancelled) {
            setDetailImage(null);
            setDetailImageError(true);
          }
        });
    }
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [selected?.id, selected?.has_image, detailRetry]);

  useEffect(() => {
    return () => {
      if (viewerUrl) URL.revokeObjectURL(viewerUrl);
    };
  }, [viewerUrl]);

  async function load() {
    setLoadingReports(true);
    setLoadError(false);
    try {
      const list = await getReports(activeMember?.id);
      setReports(list);
      const target = new URLSearchParams(window.location.search).get("id");
      if (target) {
        const found = list.find((item) => item.id === target);
        if (found) {
          setSelected(found);
        } else {
          const single = await getReport(target).catch(() => null);
          if (single) setSelected(single);
        }
      }
    } catch {
      setReports([]);
      setLoadError(true);
    } finally {
      setLoadingReports(false);
    }
  }

  async function openViewer(item: ReportRecord) {
    try {
      const blob = await getReportImage(item.id);
      setViewerUrl(URL.createObjectURL(blob));
    } catch {
      showToast("图片加载失败");
    }
  }

  function closeViewer() {
    setViewerUrl(null);
  }

  async function removeReport(item: ReportRecord) {
    if (!window.confirm(`删除报告「${item.title}」？图片与记录将一并删除，且无法恢复。`)) return;
    setDeleting(true);
    try {
      await deleteReport(item.id);
      setReports((prev) => prev.filter((entry) => entry.id !== item.id));
      setSelected((prev) => (prev?.id === item.id ? null : prev));
      closeViewer();
      showToast("报告已删除");
    } catch (err) {
      showToast(err instanceof Error ? err.message : "删除失败，请稍后再试");
    } finally {
      setDeleting(false);
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
    if (!uploadConsent) {
      showToast("请先阅读并勾选同意后再解读");
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
    <AppShell
      title="报告解读"
      subtitle="上传检查报告或皮肤照片，用普通人能懂的话解释"
      actions={<SourceBadge live />}
    >
      <div className="mx-auto max-w-4xl space-y-4">
        <h1 className="text-xl font-semibold text-ink lg:hidden">报告解读</h1>
        <OfflineBanner />
        <div className="flex gap-2">
          <button onClick={() => { setMode("report"); setSelected(null); }} aria-pressed={mode === "report"} className={clsx("chip min-h-11 px-4", mode === "report" ? "chip-active" : "")}>检查报告</button>
          <button onClick={() => { setMode("skin"); setSelected(null); }} aria-pressed={mode === "skin"} className={clsx("chip min-h-11 px-4", mode === "skin" ? "chip-active" : "")}>皮肤照片</button>
        </div>

        <div className="card p-5">
          <div className="flex items-center gap-2 text-primary">
            {mode === "skin" ? <Camera size={18} /> : <FileText size={18} />}
            <h2 className="text-[17px] font-semibold">{mode === "skin" ? "拍照识皮肤" : "上传检查报告"}</h2>
          </div>
          <p className="mt-1 text-[12px] leading-5 text-muted">
            {mode === "skin"
              ? "拍一张清晰的皮肤照片，大医会客观描述可见表现，并给出就医建议。不诊断、不推荐处方药。"
              : "支持化验单、检查报告、出院小结等。记录可随时在下方删除，完整数据可在设置页导出或删除账号。"}
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
                    className="absolute right-2 top-2 flex h-11 w-11 items-center justify-center rounded-full bg-black/50 text-white"
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
                <p className="flex items-center gap-1.5 text-ink"><ShieldCheckIcon /> 隐私与数据说明（{POLICY_VERSION}）</p>
                <p className="mt-1">图片会保存到你的健康档案，可随时删除；解读会调用第三方 AI 服务。详见
                  <Link href="/privacy/" className="mx-1 text-primary underline underline-offset-2">《隐私政策》</Link>。
                </p>
              </div>
              <label className="flex items-start gap-2 rounded-2xl border border-line p-3 text-[11px] leading-5 text-muted">
                <input
                  type="checkbox"
                  checked={uploadConsent}
                  onChange={(event) => setUploadConsent(event.target.checked)}
                  className="mt-0.5 h-4 w-4 shrink-0 accent-[var(--color-primary)]"
                />
                <span>我已知悉并同意：图片会保存到健康档案（可随时删除），解读会调用第三方 AI 服务。</span>
              </label>
              <button onClick={submit} disabled={loading || !file || !uploadConsent} className="btn-primary w-full">
                {loading ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
                {loading ? "解读中…" : mode === "skin" ? "开始观察" : "开始解读"}
              </button>
              {!uploadConsent && file ? <p className="text-center text-[11px] text-muted">请先勾选同意后再开始解读。</p> : null}
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
              <div className="flex items-center gap-2">
                <span className="badge bg-primary-soft text-primary">AI 解读</span>
                <button
                  onClick={() => removeReport(selected)}
                  disabled={deleting}
                  className="flex h-11 w-11 items-center justify-center rounded-full text-muted transition hover:bg-rose-50 hover:text-rose-600 disabled:opacity-40"
                  aria-label="删除报告"
                >
                  {deleting ? <Loader2 size={15} className="animate-spin" /> : <Trash2 size={15} />}
                </button>
              </div>
            </div>
            {(() => {
              const flags = Array.isArray(selected.critical_flags) ? selected.critical_flags.filter(Boolean) : [];
              const isCritical = String(selected.urgency || "").toLowerCase() === "critical" || flags.length > 0;
              if (!isCritical) return null;
              return (
                <div role="alert" className="mt-3 rounded-2xl border border-rose-300 bg-rose-50 p-3 text-[12px] leading-5 text-rose-800">
                  <p className="flex items-center gap-1.5 text-[13px] font-semibold"><AlertTriangle size={15} />危急值提醒：请立即联系医生或前往急诊</p>
                  {flags.length ? <p className="mt-1">危急项：{flags.join("；")}</p> : null}
                  <a href="tel:120" className="mt-2 inline-flex min-h-11 items-center gap-1.5 rounded-full bg-rose-600 px-4 text-[12px] font-medium text-white">拨打 120</a>
                </div>
              );
            })()}
            {selected.has_image ? (
              <div className="mt-3 overflow-hidden rounded-2xl border border-line bg-canvas">
                {detailImage ? (
                  <button onClick={() => openViewer(selected)} className="block w-full" aria-label="查看报告大图">
                    <img src={detailImage} alt="报告原图" className="max-h-[360px] w-full object-contain" />
                  </button>
                ) : detailImageError ? (
                  <div className="flex h-32 items-center justify-center gap-2 text-[12px] text-muted">
                    <AlertTriangle size={15} className="shrink-0" />
                    图片加载失败
                    <button onClick={() => setDetailRetry((count) => count + 1)} className="text-primary">
                      重试
                    </button>
                  </div>
                ) : (
                  <div className="flex h-32 items-center justify-center gap-2 text-[12px] text-muted">
                    <Loader2 size={15} className="animate-spin" />
                    正在加载图片…
                  </div>
                )}
              </div>
            ) : null}
            {String(selected.urgency || "").toLowerCase() === "critical" || selected.critical_flags?.length ? (
              <div className="mt-3 rounded-2xl bg-rose-50 px-3 py-2 text-[13px] leading-6 text-rose-700">
                报告提示危急值，请立即联系医生或前往急诊，不要等待在线解读。
              </div>
            ) : null}
            <p className="mt-3 whitespace-pre-wrap text-[14px] leading-7 text-ink">{selected.summary}</p>

            {selected.abnormal_values?.length ? (
              <div className="mt-4">
                <p className="text-[12px] font-medium text-muted">异常指标</p>
                <div className="mt-2 overflow-hidden rounded-2xl border border-line">
                  <table className="w-full text-left text-[12px]">
                    <thead className="bg-canvas text-muted">
                      <tr>
                        <th className="px-3 py-2 font-medium">指标</th>
                        <th className="px-3 py-2 font-medium">结果</th>
                        <th className="px-3 py-2 font-medium">参考</th>
                        <th className="px-3 py-2 font-medium">状态</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selected.abnormal_values.map((item, index) => {
                        const status = abnormalStatusMeta(item.status);
                        return (
                          <tr key={index} className="border-t border-line">
                            <td className="px-3 py-2 text-ink">{item.name}</td>
                            <td className="px-3 py-2 text-ink">{item.value}</td>
                            <td className="px-3 py-2 text-muted">{item.reference}</td>
                            <td className="px-3 py-2"><span className={clsx("badge", status.className)}>{status.label}</span></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                {selected.abnormal_values.some((item) => item.explanation) ? (
                  <ul className="mt-2 space-y-1">
                    {selected.abnormal_values.filter((item) => item.explanation).map((item, index) => (
                      <li key={index} className="text-[12px] leading-5 text-muted">· {item.name}：{item.explanation}</li>
                    ))}
                  </ul>
                ) : null}
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
            <span className="text-[11px] text-muted">{loadingReports || loadError ? "—" : `${reports.length} 条`}</span>
          </div>
          <div className="mt-3 space-y-2">
            {loadingReports ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loadingReports && loadError ? <StateBlock kind="error" onRetry={load} /> : null}
            {!loadingReports && !loadError && reports.length === 0 ? (
              <StateBlock kind="empty" title="还没有报告记录" description="上传一份化验或检查报告，解读会出现在这里" />
            ) : null}
            {!loadError ? (
              reports.map((item) => {
                const critical = String(item.urgency || "").toLowerCase() === "critical" || Boolean(item.critical_flags?.length);
                return (
                <div key={item.id} className="flex items-center gap-2 rounded-2xl bg-canvas p-3">
                  <button
                    onClick={() => { setSelected(item); window.scrollTo({ top: 0, behavior: "smooth" }); }}
                    className="flex min-w-0 flex-1 items-center justify-between gap-3 text-left"
                  >
                    <span className="min-w-0">
                      <span className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                        {critical ? <span className="badge bg-rose-100 text-rose-700">危急</span> : null}
                        <span className="truncate">{item.title}</span>
                      </span>
                      <span className="mt-0.5 block truncate text-[11px] text-muted">{item.summary}</span>
                    </span>
                    <span className="shrink-0 text-[11px] text-muted">{formatDate(item.created_at)}</span>
                    <ChevronRight size={15} className="shrink-0 text-muted" />
                  </button>
                  {item.has_image ? (
                    <button
                      onClick={() => openViewer(item)}
                      className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-white hover:text-primary"
                      aria-label="查看图片"
                    >
                      <ImageIcon size={15} />
                    </button>
                  ) : null}
                  <button
                    onClick={() => removeReport(item)}
                    disabled={deleting}
                    className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full text-muted transition hover:bg-white hover:text-rose-600 disabled:opacity-40"
                    aria-label="删除报告"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
                );
              })
            ) : null}
          </div>
        </div>

        <div className="flex items-start gap-2 rounded-2xl border border-amber-200 bg-amber-50 p-3 text-[11px] leading-5 text-amber-800">
          <AlertTriangle size={14} className="mt-0.5 shrink-0" />
          报告和皮肤照片解读只用于辅助理解，不构成诊断。危急值、持续加重或出现呼吸困难等情况请立即就医。
        </div>
      </div>

      {viewerUrl ? (
        <div className="fixed inset-0 z-[90] flex items-center justify-center bg-black/70 p-4" onClick={closeViewer}>
          <img
            src={viewerUrl}
            alt="报告图片"
            className="max-h-full max-w-full rounded-2xl bg-white object-contain"
            onClick={(event) => event.stopPropagation()}
          />
          <button
            onClick={closeViewer}
            className="absolute right-4 top-4 flex h-11 w-11 items-center justify-center rounded-full bg-black/50 text-white"
            aria-label="关闭图片"
          >
            <X size={18} />
          </button>
        </div>
      ) : null}
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

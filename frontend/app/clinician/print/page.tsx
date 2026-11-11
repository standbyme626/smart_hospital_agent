"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ClinicianNav } from "@/components/ClinicianNav";
import { getClinicalNotes, getClinicianQueue, type ClinicalNote, type ClinicianQueueItem } from "@/lib/api";
import {
  createPrintJob,
  getPrintTemplates,
  listPrintJobs,
  markPrinted,
  type PrintJobSummary,
  type PrintTemplate,
} from "@/lib/api-clinician-ext";

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

export default function ClinicianPrintPage() {
  const [queue, setQueue] = useState<ClinicianQueueItem[]>([]);
  const [templates, setTemplates] = useState<PrintTemplate[]>([]);
  const [notes, setNotes] = useState<ClinicalNote[]>([]);
  const [jobs, setJobs] = useState<PrintJobSummary[]>([]);
  const [memberId, setMemberId] = useState<string>("");
  const [manualMember, setManualMember] = useState<string>("");
  const [noteId, setNoteId] = useState<string>("");
  const [template, setTemplate] = useState<string>("outpatient_note");
  const [content, setContent] = useState<string>("");
  const [jobId, setJobId] = useState<string>("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [queueData, templateData, jobData] = await Promise.all([
        getClinicianQueue(),
        getPrintTemplates(),
        listPrintJobs(),
      ]);
      setQueue(queueData);
      setTemplates(templateData.items);
      setJobs(jobData.items);
      if (!memberId && queueData.length) setMemberId(queueData[0].member_id);
    } catch (err) {
      const status = (err as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setError((err as Error).message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, [memberId]);

  useEffect(() => {
    void load();
  }, [load]);

  const loadNotes = useCallback(async (targetMember: string) => {
    if (!targetMember) return;
    try {
      setNotes(await getClinicalNotes(targetMember));
    } catch {
      setNotes([]);
    }
  }, []);

  const targetMember = manualMember.trim() || memberId;

  useEffect(() => {
    void loadNotes(targetMember);
  }, [targetMember, loadNotes]);

  const generate = async () => {
    setError(null);
    setMessage(null);
    try {
      const result = await createPrintJob({
        template,
        note_id: template === "outpatient_note" ? noteId || undefined : undefined,
        member_id: targetMember,
      });
      setContent(result.content);
      setJobId(result.job.id);
      setMessage("已生成打印任务 " + result.job.id + "（服务端渲染并存库）");
      setJobs((await listPrintJobs()).items);
    } catch (err) {
      setError((err as Error).message || "生成失败");
    }
  };

  const mark = async () => {
    if (!jobId) return;
    try {
      await markPrinted(jobId);
      setMessage("已标记打印完成（留痕）");
      setJobs((await listPrintJobs()).items);
    } catch (err) {
      setError((err as Error).message || "标记失败");
    }
  };

  return (
    <main className="mx-auto w-full max-w-5xl px-4 py-6">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-[20px] font-semibold text-ink">文书打印</h1>
          <p className="mt-1 text-[12px] text-muted">门诊病历 / 处方笺 / 检查申请单，正文由服务端渲染并存库</p>
        </div>
        <div className="ml-auto"><SourceBadge live /></div>
      </header>

      <div className="mt-3"><ClinicianNav /></div>
      <div className="mt-3"><OfflineBanner /></div>

      {error ? <p className="mt-3 rounded-2xl bg-rose-50 px-4 py-2 text-[12px] text-rose-700">{error}</p> : null}
      {message ? <p className="mt-3 rounded-2xl bg-primary-soft px-4 py-2 text-[12px] text-primary">{message}</p> : null}

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-24 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要医生权限" /></div>
      ) : (
        <div className="mt-4 space-y-4">
          <section className="card p-4">
            <h2 className="section-title">生成打印任务</h2>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              <label>
                <span className="text-[11px] text-muted">就诊人（今日队列）</span>
                <select
                  value={memberId}
                  onChange={(event) => { setMemberId(event.target.value); setNoteId(""); }}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                >
                  {queue.map((item) => (
                    <option key={item.member_id} value={item.member_id}>
                      {item.member_name || item.member_id}
                    </option>
                  ))}
                  {queue.length === 0 ? <option value="">今日队列为空</option> : null}
                </select>
              </label>
              <label>
                <span className="text-[11px] text-muted">或输入就诊人 ID（非当日挂号）</span>
                <input
                  value={manualMember}
                  onChange={(event) => { setManualMember(event.target.value); setNoteId(""); }}
                  placeholder="mem_xxx"
                  aria-label="手动输入就诊人 ID"
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                />
              </label>
              <label>
                <span className="text-[11px] text-muted">模板</span>
                <select
                  value={template}
                  onChange={(event) => setTemplate(event.target.value)}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                >
                  {templates.map((item) => (
                    <option key={item.key} value={item.key}>{item.name}</option>
                  ))}
                </select>
              </label>
              <label>
                <span className="text-[11px] text-muted">病历（门诊病历必选）</span>
                <select
                  value={noteId}
                  onChange={(event) => setNoteId(event.target.value)}
                  disabled={template !== "outpatient_note"}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink disabled:opacity-50"
                >
                  <option value="">请选择病历</option>
                  {notes.map((note) => (
                    <option key={note.id} value={note.id}>
                      {(note.chief_complaint || "未填主诉") + " · " + (note.status === "completed" ? "已诊毕" : "草稿")}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <button
              onClick={() => void generate()}
              disabled={!targetMember || (template === "outpatient_note" && !noteId)}
              className="mt-3 rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white disabled:opacity-50"
            >
              生成打印任务
            </button>
            <p className="mt-2 text-[11px] text-muted">
              处方与检查单只包含**已签名**的医嘱；未诊毕的病历会标注草稿，避免被当成正式文书。
            </p>
          </section>

          {content ? (
            <section className="card p-4">
              <div className="flex items-center justify-between">
                <h2 className="section-title">打印预览</h2>
                <div className="flex gap-2">
                  <button onClick={() => window.print()} className="rounded-full bg-white px-3 py-1 text-[12px] text-primary ring-1 ring-line">
                    浏览器打印
                  </button>
                  <button onClick={() => void mark()} className="rounded-full bg-primary px-3 py-1 text-[12px] font-medium text-white">
                    标记已打印
                  </button>
                </div>
              </div>
              <pre className="mt-3 max-h-96 overflow-auto whitespace-pre-wrap rounded-2xl bg-canvas p-4 text-[12px] leading-6 text-ink">
                {content}
              </pre>
            </section>
          ) : null}

          <section className="card overflow-hidden">
            <div className="px-4 py-3"><h2 className="section-title">打印记录</h2></div>
            {jobs.length === 0 ? (
              <div className="border-t border-line p-4"><StateBlock kind="empty" title="还没有打印记录" /></div>
            ) : (
              <ul className="divide-y divide-line border-t border-line">
                {jobs.map((job) => (
                  <li key={job.id} className="flex flex-wrap items-center gap-2 px-4 py-2 text-[12px]">
                    <span className={job.status === "printed" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>
                      {job.status === "printed" ? "已打印" : "待打印"}
                    </span>
                    <span className="font-medium text-ink">{job.template_name}</span>
                    <span className="text-muted">{job.member_id}</span>
                    <span className="ml-auto text-[11px] text-muted">{job.doctor || "—"} · {formatTime(job.created_at)}</span>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}
    </main>
  );
}

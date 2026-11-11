"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { getClinicalNotes, getClinicianQueue, type ClinicalNote, type ClinicianQueueItem } from "@/lib/api";
import { closeVisit, getFollowupPlans, type CloseVisitResult, type FollowupPlan } from "@/lib/api-clinician-ext";

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

export default function ClinicianFollowupPage() {
  const [queue, setQueue] = useState<ClinicianQueueItem[]>([]);
  const [notes, setNotes] = useState<ClinicalNote[]>([]);
  const [plans, setPlans] = useState<FollowupPlan[]>([]);
  const [memberId, setMemberId] = useState("");
  const [noteId, setNoteId] = useState("");
  const [advice, setAdvice] = useState("按时服药，两周后复查；症状加重随时就诊。");
  const [days, setDays] = useState(14);
  const [result, setResult] = useState<CloseVisitResult | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const [queueData, planData] = await Promise.all([getClinicianQueue(), getFollowupPlans()]);
      setQueue(queueData);
      setPlans(planData.items);
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

  useEffect(() => {
    let alive = true;
    if (!memberId) {
      setNotes([]);
      return;
    }
    getClinicalNotes(memberId)
      .then((items) => {
        if (!alive) return;
        setNotes(items);
        const draft = items.find((note) => note.status !== "completed");
        setNoteId(draft ? draft.id : "");
      })
      .catch(() => {
        if (alive) setNotes([]);
      });
    return () => {
      alive = false;
    };
  }, [memberId]);

  const submit = async () => {
    setError(null);
    setMessage(null);
    setResult(null);
    try {
      const closed = await closeVisit(noteId, { advice, followup_days: days });
      setResult(closed);
      setMessage("诊毕完成，已生成随访计划（患者端随访列表可见）");
      setPlans((await getFollowupPlans()).items);
      setNotes(await getClinicalNotes(memberId));
    } catch (err) {
      setError((err as Error).message || "诊毕失败");
    }
  };

  return (
    <ConsoleShell end="clinician" title="诊毕与随访" subtitle="诊毕会把病历置为只读，并给患者生成一条随访计划">

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
            <h2 className="section-title">诊毕</h2>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <label>
                <span className="text-[11px] text-muted">就诊人</span>
                <select
                  value={memberId}
                  onChange={(event) => setMemberId(event.target.value)}
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
                <span className="text-[11px] text-muted">待诊毕病历</span>
                <select
                  value={noteId}
                  onChange={(event) => setNoteId(event.target.value)}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                >
                  <option value="">请选择病历</option>
                  {notes.map((note) => (
                    <option key={note.id} value={note.id} disabled={note.status === "completed"}>
                      {(note.chief_complaint || "未填主诉") + " · " + (note.status === "completed" ? "已诊毕" : "草稿")}
                    </option>
                  ))}
                </select>
              </label>
              <label className="sm:col-span-2">
                <span className="text-[11px] text-muted">随访医嘱（会显示在患者随访里）</span>
                <textarea
                  value={advice}
                  onChange={(event) => setAdvice(event.target.value)}
                  rows={3}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                />
              </label>
              <label>
                <span className="text-[11px] text-muted">随访天数（1-180）</span>
                <input
                  type="number"
                  min={1}
                  max={180}
                  value={days}
                  onChange={(event) => setDays(Number(event.target.value))}
                  className="mt-1 w-full rounded-xl bg-canvas px-3 py-2 text-[12px] text-ink"
                />
              </label>
            </div>
            <button
              onClick={() => void submit()}
              disabled={!noteId || days < 1 || days > 180}
              className="mt-3 rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white disabled:opacity-50"
            >
              诊毕并生成随访
            </button>
            <p className="mt-2 text-[11px] text-muted">
              已诊毕的病历不能再改（只读）；重复诊毕会被拒绝（409），避免生成重复随访。
            </p>
          </section>

          {result ? (
            <section className="card p-4">
              <h2 className="section-title">本次生成</h2>
              <div className="mt-2 rounded-2xl bg-canvas p-3 text-[12px] leading-6 text-ink">
                <p>病历：{result.note.id} → {result.note.status === "completed" ? "已诊毕" : result.note.status}</p>
                <p>随访计划：{result.followup.title}</p>
                <p>随访到期：{result.followup.due_date}（{result.closure.followup_days} 天）</p>
              </div>
            </section>
          ) : null}

          <section className="card overflow-hidden">
            <div className="px-4 py-3"><h2 className="section-title">随访计划</h2></div>
            {plans.length === 0 ? (
              <div className="border-t border-line p-4"><StateBlock kind="empty" title="还没有随访计划" /></div>
            ) : (
              <ul className="divide-y divide-line border-t border-line">
                {plans.map((plan) => (
                  <li key={plan.id} className="px-4 py-2 text-[12px]">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className={plan.status === "pending" ? "badge bg-amber-50 text-amber-700" : "badge bg-emerald-50 text-emerald-700"}>
                        {plan.status === "pending" ? "待随访" : plan.status || "—"}
                      </span>
                      <span className="font-medium text-ink">{plan.title || "随访"}</span>
                      <span className="text-muted">{plan.member_name || plan.member_id}</span>
                      <span className="text-muted">到期 {plan.due_date || "—"}</span>
                      <span className="ml-auto text-[11px] text-muted">{plan.doctor || "—"} · {formatTime(plan.created_at)}</span>
                    </div>
                    {plan.advice ? <p className="mt-1 text-[11px] text-muted">{plan.advice}</p> : null}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
      )}
    </ConsoleShell>
  );
}

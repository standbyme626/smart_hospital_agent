"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { CheckCircle2, ClipboardList, Loader2 } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { useApp } from "@/lib/app-context";
import { createPrevisit, getPrevisits, track, type PrevisitForm } from "@/lib/api";
import { relativeTime } from "@/lib/format";

export default function PrevisitPage() {
  const { activeMember, showToast } = useApp();
  const [forms, setForms] = useState<PrevisitForm[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState<PrevisitForm | null>(null);

  const [chief, setChief] = useState("");
  const [duration, setDuration] = useState("");
  const [medications, setMedications] = useState("");
  const [allergies, setAllergies] = useState("");
  const [history, setHistory] = useState("");
  const [questions, setQuestions] = useState("");

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setForms(await getPrevisits());
    } catch {
      setForms([]);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeMember?.id]);

  async function submit() {
    if (!chief.trim()) {
      showToast("请填写主诉");
      return;
    }
    setSubmitting(true);
    setSubmitted(null);
    try {
      const form = await createPrevisit({
        chief_complaint: chief.trim(),
        duration: duration.trim() || undefined,
        medications: medications.trim() || undefined,
        allergies: allergies.trim() || undefined,
        history: history.trim() || undefined,
        questions: questions.trim() || undefined,
        member_id: activeMember?.id,
      });
      setSubmitted(form);
      setChief("");
      setDuration("");
      setMedications("");
      setAllergies("");
      setHistory("");
      setQuestions("");
      track("previsit_submit", { member_id: activeMember?.id || null });
      await load();
    } catch {
      showToast("提交失败，请稍后再试");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell title="就诊准备" subtitle="提前把情况写清楚，医生接诊时直接看到" actions={<SourceBadge live />}>
      <div className="readable space-y-4">
        <OfflineBanner />

        {submitted ? (
          <div className="card border-emerald-200 bg-emerald-50 p-4">
            <p className="flex items-center gap-2 text-[14px] font-medium text-emerald-700">
              <CheckCircle2 size={16} /> 已提交，医生接诊时会看到这份摘要
            </p>
            <p className="mt-1 text-[12px] text-emerald-700/80">
              主诉：{submitted.chief_complaint}
            </p>
          </div>
        ) : null}

        <section className="card p-5">
          <div className="flex items-center gap-2 text-primary">
            <ClipboardList size={18} />
            <h2 className="text-[17px] font-semibold">就诊准备</h2>
          </div>
          <p className="mt-1 text-[12px] leading-5 text-muted">
            这些信息会进入医生端「患者 360」；不替代面诊，也不会用于自动诊断。
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <label className="block sm:col-span-2">
              <span className="label">主诉（必填）</span>
              <input className="input" value={chief} onChange={(event) => setChief(event.target.value)} placeholder="例如：胸痛 30 分钟，活动后加重" />
            </label>
            <label className="block">
              <span className="label">持续时间</span>
              <input className="input" value={duration} onChange={(event) => setDuration(event.target.value)} placeholder="例如：30 分钟 / 2 天" />
            </label>
            <label className="block">
              <span className="label">在用药物</span>
              <input className="input" value={medications} onChange={(event) => setMedications(event.target.value)} placeholder="例如：氨氯地平 5mg 每日一次" />
            </label>
            <label className="block">
              <span className="label">过敏史</span>
              <input className="input" value={allergies} onChange={(event) => setAllergies(event.target.value)} placeholder="例如：青霉素（皮疹）" />
            </label>
            <label className="block">
              <span className="label">既往史</span>
              <input className="input" value={history} onChange={(event) => setHistory(event.target.value)} placeholder="例如：高血压 8 年" />
            </label>
            <label className="block sm:col-span-2">
              <span className="label">想问医生的问题（选填）</span>
              <textarea className="input min-h-[80px]" value={questions} onChange={(event) => setQuestions(event.target.value)} placeholder="例如：需要做冠脉 CT 吗？" />
            </label>
          </div>
          <button onClick={submit} disabled={submitting} className="btn-primary mt-4">
            {submitting ? <Loader2 size={15} className="animate-spin" /> : <CheckCircle2 size={15} />}
            提交就诊准备
          </button>
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">最近提交</h2>
            <span className="text-[11px] text-muted">{loading || failed ? "—" : forms.length + " 条"}</span>
          </div>
          <div className="mt-3 space-y-2">
            {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loading && failed ? <StateBlock kind="error" onRetry={load} /> : null}
            {!loading && !failed && forms.length === 0 ? (
              <StateBlock kind="empty" title="还没有提交过" description="填写上面的表单，医生接诊时就能看到" />
            ) : null}
            {forms.map((item) => (
              <div key={item.id} className="rounded-2xl bg-canvas p-3">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-[13px] font-medium text-ink">{item.chief_complaint}</p>
                  <span className="shrink-0 text-[11px] text-muted">{relativeTime(item.created_at)}</span>
                </div>
                {item.questions ? <p className="mt-1 truncate text-[11px] text-muted">想问：{item.questions}</p> : null}
              </div>
            ))}
          </div>
          <p className="mt-3 text-[11px] leading-5 text-muted">
            就诊准备属于平台数据；医生端与患者端共用同一份，不需要重复填写。
          </p>
        </section>

        <p className="text-center text-[11px] text-muted">
          <Link href="/" className="text-primary hover:underline">返回首页</Link>
        </p>
      </div>
    </AppShell>
  );
}

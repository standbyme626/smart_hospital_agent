"use client";

import { useEffect, useState } from "react";
import { LifeBuoy, Loader2 } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { createFeedback, getFeedback, type FeedbackTicket } from "@/lib/api";

const CATEGORIES = ["服务态度", "排队等候", "费用问题", "隐私疑虑", "其他"];
const FAQ = [
  { q: "号源显示已满，还能约到吗？", a: "号源由医院系统实时返回；已满时可关注退号，或选择其他医生。" },
  { q: "付费后多久生效？能退吗？", a: "以医院收费记录为准；未就诊可申请取消，退款由运营台处理。" },
  { q: "报告解读准不准？能当诊断吗？", a: "解读只做说明与就医建议，不构成诊断；以医生判断为准。" },
  { q: "我的数据会被谁看到？", a: "仅本人与已授权的家庭成员可见；医生只能看当班患者。" },
];

export default function SupportPage() {
  const [items, setItems] = useState<FeedbackTicket[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [category, setCategory] = useState(CATEGORIES[0]);
  const [content, setContent] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getFeedback());
    } catch {
      setItems([]);
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function submit() {
    if (!content.trim()) return;
    setSubmitting(true);
    setSubmitted(false);
    try {
      await createFeedback({ category, content: content.trim() });
      setContent("");
      setSubmitted(true);
      await load();
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <AppShell title="客服与反馈" subtitle="常见问题、提交反馈与投诉" actions={<SourceBadge live />}>
      <div className="readable space-y-4">
        <OfflineBanner />

        <section className="card p-5">
          <div className="flex items-center gap-2 text-primary"><LifeBuoy size={18} /><h2 className="text-[16px] font-semibold">常见问题</h2></div>
          <ul className="mt-3 space-y-2">
            {FAQ.map((item) => (
              <li key={item.q} className="rounded-2xl bg-canvas px-3 py-2">
                <p className="text-[13px] font-medium text-ink">{item.q}</p>
                <p className="mt-0.5 text-[12px] leading-5 text-muted">{item.a}</p>
              </li>
            ))}
          </ul>
        </section>

        <section className="card p-5">
          <h2 className="section-title">提交反馈 / 投诉</h2>
          <div className="mt-3 flex flex-wrap gap-2">
            {CATEGORIES.map((item) => (
              <button key={item} onClick={() => setCategory(item)} className={category === item ? "chip border-primary bg-primary text-white" : "chip hover:border-primary/40 hover:text-primary"}>{item}</button>
            ))}
          </div>
          <textarea className="input mt-3 min-h-[96px]" value={content} onChange={(event) => setContent(event.target.value)} placeholder="请描述发生了什么、时间与地点…" />
          <button onClick={submit} disabled={submitting || !content.trim()} className="btn-primary mt-3 disabled:opacity-50">
            {submitting ? <Loader2 size={15} className="animate-spin" /> : null}
            提交
          </button>
          {submitted ? <p className="mt-2 text-[12px] text-emerald-600">已提交，24 小时内受理，7 个工作日内答复。</p> : null}
        </section>

        <section className="card p-5">
          <div className="flex items-center justify-between">
            <h2 className="section-title">我的反馈</h2>
            <span className="text-[11px] text-muted">{loading || failed ? "—" : items.length + " 条"}</span>
          </div>
          <div className="mt-3 space-y-2">
            {loading ? <div className="h-16 animate-pulse rounded-2xl bg-canvas" /> : null}
            {!loading && failed ? <StateBlock kind="error" onRetry={load} /> : null}
            {!loading && !failed && items.length === 0 ? (
              <StateBlock kind="empty" title="还没有提交过反馈" description="提交后会显示受理状态" />
            ) : null}
            {items.map((item) => (
              <div key={item.id} className="rounded-2xl bg-canvas p-3">
                <div className="flex items-center gap-2">
                  <span className="badge">{item.category}</span>
                  <span className={item.status === "submitted" ? "badge bg-amber-50 text-amber-700" : "badge bg-emerald-50 text-emerald-700"}>
                    {item.status === "submitted" ? "已受理" : item.status}
                  </span>
                  <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 10)}</span>
                </div>
                <p className="mt-1 text-[12px] leading-5 text-muted">{item.content}</p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </AppShell>
  );
}

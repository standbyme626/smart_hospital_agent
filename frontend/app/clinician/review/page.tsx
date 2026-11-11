"use client";

import { useCallback, useEffect, useState } from "react";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import {
  getPrescriptionReviews,
  reviewOrder,
  type PrescriptionReview,
} from "@/lib/api-clinician-ext";

type Filter = "pending" | "approved" | "rejected" | "all";

const FILTERS: Array<{ key: Filter; label: string }> = [
  { key: "pending", label: "待审" },
  { key: "approved", label: "通过" },
  { key: "rejected", label: "退回" },
  { key: "all", label: "全部" },
];

function formatTime(value: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", { hour12: false });
}

export default function ClinicianReviewPage() {
  const [filter, setFilter] = useState<Filter>("pending");
  const [items, setItems] = useState<PrescriptionReview[]>([]);
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [hisNote, setHisNote] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [denied, setDenied] = useState(false);

  const load = useCallback(async (target: Filter) => {
    setLoading(true);
    setError(null);
    setDenied(false);
    try {
      const data = await getPrescriptionReviews(target === "all" ? undefined : target);
      setItems(data.items);
      setSummary(data.summary);
      setHisNote(data.his_note);
    } catch (err) {
      const status = (err as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setError((err as Error).message || "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(filter);
  }, [filter, load]);

  const decide = async (orderId: string, status: "approved" | "rejected") => {
    setError(null);
    setMessage(null);
    try {
      await reviewOrder(orderId, status, status === "approved" ? "审方通过" : "需调整");
      setMessage(status === "approved" ? "已记录审方通过" : "已退回，请医生调整医嘱");
      await load(filter);
    } catch (err) {
      setError((err as Error).message || "审方失败");
    }
  };

  return (
    <ConsoleShell end="clinician" title="药师审方" subtitle="已签名医嘱的审方状态；HIS 审方记录待院方开通">

      <div className="mt-3"><OfflineBanner /></div>

      {error ? <p className="mt-3 rounded-2xl bg-rose-50 px-4 py-2 text-[12px] text-rose-700">{error}</p> : null}
      {message ? <p className="mt-3 rounded-2xl bg-primary-soft px-4 py-2 text-[12px] text-primary">{message}</p> : null}

      <p className="mt-3 rounded-2xl bg-amber-50 px-4 py-2 text-[12px] text-amber-700">
        {hisNote || "HIS 药师审方记录待院方开通；当前为平台侧审方状态。"} 平台不替代药师判断，只做记录与提醒。
      </p>

      <div className="mt-3 flex flex-wrap gap-2">
        {FILTERS.map((item) => (
          <button
            key={item.key}
            onClick={() => setFilter(item.key)}
            className={
              filter === item.key
                ? "rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white"
                : "rounded-full bg-white px-4 py-1.5 text-[12px] text-ink ring-1 ring-line"
            }
          >
            {item.label}
            {item.key !== "all" ? <span className="ml-1 text-[11px] opacity-70">{summary[item.key] ?? 0}</span> : null}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          {[0, 1, 2].map((i) => <div key={i} className="card h-24 animate-pulse bg-canvas" />)}
        </div>
      ) : denied ? (
        <div className="mt-4 card p-5"><StateBlock kind="denied" title="需要医生权限" /></div>
      ) : (
        <section className="mt-4 card overflow-hidden">
          {items.length === 0 ? (
            <div className="p-4">
              <StateBlock kind="empty" title="没有待审医嘱" description="医生签名后的医嘱会出现在这里" />
            </div>
          ) : (
            <ul className="divide-y divide-line">
              {items.map((item) => (
                <li key={item.id} className="px-4 py-3">
                  <div className="flex flex-wrap items-center gap-2 text-[12px]">
                    <span
                      className={
                        item.review_status === "approved"
                          ? "badge bg-emerald-50 text-emerald-700"
                          : item.review_status === "rejected"
                            ? "badge bg-rose-50 text-rose-700"
                            : "badge bg-amber-50 text-amber-700"
                      }
                    >
                      {item.review_status === "approved" ? "审方通过" : item.review_status === "rejected" ? "已退回" : "待审"}
                    </span>
                    <span className="font-medium text-ink">{item.name}</span>
                    <span className="text-muted">{item.dose || ""} {item.usage || ""}</span>
                    <span className="badge bg-canvas text-muted">{item.member_name || item.member_id}</span>
                    <span className="ml-auto text-[11px] text-muted">{item.doctor || "—"} · {formatTime(item.signed_at)}</span>
                  </div>
                  {item.review_note ? <p className="mt-1 text-[11px] text-muted">审方意见：{item.review_note}</p> : null}
                  {item.review_status === "pending" ? (
                    <div className="mt-2 flex gap-3 text-[12px]">
                      <button onClick={() => void decide(item.id, "approved")} className="font-medium text-emerald-600">
                        审方通过
                      </button>
                      <button onClick={() => void decide(item.id, "rejected")} className="font-medium text-rose-600">
                        退回调整
                      </button>
                    </div>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </ConsoleShell>
  );
}

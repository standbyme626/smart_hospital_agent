"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ShieldCheck } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { decideOpsReview, getOpsReviews, type OpsReview } from "@/lib/api";

export default function OpsReviewsPage() {
  const [items, setItems] = useState<OpsReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [denied, setDenied] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setFailed(false);
    setDenied(false);
    try {
      setItems(await getOpsReviews());
    } catch (error) {
      const status = (error as { status?: number } | null)?.status;
      if (status === 401 || status === 403) setDenied(true);
      else setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function decide(id: string, action: "approve" | "reject") {
    setBusy(id + action);
    try {
      await decideOpsReview(id, action, action === "approve" ? "人工放行" : "人工驳回");
      await load();
    } finally {
      setBusy(null);
    }
  }

  return (
    <ConsoleShell end="ops" title="运营 / 质控台" subtitle="支付与异常逐笔审核">
      <div className="space-y-4">
        {denied ? <div className="card p-3"><StateBlock kind="denied" title="当前账号无运营权限" /></div> : loading ? (
          <div className="card h-24 animate-pulse" />
        ) : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="没有待审核事项" description="正常支付自动放行，不进入人工队列" /></div>
        ) : items.map((item) => (
          <section key={item.id} className="card p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="badge">{item.kind}</span>
              <span className="text-[13px] font-medium text-ink">单号 {item.ref_id}</span>
              <span className="badge bg-amber-50 text-amber-700">{item.status}</span>
              <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 16).replace("T", " ")}</span>
            </div>
            <p className="mt-1 text-[11px] text-muted">申请人 {item.user_id} · 就诊人 {item.member_id || "—"}</p>
            <div className="mt-3 flex gap-2">
              <button onClick={() => decide(item.id, "approve")} disabled={busy !== null} className="rounded-full bg-emerald-600 px-4 py-1.5 text-[12px] font-medium text-white disabled:opacity-50">批准</button>
              <button onClick={() => decide(item.id, "reject")} disabled={busy !== null} className="rounded-full bg-white px-4 py-1.5 text-[12px] font-medium text-rose-600 ring-1 ring-rose-200 disabled:opacity-50">驳回</button>
            </div>
          </section>
        ))}
        <p className="px-1 text-[11px] leading-5 text-muted">仅异常进入人工：退款/改价、金额不一致、重复支付、超阈值、对账差异。</p>
      </div>
    </ConsoleShell>
  );
}

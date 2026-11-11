"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Send } from "lucide-react";
import { OpsNav } from "@/components/OpsNav";
import { StateBlock } from "@/components/ui";
import { getOpsNotifications, retryOpsNotification, type OpsDelivery } from "@/lib/api";

export default function OpsNotificationsPage() {
  const [items, setItems] = useState<OpsDelivery[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getOpsNotifications());
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function retry(id: string) {
    setBusy(id);
    try {
      await retryOpsNotification(id);
      await load();
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="border-b border-line bg-white/90">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <Send size={18} className="text-primary" />
          <div><p className="text-[15px] font-semibold text-ink">通知记录</p><p className="text-[11px] text-muted">站内信发送与重试（短信/微信待开通）</p></div>
          <Link href="/" className="ml-auto inline-flex items-center gap-1 rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary"><ArrowLeft size={12} /> 患者端</Link>
        </div>
        <div className="mx-auto max-w-5xl px-4 pb-3"><OpsNav /></div>
      </header>
      <main className="mx-auto max-w-5xl space-y-3 px-4 py-5">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div> : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="没有发送记录" /></div>
        ) : (
          <section className="card divide-y divide-line">
            {items.map((item) => (
              <div key={item.id} className="flex flex-wrap items-center gap-2 px-4 py-3">
                <span className="badge">{item.channel}</span>
                <span className={item.status === "sent" ? "badge bg-emerald-50 text-emerald-700" : "badge bg-amber-50 text-amber-700"}>{item.status === "sent" ? "已发送" : "失败"}</span>
                <span className="text-[11px] text-muted">尝试 {item.attempts} 次</span>
                {item.error ? <span className="min-w-0 flex-1 truncate text-[11px] text-rose-600">{item.error}</span> : null}
                <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 16).replace("T", " ")}</span>
                {item.status !== "sent" ? (
                  <button onClick={() => retry(item.id)} disabled={busy === item.id} className="rounded-full bg-primary px-3 py-1 text-[11px] font-medium text-white disabled:opacity-50">重试</button>
                ) : null}
              </div>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}

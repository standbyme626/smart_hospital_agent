"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, Send } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
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
    <ConsoleShell end="ops" title="通知记录" subtitle="站内信发送与重试（短信/微信待开通）">
      <div className="space-y-4">
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
      </div>
    </ConsoleShell>
  );
}

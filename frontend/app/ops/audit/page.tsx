"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ScrollText } from "lucide-react";
import { OpsNav } from "@/components/OpsNav";
import { StateBlock } from "@/components/ui";
import { getOpsAudit, type OpsAuditEntry } from "@/lib/api";

export default function OpsAuditPage() {
  const [items, setItems] = useState<OpsAuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getOpsAudit());
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="min-h-dvh bg-canvas">
      <header className="border-b border-line bg-white/90">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-4 py-3">
          <ScrollText size={18} className="text-primary" />
          <div><p className="text-[15px] font-semibold text-ink">审计复盘</p><p className="text-[11px] text-muted">最近 50 条审计事件</p></div>
          <Link href="/" className="ml-auto inline-flex items-center gap-1 rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary"><ArrowLeft size={12} /> 患者端</Link>
        </div>
        <div className="mx-auto max-w-5xl px-4 pb-3"><OpsNav /></div>
      </header>
      <main className="mx-auto max-w-5xl space-y-3 px-4 py-5">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="还没有审计事件" /></div>
        ) : (
          <section className="card divide-y divide-line">
            {items.map((item) => (
              <div key={item.id} className="flex flex-wrap items-center gap-2 px-4 py-2.5">
                <span className="badge">{item.action}</span>
                <span className="text-[12px] text-muted">{item.actor || "system"}</span>
                <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 19).replace("T", " ")}</span>
              </div>
            ))}
          </section>
        )}
        <p className="px-1 text-[11px] leading-5 text-muted">详情已脱敏；完整回放与导出在后续版本开放。</p>
      </main>
    </div>
  );
}

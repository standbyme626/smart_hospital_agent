"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ScrollText } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
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
    <ConsoleShell end="ops" title="审计复盘" subtitle="最近 50 条审计事件">
      <div className="space-y-4">
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
      </div>
    </ConsoleShell>
  );
}

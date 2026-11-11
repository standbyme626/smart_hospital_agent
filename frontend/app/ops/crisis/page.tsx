"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ShieldAlert } from "lucide-react";
import { StateBlock } from "@/components/ui";
import { ConsoleShell } from "@/components/ConsoleShell";
import { getOpsCrisis, resolveOpsCrisis, type OpsCrisisEvent } from "@/lib/api";

export default function OpsCrisisPage() {
  const [items, setItems] = useState<OpsCrisisEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      setItems(await getOpsCrisis());
    } catch {
      setFailed(true);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function resolve(id: string) {
    setBusy(id);
    try {
      await resolveOpsCrisis(id);
      await load();
    } finally {
      setBusy(null);
    }
  }

  return (
    <ConsoleShell end="ops" title="危机事件" subtitle="红旗信号与人工介入记录">
      <div className="space-y-4">
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div> : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="没有危机事件" /></div>
        ) : items.map((item) => (
          <section key={item.id} className="card p-4">
            <div className="flex flex-wrap items-center gap-2">
              <span className="badge bg-rose-50 text-rose-700">{item.kind}</span>
              <span className={item.status === "open" ? "badge bg-amber-50 text-amber-700" : "badge bg-emerald-50 text-emerald-700"}>{item.status === "open" ? "待处理" : "已处理"}</span>
              <span className="ml-auto text-[11px] text-muted">{item.created_at?.slice(0, 16).replace("T", " ")}</span>
            </div>
            <p className="mt-2 text-[13px] text-ink">{item.detail || "无详情"}</p>
            {item.status === "open" ? (
              <button onClick={() => resolve(item.id)} disabled={busy === item.id} className="mt-3 rounded-full bg-primary px-4 py-1.5 text-[12px] font-medium text-white disabled:opacity-50">标记已处理</button>
            ) : (
              <p className="mt-2 text-[11px] text-muted">处理人：{item.handled_by || "—"}</p>
            )}
          </section>
        ))}
      </div>
    </ConsoleShell>
  );
}

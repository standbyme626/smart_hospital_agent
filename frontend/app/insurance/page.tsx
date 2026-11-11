"use client";

import { useEffect, useState } from "react";
import { ReceiptText } from "lucide-react";
import { AppShell } from "@/components/AppShell";
import { OfflineBanner, SourceBadge, StateBlock } from "@/components/ui";
import { getSettlements, type SettlementItem } from "@/lib/api";

function money(value: number | string): string {
  const n = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(n) ? "¥" + n.toFixed(2) : "—";
}

export default function InsurancePage() {
  const [items, setItems] = useState<SettlementItem[]>([]);
  const [source, setSource] = useState("");
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      const data = await getSettlements();
      setItems(data.items);
      setSource(data.source || "");
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

  return (
    <AppShell title="医保结算" subtitle="挂号费结算明细" actions={<SourceBadge live />}>
      <div className="mx-auto max-w-3xl space-y-4">
        <OfflineBanner />
        {source ? <p className="rounded-2xl bg-amber-50 px-3 py-2 text-[12px] text-amber-800">{source}</p> : null}
        {loading ? <div className="card h-24 animate-pulse" /> : failed ? (
          <div className="card p-3"><StateBlock kind="error" onRetry={load} /></div>
        ) : items.length === 0 ? (
          <div className="card p-3"><StateBlock kind="empty" title="暂无结算记录" description="挂号缴费后，医保结算明细会显示在这里" /></div>
        ) : (
          items.map((item) => (
            <section key={item.id} className="card p-5">
              <div className="flex items-center gap-2 text-primary">
                <ReceiptText size={18} />
                <h2 className="text-[16px] font-semibold">挂号费结算</h2>
                <span className="ml-auto text-[11px] text-muted">{item.settled_at?.slice(0, 10)}</span>
              </div>
              <div className="mt-4 grid gap-2 sm:grid-cols-4">
                {[
                  { label: "总额", value: item.total },
                  { label: "医保统筹", value: item.insurance },
                  { label: "个人账户", value: item.account },
                  { label: "个人自付", value: item.self_pay },
                ].map((row) => (
                  <div key={row.label} className="rounded-2xl bg-canvas p-3">
                    <p className="text-[11px] text-muted">{row.label}</p>
                    <p className="mt-1 text-[16px] font-semibold text-ink">{money(row.value)}</p>
                  </div>
                ))}
              </div>
              <p className="mt-3 text-[11px] text-muted">结算状态：{item.status === "settled" ? "已结算" : item.status}</p>
            </section>
          ))
        )}
        <p className="text-center text-[11px] leading-5 text-muted">医保实时结算需院方与医保专网开通；当前为平台模拟结算数据。</p>
      </div>
    </AppShell>
  );
}

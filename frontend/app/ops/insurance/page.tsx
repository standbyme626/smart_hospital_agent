"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, ReceiptText } from "lucide-react";
import { OpsNav } from "@/components/OpsNav";
import { StateBlock } from "@/components/ui";
import { getOpsInsuranceRecon, type OpsInsuranceItem } from "@/lib/api";

export default function OpsInsurancePage() {
  const [items, setItems] = useState<OpsInsuranceItem[]>([]);
  const [diffCount, setDiffCount] = useState(0);
  const [checked, setChecked] = useState(0);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);

  async function load() {
    setLoading(true);
    setFailed(false);
    try {
      const data = await getOpsInsuranceRecon();
      setItems(data.items);
      setDiffCount(data.diff_count);
      setChecked(data.checked);
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
          <ReceiptText size={18} className="text-primary" />
          <div><p className="text-[15px] font-semibold text-ink">医保对账</p><p className="text-[11px] text-muted">结算快照 vs 挂号金额</p></div>
          <Link href="/" className="ml-auto inline-flex items-center gap-1 rounded-full bg-white px-3 py-1 text-[11px] text-muted ring-1 ring-line hover:text-primary"><ArrowLeft size={12} /> 患者端</Link>
        </div>
        <div className="mx-auto max-w-5xl px-4 pb-3"><OpsNav /></div>
      </header>
      <main className="mx-auto max-w-5xl space-y-3 px-4 py-5">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="card p-4"><p className="text-[12px] text-muted">核对笔数</p><p className="mt-1 text-[24px] font-semibold text-ink">{checked}</p></div>
          <div className="card p-4"><p className="text-[12px] text-muted">差异笔数</p><p className={diffCount ? "mt-1 text-[24px] font-semibold text-rose-600" : "mt-1 text-[24px] font-semibold text-emerald-600"}>{diffCount}</p></div>
          <div className="card p-4"><p className="text-[12px] text-muted">说明</p><p className="mt-1 text-[12px] leading-5 text-muted">医保为平台模拟结算；真实医保联网待院方开通。</p></div>
        </div>
        <section className="card overflow-hidden">
          {loading ? <div className="h-24 animate-pulse" /> : failed ? <div className="p-3"><StateBlock kind="error" onRetry={load} /></div> : items.length === 0 ? (
            <div className="p-3"><StateBlock kind="empty" title="没有结算记录" /></div>
          ) : (
            <table className="w-full text-left text-[12px]">
              <thead className="bg-canvas text-muted"><tr><th className="px-3 py-2">结算单</th><th className="px-3 py-2">结算金额</th><th className="px-3 py-2">挂号金额</th><th className="px-3 py-2">差异</th><th className="px-3 py-2">时间</th></tr></thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-t border-line">
                    <td className="px-3 py-2 text-ink">{item.id}</td>
                    <td className="px-3 py-2 text-muted">{Number(item.settlement_total).toFixed(2)}</td>
                    <td className="px-3 py-2 text-muted">{Number(item.appointment_amount).toFixed(2)}</td>
                    <td className={Math.abs(item.diff) > 0.01 ? "px-3 py-2 text-rose-600" : "px-3 py-2 text-emerald-600"}>{item.diff.toFixed(2)}</td>
                    <td className="px-3 py-2 text-muted">{item.settled_at?.slice(0, 10)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      </main>
    </div>
  );
}
